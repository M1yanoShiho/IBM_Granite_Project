from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import g3_baseline_comparison as g3  # noqa: E402, I001
import generator_a1_a2_ablation as ablation  # noqa: E402, I001
from evidence_rag.contracts.models import EvidenceCandidate, SelectedEvidenceSet  # noqa: E402


class PromptAwareLLM:
    def generate(self, prompt: str) -> str:
        if prompt.startswith("Answer the question"):
            if "one independently verifiable factual claim" in prompt:
                return "Acme raised revenue. Acme launched Product X."
            return "Acme raised revenue and launched Product X."
        if prompt.startswith("Split the answer"):
            answer = prompt.split("\n\nAnswer:\n", maxsplit=1)[1]
            return json.dumps(
                {
                    "claims": [
                        {"source_text": answer, "text": answer},
                    ]
                }
            )
        if prompt.startswith("Check whether"):
            items = json.loads(prompt.split("\n\nItems:\n", maxsplit=1)[1])
            return json.dumps(
                {
                    "results": [
                        {"claim_id": item["claim_id"], "faithful": True}
                        for item in items
                    ]
                }
            )
        raise AssertionError(f"unexpected prompt: {prompt[:80]}")


class QueueLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)

    def generate(self, prompt: str) -> str:
        return next(self.responses)


def a_case() -> g3.G3Case:
    selected = SelectedEvidenceSet(
        query_id="q1",
        evidence=(
            EvidenceCandidate(
                evidence_id="e1",
                document_id="d1",
                chunk_id="c1",
                text="Acme raised revenue and launched Product X.",
                source_uri="test://e1",
                retrieval_score=1.0,
                retrieval_rank=1,
            ),
        ),
    )
    return g3.G3Case(
        query_id="q1",
        question="What did Acme do?",
        required_facts=("Acme raised revenue.",),
        constraints=(),
        selected=selected,
        gold_answers=(("raised revenue",),),
    )


def test_git_head_falls_back_to_repository_metadata_without_git(
    tmp_path: Path,
) -> None:
    expected = "1" * 40
    git_dir = tmp_path / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "refs" / "heads" / "main").write_text(
        f"{expected}\n", encoding="utf-8"
    )

    with (
        patch.dict("os.environ", {}, clear=True),
        patch.object(ablation.subprocess, "run", side_effect=FileNotFoundError),
    ):
        assert ablation.git_head(tmp_path) == expected


def test_git_head_can_be_supplied_by_environment(tmp_path: Path) -> None:
    expected = "a" * 40

    with patch.dict("os.environ", {"EXPERIMENT_GIT_HEAD": expected}, clear=True):
        assert ablation.git_head(tmp_path) == expected


def test_runner_pairs_each_a1_answer_with_both_a2_versions() -> None:
    recorder = ablation.RecordingLLM(PromptAwareLLM())

    outputs, summary = ablation.run_cases([a_case()], recorder)

    by_arm = {row["arm"]: row for row in outputs}
    assert set(by_arm) == set(ablation.ARMS)
    assert by_arm["old_old"]["answer_sha256"] == by_arm["old_new"]["answer_sha256"]
    assert by_arm["new_old"]["answer_sha256"] == by_arm["new_new"]["answer_sha256"]
    assert by_arm["old_old"]["answer_sha256"] != by_arm["new_old"]["answer_sha256"]
    assert summary["cases"] == 1
    assert summary["records"] == 4
    assert len(recorder.records) == 10  # two A1 calls plus split+faithfulness for four arms


def test_new_splitter_can_anchor_two_paraphrased_claims_to_one_sentence() -> None:
    answer = "Pfizer raised revenue and launched Product X."
    split = json.dumps(
        {
            "claims": [
                {"source_text": "Revenue grew at Pfizer.", "text": "Pfizer raised revenue."},
                {
                    "source_text": "Pfizer introduced Product X.",
                    "text": "Pfizer launched Product X.",
                },
            ]
        }
    )
    old_faithful = json.dumps(
        {"results": [{"claim_id": "claim-1", "faithful": True}]}
    )
    new_faithful = json.dumps(
        {
            "results": [
                {"claim_id": "claim-1", "faithful": True},
                {"claim_id": "claim-2", "faithful": True},
            ]
        }
    )
    old_llm = ablation.RecordingLLM(QueueLLM([split, old_faithful]))
    new_llm = ablation.RecordingLLM(QueueLLM([split, new_faithful]))

    old_claims = ablation.HistoricalClaimSplitter(old_llm, "old").split(
        "q1", answer, "old_old"
    )
    new_claims = ablation.HistoricalClaimSplitter(new_llm, "new").split(
        "q1", answer, "old_new"
    )

    assert len(old_claims) == 1
    assert len(new_claims) == 2
    assert all(answer[claim.span.start : claim.span.end] == answer for claim in new_claims)


def test_new_splitter_checks_faithfulness_against_real_answer_span() -> None:
    answer = "Pfizer's revenue increased by eight percent this year."
    split = json.dumps(
        {
            "claims": [
                {
                    "source_text": "Revenue rose 8%.",
                    "text": "Pfizer revenue rose by 8% this year.",
                }
            ]
        }
    )
    faithful = json.dumps(
        {"results": [{"claim_id": "claim-1", "faithful": True}]}
    )
    llm = ablation.RecordingLLM(QueueLLM([split, faithful]))

    ablation.HistoricalClaimSplitter(llm, "new").split("q1", answer, "old_new")

    faithfulness_prompt = llm.records[1]["prompt"]
    assert answer in faithfulness_prompt
    assert '"source_text": "Revenue rose 8%."' not in faithfulness_prompt


def test_blind_packet_does_not_expose_arm_or_query_id() -> None:
    outputs = [
        {
            "query_id": "q-secret",
            "question": "What happened?",
            "arm": "new_new",
            "a1_version": "new",
            "answer": "Acme grew.",
            "claims": [
                {
                    "text": "Acme grew.",
                    "source_span_text": "Acme grew.",
                }
            ],
        }
    ]

    rows, key = ablation.build_blind_packet(outputs, seed=13)

    assert len(rows) == 3
    assert all("arm" not in row and "query_id" not in row for row in rows)
    assert {item["arm"] for item in key} == {"new", "new_new"}
    assert {item["query_id"] for item in key} == {"q-secret"}
    assert {row["unit_type"] for row in rows} == {"a1_sentence", "a2_claim", "a2_case"}
