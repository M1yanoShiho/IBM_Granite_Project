from pathlib import Path
from types import SimpleNamespace

from evidence_rag.contracts.models import Query
from evidence_rag.evaluation.experiment05_generation import (
    answer_from_routings,
    build_system_output,
    prepare_prompt_evidence,
    prompt_template,
)
from evidence_rag.evaluation.experiment05_runtime import (
    PreparedArm,
    PreparedEvidence,
    PreparedQuery,
)
from evidence_rag.evaluation.experiment05_scorer import CANONICAL_ABSTENTION


class _Tokenizer:
    def __call__(self, text: str, **_kwargs: object) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text.split())))}


class _Llm:
    _tokenizer = _Tokenizer()

    def input_token_count(self, prompt: str) -> int:
        return len(prompt.split())


def _prepared() -> PreparedQuery:
    evidence = tuple(
        PreparedEvidence(
            evidence_id=f"e{index}",
            text=f"evidence text {index}",
            retrieval_score=float(10 - index),
            retrieval_rank=index,
        )
        for index in range(1, 4)
    )
    keep = PreparedArm(retrieved_evidence_ids=("e1", "e2", "e3"), selected=evidence)
    return PreparedQuery(
        dataset="kilt-nq",
        query_id="q1",
        arms={
            "bm25_rag": keep,
            "hybrid_rag": keep,
            "granite_rerank_rag": keep,
            "provence_rag": keep,
            "ours_seed13": keep,
            "ours_seed42": keep,
            "ours_seed73": keep,
            "ablation_bm25_retriever": keep,
            "ablation_no_selector": keep,
            "ablation_direct_generator": keep,
        },
        selector_traces={
            "hybrid": {
                "query_id": "q1",
                "threshold": 0.9,
                "status": "normal",
                "baseline_evidence_ids": ("e1", "e2", "e3"),
                "selected_evidence_ids": ("e1", "e2", "e3"),
                "dropped_evidence_ids": (),
                "decisions": tuple(
                    {
                        "evidence_id": f"e{index}",
                        "retrieval_rank": index,
                        "protect_score": 0.5,
                        "harm_score": 0.5,
                        "action": "KEEP",
                    }
                    for index in range(1, 4)
                ),
            },
            "bm25": {
                "query_id": "q1",
                "threshold": 0.9,
                "status": "normal",
                "baseline_evidence_ids": ("e1", "e2", "e3"),
                "selected_evidence_ids": ("e1", "e2", "e3"),
                "dropped_evidence_ids": (),
                "decisions": tuple(
                    {
                        "evidence_id": f"e{index}",
                        "retrieval_rank": index,
                        "protect_score": 0.5,
                        "harm_score": 0.5,
                        "action": "KEEP",
                    }
                    for index in range(1, 4)
                ),
            },
        },
    )


def test_presented_records_match_exact_prompt_prefix_and_hashes(tmp_path: Path) -> None:
    prepared = _prepared()
    query = Query(query_id="q1", text="question")
    template = prompt_template("kilt-nq")
    presented, selected_records, presented_records, prompt, prompt_tokens = prepare_prompt_evidence(
        prepared=prepared,
        arm_id="hybrid_rag",
        query=query,
        template=template,
        llm=_Llm(),
        artifact_root=tmp_path,
        max_input_tokens=158,
    )

    assert tuple(item.evidence_id for item in presented.evidence) == tuple(
        item.evidence_id for item in presented_records
    )
    assert prompt_tokens == len(prompt.split()) <= 158
    assert all((tmp_path / item.text_sha256[:2] / f"{item.text_sha256}.txt").is_file() for item in selected_records)


def test_routing_citations_are_inserted_before_sentence_punctuation() -> None:
    answer = answer_from_routings(
        (
            SimpleNamespace(sentence="Paris is in France.", outcome="verified", citation="e2"),
            SimpleNamespace(sentence="Unverified sentence.", outcome="unverified", citation=None),
        ),
        ("e1", "e2"),
    )

    assert answer == "Paris is in France [2]. Unverified sentence."


def test_review_annotation_stays_after_the_cited_sentence() -> None:
    answer = answer_from_routings(
        (
            SimpleNamespace(
                sentence="Paris is in France. [may warrant review]",
                outcome="verified",
                citation="e1",
            ),
        ),
        ("e1",),
    )

    assert answer == "Paris is in France [1]. [may warrant review]"


def test_empty_answer_uses_canonical_abstention(tmp_path: Path) -> None:
    prepared = _prepared()
    output = build_system_output(
        prepared=prepared,
        arm_id="hybrid_rag",
        answer="",
        runtime_error=None,
        selected_records=(),
        presented_records=(),
        prompt="prompt",
        prompt_tokens=1,
        model_fingerprints={"generator": "abc"},
        config_fingerprint="config",
        prompt_fingerprint="prompt",
    )

    assert output.abstained is True
    assert output.answer_text == CANONICAL_ABSTENTION
