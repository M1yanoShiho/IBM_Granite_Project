from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from evidence_rag.contracts.models import (
    EvidenceCandidate,
    GenerationResult,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
)
from evidence_rag.infrastructure.datasets import GoldCase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_flow_joint as joint  # noqa: E402


class FakeGenerator:
    def __init__(self, *, advanced: bool = False) -> None:
        self.advanced = advanced
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.last_routings: list[object] = []

    def generate(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> GenerationResult:
        del checklist
        ids = tuple(item.evidence_id for item in selected.evidence)
        self.calls.append((query.query_id, ids))
        answer = "right" if self.advanced or len(ids) == 9 else "wrong"
        return GenerationResult(
            query_id=query.query_id,
            answer=answer,
            cited_evidence_ids=(ids[0],),
        )


def _candidate(index: int) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=f"e{index}",
        document_id=f"d{index}",
        chunk_id=f"c{index}",
        text=f"evidence {index}",
        source_uri=f"memory://{index}",
        retrieval_score=float(11 - index),
        retrieval_rank=index,
    )


def _cases() -> tuple[joint.JointCase, ...]:
    topk = tuple(_candidate(index) for index in range(1, 11))
    return (
        joint.JointCase("q1", "question 1", "component-1", topk, topk),
        joint.JointCase("q2", "question 2", "component-2", topk, topk[:-1]),
    )


def test_unchanged_context_reuses_both_generator_results() -> None:
    basic = FakeGenerator()
    advanced = FakeGenerator(advanced=True)
    rows = joint.run_cases(_cases(), basic_generator=basic, verify_generator=advanced)

    assert len(basic.calls) == 3
    assert len(advanced.calls) == 3
    assert rows[0]["arms"][joint.ARMS[1]]["reused_from"] == joint.ARMS[0]
    assert rows[0]["arms"][joint.ARMS[3]]["reused_from"] == joint.ARMS[2]
    assert rows[1]["arms"][joint.ARMS[1]]["reused_from"] is None
    assert rows[1]["arms"][joint.ARMS[3]]["reused_from"] is None


def test_score_reports_selector_effect_and_interaction() -> None:
    rows = joint.run_cases(
        _cases(),
        basic_generator=FakeGenerator(),
        verify_generator=FakeGenerator(advanced=True),
    )
    gold = {
        "q1": GoldCase(
            query_id="q1", relevant_document_ids=("d1",), reference_answers=("right",)
        ),
        "q2": GoldCase(
            query_id="q2", relevant_document_ids=("d1",), reference_answers=("right",)
        ),
    }
    report = joint.score_rows(rows, gold)

    changed = report["selector_changed"]
    assert changed["queries"] == 1
    assert changed["answer_comparisons"]["B_minus_A"]["delta"] == pytest.approx(1.0)
    assert changed["answer_comparisons"]["D_minus_C"]["delta"] == pytest.approx(0.0)
    assert changed["answer_comparisons"]["interaction"]["delta"] == pytest.approx(-1.0)


def test_runtime_parser_has_no_gold_argument() -> None:
    parser = joint._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--queries",
                "q.jsonl",
                "--candidate-pool",
                "p.jsonl",
                "--roles",
                "r.jsonl",
                "--decision-trace",
                "d.jsonl",
                "--gold",
                "g.jsonl",
                "--granite-snapshot",
                "model",
                "--output-dir",
                "out",
            ]
        )


def test_load_runtime_cases_rejects_selector_evidence_outside_topk(tmp_path: Path) -> None:
    queries = tmp_path / "queries.jsonl"
    pools = tmp_path / "pool.jsonl"
    roles = tmp_path / "roles.jsonl"
    decisions = tmp_path / "decisions.jsonl"
    queries.write_text('{"query_id":"q","text":"question"}\n', encoding="utf-8")
    pools.write_text(
        json.dumps(
            {
                "query_id": "q",
                "candidates": [item.model_dump(mode="json") for item in _cases()[0].topk10],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    roles.write_text(
        '{"query_id":"q","component_id":"c","role":"decision-dev"}\n',
        encoding="utf-8",
    )
    decisions.write_text(
        '{"dataset_kind":"niah","query_id":"q","selected_evidence_ids":["outside"]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside TopK10"):
        joint.load_runtime_cases(
            queries_path=queries,
            candidate_pool_path=pools,
            roles_path=roles,
            decision_trace_path=decisions,
            require_expected_counts=False,
        )
