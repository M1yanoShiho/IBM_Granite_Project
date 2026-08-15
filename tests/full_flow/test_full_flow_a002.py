from __future__ import annotations

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
from evidence_rag.generator.trace import GeneratorTrace, unavailable_draft_trace
from evidence_rag.infrastructure.datasets import GoldCase

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import full_flow_a002 as a002  # noqa: E402
import full_flow_joint as joint  # noqa: E402


def _candidate(index: int) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=f"e{index}",
        document_id=f"d{index}",
        chunk_id=f"c{index}",
        text=f"evidence {index}",
        source_uri=f"fixture://{index}",
        retrieval_score=float(11 - index),
        retrieval_rank=index,
    )


def _cases() -> tuple[joint.JointCase, ...]:
    topk = tuple(_candidate(index) for index in range(1, 11))
    return (
        joint.JointCase("q1", "question 1", "component-1", topk, topk),
        joint.JointCase("q2", "question 2", "component-2", topk, topk[:-1]),
    )


class FakeTracedGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.last_trace: GeneratorTrace | None = None

    def generate(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> GenerationResult:
        del checklist
        ids = tuple(item.evidence_id for item in selected.evidence)
        self.calls.append((query.query_id, ids))
        answer = "right" if len(ids) == 9 else "wrong"
        result = GenerationResult(
            query_id=query.query_id,
            answer=answer,
            cited_evidence_ids=(ids[0],),
        )
        self.last_trace = GeneratorTrace(
            query_id=query.query_id,
            selected_evidence_ids=ids,
            draft=unavailable_draft_trace(answer),
            claims=(),
            final_answer=answer,
            final_cited_evidence_ids=result.cited_evidence_ids,
        )
        return result


def test_repeat_selection_is_stratified_and_deterministic() -> None:
    base = _cases()
    cases = tuple(
        joint.JointCase(
            f"{case.query_id}-{index}",
            case.question,
            f"{case.component_id}-{index}",
            case.topk10,
            case.selected,
        )
        for index in range(5)
        for case in base
    )
    first = a002.choose_repeat_ids(cases, fraction=0.4)
    second = a002.choose_repeat_ids(cases, fraction=0.4)

    assert first == second
    assert len(first) == 4
    assert sum(query_id.startswith("q1") for query_id in first) == 2
    assert sum(query_id.startswith("q2") for query_id in first) == 2


def test_repeat_pass_reverses_arm_order_and_saves_per_query_trace() -> None:
    generator = FakeTracedGenerator()
    rows = a002.run_cases(_cases(), generator=generator, repeat_ids={"q1"})

    assert rows[0]["repeat_arm_order"] == list(reversed(rows[0]["primary_arm_order"]))
    assert len(rows[0]["arms"][a002.ARMS[0]]) == 2
    assert len(rows[0]["arms"][a002.ARMS[1]]) == 2
    assert rows[0]["arms"][a002.ARMS[0]][0]["trace"]["query_id"] == "q1"
    assert len(rows[1]["arms"][a002.ARMS[0]]) == 1


def test_score_separates_selector_effect_from_repeat_stability() -> None:
    rows = a002.run_cases(
        _cases(),
        generator=FakeTracedGenerator(),
        repeat_ids={"q1", "q2"},
    )
    gold = {
        query_id: GoldCase(
            query_id=query_id,
            relevant_document_ids=("d1",),
            reference_answers=("right",),
        )
        for query_id in ("q1", "q2")
    }

    report = a002.score_rows(rows, gold)

    assert report["selector_changed"]["L_minus_K_answer"]["delta"] == pytest.approx(1.0)
    assert report["same_input_repeat"][a002.ARMS[0]]["exact_answer"] == 2
    assert report["same_input_cross_arm_unchanged"]["exact_answer"] == 1


def test_runtime_command_does_not_accept_gold() -> None:
    with pytest.raises(SystemExit):
        a002._parser().parse_args(
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
                "--granite-snapshot",
                "granite",
                "--true-snapshot",
                "true",
                "--output-dir",
                "out",
                "--gold",
                "gold.jsonl",
            ]
        )


def test_git_identity_is_resolved_from_the_repository() -> None:
    assert len(a002._git_commit()) == 40
