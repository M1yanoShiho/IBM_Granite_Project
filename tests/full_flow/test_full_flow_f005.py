from __future__ import annotations

import sys
from pathlib import Path

from evidence_rag.contracts.models import EvidenceCandidate, GenerationResult
from evidence_rag.infrastructure.datasets import GoldCase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_flow_f005 as f005  # noqa: E402


def _candidate(evidence_id: str, document_id: str, rank: int) -> dict[str, object]:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=document_id,
        chunk_id=f"chunk-{rank}",
        text=f"text {rank}",
        source_uri=f"test://{document_id}",
        retrieval_score=float(11 - rank),
        retrieval_rank=rank,
    ).model_dump(mode="json")


def _arm(query_id: str, answer: str, notes: bool = False) -> dict[str, object]:
    return {
        "generation": GenerationResult(
            query_id=query_id, answer=answer, cited_evidence_ids=("e1",)
        ).model_dump(mode="json"),
        "notes": [{"slot": "PERSON"}] if notes else [],
        "error": None,
    }


def _row(query_id: str, *, base: str, mixed: str, selected: str) -> dict[str, object]:
    candidates = [_candidate(f"e{rank}", f"d{rank}", rank) for rank in range(1, 11)]
    return {
        "query_id": query_id,
        "selector_changed": True,
        "topk10": candidates,
        "selected_evidence_ids": [f"e{rank}" for rank in range(1, 10)],
        "arms": {
            f005.ARMS[0]: _arm(query_id, base),
            f005.ARMS[1]: _arm(query_id, mixed, True),
            f005.ARMS[2]: _arm(query_id, selected, True),
        },
    }


def test_score_separates_generator_selector_and_full_system_effects() -> None:
    rows = [
        _row("q1", base="wrong", mixed="right", selected="right"),
        _row("q2", base="wrong", mixed="wrong", selected="right"),
    ]
    gold = {
        query_id: GoldCase(
            query_id=query_id,
            relevant_document_ids=("d1",),
            reference_answers=("right",),
        )
        for query_id in ("q1", "q2")
    }
    report = f005.score(
        rows,
        gold,
        {"d1": "parent"},
        {"q1": "cf1", "q2": "cf2"},
    )

    assert report["comparisons"]["B_minus_A_generator_only"]["answer_match"]["delta"] == 0.5
    assert report["comparisons"]["C_minus_B_selector_only"]["answer_match"]["delta"] == 0.5
    assert report["comparisons"]["C_minus_A_full_system"]["answer_match"]["delta"] == 1.0
    assert report["confirmation"]["point_estimate_success"] is True


def test_point_success_requires_positive_selector_effect() -> None:
    rows = [_row("q1", base="wrong", mixed="right", selected="right")]
    gold = {
        "q1": GoldCase(
            query_id="q1", relevant_document_ids=("d1",), reference_answers=("right",)
        )
    }
    report = f005.score(rows, gold, {"d1": "parent"}, {"q1": "cf1"})

    assert report["comparisons"]["C_minus_A_full_system"]["answer_match"]["delta"] == 1.0
    assert report["comparisons"]["C_minus_B_selector_only"]["answer_match"]["delta"] == 0.0
    assert report["confirmation"]["point_estimate_success"] is False


def test_parent_components_join_queries_sharing_source_page() -> None:
    gold = {
        "q1": GoldCase(query_id="q1", relevant_document_ids=("d1",)),
        "q2": GoldCase(query_id="q2", relevant_document_ids=("d2",)),
        "q3": GoldCase(query_id="q3", relevant_document_ids=("d3",)),
    }
    components = f005._component_ids(
        gold,
        {"d1": "shared", "d2": "shared", "d3": "separate"},
    )

    assert components["q1"] == components["q2"]
    assert components["q1"] != components["q3"]
