from __future__ import annotations

import sys
from pathlib import Path

import pytest

from evidence_rag.contracts.models import EvidenceCandidate, GenerationResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_flow_citation_score as citation  # noqa: E402


def _candidate(index: int) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=f"e{index}",
        document_id=f"d{index}",
        chunk_id=f"c{index}",
        text=f"Supported fact {index}.",
        source_uri=f"memory://{index}",
        retrieval_score=float(11 - index),
        retrieval_rank=index,
    )


def _arm(answer: str, cited: tuple[str, ...], routing: list[dict[str, object]]) -> dict[str, object]:
    return {
        "generation": GenerationResult(
            query_id="q1", answer=answer, cited_evidence_ids=cited
        ).model_dump(mode="json"),
        "routing": routing,
        "error": None,
        "seconds": 0.1,
    }


def _row() -> dict[str, object]:
    routing = [
        {
            "claim_id": "c1",
            "outcome": "verified",
            "sentence": "Supported fact 1.",
            "citation": "e1",
        },
        {
            "claim_id": "c2",
            "outcome": "unverified",
            "sentence": "Unknown fact. [unverified]",
            "citation": None,
        },
    ]
    basic = _arm("Supported fact 1. Unknown fact.", ("e1",), [])
    verify = _arm("Supported fact 1. Unknown fact. [unverified]", ("e1",), routing)
    return {
        "query_id": "q1",
        "question": "Question?",
        "component_id": "component-1",
        "selector_changed": True,
        "topk10": [_candidate(index).model_dump(mode="json") for index in range(1, 11)],
        "selected_evidence_ids": [f"e{index}" for index in range(1, 10)],
        "arms": {
            "A_topk_basic": basic,
            "B_selector_basic": basic,
            "C_topk_verify": verify,
            "D_selector_verify": verify,
        },
    }


def test_basic_mapping_assigns_all_answer_citations_to_every_sentence() -> None:
    examples, _components, _changed = citation.build_examples([_row()], "A_topk_basic")

    assert examples[0].sentences == ("Supported fact 1.", "Unknown fact.")
    assert examples[0].citations == (("e1",), ("e1",))


def test_verify_mapping_uses_exact_routing_and_counts_uncited_sentence() -> None:
    examples, _components, _changed = citation.build_examples([_row()], "C_topk_verify")

    assert examples[0].sentences == ("Supported fact 1.", "Unknown fact.")
    assert examples[0].citations == (("e1",), ())


def test_score_reports_alce_precision_and_recall_by_scope() -> None:
    def entails(premise: str, hypothesis: str) -> bool:
        return hypothesis in premise

    report, per_case = citation.score_rows([_row()], entails)

    assert report["selector_changed"]["queries"] == 1
    assert report["selector_changed"]["aggregate"]["A_topk_basic"][
        "citation_precision_alce"
    ] == pytest.approx(0.5)
    assert report["selector_changed"]["aggregate"]["C_topk_verify"][
        "citation_recall_alce"
    ] == pytest.approx(0.5)
    assert len(per_case) == 1
