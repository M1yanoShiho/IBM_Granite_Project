from __future__ import annotations

import sys
from pathlib import Path

from evidence_rag.contracts.models import EvidenceCandidate, GenerationResult
from evidence_rag.infrastructure.datasets import GoldCase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_flow_diagnose as diagnose  # noqa: E402


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


def _arm(answer: str, citation: str) -> dict[str, object]:
    return {
        "generation": GenerationResult(
            query_id="q1", answer=answer, cited_evidence_ids=(citation,)
        ).model_dump(mode="json"),
        "routing": [],
        "error": None,
        "seconds": 0.1,
    }


def test_diagnosis_distinguishes_removed_evidence_from_generator_sensitivity() -> None:
    topk = [_candidate(index).model_dump(mode="json") for index in range(1, 11)]
    rows = [
        {
            "query_id": "q1",
            "question": "When did it happen?",
            "component_id": "component-1",
            "selector_changed": True,
            "topk10": topk,
            "selected_evidence_ids": [f"e{index}" for index in range(1, 10)],
            "arms": {
                "A_topk_basic": _arm("It happened in 1973.", "e10"),
                "B_selector_basic": _arm("It happened later.", "e1"),
                "C_topk_verify": _arm("It happened in 1973.", "e10"),
                "D_selector_verify": _arm("It happened later.", "e1"),
            },
        }
    ]
    gold = {
        "q1": GoldCase(
            query_id="q1", relevant_document_ids=("d10",), reference_answers=("1973",)
        )
    }

    diagnosed, summary = diagnose.diagnose_rows(rows, gold)

    assert summary["queries"] == 1
    assert diagnosed[0]["removed_relevant_document_ids"] == ["d10"]
    assert (
        diagnosed[0]["comparisons"]["basic"]["primary_diagnosis"]
        == "selector_removed_relevant_evidence"
    )
    assert summary["comparisons"]["verify"]["transitions"]["right_to_wrong"] == 1


def test_diagnosis_marks_wrong_answer_with_all_evidence_as_extraction_candidate() -> None:
    topk = [_candidate(index).model_dump(mode="json") for index in range(1, 11)]
    rows = [
        {
            "query_id": "q1",
            "question": "Who won?",
            "component_id": "component-1",
            "selector_changed": True,
            "topk10": topk,
            "selected_evidence_ids": [f"e{index}" for index in range(1, 10)],
            "arms": {
                "A_topk_basic": _arm("Someone won.", "e1"),
                "B_selector_basic": _arm("Someone won.", "e1"),
                "C_topk_verify": _arm("Someone won.", "e1"),
                "D_selector_verify": _arm("Someone won.", "e1"),
            },
        }
    ]
    gold = {
        "q1": GoldCase(
            query_id="q1", relevant_document_ids=("d1",), reference_answers=("Ada Lovelace",)
        )
    }

    diagnosed, _summary = diagnose.diagnose_rows(rows, gold)

    assert diagnosed[0]["all_relevant_documents_remain"] is True
    assert (
        diagnosed[0]["comparisons"]["verify"]["primary_diagnosis"]
        == "answer_extraction_failure_with_evidence_present"
    )
