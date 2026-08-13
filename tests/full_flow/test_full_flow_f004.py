from __future__ import annotations

import json
import sys
from pathlib import Path

from evidence_rag.contracts.models import EvidenceCandidate, GenerationResult
from evidence_rag.infrastructure.datasets import GoldCase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_flow_f004 as f004  # noqa: E402


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


def _arm(answer: str) -> dict[str, object]:
    return {
        "generation": GenerationResult(
            query_id="q1", answer=answer, cited_evidence_ids=("e1",)
        ).model_dump(mode="json"),
        "routing": [],
        "notes": [],
        "error": None,
        "seconds": 0.1,
    }


def test_trace_loader_ignores_gold_derived_fields(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    candidates = [
        {
            "evidence_id": f"e{index}",
            "retrieval_rank": index,
            "protect_score": 0.9,
            "harm_score": 0.1,
            "safe_score": 0.1,
            "action": "DROP_HARM" if index == 10 else "KEEP",
        }
        for index in range(1, 11)
    ]
    trace.write_text(
        json.dumps(
            {
                "dataset_kind": "niah",
                "query_id": "q1",
                "selected_evidence_ids": [f"e{index}" for index in range(1, 10)],
                "dropped_evidence_ids": ["e10"],
                "candidates": candidates,
                "recall_loss": 0.0,
                "chain_loss": 0.0,
                "gold_answer": "must be ignored",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = f004._load_trace_runtime_fields(trace)
    serialized = json.dumps(loaded)

    assert "recall_loss" not in serialized
    assert "chain_loss" not in serialized
    assert "gold_answer" not in serialized
    assert set(loaded["q1"]) == {
        "query_id",
        "candidates",
        "selected_evidence_ids",
        "dropped_evidence_ids",
    }


def test_score_reports_notes_and_guidance_effects() -> None:
    topk = [_candidate(index).model_dump(mode="json") for index in range(1, 11)]
    rows = [
        {
            "query_id": "q1",
            "question": "When?",
            "component_id": "c1",
            "topk10": topk,
            "arms": {
                "G0_current": _arm("wrong"),
                "G1_notes": {**_arm("right"), "notes": [{"slot": "DATE_OR_YEAR"}]},
                "G2_guided_notes": {
                    **_arm("right"),
                    "notes": [{"slot": "DATE_OR_YEAR"}],
                },
            },
        }
    ]
    gold = {
        "q1": GoldCase(query_id="q1", relevant_document_ids=("d1",), reference_answers=("right",))
    }

    report = f004.score_rows(rows, gold)

    assert report["comparisons"]["G1_minus_G0"]["answer_match"]["delta"] == 1.0
    assert report["comparisons"]["G2_minus_G1"]["answer_match"]["delta"] == 0.0
    assert report["notes_nonempty_queries"] == {
        "G0_current": 0,
        "G1_notes": 1,
        "G2_guided_notes": 1,
    }
