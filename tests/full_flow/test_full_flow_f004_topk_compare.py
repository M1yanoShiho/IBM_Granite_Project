from __future__ import annotations

import sys
from pathlib import Path

from evidence_rag.contracts.models import GenerationResult
from evidence_rag.infrastructure.datasets import GoldCase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_flow_f004_topk_compare as topk  # noqa: E402


def _generation(answer: str) -> dict[str, object]:
    return GenerationResult(
        query_id="q1", answer=answer, cited_evidence_ids=("e1",)
    ).model_dump(mode="json")


def test_compare_to_topk_reports_paired_transition() -> None:
    f004 = [
        {
            "query_id": "q1",
            "component_id": "c1",
            "arms": {
                "G1_notes": {"generation": _generation("right")},
                "G2_guided_notes": {"generation": _generation("right")},
            },
        }
    ]
    f001 = {"q1": {"arms": {"C_topk_verify": {"generation": _generation("wrong")}}}}
    gold = {"q1": GoldCase(query_id="q1", reference_answers=("right",))}

    report = topk.compare_to_topk(f004, f001, gold)

    assert report["comparisons"]["G1_notes"]["paired"]["delta"] == 1.0
    assert report["comparisons"]["G1_notes"]["transitions"]["net"] == 1
    assert report["gold_loaded_at_runtime"] is False
