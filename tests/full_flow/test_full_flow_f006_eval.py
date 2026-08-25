from __future__ import annotations

import sys
from pathlib import Path

from evidence_rag.contracts.models import GenerationResult
from evidence_rag.infrastructure.datasets import GoldCase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_flow_f006_eval as f006  # noqa: E402


def _arm(answer: str, notes: bool = False) -> dict[str, object]:
    return {
        "generation": GenerationResult(
            query_id="q1", answer=answer, cited_evidence_ids=("e1",)
        ).model_dump(mode="json"),
        "notes": [{"slot": "PERSON"}] if notes else [],
    }


def test_score_requires_mixed_to_beat_all_three_controls() -> None:
    row = {
        "query_id": "q1",
        "component_id": "c1",
        "arms": {
            "TopK_frozen": _arm("wrong"),
            "Base_notes_frozen": _arm("wrong"),
            "Clean_LoRA": _arm("wrong", True),
            "Mixed_LoRA": _arm("right", True),
        },
    }
    report = f006.score(
        [row], {"q1": GoldCase(query_id="q1", reference_answers=("right",))}
    )

    assert report["comparisons"]["Mixed_minus_Clean"]["answer_match"]["delta"] == 1.0
    assert report["development_gate"] == {
        "mixed_beats_base_point": True,
        "mixed_beats_clean_point": True,
        "mixed_beats_topk_point": True,
        "pass_for_f005": True,
    }


def test_gate_fails_when_mixed_only_ties_clean() -> None:
    row = {
        "query_id": "q1",
        "component_id": "c1",
        "arms": {
            "TopK_frozen": _arm("wrong"),
            "Base_notes_frozen": _arm("wrong"),
            "Clean_LoRA": _arm("right", True),
            "Mixed_LoRA": _arm("right", True),
        },
    }
    report = f006.score(
        [row], {"q1": GoldCase(query_id="q1", reference_answers=("right",))}
    )

    assert report["development_gate"]["mixed_beats_clean_point"] is False
    assert report["development_gate"]["pass_for_f005"] is False
