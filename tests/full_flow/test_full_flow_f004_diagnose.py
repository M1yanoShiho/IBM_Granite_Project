from __future__ import annotations

import sys
from pathlib import Path

from evidence_rag.contracts.models import GenerationResult
from evidence_rag.infrastructure.datasets import GoldCase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_flow_f004_diagnose as diagnose  # noqa: E402


def _arm(answer: str, notes: bool = False) -> dict[str, object]:
    cited = ("e1",) if answer else ()
    return {
        "generation": GenerationResult(
            query_id="q1", answer=answer, cited_evidence_ids=cited
        ).model_dump(mode="json"),
        "notes": [{"slot": "DATE_OR_YEAR"}] if notes else [],
    }


def test_diagnosis_counts_empty_recovery_and_selector_help_preservation() -> None:
    generation = {
        "query_id": "q1",
        "question": "When?",
        "component_id": "c1",
        "question_slots": ["DATE_OR_YEAR"],
        "arms": {
            "G0_current": _arm(""),
            "G1_notes": _arm("in 2009", True),
            "G2_guided_notes": _arm("wrong", True),
        },
    }
    f002 = {
        "query_id": "q1",
        "reference_shape": "date_or_year",
        "comparisons": {
            "verify": {
                "primary_diagnosis": "selector_helped_generation",
                "transition": "wrong_to_right",
            }
        },
    }
    summary, rows = diagnose.diagnose_rows(
        [generation],
        {"q1": f002},
        {"q1": GoldCase(query_id="q1", reference_answers=("in 2009",))},
    )

    assert summary["g0_empty_cases"]["G1_notes"] == {
        "became_nonempty": 1,
        "became_correct": 1,
    }
    assert summary["transition_counts"]["G1_vs_G0"]["0->1"] == 1
    assert summary["previous_selector_help_cases"]["G2_guided_notes"]["correct"] == 0
    assert len(rows) == 1
