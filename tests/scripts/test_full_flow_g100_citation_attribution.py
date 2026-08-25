from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from full_flow_g100_citation_attribution import (  # noqa: E402
    classify_claim,
)


def test_classify_claim_prioritises_draft_citation_before_evaluator() -> None:
    stage, reason = classify_claim(
        run_row={"trace": {"draft": {"splitter": {"status": "structured"}}}},
        claim={
            "declared_indices": [1],
            "declared_verified": False,
            "final_disposition": "verified",
            "routing_outcome": "verified",
            "gated_outcome": "verified",
            "citation": "e1",
        },
        candidate_supported=False,
    )

    assert stage == "DRAFT_CITATION_MISSING_OR_WRONG"
    assert "draft-declared" in reason


def test_classify_claim_marks_true_verified_minicheck_failure_as_disagreement() -> None:
    stage, _reason = classify_claim(
        run_row={"trace": {"draft": {"splitter": {"status": "structured"}}}},
        claim={
            "declared_indices": [1],
            "declared_verified": True,
            "final_disposition": "verified",
            "routing_outcome": "verified",
            "gated_outcome": "verified",
            "citation": "e1",
        },
        candidate_supported=False,
    )

    assert stage == "EVALUATOR_DISAGREEMENT"
