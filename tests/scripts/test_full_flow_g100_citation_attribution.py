from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from full_flow_g100_citation_attribution import (  # noqa: E402
    build_outputs,
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


def test_real_g100_outputs_are_audit_candidates_not_route_decisions() -> None:
    summary, rows, markdown = build_outputs(ROOT)

    assert summary["status"] == "PASS"
    assert summary["runtime_boundary"]["training_started"] is False
    assert summary["interpretation"]["g110_required_before_repair"] is True
    assert summary["scope"]["claim_level_rows"] == len(rows)
    assert summary["scope"]["claim_level_rows"] > 0
    assert "G110 must audit" in markdown
