from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from full_flow_g130_routing_repair import (  # noqa: E402
    REQUIRED_TRACE_FIELDS,
    true_routing_breakdown,
)

from evidence_rag.generator.trace import ClaimTrace  # noqa: E402


def test_trace_contract_exposes_g130_audit_fields() -> None:
    assert REQUIRED_TRACE_FIELDS <= set(ClaimTrace.model_fields)


def test_true_routing_breakdown_separates_gate_warning_from_missing_attachment() -> None:
    sample_rows = [
        {
            "audit_row_id": "a",
            "routing_outcome": "verified",
            "citation": "ev-1",
            "gated_outcome": "dropped_entity_conflict",
        },
        {
            "audit_row_id": "b",
            "routing_outcome": "unverified",
            "citation": None,
            "gated_outcome": "unverified",
        },
        {
            "audit_row_id": "c",
            "routing_outcome": "verified",
            "citation": "ev-2",
            "gated_outcome": "verified",
        },
        {
            "audit_row_id": "d",
            "routing_outcome": "verified",
            "citation": "ev-3",
            "gated_outcome": "dropped_entity_conflict",
        },
    ]
    adjudicated_rows = [
        {"audit_row_id": "a", "final_label": "TRUE_ROUTING_OR_ATTACHMENT"},
        {"audit_row_id": "b", "final_label": "TRUE_ROUTING_OR_ATTACHMENT"},
        {"audit_row_id": "c", "final_label": "EVALUATOR_DISAGREEMENT"},
        {"audit_row_id": "d", "final_label": "TRUE_ROUTING_OR_ATTACHMENT"},
    ]

    assert true_routing_breakdown(sample_rows, adjudicated_rows) == {
        "total_true_routing_or_attachment": 3,
        "true_unverified_no_attachment": 1,
        "verified_attachment_with_observe_gate_warning": 2,
    }
