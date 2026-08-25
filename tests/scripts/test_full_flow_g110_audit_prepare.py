from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from full_flow_g110_audit_prepare import (  # noqa: E402
    row_key,
    stratified_sample,
)


def test_stratified_sample_is_deterministic() -> None:
    rows = [
        {"failure_stage": "TRUE_ROUTING_OR_ATTACHMENT", "config": "GC13", "task_id": f"t{i}", "claim_id": "c"}
        for i in range(30)
    ]
    first = stratified_sample(rows, target=10, namespace="fixture", min_per_stratum=5)
    second = stratified_sample(rows, target=10, namespace="fixture", min_per_stratum=5)

    assert [row_key(row) for row in first] == [row_key(row) for row in second]
