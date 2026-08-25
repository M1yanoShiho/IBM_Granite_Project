from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from full_flow_g000_freeze import (  # noqa: E402
    _g230_archive_verification,
    build_denylist,
    build_outputs,
)


def test_denylist_preserves_heldout_hashes_and_retired_sealed_boundary() -> None:
    denylist = build_denylist(ROOT)

    for block in denylist["system_heldout"].values():
        assert block["ordered_ids_sha256"] == block["config_sha256"]
        assert block["content_or_score_loaded"] is False

    assert denylist["retired"]["sealed600"]["count"] == 600
    assert denylist["retired"]["sealed600"]["new_training_or_selection_allowed"] is False
    assert (
        sum(denylist["role_assignments"]["niah_train_r002"]["roles"].values())
        == denylist["role_assignments"]["niah_train_r002"]["count"]
        == 1023
    )


def test_g230_archive_verification_recomputes_decompressed_hashes() -> None:
    verification = _g230_archive_verification(ROOT)

    assert verification["status"] == "PASS"
    assert set(verification["generation_archives"]) == {"gn", "seed13", "seed42", "seed73"}
    assert all(
        item["decompressed_sha256"] == item["expected_decompressed_sha256"]
        for item in verification["generation_archives"].values()
    )


def test_g000_without_server_audit_stops_before_next_stage() -> None:
    manifest, _denylist, preflight, report = build_outputs(ROOT)

    assert manifest["server_entity_verification"]["status"] == "PENDING"
    assert preflight["status"] == "FAIL"
    assert preflight["next_allowed_stage"] == "STOP"
    assert manifest["runtime_contract"]["generator_runtime_gold_loaded"] is False
    assert "未启动训练" in report
