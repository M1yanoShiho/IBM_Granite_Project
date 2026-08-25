from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "docs/full-flow/experiments/04_frozen_three_module_system_evaluation_2026-08-21"
)


def test_committed_manifest_proves_frozen_ids_and_separation() -> None:
    manifest = json.loads(
        (EXPERIMENT_DIR / "artifacts/goal1_data_manifest.json").read_text(encoding="utf-8")
    )
    expected = {
        "hotpotqa": (400, "1c1b822c8942b3e14979de187b63d06f5b64ebcdd2303cf5b6b56c9bea92755d"),
        "musique-answerable": (
            400,
            "37626d4b1890bdc239d0904c9f406a06023f73382ccd9aabfa44bf65189761fb",
        ),
        "rgb-noise": (300, "25ed22988cb9684a8de9fc74c603e79ef44792f0e3bfd4c3b9e74c2e2a301c87"),
    }

    assert manifest["overall_pass"] is True
    for dataset, (count, digest) in expected.items():
        audit = manifest["datasets"][dataset]
        assert audit["frozen_ids"]["observed_count"] == count
        assert audit["frozen_ids"]["observed_ordered_ids_sha256"] == digest
        assert audit["frozen_ids"]["ordered_ids_match"] is True
        assert audit["runtime"]["forbidden_gold_field_count"] == 0
        assert audit["runtime"]["candidate_pool_isolated"] is True
        assert audit["cross_checks"]["runtime_sidecar_ordered_ids_equal"] is True
        assert audit["cross_checks"]["physical_parent_directories_distinct"] is True
    assert manifest["isolation"]["only_alignment_key"] == "query_id"
    assert manifest["scope_attestation"]["heldout_scored"] is False


def test_committed_scorer_validation_covers_all_metrics_and_failure_denominators() -> None:
    validation = json.loads(
        (EXPERIMENT_DIR / "artifacts/goal1_scorer_validation.json").read_text(
            encoding="utf-8"
        )
    )

    assert validation["overall_pass"] is True
    assert set(validation["observed_aggregate"]) == {"ret", "sel", "ans", "cit", "rar"}
    assert validation["denominators"] == {
        "ret": 8,
        "sel": 8,
        "ans": 8,
        "cit": 8,
        "rar": 8,
    }
    assert all(validation["checks"].values())
    assert validation["minicheck_model_run"] is False
