from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from full_flow_a000_freeze import (  # noqa: E402
    build_outputs,
    minimum_detectable_effect,
    required_paired_n,
)


def test_required_n_increases_with_discordance_and_smaller_effect() -> None:
    low_discordance = required_paired_n(discordance_rate=0.05, effect=0.02)
    high_discordance = required_paired_n(discordance_rate=0.15, effect=0.02)
    smaller_effect = required_paired_n(discordance_rate=0.05, effect=0.01)
    assert high_discordance > low_discordance
    assert smaller_effect > low_discordance


def test_reserved_sample_mde_exceeds_two_points() -> None:
    assert minimum_detectable_effect(n=400, discordance_rate=0.05) > 0.02
    assert minimum_detectable_effect(n=300, discordance_rate=0.10) > 0.04


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"discordance_rate": 0.0, "effect": 0.02}, "discordance_rate"),
        ({"discordance_rate": 0.1, "effect": 0.0}, "effect"),
        ({"discordance_rate": 0.1, "effect": 0.02, "design_effect": 0.9}, "design_effect"),
    ],
)
def test_power_inputs_are_validated(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        required_paired_n(**kwargs)


def test_real_a000_sources_preserve_runtime_and_sealed_boundaries() -> None:
    data, power = build_outputs(ROOT)
    assert data["runtime_gold_boundary"]["generation_reads_gold"] is False
    assert data["datasets"]["sealed600"]["new_training_or_selection_allowed"] is False
    assert data["datasets"]["system_heldout"]["status"].endswith("NEVER_SCORED")
    assert (
        data["datasets"]["system_heldout"]["historical_exposure"][
            "generated_records_per_dataset_per_arm"
        ]
        == 3
    )
    assert data["models"]["generator"]["other_model_family_allowed"] is False
    assert data["server_entity_verification"]["status"] == "PENDING"
    assert power["status"] == "FINAL_SAMPLE_UNDERPOWERED_FOR_2PP_PER_DATASET"


def test_real_server_audit_promotes_manifest_to_pass() -> None:
    audit = (
        ROOT
        / "docs/full-flow/experiments/02_generator_selector_alignment_2026-08-15"
        / "artifacts/A000/A000_SERVER_AUDIT.json"
    )
    data, _ = build_outputs(ROOT, audit)
    verification = data["server_entity_verification"]
    assert verification["status"] == "PASS"
    assert verification["audit_sha256"]
    assert verification["heldout_exposure_boundary"]["metric_or_scorer_run"] is False
