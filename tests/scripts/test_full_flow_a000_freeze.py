from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from full_flow_a000_freeze import (  # noqa: E402
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
