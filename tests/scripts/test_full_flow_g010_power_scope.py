from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from full_flow_g010_power_scope import (  # noqa: E402
    minimum_detectable_effect,
    required_paired_n,
)


def test_mde_increases_with_discordance_and_shrinks_with_sample_size() -> None:
    small = minimum_detectable_effect(n=739, discordance_rate=0.05)
    high_discordance = minimum_detectable_effect(n=739, discordance_rate=0.20)
    bigger_sample = minimum_detectable_effect(n=2000, discordance_rate=0.05)

    assert high_discordance > small
    assert bigger_sample < small


def test_required_n_rejects_invalid_effects() -> None:
    with pytest.raises(ValueError, match="effect"):
        required_paired_n(effect=0.0, discordance_rate=0.10)
