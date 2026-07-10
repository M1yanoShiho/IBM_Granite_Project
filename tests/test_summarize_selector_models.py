"""Tests for selector feature-importance summaries."""

from __future__ import annotations

from eval.summarize_selector_models import normalize_importance


def test_normalize_importance_sums_each_model_to_one() -> None:
    assert normalize_importance([2.0, 1.0, 1.0]) == [0.5, 0.25, 0.25]


def test_normalize_importance_handles_unused_features() -> None:
    assert normalize_importance([0.0, 0.0]) == [0.0, 0.0]

