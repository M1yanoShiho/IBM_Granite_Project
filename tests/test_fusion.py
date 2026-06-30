"""Tests for src/retrieval/fusion.py — convex-combination score fusion.

Pure maths on hand-checkable scores; no models. Pins min-max normalisation, the
alpha endpoints (alpha=1 -> dense order, alpha=0 -> lexical order), the union of
doc ids, and the missing-doc-contributes-0 rule.
"""
from __future__ import annotations

import pytest

from src.retrieval.fusion import convex_fuse, fuse_one, minmax_normalize


def test_minmax_normalize_scales_to_unit_range() -> None:
    assert minmax_normalize({"a": 2.0, "b": 4.0, "c": 6.0}) == {"a": 0.0, "b": 0.5, "c": 1.0}


def test_minmax_normalize_empty_is_empty() -> None:
    assert minmax_normalize({}) == {}


def test_minmax_normalize_all_equal_maps_to_one() -> None:
    # No spread (single doc or ties) -> treat each as maximally relevant for that arm.
    assert minmax_normalize({"a": 3.0, "b": 3.0}) == {"a": 1.0, "b": 1.0}


def test_fuse_one_alpha_one_is_pure_dense() -> None:
    dense = {"d1": 0.9, "d2": 0.1}      # -> d1=1.0, d2=0.0
    lexical = {"d2": 5.0, "d3": 1.0}    # ignored at alpha=1
    fused = fuse_one(dense, lexical, alpha=1.0)
    assert fused["d1"] == pytest.approx(1.0)
    assert fused["d1"] > fused["d2"] and fused["d2"] >= fused["d3"]


def test_fuse_one_alpha_zero_is_pure_lexical() -> None:
    dense = {"d1": 0.9, "d2": 0.1}
    lexical = {"d2": 5.0, "d3": 1.0}    # -> d2=1.0, d3=0.0
    fused = fuse_one(dense, lexical, alpha=0.0)
    assert fused["d2"] == pytest.approx(1.0)
    assert fused["d1"] == pytest.approx(0.0)  # d1 absent from lexical -> 0


def test_fuse_one_blends_union_of_docs() -> None:
    dense = {"d1": 1.0, "d2": 0.0}      # already unit-scaled
    lexical = {"d2": 1.0, "d3": 0.0}
    fused = fuse_one(dense, lexical, alpha=0.5)
    assert set(fused) == {"d1", "d2", "d3"}
    assert fused["d1"] == pytest.approx(0.5)
    assert fused["d2"] == pytest.approx(0.5)
    assert fused["d3"] == pytest.approx(0.0)


def test_fuse_one_rejects_alpha_out_of_range() -> None:
    with pytest.raises(ValueError, match="alpha"):
        fuse_one({"d1": 1.0}, {"d1": 1.0}, alpha=1.5)


def test_convex_fuse_fuses_per_query() -> None:
    dense_run = {"q1": {"d1": 1.0, "d2": 0.0}}
    lexical_run = {"q1": {"d2": 1.0, "d3": 0.0}}
    fused = convex_fuse(dense_run, lexical_run, alpha=0.5)
    assert set(fused["q1"]) == {"d1", "d2", "d3"}
    assert fused["q1"]["d1"] == pytest.approx(0.5)


def test_convex_fuse_handles_query_in_one_run_only() -> None:
    fused = convex_fuse({"q1": {"d1": 1.0, "d2": 0.0}}, {}, alpha=0.5)
    assert fused["q1"]["d1"] == pytest.approx(0.5)  # empty lexical contributes 0
