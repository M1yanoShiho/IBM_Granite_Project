"""Tests for the offline gated dynamic-alpha corroboration analysis."""
from eval.gate_corroboration import margin_signal, max_votes_signal


def test_max_votes_signal_takes_the_per_query_max():
    corr = {"q1": {"a": 0.0, "b": 2.0, "c": 1.0}, "q2": {"a": 0.0, "b": 0.0}}
    assert max_votes_signal(corr) == {"q1": 2.0, "q2": 0.0}


def test_max_votes_signal_empty_query_scores_zero():
    assert max_votes_signal({"q1": {}}) == {"q1": 0.0}


def test_margin_signal_is_top1_minus_top2_after_minmax():
    # exactly-representable scores: minmax -> 1.0/0.5/0.0 -> margin 0.5 (no float dust)
    rel = {"q1": {"a": 1.0, "b": 0.5, "c": 0.0}}
    assert margin_signal(rel) == {"q1": 0.5}


def test_margin_signal_degenerate_cases_are_zero():
    # all-equal scores minmax to all-1.0 (margin 0); a single doc has no top2.
    rel = {"q1": {"a": 0.7, "b": 0.7}, "q2": {"a": 0.9}}
    assert margin_signal(rel) == {"q1": 0.0, "q2": 0.0}
