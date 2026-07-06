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


import pytest

from eval.gate_corroboration import gate_mask, gated_fuse
from src.retrieval.fusion import fuse_one


def test_gate_mask_global_always_fires():
    votes, margins = {"q1": 0.0, "q2": 3.0}, {"q1": 0.5, "q2": 0.01}
    assert gate_mask("global", None, votes, margins) == {"q1": True, "q2": True}


def test_gate_mask_votes_fires_at_or_above_tau():
    votes, margins = {"q1": 1.0, "q2": 2.0, "q3": 0.0}, {}
    assert gate_mask("votes", 2.0, votes, margins) == {"q1": False, "q2": True, "q3": False}


def test_gate_mask_margin_fires_below_threshold():
    votes, margins = {}, {"q1": 0.04, "q2": 0.5}
    assert gate_mask("margin", 0.05, votes, margins) == {"q1": True, "q2": False}


def test_gate_mask_rejects_unknown_family():
    with pytest.raises(ValueError):
        gate_mask("nonsense", 1.0, {}, {})


def test_gated_fuse_gate_off_keeps_pure_relevance_order():
    rel = {"q1": {"cf": 0.99, "n1": 0.5}}
    cor = {"q1": {"cf": 0.0, "n1": 5.0}}
    fused = gated_fuse(rel, cor, alpha=0.5, gate={"q1": False})
    assert fused["q1"] == rel["q1"]  # untouched: first-stage scores pass through


def test_gated_fuse_gate_on_matches_fuse_one():
    rel = {"q1": {"cf": 0.99, "n1": 0.5}}
    cor = {"q1": {"cf": 0.0, "n1": 5.0}}
    fused = gated_fuse(rel, cor, alpha=0.5, gate={"q1": True})
    assert fused["q1"] == fuse_one(rel["q1"], cor["q1"], 0.5)


def test_gated_fuse_mixed_gate():
    rel = {"q1": {"a": 0.9, "b": 0.1}, "q2": {"a": 0.9, "b": 0.1}}
    cor = {"q1": {"a": 0.0, "b": 3.0}, "q2": {"a": 0.0, "b": 3.0}}
    fused = gated_fuse(rel, cor, alpha=0.0, gate={"q1": True, "q2": False})
    assert fused["q1"] == fuse_one(rel["q1"], cor["q1"], 0.0)
    assert fused["q2"] == rel["q2"]


from eval.gate_corroboration import split_queries


def test_split_queries_is_deterministic_disjoint_and_covers_all():
    qids = [f"q{i}" for i in range(300)]
    dev1, test1 = split_queries(qids, seed=0)
    dev2, test2 = split_queries(list(reversed(qids)), seed=0)  # input order irrelevant
    assert (dev1, test1) == (dev2, test2)
    assert len(dev1) == len(test1) == 150
    assert set(dev1).isdisjoint(test1)
    assert set(dev1) | set(test1) == set(qids)


def test_split_queries_seed_changes_the_split():
    qids = [f"q{i}" for i in range(300)]
    assert split_queries(qids, seed=0) != split_queries(qids, seed=1)


def test_split_queries_dev_fraction():
    dev, test = split_queries([f"q{i}" for i in range(10)], seed=0, dev_fraction=0.3)
    assert len(dev) == 3 and len(test) == 7
