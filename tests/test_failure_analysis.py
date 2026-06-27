"""Tests for eval/failure_analysis.py — per-query failure-mode analysis.

Once the aggregate comparison shows granite ~= gte, the interesting question is
*behavioural*: do they win the same queries (redundant) or different ones
(complementary)? And where does BM25 still beat dense? These are pure functions
over per-query score dicts, so they need no models or datasets.
"""
from __future__ import annotations

import pytest

from eval.failure_analysis import (
    head_to_head,
    per_query_deltas,
    query_features,
    summarize_by_bucket,
    top_disagreements,
    upsets,
)


def test_per_query_deltas_computes_shared_differences() -> None:
    a = {"q1": 0.9, "q2": 0.5, "qx": 0.3}     # qx only in a
    b = {"q1": 0.6, "q2": 0.8, "qy": 0.1}     # qy only in b

    deltas = per_query_deltas(a, b)

    assert deltas == {"q1": pytest.approx(0.3), "q2": pytest.approx(-0.3)}


def test_per_query_deltas_raises_without_overlap() -> None:
    with pytest.raises(ValueError, match="shared"):
        per_query_deltas({"q1": 0.5}, {"q2": 0.5})


def test_head_to_head_counts_wins_losses_ties() -> None:
    # q1 +0.3 (a wins), q2 -0.3 (b wins), q3 0 (tie), q4 +0.2 (a wins).
    a = {"q1": 0.9, "q2": 0.5, "q3": 0.7, "q4": 0.6}
    b = {"q1": 0.6, "q2": 0.8, "q3": 0.7, "q4": 0.4}

    h = head_to_head(a, b)

    assert (h.a_wins, h.b_wins, h.ties, h.n) == (2, 1, 1, 4)
    assert h.mean_delta == pytest.approx((0.3 - 0.3 + 0.0 + 0.2) / 4)


def test_head_to_head_correlation_high_when_scores_move_together() -> None:
    # b = a + 0.1: perfectly rank-aligned -> correlation 1.0 (redundant), even
    # though b wins every query.
    a = {"q1": 0.2, "q2": 0.4, "q3": 0.6, "q4": 0.8}
    b = {"q1": 0.3, "q2": 0.5, "q3": 0.7, "q4": 0.9}

    assert head_to_head(a, b).correlation == pytest.approx(1.0)


def test_head_to_head_correlation_negative_when_winners_differ() -> None:
    # a is high exactly where b is low -> negative correlation (complementary).
    a = {"q1": 0.9, "q2": 0.1, "q3": 0.8, "q4": 0.2}
    b = {"q1": 0.1, "q2": 0.9, "q3": 0.2, "q4": 0.8}

    assert head_to_head(a, b).correlation < 0


def test_top_disagreements_ranks_biggest_margins_each_way() -> None:
    # deltas: q1 +0.6, q2 -0.6, q3 +0.05, q4 0.0
    a = {"q1": 0.9, "q2": 0.2, "q3": 0.55, "q4": 0.5}
    b = {"q1": 0.3, "q2": 0.8, "q3": 0.50, "q4": 0.5}

    a_wins_most, b_wins_most = top_disagreements(a, b, n=2)

    assert a_wins_most[0] == ("q1", pytest.approx(0.6))    # a's biggest win first
    assert b_wins_most[0] == ("q2", pytest.approx(-0.6))   # b's biggest win first
    assert all(d > 0 for _, d in a_wins_most)              # ties/losses excluded
    assert all(d < 0 for _, d in b_wins_most)


def test_upsets_finds_where_weak_beats_strong() -> None:
    # bm25 - granite: q1 +0.4 (upset), q2 -0.7, q3 0.0
    bm25 = {"q1": 0.9, "q2": 0.1, "q3": 0.4}
    granite = {"q1": 0.5, "q2": 0.8, "q3": 0.4}

    assert upsets(bm25, granite) == [("q1", pytest.approx(0.4))]


def test_query_features_counts_words_and_detects_digits() -> None:
    assert query_features("what is COVID 19") == {"n_words": 4, "has_digit": True}
    assert query_features("renal failure treatment") == {"n_words": 3, "has_digit": False}


def test_summarize_by_bucket_means_per_group() -> None:
    deltas = {"q1": 0.2, "q2": -0.1, "q3": 0.6, "q4": 0.0}
    buckets = {"q1": "num", "q2": "text", "q3": "num", "q4": "text"}

    out = summarize_by_bucket(deltas, buckets)

    assert out["num"].n == 2
    assert out["num"].mean_delta == pytest.approx(0.4)      # (0.2 + 0.6) / 2
    assert out["text"].mean_delta == pytest.approx(-0.05)   # (-0.1 + 0.0) / 2
