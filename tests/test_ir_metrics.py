"""Tests for per-query metric extraction (eval/ir_metrics.per_query_scores).

The aggregate metrics (evaluate_run etc.) wrap ranx directly and are exercised via
run_benchmark; the new piece is per-query scores, which feed the failure analysis
(which queries a retriever wins/loses) and paired significance testing.
"""
from __future__ import annotations

import pytest

from eval.ir_metrics import ndcg_at_k, per_query_scores


def test_per_query_scores_maps_each_query_to_its_own_ndcg() -> None:
    # q1 and q3 rank the relevant doc first (ndcg@10 = 1.0); q2 ranks an irrelevant
    # doc first and the relevant one second (ndcg@10 = 1/log2(3) = 0.6309). Keying
    # by the RIGHT query id is the point -- an ordering bug would swap them.
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}, "q3": {"d3": 1}}
    run = {
        "q1": {"d1": 0.9, "dx": 0.1},
        "q2": {"dx": 0.9, "d2": 0.1},
        "q3": {"d3": 0.9},
    }

    scores = per_query_scores(run, qrels, "ndcg@10")

    assert set(scores) == {"q1", "q2", "q3"}
    assert scores["q1"] == pytest.approx(1.0)
    assert scores["q3"] == pytest.approx(1.0)
    assert scores["q2"] == pytest.approx(0.6309297, abs=1e-6)


def test_per_query_scores_mean_matches_aggregate() -> None:
    # The mean of the per-query scores must equal the aggregate metric, so the
    # significance analysis and the headline numbers can never silently diverge.
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}, "q3": {"d3": 1}}
    run = {
        "q1": {"d1": 0.9, "dx": 0.1},
        "q2": {"dx": 0.9, "d2": 0.1},
        "q3": {"d3": 0.9},
    }

    per_q = per_query_scores(run, qrels, "ndcg@10")

    assert sum(per_q.values()) / len(per_q) == pytest.approx(ndcg_at_k(run, qrels, 10))
