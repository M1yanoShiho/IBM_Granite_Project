"""Tests for eval.compare_rules — the offline combination-rule comparison (Exp A/B).

Pure arithmetic on a (relevance, corroboration) runs dump: blend vs hard cascade vs
lexicographic tie-break, scored by needle-found@k AND MRR. The rule logic is what needs
proving; the metrics (per_query_hits / per_query_reciprocal_rank) are reused verbatim so
the numbers agree with the certification.
"""
from __future__ import annotations

import json

from eval.compare_rules import (
    _order_to_scores,
    build_rule_runs,
    cascade_run,
    lexicographic_run,
    main,
    sweep_blend,
    sweep_lexicographic,
)
from eval.gate_corroboration import per_query_reciprocal_rank
from eval.tune_corroboration import per_query_hits


def test_order_to_scores_is_reproduced_by_the_shared_metric():
    scores = _order_to_scores(["a", "b", "c"])
    assert scores["a"] > scores["b"] > scores["c"]
    run = {"q": scores}
    assert per_query_hits(run, {"q": "a"}, 1)["q"] == 1.0
    assert per_query_hits(run, {"q": "c"}, 1)["q"] == 0.0


def test_cascade_promotes_corroborated_needle_over_more_relevant_counterfactual():
    # The mechanism: a lone counterfactual is MORE relevant but uncorroborated; cascade
    # ranks by votes first, so the corroborated needle wins.
    rel = {"q": {"needle": 0.90, "cf": 0.95, "noise": 0.10}}
    corr = {"q": {"needle": 2.0, "cf": 0.0, "noise": 0.0}}
    run = cascade_run(rel, corr)
    assert per_query_hits(run, {"q": "needle"}, 1)["q"] == 1.0


def test_cascade_breaks_equal_vote_ties_by_relevance():
    rel = {"q": {"a": 0.9, "b": 0.1}}
    corr = {"q": {"a": 1.0, "b": 1.0}}  # equal votes -> relevance decides
    run = cascade_run(rel, corr)
    assert per_query_hits(run, {"q": "a"}, 1)["q"] == 1.0


def test_lexicographic_swaps_within_epsilon_band_but_keeps_cross_band_relevance():
    # hi is clearly most relevant (own band); needle and cf sit in the same eps band,
    # so the more-corroborated needle jumps the (barely) more-relevant cf.
    rel = {"q": {"hi": 1.0, "cf": 0.55, "needle": 0.52, "lo": 0.0}}
    corr = {"q": {"needle": 2.0, "cf": 0.0, "hi": 0.0, "lo": 0.0}}
    run = lexicographic_run(rel, corr, epsilon=0.1)
    assert per_query_reciprocal_rank(run, {"q": "hi"})["q"] == 1.0
    assert per_query_reciprocal_rank(run, {"q": "needle"})["q"] == 0.5
    assert per_query_reciprocal_rank(run, {"q": "cf"})["q"] == 1.0 / 3.0


def test_lexicographic_tiny_epsilon_preserves_relevance_order():
    # A relevance gap wider than the eps resolution is NOT overridden by votes.
    rel = {"q": {"hi": 1.0, "cf": 0.55, "needle": 0.52, "lo": 0.0}}
    corr = {"q": {"needle": 9.0, "cf": 0.0, "hi": 0.0, "lo": 0.0}}
    run = lexicographic_run(rel, corr, epsilon=0.001)
    assert per_query_reciprocal_rank(run, {"q": "cf"})["q"] == 0.5
    assert per_query_reciprocal_rank(run, {"q": "needle"})["q"] == 1.0 / 3.0


def test_lexicographic_rejects_out_of_range_epsilon():
    import pytest

    with pytest.raises(ValueError):
        lexicographic_run({"q": {"a": 1.0}}, {"q": {"a": 0.0}}, epsilon=0.0)


def test_sweep_blend_reports_found_and_mrr_and_alpha1_is_pure_relevance():
    rel = {"q": {"needle": 0.9, "cf": 0.95}}  # pure relevance: cf rank1, needle rank2
    corr = {"q": {"needle": 2.0, "cf": 0.0}}
    curve = sweep_blend(rel, corr, {"q": "needle"}, grid=[1.0, 0.0], k=1)
    d = {alpha: (found, mrr) for alpha, found, mrr in curve}
    assert d[1.0] == (0.0, 0.5)  # pure relevance: needle rank2
    assert d[0.0] == (1.0, 1.0)  # pure corroboration: needle rank1


def test_sweep_lexicographic_reports_found_and_mrr_over_grid():
    rel = {"q": {"hi": 1.0, "cf": 0.55, "needle": 0.52, "lo": 0.0}}
    corr = {"q": {"needle": 2.0, "cf": 0.0, "hi": 0.0, "lo": 0.0}}
    curve = sweep_lexicographic(rel, corr, {"q": "needle"}, eps_grid=[0.1, 0.001], k=2)
    d = {eps: (found, mrr) for eps, found, mrr in curve}
    assert d[0.1] == (1.0, 0.5)  # needle rescued to rank2 -> in top2
    assert d[0.001][0] == 0.0  # needle rank3 -> not in top2


def test_build_rule_runs_returns_four_named_rules():
    rel = {"q": {"needle": 0.9, "cf": 0.95, "noise": 0.1}}
    corr = {"q": {"needle": 2.0, "cf": 0.0, "noise": 0.0}}
    runs = build_rule_runs(rel, corr, alpha=0.6, epsilon=0.1)
    assert set(runs) == {"q2d", "blend", "cascade", "lexicographic"}
    # q2d = pure relevance (alpha=1): cf outranks needle.
    assert per_query_hits(runs["q2d"], {"q": "needle"}, 1)["q"] == 0.0
    # cascade lifts the corroborated needle to rank 1.
    assert per_query_hits(runs["cascade"], {"q": "needle"}, 1)["q"] == 1.0


def test_main_writes_per_query_and_curve_csvs(tmp_path):
    dump = {
        "relevance_run": {
            "q1": {"needle": 0.90, "cf": 0.95, "x": 0.10},
            "q2": {"needle": 0.80, "cf": 0.70, "x": 0.10},
        },
        "corroboration_run": {
            "q1": {"needle": 2.0, "cf": 0.0, "x": 0.0},
            "q2": {"needle": 1.0, "cf": 0.0, "x": 0.0},
        },
        "needles": {"q1": "needle", "q2": "needle"},
    }
    dump_path = tmp_path / "runs.json"
    dump_path.write_text(json.dumps(dump), encoding="utf-8")
    out = tmp_path / "out"
    main([
        "--from-runs", str(dump_path), "--k", "10", "--alpha", "0.6",
        "--epsilon", "0.1", "--alpha-step", "0.5", "--out-dir", str(out), "--tag", "t",
    ])
    hits = out / "compare_rules_t_per_query_hits.csv"
    assert hits.exists()
    assert (out / "compare_rules_t_per_query_mrr.csv").exists()
    assert (out / "compare_rules_t_alpha_curve.csv").exists()
    assert (out / "compare_rules_t_eps_curve.csv").exists()
    header = hits.read_text(encoding="utf-8").splitlines()[0]
    for rule in ("q2d", "blend", "cascade", "lexicographic"):
        assert rule in header
