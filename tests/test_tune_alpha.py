"""Tests for eval/tune_alpha.py — the convex-hybrid alpha sweep/tuning.

Tests the pure sweep arithmetic on synthetic runs where the optimal alpha is known,
plus the grid and CSV writer. The retrieval-bearing _arm_runs (models + datasets) is
exercised only on the HPC run, not here.
"""
from __future__ import annotations

from pathlib import Path

from eval.tune_alpha import _grid, _parse_args, best_alpha, sweep, write_curve


def test_grid_is_inclusive_unit_interval() -> None:
    assert _grid(0.5) == [0.0, 0.5, 1.0]
    g = _grid(0.05)
    assert g[0] == 0.0 and g[-1] == 1.0 and len(g) == 21


def test_sweep_prefers_lexical_when_only_lexical_finds_the_gold() -> None:
    # Gold for q1 is d2. Dense ranks d1 top and d2 last; BM25 ranks d2 top.
    # alpha=0 (pure lexical) puts the gold first -> highest nDCG.
    dense_run = {"q1": {"d1": 1.0, "d2": 0.0}}
    bm25_run = {"q1": {"d2": 1.0, "d1": 0.0}}
    qrels = {"q1": {"d2": 1}}
    a_star, _ = best_alpha(sweep(dense_run, bm25_run, qrels, [0.0, 1.0]))
    assert a_star == 0.0


def test_sweep_prefers_dense_when_only_dense_finds_the_gold() -> None:
    dense_run = {"q1": {"d1": 1.0, "d2": 0.0}}
    bm25_run = {"q1": {"d2": 1.0, "d1": 0.0}}
    qrels = {"q1": {"d1": 1}}  # gold is d1, which dense ranks top
    a_star, _ = best_alpha(sweep(dense_run, bm25_run, qrels, [0.0, 1.0]))
    assert a_star == 1.0


def test_best_alpha_breaks_ties_toward_smaller_alpha() -> None:
    # Flat curve -> tie; prefer the smaller alpha (less reliance on the weaker arm).
    assert best_alpha([(0.0, 0.5), (0.5, 0.5), (1.0, 0.5)])[0] == 0.0


def test_write_curve_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "curve.csv"
    write_curve([(0.0, 0.1), (1.0, 0.9)], path)
    lines = path.read_text().splitlines()
    assert lines[0] == "alpha,ndcg@10"
    assert lines[1] == "0.0,0.1"


def test_parse_args_lexical_defaults_to_bm25_and_accepts_splade() -> None:
    assert _parse_args(["--dataset", "scifact"]).lexical == "bm25"
    assert _parse_args(["--dataset", "scifact", "--lexical", "splade"]).lexical == "splade"


def test_parse_args_lexical_accepts_strong_bm25() -> None:
    # The fair BM25 arm must be tunable as the convex lexical arm too, so its
    # deployment alpha is picked on dev like the naive-bm25 and splade arms.
    args = _parse_args(["--dataset", "scifact", "--lexical", "strong_bm25"])
    assert args.lexical == "strong_bm25"
