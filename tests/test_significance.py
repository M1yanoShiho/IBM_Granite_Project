"""Tests for eval/significance.py — paired significance for retriever comparisons.

Pure-numpy randomization (sign-flip permutation) test + percentile bootstrap CI on
the per-query metric differences — the IR-standard way (Smucker et al. 2007) to ask
"is retriever A's edge over B real, or noise?". Seeded, so results reproduce.
"""
from __future__ import annotations

import numpy as np
import pytest

from eval.significance import (
    bootstrap_ci,
    compare_to_reference,
    load_per_query_csv,
    paired_significance,
    randomization_test,
)


def test_randomization_test_small_p_for_consistent_difference() -> None:
    # Every query favours A by a clear, consistent margin -> very unlikely under
    # the sign-flip null -> small p.
    diffs = np.array([0.20, 0.15, 0.25, 0.18, 0.22, 0.19, 0.21, 0.17, 0.23, 0.16])
    assert randomization_test(diffs, n_permutations=5000, seed=0) < 0.05


def test_randomization_test_large_p_for_symmetric_difference() -> None:
    # Differences symmetric about 0 -> no systematic effect -> p near 1.
    diffs = np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
    assert randomization_test(diffs, n_permutations=5000, seed=0) > 0.5


def test_randomization_test_reproducible_with_seed() -> None:
    diffs = np.array([0.1, -0.05, 0.2, 0.0, 0.15, -0.1])
    a = randomization_test(diffs, n_permutations=2000, seed=42)
    b = randomization_test(diffs, n_permutations=2000, seed=42)
    assert a == b


def test_bootstrap_ci_excludes_zero_for_consistent_difference() -> None:
    diffs = np.array([0.20, 0.15, 0.25, 0.18, 0.22, 0.19, 0.21, 0.17])
    low, high = bootstrap_ci(diffs, n_resamples=5000, confidence=0.95, seed=0)
    assert low > 0.0  # whole CI above 0 -> A reliably beats B
    assert high >= low


def test_bootstrap_ci_straddles_zero_for_symmetric_difference() -> None:
    diffs = np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
    low, high = bootstrap_ci(diffs, n_resamples=5000, confidence=0.95, seed=0)
    assert low < 0.0 < high


def test_paired_significance_pairs_on_shared_queries_and_flags_real_difference() -> None:
    # A beats B on every shared query; qx exists only for A and must be ignored.
    scores_a = {
        "q0": 0.92, "q1": 0.88, "q2": 0.95, "q3": 0.90, "q4": 0.85,
        "q5": 0.93, "q6": 0.89, "q7": 0.91, "q8": 0.87, "q9": 0.94, "qx": 0.50,
    }
    scores_b = {
        "q0": 0.60, "q1": 0.62, "q2": 0.58, "q3": 0.61, "q4": 0.59,
        "q5": 0.63, "q6": 0.57, "q7": 0.60, "q8": 0.62, "q9": 0.58,
    }

    res = paired_significance(scores_a, scores_b, n_permutations=5000, n_resamples=5000, seed=0)

    assert res.n_queries == 10           # qx (A-only) excluded
    assert res.mean_diff > 0
    assert res.p_value < 0.05
    assert res.ci_low > 0


def test_paired_significance_not_significant_for_few_noisy_queries() -> None:
    # Only 5 queries with small mixed-sign gaps (like granite vs gte): the gap can't
    # clear significance -- the test must NOT cry wolf on a near-tie.
    scores_a = {"q0": 0.72, "q1": 0.66, "q2": 0.71, "q3": 0.65, "q4": 0.70}
    scores_b = {"q0": 0.70, "q1": 0.69, "q2": 0.68, "q3": 0.69, "q4": 0.69}
    res = paired_significance(scores_a, scores_b, n_permutations=5000, n_resamples=5000, seed=0)
    assert res.p_value > 0.05


def test_paired_significance_raises_without_shared_queries() -> None:
    with pytest.raises(ValueError, match="shared"):
        paired_significance({"q1": 0.5}, {"q2": 0.5})


def test_load_per_query_csv_round_trips(tmp_path) -> None:
    # The wide per-query CSV (qid + one column per retriever) loads back into
    # {retriever: {qid: score}}, ready for compare_to_reference.
    path = tmp_path / "per_query.csv"
    path.write_text(
        "qid,granite_dense,gte_dense\n"
        "q1,1.0,0.8\n"
        "q2,0.0,0.5\n"
    )

    loaded = load_per_query_csv(path)

    assert set(loaded) == {"granite_dense", "gte_dense"}
    assert loaded["granite_dense"] == {"q1": 1.0, "q2": 0.0}
    assert loaded["gte_dense"] == {"q1": 0.8, "q2": 0.5}


def test_compare_to_reference_scores_each_retriever_against_the_reference(tmp_path) -> None:
    # Each non-reference retriever is compared to the reference; mean_diff is
    # (retriever - reference), so positive = beats the reference.
    per_query = {
        "gte_dense": {f"q{i}": 0.60 for i in range(10)},
        "granite_dense": {f"q{i}": 0.75 for i in range(10)},
    }

    results = compare_to_reference(per_query, "gte_dense", n_permutations=2000, n_resamples=2000, seed=0)

    assert set(results) == {"granite_dense"}        # the reference is not compared to itself
    assert results["granite_dense"].mean_diff == pytest.approx(0.15)


def test_compare_to_reference_rejects_unknown_reference() -> None:
    with pytest.raises(ValueError, match="[Rr]eference"):
        compare_to_reference({"a": {"q1": 0.5}}, "missing")
