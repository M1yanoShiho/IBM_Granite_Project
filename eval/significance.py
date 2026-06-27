"""Paired significance testing for retriever comparisons.

Tier-1 statistics on top of per-query scores (``eval.ir_metrics.per_query_scores``).
The central question after the fair comparison is: *is retriever A's edge over B
real, or noise?* A 1.3-point nDCG gap on 300 queries (e.g. granite_dense vs
gte_dense) is meaningless without this.

Two pure-numpy, seeded (reproducible) tools, the pair recommended for IR evaluation
by Smucker, Allan & Carterette (2007), "A Comparison of Statistical Significance
Tests for IR Evaluation":

- :func:`randomization_test` — a two-sided paired *sign-flip permutation* test. H0:
  the per-query differences are symmetric about 0 (the two retrievers are
  interchangeable). Reported p = fraction of sign-flips whose mean is at least as
  extreme as observed.
- :func:`bootstrap_ci` — a percentile bootstrap confidence interval on the mean
  per-query difference. A CI that excludes 0 is the visual companion to a small p.

:func:`compare_to_reference` runs both for every retriever against one reference
(e.g. each model vs gte_dense), reading the wide per-query CSV that
``eval.run_benchmark --per-query-out`` writes. Run it locally — no GPU needed.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# {retriever_name: {query_id: per_query_score}}
PerQuery = Dict[str, Dict[str, float]]


@dataclass
class SignificanceResult:
    """Outcome of comparing retriever A to retriever B on shared queries.

    ``mean_diff`` and the CI are for ``A - B``, so a positive ``mean_diff`` means A
    scores higher than B. ``p_value`` is the two-sided randomization-test p.
    """

    mean_a: float
    mean_b: float
    mean_diff: float
    p_value: float
    ci_low: float
    ci_high: float
    n_queries: int
    confidence: float

    @property
    def significant(self) -> bool:
        """True if the difference clears the (1 - confidence) significance level."""
        return self.p_value < (1.0 - self.confidence)


def randomization_test(
    diffs: "np.ndarray | List[float]",
    n_permutations: int = 10000,
    seed: int = 0,
) -> float:
    """Two-sided paired randomization (sign-flip permutation) test p-value.

    Under H0 the sign of each per-query difference is arbitrary, so we draw
    ``n_permutations`` random sign vectors, recompute the mean each time, and report
    the fraction whose magnitude is >= the observed magnitude. The observed
    assignment is counted too (the ``+1`` on both sides), so the p-value is never 0.
    Returns 1.0 when the observed mean difference is exactly 0.
    """
    diffs = np.asarray(diffs, dtype=float)
    if diffs.size == 0:
        raise ValueError("Cannot run a randomization test on zero differences.")
    observed = abs(float(diffs.mean()))
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([1.0, -1.0], size=(n_permutations, diffs.size))
    permuted = np.abs((signs * diffs).mean(axis=1))
    count = int(np.count_nonzero(permuted >= observed))
    return (count + 1) / (n_permutations + 1)


def bootstrap_ci(
    diffs: "np.ndarray | List[float]",
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean of the paired per-query differences.

    Resamples the differences with replacement ``n_resamples`` times and takes the
    central ``confidence`` interval of the resample means. A CI that excludes 0 is
    the interval-estimate companion to a small randomization p-value.
    """
    diffs = np.asarray(diffs, dtype=float)
    if diffs.size == 0:
        raise ValueError("Cannot bootstrap zero differences.")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_resamples, diffs.size))
    means = diffs[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def _shared_qids(scores_a: Dict[str, float], scores_b: Dict[str, float]) -> List[str]:
    """Query ids present for both retrievers, in ``scores_a`` order."""
    shared = [qid for qid in scores_a if qid in scores_b]
    if not shared:
        raise ValueError("No shared query ids between the two score dicts.")
    return shared


def paired_significance(
    scores_a: Dict[str, float],
    scores_b: Dict[str, float],
    *,
    n_permutations: int = 10000,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> SignificanceResult:
    """Compare two retrievers' per-query scores: mean difference, randomization
    p-value, and bootstrap CI — all on the queries the two share (paired).

    ``scores_a``/``scores_b`` map ``{query_id: score}`` (e.g. from
    :func:`eval.ir_metrics.per_query_scores`). The comparison is paired on the
    shared query ids; queries present for only one retriever are ignored.
    """
    shared = _shared_qids(scores_a, scores_b)
    a = np.array([scores_a[qid] for qid in shared], dtype=float)
    b = np.array([scores_b[qid] for qid in shared], dtype=float)
    diffs = a - b
    low, high = bootstrap_ci(diffs, n_resamples=n_resamples, confidence=confidence, seed=seed)
    return SignificanceResult(
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_diff=float(diffs.mean()),
        p_value=randomization_test(diffs, n_permutations=n_permutations, seed=seed),
        ci_low=low,
        ci_high=high,
        n_queries=len(shared),
        confidence=confidence,
    )


def compare_to_reference(
    per_query: PerQuery,
    reference: str,
    **kwargs,
) -> Dict[str, SignificanceResult]:
    """Compare every other retriever in ``per_query`` to ``reference``.

    Each result is ``retriever - reference``, so a positive ``mean_diff`` means the
    retriever beats the reference. ``kwargs`` pass through to
    :func:`paired_significance` (n_permutations, n_resamples, confidence, seed).
    """
    if reference not in per_query:
        raise ValueError(
            f"Reference retriever {reference!r} not in {list(per_query)}."
        )
    ref = per_query[reference]
    return {
        name: paired_significance(scores, ref, **kwargs)
        for name, scores in per_query.items()
        if name != reference
    }


def load_per_query_csv(path: str | Path) -> PerQuery:
    """Load the wide per-query CSV (``qid`` + one column per retriever) written by
    ``eval.run_benchmark --per-query-out`` into ``{retriever: {qid: score}}``.

    Blank cells (a retriever missing a query) are skipped, so the dicts only hold
    the queries each retriever actually scored.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    retrievers = [c for c in rows[0] if c != "qid"]
    out: PerQuery = {name: {} for name in retrievers}
    for row in rows:
        qid = row["qid"]
        for name in retrievers:
            cell = row[name]
            if cell != "":
                out[name][qid] = float(cell)
    return out


def format_report(results: Dict[str, SignificanceResult], reference: str) -> str:
    """Render :func:`compare_to_reference` output as a fixed-width table."""
    lines = [
        f"vs {reference} (reference)",
        f"{'retriever':<22}{'mean':>8}{'d_vs_ref':>11}{'95% CI':>22}{'p':>9}  sig",
    ]
    for name, res in sorted(results.items(), key=lambda kv: kv[1].mean_a, reverse=True):
        ci = f"[{res.ci_low:+.4f}, {res.ci_high:+.4f}]"
        star = "*" if res.significant else ""
        lines.append(
            f"{name:<22}{res.mean_a:>8.4f}{res.mean_diff:>+11.4f}{ci:>22}"
            f"{res.p_value:>9.4f}  {star}"
        )
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> None:
    """CLI: read a per-query CSV and print each retriever's significance vs a
    reference. Runs locally (no GPU): the embedding work is already in the CSV.

    Example::

        python -m eval.significance --per-query-csv results/scifact_per_query.csv \\
            --reference gte_dense
    """
    parser = argparse.ArgumentParser(
        prog="python -m eval.significance",
        description="Paired randomization test + bootstrap CI between retrievers.",
    )
    parser.add_argument(
        "--per-query-csv",
        type=Path,
        required=True,
        help="Wide per-query CSV from 'run_benchmark --per-query-out'.",
    )
    parser.add_argument(
        "--reference",
        default="gte_dense",
        help="Retriever to compare every other against (default: %(default)s).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed (default: %(default)s)."
    )
    args = parser.parse_args(argv)

    per_query = load_per_query_csv(args.per_query_csv)
    results = compare_to_reference(per_query, args.reference, seed=args.seed)
    print(format_report(results, args.reference))


if __name__ == "__main__":
    main()
