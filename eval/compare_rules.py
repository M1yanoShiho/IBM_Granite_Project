"""Offline combination-rule comparison for the corroboration reranker (Experiment A/B).

Pure arithmetic on a runs dump (``eval.tune_corroboration --dump-runs``): no LLM, no
GPU, no index. Given each candidate's (relevance, corroboration-vote) per query, it
compares three ways to COMBINE the two signals into a ranking:

* ``blend``         -- convex ``final = alpha*relevance + (1-alpha)*corroboration`` (the
  certified method; ``alpha = 1.0`` is pure q2d, ``alpha = 0.6`` the certified point).
* ``cascade``       -- hard: corroboration votes are the primary sort key, first-stage
  relevance only the tie-break (the ``alpha = 0`` extreme, but relevance -- not doc_id --
  orders equal-vote docs).
* ``lexicographic`` -- conservative: min-max relevance coarsened into ``epsilon`` buckets
  is primary, corroboration secondary, relevance tertiary -- so votes re-order only
  candidates whose relevance sits in the same ``epsilon`` band.

Reports needle-found@k AND MRR for each rule (whether the gain is a top-k-boundary
effect or a genuine rank change -- the MRR-flatness question -- is the whole point),
plus the alpha- and epsilon- sensitivity curves. Writes per-query CSVs for
``eval.significance``. Answers "blend vs cascade vs tie-break" (Experiment A) and the
curve shapes (Experiment B) from ONE dump; the per-scale slurm reuses it at each corpus
size (Experiment C).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from eval.gate_corroboration import per_query_reciprocal_rank
from eval.ir_metrics import Run
from eval.tune_alpha import _grid
from eval.tune_corroboration import load_runs, per_query_hits
from src.retrieval.fusion import convex_fuse, minmax_normalize

# Lexicographic epsilon grid (relevance band width): mirrors the margin thresholds in
# eval.gate_corroboration -- fine near-tie bands up to a coarse third of the range.
EPS_GRID = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]

_RULE_ORDER = ("q2d", "blend", "cascade", "lexicographic")


def _order_to_scores(order: List[str]) -> Dict[str, float]:
    """Encode an explicit best-first ranking as strictly descending scores.

    The shared metric helpers (``per_query_hits`` / ``per_query_reciprocal_rank``) rank
    by score descending, then doc_id ascending; strictly descending distinct scores
    reproduce EXACTLY ``order`` (the doc_id tie-break never fires) -- so a rule that is
    naturally a *sort* (cascade, lexicographic) is scored by the same tested code as the
    convex blend, with no magic-constant score encoding.
    """
    n = len(order)
    return {doc: float(n - i) for i, doc in enumerate(order)}


def cascade_run(relevance_run: Run, corroboration_run: Run) -> Run:
    """Hard cascade: corroboration votes primary, first-stage relevance the tie-break.

    The ``alpha = 0`` extreme, except equal-vote docs keep their *relevance* order
    rather than an arbitrary one -- so a corroborated needle overtakes a more-relevant
    but uncorroborated counterfactual, while relevance still orders same-vote docs.
    """
    fused: Run = {}
    for qid, rel in relevance_run.items():
        corr = corroboration_run.get(qid, {})
        # ascending sort on negated numeric keys => votes desc, relevance desc, doc_id asc.
        order = sorted(rel, key=lambda d: (-corr.get(d, 0.0), -rel[d], d))
        fused[qid] = _order_to_scores(order)
    return fused


def lexicographic_run(relevance_run: Run, corroboration_run: Run, epsilon: float) -> Run:
    """Conservative tie-break: relevance coarsened to ``epsilon`` bands is primary.

    Per query: min-max the relevance, bucket it by ``floor(nd / epsilon)`` (higher
    relevance -> higher bucket), then order by (bucket desc, votes desc, relevance desc,
    doc_id asc). Votes therefore re-order only candidates in the same ``epsilon`` band;
    a relevance gap wider than the band is never overridden. ``epsilon`` in (0, 1].

    Fixed-bin caveat: two values within ``epsilon`` that straddle a bin edge land in
    different buckets and are not swapped -- the epsilon sweep characterises this.
    """
    if not 0.0 < epsilon <= 1.0:
        raise ValueError(f"epsilon must be in (0, 1]; got {epsilon}.")
    fused: Run = {}
    for qid, rel in relevance_run.items():
        corr = corroboration_run.get(qid, {})
        nd = minmax_normalize(rel)
        order = sorted(
            rel, key=lambda d: (-int(nd[d] / epsilon), -corr.get(d, 0.0), -nd[d], d)
        )
        fused[qid] = _order_to_scores(order)
    return fused


def build_rule_runs(
    relevance_run: Run, corroboration_run: Run, alpha: float, epsilon: float
) -> Dict[str, Run]:
    """The four fused runs compared in Experiment A, keyed by rule name."""
    return {
        "q2d": convex_fuse(relevance_run, corroboration_run, 1.0),
        "blend": convex_fuse(relevance_run, corroboration_run, alpha),
        "cascade": cascade_run(relevance_run, corroboration_run),
        "lexicographic": lexicographic_run(relevance_run, corroboration_run, epsilon),
    }


def _mean(values: Dict[str, float]) -> float:
    return sum(values.values()) / len(values) if values else 0.0


def sweep_blend(
    relevance_run: Run,
    corroboration_run: Run,
    needles: Dict[str, str],
    grid: List[float],
    k: int,
) -> List[Tuple[float, float, float]]:
    """``[(alpha, needle-found@k, MRR)]`` over the alpha grid (Experiment B).

    ``alpha`` = relevance weight; ``alpha = 1.0`` is pure q2d. MRR sits alongside
    found@k so the curve exposes BOTH the top-k gain and whether the rank moves.
    """
    out: List[Tuple[float, float, float]] = []
    for alpha in grid:
        fused = convex_fuse(relevance_run, corroboration_run, alpha)
        out.append(
            (
                alpha,
                _mean(per_query_hits(fused, needles, k)),
                _mean(per_query_reciprocal_rank(fused, needles)),
            )
        )
    return out


def sweep_lexicographic(
    relevance_run: Run,
    corroboration_run: Run,
    needles: Dict[str, str],
    eps_grid: List[float],
    k: int,
) -> List[Tuple[float, float, float]]:
    """``[(epsilon, needle-found@k, MRR)]`` over the epsilon grid (Experiment B)."""
    out: List[Tuple[float, float, float]] = []
    for eps in eps_grid:
        fused = lexicographic_run(relevance_run, corroboration_run, eps)
        out.append(
            (
                eps,
                _mean(per_query_hits(fused, needles, k)),
                _mean(per_query_reciprocal_rank(fused, needles)),
            )
        )
    return out


def _write_curve(
    rows: List[Tuple[float, float, float]], path: Path, xname: str, k: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([xname, f"needle_found@{k}", "mrr"])
        for x, found, mrr in rows:
            writer.writerow([x, round(found, 4), round(mrr, 4)])


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m eval.compare_rules",
        description="Offline comparison of corroboration combination rules (blend vs "
        "cascade vs lexicographic tie-break) from a runs dump: needle-found@k + MRR + "
        "alpha/epsilon sensitivity curves. Pure arithmetic, no GPU.",
    )
    p.add_argument("--from-runs", type=Path, required=True, dest="from_runs",
                   help="Runs dump JSON from eval.tune_corroboration --dump-runs.")
    p.add_argument("--k", type=int, default=10, help="needle-found cut-off (default: %(default)s).")
    p.add_argument("--alpha", type=float, default=0.6,
                   help="Blend relevance weight (default: certified %(default)s).")
    p.add_argument("--epsilon", type=float, default=0.10,
                   help="Lexicographic relevance band (default: %(default)s).")
    p.add_argument("--alpha-step", type=float, default=0.1, dest="alpha_step",
                   help="Alpha-curve grid step (default: %(default)s).")
    p.add_argument("--out-dir", type=Path, default=Path("results"), dest="out_dir")
    p.add_argument("--tag", default="nq300", help="Filename tag (default: %(default)s).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    from eval.run_benchmark import write_per_query_csv  # lazy: run_benchmark is heavy

    args = _parse_args(argv)
    relevance_run, corroboration_run, needles = load_runs(args.from_runs)

    # Experiment A: the four rules at fixed (alpha, epsilon), per-query hits + MRR.
    runs = build_rule_runs(relevance_run, corroboration_run, args.alpha, args.epsilon)
    hits = {name: per_query_hits(runs[name], needles, args.k) for name in _RULE_ORDER}
    mrr = {name: per_query_reciprocal_rank(runs[name], needles) for name in _RULE_ORDER}
    hits_csv = args.out_dir / f"compare_rules_{args.tag}_per_query_hits.csv"
    mrr_csv = args.out_dir / f"compare_rules_{args.tag}_per_query_mrr.csv"
    write_per_query_csv(hits, hits_csv)
    write_per_query_csv(mrr, mrr_csv)

    # Experiment B: alpha- and epsilon- sensitivity curves (found@k AND MRR).
    alpha_curve = sweep_blend(
        relevance_run, corroboration_run, needles, _grid(args.alpha_step), args.k
    )
    eps_curve = sweep_lexicographic(
        relevance_run, corroboration_run, needles, EPS_GRID, args.k
    )
    _write_curve(alpha_curve, args.out_dir / f"compare_rules_{args.tag}_alpha_curve.csv", "alpha", args.k)
    _write_curve(eps_curve, args.out_dir / f"compare_rules_{args.tag}_eps_curve.csv", "epsilon", args.k)

    print(f"queries={len(needles)}  k={args.k}  alpha={args.alpha}  epsilon={args.epsilon}")
    print(f"rule            needle_found@{args.k}   MRR")
    for name in _RULE_ORDER:
        print(f"  {name:<14} {_mean(hits[name]):.4f}           {_mean(mrr[name]):.4f}")
    print(f"wrote {hits_csv}, {mrr_csv}, and the alpha/epsilon curves")
    print("significance (run on both the hits and the MRR CSV):")
    for ref in ("q2d", "blend"):
        print(f"  python -m eval.significance --per-query-csv {hits_csv} --reference {ref}")
        print(f"  python -m eval.significance --per-query-csv {mrr_csv} --reference {ref}")


if __name__ == "__main__":
    main()
