"""Offline alpha sweep + dev-tuning for the convex-combination hybrid.

Convex fusion is deterministic given each arm's per-query scores, so we retrieve
ONCE per arm and then sweep alpha as pure arithmetic — no re-retrieval per alpha.

Outputs:
- tune: the best alpha on a dev/train split (argmax nDCG@10) — the deploy point.
- curve: nDCG@10 vs alpha on the test split — the headline artifact ("does ANY
  alpha beat pure dense?"). alpha = 1.0 in the curve IS pure dense, so it should
  match the published granite_dense nDCG@10 (a consistency check).

Reuses the benchmark's arm-building (``run_benchmark._build_component`` + ``build_run``)
and metrics (``ir_metrics.ndcg_at_k``); the fusion maths come from
``src.retrieval.fusion.convex_fuse``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple

from eval.benchmarks.loader import load_benchmark
from eval.ir_metrics import Qrels, Run, ndcg_at_k
from eval.run_benchmark import BenchmarkConfig, _build_component, build_run
from src.retrieval.fusion import convex_fuse


def _arm_runs(
    dataset: str,
    split: str,
    chunk_unit: str,
    lexical: str,
    pool_depth: int,
    cache_dir: Optional[Path],
    max_queries: Optional[int],
    max_docs: Optional[int],
) -> Tuple[Run, Run, Qrels]:
    """Retrieve once per arm on a split; return ``(dense_run, lexical_run, qrels)``.

    Each arm is built and its run trimmed to ``pool_depth`` docs per query, so fusion
    has a deep, equal-depth pool to reorder. ``lexical`` is the lexical arm's retriever
    name (``"bm25"`` or ``"splade"``). Reuses the benchmark's component builder (so the
    dense arm is identical to the audited ``granite_dense``) and its index cache, so the
    test-split index is reused by the final ``run_benchmark`` run.
    """
    data = load_benchmark(
        dataset, split=split, max_queries=max_queries, max_docs=max_docs
    )
    config = BenchmarkConfig(
        dataset=dataset, split=split, chunk_unit=chunk_unit, index_cache_dir=cache_dir
    )
    doc_ids = list(data.corpus.keys())
    corpus = list(data.corpus.values())
    dense = _build_component("granite_dense", config, data, corpus, doc_ids, pool_depth)
    lex = _build_component(lexical, config, data, corpus, doc_ids, pool_depth)
    dense_run = build_run(dense, data.queries, pooling="max", top_n_docs=pool_depth)
    lex_run = build_run(lex, data.queries, pooling="max", top_n_docs=pool_depth)
    return dense_run, lex_run, data.qrels


def sweep(
    dense_run: Run,
    bm25_run: Run,
    qrels: Qrels,
    grid: List[float],
) -> List[Tuple[float, float]]:
    """Return ``[(alpha, ndcg@10)]`` over the grid (pure arithmetic, no retrieval)."""
    out: List[Tuple[float, float]] = []
    for alpha in grid:
        fused = convex_fuse(dense_run, bm25_run, alpha)
        out.append((alpha, ndcg_at_k(fused, qrels, 10)))
    return out


def best_alpha(curve: List[Tuple[float, float]]) -> Tuple[float, float]:
    """The ``(alpha, ndcg)`` with the highest nDCG; ties break to the smaller alpha
    (less reliance on the weaker lexical arm)."""
    return max(curve, key=lambda pair: (pair[1], -pair[0]))


def _grid(step: float) -> List[float]:
    """alpha grid ``0.0, step, ..., 1.0`` inclusive."""
    n = round(1.0 / step)
    return [round(i * step, 4) for i in range(n + 1)]


def write_curve(curve: List[Tuple[float, float]], path: Path) -> None:
    """Write the ``alpha,ndcg@10`` curve CSV (the headline figure)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "ndcg@10"])
        for alpha, ndcg in curve:
            writer.writerow([alpha, ndcg])


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m eval.tune_alpha",
        description="Sweep/tune the convex-hybrid alpha (dense vs a lexical arm) without test leakage.",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tune-split", default="dev",
                        help="Split to pick alpha on, e.g. dev or train (default: %(default)s).")
    parser.add_argument("--test-split", default="test",
                        help="Split to report the alpha-curve on (default: %(default)s).")
    parser.add_argument("--chunk-unit", default="word", choices=["word", "token"],
                        help="Chunking unit for the dense arm (default: %(default)s).")
    parser.add_argument("--lexical", default="bm25",
                        choices=["bm25", "strong_bm25", "splade"],
                        help="Lexical arm fused with granite_dense (default: %(default)s).")
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--pool-depth", type=int, default=100)
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Cache dense indexes here (shared with run_benchmark).")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="Write the TEST alpha-curve CSV here (alpha,ndcg@10).")
    parser.add_argument("--best-alpha-out", type=Path, default=None,
                        help="Write the chosen alpha (a float) here, for the run script.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    grid = _grid(args.alpha_step)

    # Tune on the dev/train split.
    d_run, b_run, qrels = _arm_runs(
        args.dataset, args.tune_split, args.chunk_unit, args.lexical, args.pool_depth,
        args.cache_dir, args.max_queries, args.max_docs,
    )
    a_star, ndcg_dev = best_alpha(sweep(d_run, b_run, qrels, grid))
    print(f"[tune:{args.tune_split}] best alpha = {a_star} (nDCG@10 = {ndcg_dev:.4f})")

    # Report the curve on the test split.
    td_run, tb_run, tqrels = _arm_runs(
        args.dataset, args.test_split, args.chunk_unit, args.lexical, args.pool_depth,
        args.cache_dir, args.max_queries, args.max_docs,
    )
    test_curve = sweep(td_run, tb_run, tqrels, grid)
    by_alpha = dict(test_curve)
    delta = by_alpha[a_star] - by_alpha[1.0]
    print(
        f"[test:{args.test_split}] nDCG@10 @ alpha*={a_star}: {by_alpha[a_star]:.4f} "
        f"| pure dense (alpha=1.0): {by_alpha[1.0]:.4f} | delta: {delta:+.4f}"
    )
    if args.out is not None:
        write_curve(test_curve, args.out)
        print(f"wrote test alpha-curve to {args.out}")
    if args.best_alpha_out is not None:
        args.best_alpha_out.parent.mkdir(parents=True, exist_ok=True)
        args.best_alpha_out.write_text(str(a_star))


if __name__ == "__main__":
    main()
