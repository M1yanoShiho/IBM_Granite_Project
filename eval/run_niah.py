"""NIAH evaluation harness — how well a retriever finds the DESIGNATED needle.

Loads a task recipe (:func:`eval.build_niah_task.load_niah_task`), rebuilds the
haystack (background + needles + distractors) — optionally at a different corpus
size for the Phase-1 scale sweep (``--max-docs``) — builds the named retrievers
over it (reusing ``run_benchmark._build_retrievers``, so every retriever/index
type works unchanged), and scores each by **needle-found@k** and **MRR** of each
query's single designated needle. Writes an aggregate CSV + a per-query CSV for
``eval.significance`` (so a cross-retriever gain gets a p-value).

    python -m eval.run_niah --task results/niah_nq_task.json \
        --retrievers granite_dense splade convex_hybrid_granite_splade --k 10
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

from src.niah.hardness_gate import hit_at_k, mean_recall, reciprocal_rank
from src.niah.types import NiahTask


def evaluate_retriever_on_task(retriever, task: NiahTask, k: int) -> Dict[str, Dict[str, float]]:
    """Per-query needle-found@k and reciprocal-rank of each DESIGNATED needle.

    Deduplicates the retriever's chunks to a doc-level best-first ranking, then
    scores the single designated needle per query (queries without one are skipped).
    Returns ``{"needle_found": {qid: 0/1}, "mrr": {qid: 1/rank}}``.
    """
    hits: Dict[str, float] = {}
    rrs: Dict[str, float] = {}
    for ex in task.examples:
        if ex.needle_id is None:
            continue
        ranked = list(dict.fromkeys(c.doc_id for c in retriever.retrieve(ex.query)))
        hits[ex.query_id] = hit_at_k(ranked, ex.needle_id, k)
        rrs[ex.query_id] = reciprocal_rank(ranked, ex.needle_id)
    return {"needle_found": hits, "mrr": rrs}


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m eval.run_niah",
        description="Score retrievers on a NIAH task by needle-found@k / MRR.",
    )
    p.add_argument("--task", type=Path, required=True, help="Task recipe JSON from build_niah_task.")
    p.add_argument("--retrievers", nargs="+", default=["granite_dense"])
    p.add_argument("--k", type=int, default=10, help="needle-found cut-off (default: %(default)s).")
    p.add_argument(
        "--max-docs", type=int, default=None, dest="max_docs",
        help="Override the recipe corpus size — one point of the scale sweep (default: recipe's).",
    )
    p.add_argument(
        "--index-type", default="hnsw", dest="index_type",
        choices=["flat", "hnsw", "ivf", "ivfpq"],
    )
    p.add_argument("--pq-nbits", type=int, default=8, dest="pq_nbits")
    p.add_argument("--out", type=Path, default=Path("results/niah_eval.csv"))
    p.add_argument("--per-query-out", type=Path, default=None, dest="per_query_out")
    p.add_argument("--cache-dir", type=Path, default=None, dest="cache_dir")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    # Heavy deps imported lazily so unit tests of the scoring core stay light.
    from eval.benchmarks.loader import BenchmarkData
    from eval.build_niah_task import load_niah_task
    from eval.run_benchmark import BenchmarkConfig, _build_retrievers, write_per_query_csv

    task = load_niah_task(args.task, max_docs=args.max_docs)
    data = BenchmarkData(corpus=task.corpus, queries=task.queries, qrels=task.qrels)
    config = BenchmarkConfig(
        dataset="niah",
        retrievers=args.retrievers,
        k_values=[args.k],
        index_type=args.index_type,
        pq_nbits=args.pq_nbits,
        chunk_unit="token",
        index_cache_dir=args.cache_dir,
    )
    retrievers = _build_retrievers(config, data)

    rows = []
    per_query: Dict[str, Dict[str, float]] = {}
    n_docs = len(task.corpus)
    for name in args.retrievers:
        scored = evaluate_retriever_on_task(retrievers[name], task, args.k)
        nf = mean_recall(scored["needle_found"])
        mrr = mean_recall(scored["mrr"])
        rows.append(
            {"retriever": name, "n_docs": n_docs,
             f"needle_found@{args.k}": round(nf, 4), "mrr": round(mrr, 4)}
        )
        per_query[name] = scored["needle_found"]
        print(f"{name}: needle_found@{args.k}={nf:.3f} MRR={mrr:.3f} (n_docs={n_docs})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")

    if args.per_query_out is not None:
        write_per_query_csv(per_query, args.per_query_out)
        print(f"Wrote {args.per_query_out} (per-query needle-found for eval.significance)")


if __name__ == "__main__":
    main()
