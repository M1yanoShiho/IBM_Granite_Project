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
import json
from pathlib import Path
from typing import Dict

from src.niah.hardness_gate import hit_at_k, mean_recall, reciprocal_rank
from src.niah.types import NiahTask


def evaluate_retriever_on_task(retriever, task: NiahTask, k: int) -> Dict[str, Dict]:
    """Per-query needle-found@k, reciprocal-rank, and the FULL doc-level run.

    Deduplicates the retriever's chunks to a doc-level best-first ranking (a doc's
    duplicate chunks collapse to its best = first-seen score), then scores the single
    designated needle per query (queries without one are skipped). Returns
    ``{"needle_found": {qid: 0/1}, "mrr": {qid: 1/rank}, "run": {qid: {doc_id: score}}}``
    -- the run is what ``--dump-runs`` persists (WS-0 item 2) so the burial /
    migration / cascade / oracle analyses (WS-2/3/4/5) replay offline.
    """
    hits: Dict[str, float] = {}
    rrs: Dict[str, float] = {}
    runs: Dict[str, Dict[str, float]] = {}
    for ex in task.examples:
        if ex.needle_id is None:
            continue
        doc_scores: Dict[str, float] = {}
        for c in retriever.retrieve(ex.query):
            if c.doc_id not in doc_scores:
                doc_scores[c.doc_id] = float(c.score)
        ranked = list(doc_scores)
        hits[ex.query_id] = hit_at_k(ranked, ex.needle_id, k)
        rrs[ex.query_id] = reciprocal_rank(ranked, ex.needle_id)
        runs[ex.query_id] = doc_scores
    return {"needle_found": hits, "mrr": rrs, "run": runs}


def _dump_path(base: Path, retriever_name: str) -> Path:
    """``results/x.json`` + ``granite_dense`` -> ``results/x_granite_dense.json``."""
    return base.with_name(f"{base.stem}_{retriever_name}{base.suffix}")


def dump_niah_run(
    run: Dict[str, Dict[str, float]],
    needles: Dict[str, str],
    retriever_name: str,
    k: int,
    path: Path,
) -> None:
    """Persist one retriever's full doc-level ranking + needles as a self-contained
    JSON (rank order = dict insertion order; scores keep top1-top2 margins for WS-4)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"retriever": retriever_name, "k": k, "needles": needles, "run": run}
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_niah_run(path: Path) -> Dict:
    """Load a :func:`dump_niah_run` payload (``retriever`` / ``k`` / ``needles`` / ``run``)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
    p.add_argument(
        "--dump-runs", type=Path, default=None, dest="dump_runs",
        help="Dump each retriever's FULL doc-level ranking (run + needles) to "
             "<stem>_<retriever><suffix> JSON — the WS-0 raw material for the offline "
             "burial/migration/cascade/oracle analyses (WS-2/3/4/5).",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    # Heavy deps imported lazily so unit tests of the scoring core stay light.
    from eval.benchmarks.loader import BenchmarkData
    from eval.run_benchmark import (
        BenchmarkConfig,
        _build_retrievers,
        retrievers_need_llm,
        write_per_query_csv,
    )
    from eval.build_niah_task import load_niah_task

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
    # Build ONE LLMClient and share it across every transform / LLM reranker in this
    # run; without this each (hyde/q2d/listrank) loads its own 3B and several in one
    # job OOM the GPU. Pure dense/sparse/cross-encoder runs need no LLM at all.
    llm = None
    if retrievers_need_llm(args.retrievers):
        from src.llm_client import LLMClient

        llm = LLMClient()
    retrievers = _build_retrievers(config, data, llm=llm)

    rows = []
    per_query: Dict[str, Dict[str, float]] = {}
    n_docs = len(task.corpus)
    needles = {ex.query_id: ex.needle_id for ex in task.examples if ex.needle_id is not None}
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
        if args.dump_runs is not None:
            dump_path = _dump_path(args.dump_runs, name)
            dump_niah_run(scored["run"], needles, name, args.k, dump_path)
            print(f"dumped {name} run to {dump_path} (offline WS-2/3/4/5 raw material)")

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
