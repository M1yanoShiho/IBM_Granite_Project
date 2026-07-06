"""Offline lambda-sweep for the corroboration reranker (mirrors ``eval/tune_alpha``).

The blend ``final = alpha*relevance + (1-alpha)*corroboration`` is deterministic given
each candidate's (relevance, corroboration) scores, so we extract answers ONCE per query
and sweep ``alpha`` as pure arithmetic -- no re-extraction per alpha. ``alpha`` is the
RELEVANCE weight: ``alpha = 1.0`` is pure first-stage (q2d) and should reproduce its
needle-found@k (a consistency check); the curve answers "does ANY alpha beat pure q2d?".

Report the whole curve (a sensitivity analysis, no test-set cherry-picking) -- a single
blind alpha can be deceptive (the convex-hybrid lesson). Reuses
``src.retrieval.fusion.convex_fuse`` (min-max + convex blend) and ``eval.tune_alpha._grid``.
Metric = needle-found@k of each query's DESIGNATED needle (single-target NIAH), not ndcg.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from eval.ir_metrics import Run
from eval.tune_alpha import _grid
from src.retrieval.base import RetrievedChunk
from src.retrieval.fusion import convex_fuse


def dedup_docs(chunks: List[RetrievedChunk], top_n: int) -> List[RetrievedChunk]:
    """First (best) chunk per doc_id, best-first, capped at ``top_n`` (doc-level pool).

    Matches ``run_niah``'s doc-level dedup so the sweep scores the same units the metric
    does; chunk order is best-first, so the first occurrence of a doc is its best chunk.
    """
    seen: set[str] = set()
    out: List[RetrievedChunk] = []
    for chunk in chunks:
        if chunk.doc_id in seen:
            continue
        seen.add(chunk.doc_id)
        out.append(chunk)
        if len(out) >= top_n:
            break
    return out


def build_corroboration_runs(
    task, retriever, reranker, top_n: int, max_queries: Optional[int] = None
) -> Tuple[Run, Run, Dict[str, str]]:
    """Extract answers ONCE per query; return ``(relevance_run, corroboration_run, needles)``.

    ``reranker`` needs a ``score_docs(query, docs) -> (relevance, corroboration)`` method
    (:class:`~src.retrieval.reranker.CorroborationReranker`). ``needles`` maps qid -> the
    designated needle doc_id; examples without a designated needle are skipped.
    ``max_queries`` caps the number of scored queries (in task-example order) so a fast
    first-read on a subset finishes within the wall clock -- the extraction is the cost.
    """
    relevance_run: Run = {}
    corroboration_run: Run = {}
    needles: Dict[str, str] = {}
    for ex in task.examples:
        if ex.needle_id is None:
            continue
        if max_queries is not None and len(needles) >= max_queries:
            break
        docs = dedup_docs(retriever.retrieve(ex.query), top_n)
        relevance, corroboration = reranker.score_docs(ex.query, docs)
        relevance_run[ex.query_id] = {docs[i].doc_id: relevance[i] for i in range(len(docs))}
        corroboration_run[ex.query_id] = {docs[i].doc_id: corroboration[i] for i in range(len(docs))}
        needles[ex.query_id] = ex.needle_id
    return relevance_run, corroboration_run, needles


def per_query_hits(fused: Run, needles: Dict[str, str], k: int) -> Dict[str, float]:
    """Per-query ``1.0``/``0.0`` -- is each query's designated needle in ``fused``'s
    top-``k``. The per-query vector a paired significance test consumes (vs the scalar
    mean below)."""
    hits: Dict[str, float] = {}
    for qid, needle in needles.items():
        scores = fused.get(qid, {})
        # Deterministic tie-break: score descending, then doc_id ascending, so an
        # exact score tie no longer resolves by hash-seed-dependent dict order
        # (needle-found@k must be reproducible across runs).
        ranked = sorted(sorted(scores), key=lambda d: scores[d], reverse=True)
        hits[qid] = 1.0 if needle in ranked[:k] else 0.0
    return hits


def needle_found_at_k(fused: Run, needles: Dict[str, str], k: int) -> float:
    """Fraction of queries whose designated needle is in the top-``k`` of ``fused``."""
    hits = per_query_hits(fused, needles, k)
    return sum(hits.values()) / len(hits) if hits else 0.0


def dump_runs(
    relevance_run: Run, corroboration_run: Run, needles: Dict[str, str], path: Path
) -> None:
    """Persist the two runs + needles as JSON, so significance / re-sweeps run OFFLINE
    (``--from-runs``) without re-extracting -- the extraction is the expensive step."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "relevance_run": relevance_run,
                "corroboration_run": corroboration_run,
                "needles": needles,
            }
        ),
        encoding="utf-8",
    )


def load_runs(path: Path) -> Tuple[Run, Run, Dict[str, str]]:
    """Load ``(relevance_run, corroboration_run, needles)`` dumped by :func:`dump_runs`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["relevance_run"], payload["corroboration_run"], payload["needles"]


def sweep_corroboration(
    relevance_run: Run,
    corroboration_run: Run,
    needles: Dict[str, str],
    grid: List[float],
    k: int = 10,
) -> List[Tuple[float, float]]:
    """``[(alpha, needle-found@k)]`` over the grid; ``alpha`` = relevance weight.

    Pure arithmetic on the two cached runs -- no retrieval or extraction per alpha.
    """
    out: List[Tuple[float, float]] = []
    for alpha in grid:
        fused = convex_fuse(relevance_run, corroboration_run, alpha)
        out.append((alpha, needle_found_at_k(fused, needles, k)))
    return out


def best_alpha(curve: List[Tuple[float, float]]) -> Tuple[float, float]:
    """The ``(alpha, score)`` with the highest needle-found; ties break to the LARGER
    alpha (prefer the established relevance signal -- conservative / Occam)."""
    return max(curve, key=lambda pair: (pair[1], pair[0]))


def write_curve(curve: List[Tuple[float, float]], path: Path, k: int) -> None:
    """Write the ``alpha,needle_found@k`` curve CSV (the sensitivity-analysis artifact)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", f"needle_found@{k}"])
        for alpha, score in curve:
            writer.writerow([alpha, round(score, 4)])


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m eval.tune_corroboration",
        description="Sweep the corroboration reranker's blend weight (relevance vs "
        "corroboration) without re-extraction; report the needle-found@k curve.",
    )
    p.add_argument("--task", type=Path, default=None, help="NIAH task recipe JSON (or use --from-runs).")
    p.add_argument("--first-stage", default="q2d_granite", dest="first_stage")
    p.add_argument("--k", type=int, default=10, help="needle-found cut-off (default: %(default)s).")
    p.add_argument("--top-n", type=int, default=20, dest="top_n",
                   help="Docs re-scored per query (default: %(default)s).")
    p.add_argument("--max-queries", type=int, default=None, dest="max_queries",
                   help="Cap scored queries for a fast first-read subset (default: all).")
    p.add_argument("--alpha-step", type=float, default=0.1, dest="alpha_step")
    p.add_argument("--no-parametric", dest="use_parametric", action="store_false",
                   help="Disable the parametric-knowledge voter.")
    p.set_defaults(use_parametric=True)
    p.add_argument("--max-docs", type=int, default=None, dest="max_docs")
    p.add_argument("--index-type", default="hnsw", dest="index_type",
                   choices=["flat", "hnsw", "ivf", "ivfpq"])
    p.add_argument("--cache-dir", type=Path, default=None, dest="cache_dir")
    p.add_argument("--out", type=Path, default=Path("results/corroboration_alpha_curve.csv"))
    p.add_argument("--per-query-out", type=Path, default=None, dest="per_query_out",
                   help="Write per-query needle-found at alpha=1.0 (q2d) and alpha* "
                        "(q2d_corroborate) -> feed eval.significance for the p-value.")
    p.add_argument("--dump-runs", type=Path, default=None, dest="dump_runs",
                   help="Dump the extracted (relevance, corroboration) runs as JSON for "
                        "offline re-analysis (avoids re-extraction).")
    p.add_argument("--from-runs", type=Path, default=None, dest="from_runs",
                   help="Load dumped runs and sweep OFFLINE (skips the LLM extraction).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    if args.from_runs is not None:
        # Offline: reuse a previous extraction; no model loads, no GPU.
        relevance_run, corroboration_run, needles = load_runs(args.from_runs)
    else:
        if args.task is None:
            raise SystemExit("give --task (to extract) or --from-runs (to reuse a dump).")
        # Heavy deps imported lazily so the unit tests of the pure logic stay light.
        from eval.benchmarks.loader import BenchmarkData
        from eval.build_niah_task import load_niah_task
        from eval.run_benchmark import BenchmarkConfig, _build_named
        from src.llm_client import LLMClient
        from src.retrieval.reranker import CorroborationReranker

        task = load_niah_task(args.task, max_docs=args.max_docs)
        data = BenchmarkData(corpus=task.corpus, queries=task.queries, qrels=task.qrels)
        config = BenchmarkConfig(
            dataset="niah",
            retrievers=[args.first_stage],
            k_values=[args.k],
            index_type=args.index_type,
            chunk_unit="token",
            index_cache_dir=args.cache_dir,
        )
        llm = LLMClient()
        doc_ids = list(data.corpus.keys())
        corpus = list(data.corpus.values())
        retriever = _build_named(args.first_stage, config, data, corpus, doc_ids, args.k, llm=llm)
        reranker = CorroborationReranker(llm, top_n=args.top_n, use_parametric=args.use_parametric)
        relevance_run, corroboration_run, needles = build_corroboration_runs(
            task, retriever, reranker, args.top_n, max_queries=args.max_queries
        )
        if args.dump_runs is not None:
            dump_runs(relevance_run, corroboration_run, needles, args.dump_runs)
            print(f"dumped runs to {args.dump_runs} (re-analyse offline with --from-runs)")

    curve = sweep_corroboration(relevance_run, corroboration_run, needles, _grid(args.alpha_step), args.k)
    a_star, best = best_alpha(curve)
    pure_rel = dict(curve).get(1.0)
    print(f"first_stage={args.first_stage}  queries={len(needles)}  top_n={args.top_n}")
    for alpha, score in curve:
        print(f"  alpha={alpha:.2f}  needle_found@{args.k}={score:.4f}")
    if pure_rel is not None:
        print(
            f"alpha*={a_star} needle_found@{args.k}={best:.4f} | "
            f"pure relevance (alpha=1.0)={pure_rel:.4f} | delta={best - pure_rel:+.4f}"
        )
    write_curve(curve, args.out, args.k)
    print(f"wrote {args.out}")

    if args.per_query_out is not None:
        from eval.run_benchmark import write_per_query_csv

        hits_q2d = per_query_hits(convex_fuse(relevance_run, corroboration_run, 1.0), needles, args.k)
        hits_corrob = per_query_hits(convex_fuse(relevance_run, corroboration_run, a_star), needles, args.k)
        write_per_query_csv({"q2d": hits_q2d, "q2d_corroborate": hits_corrob}, args.per_query_out)
        print(
            f"wrote per-query {args.per_query_out} -> significance: "
            f"python -m eval.significance --per-query-csv {args.per_query_out} --reference q2d"
        )


if __name__ == "__main__":
    main()
