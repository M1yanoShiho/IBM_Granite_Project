"""Bridge a NIAH task into eval.run_rag: measure RAG answer quality (cover-EM) over the
counterfactual-distractor haystack, across retrievers (dense -> q2d -> q2d_corroborate).

A ``NiahTask`` is an ordinary (corpus, queries, qrels) benchmark plus gold answers, so
it becomes a ``BenchmarkData`` that ``run_rag.run`` scores unchanged. One shared
``LLMClient`` serves the q2d transform, the corroboration reranking, and generation (the
OOM-safe pattern from run_niah). Design: docs/superpowers/specs/2026-07-07-niah-rag-bridge-design.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List, Optional

from eval.benchmarks.loader import BenchmarkData
from src.niah.types import NiahTask


def niah_to_benchmark_data(task: NiahTask) -> BenchmarkData:
    """Wrap a NIAH task (corpus WITH counterfactual distractors + queries + qrels + gold
    answers) as a ``BenchmarkData`` for RAG scoring. Raises if the task has no answers --
    cover-EM needs gold to score against."""
    if not task.answers:
        raise ValueError(
            "NIAH task carries no gold answers -- cover-EM needs them. Load a recipe "
            "whose source dataset provides answers (e.g. nq)."
        )
    return BenchmarkData(
        corpus=task.corpus, queries=task.queries, qrels=task.qrels, answers=task.answers
    )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m eval.run_niah_rag",
        description="RAG answer quality (cover-EM) over the NIAH counterfactual haystack, "
        "across retrievers. Bridges a NIAH task into eval.run_rag.",
    )
    p.add_argument("--task", type=Path, required=True,
                   help="NIAH recipe JSON (loaded via load_niah_task).")
    p.add_argument("--retrievers", nargs="+",
                   default=["granite_dense", "q2d_granite", "q2d_corroborate"],
                   help="Retriever arms to compare (default: %(default)s).")
    p.add_argument("--top-k", type=int, default=4, dest="top_k",
                   help="RAG context size -- chunks passed to the generator (default: %(default)s).")
    p.add_argument("--max-docs", type=int, default=None, dest="max_docs",
                   help="Override the recipe corpus cap (scale sweep).")
    p.add_argument("--out", type=Path, default=Path("results/rag_niah_agg.csv"))
    p.add_argument("--per-query-out", type=Path, default=Path("results/rag_niah_cmp"),
                   dest="per_query_out",
                   help="Per-metric per-query CSV prefix (one column per retriever).")
    p.add_argument("--predictions-out", type=Path, default=None, dest="predictions_out")
    return p.parse_args(argv)


def run_niah_rag(
    task: NiahTask,
    retrievers: List[str],
    llm,
    top_k: int,
    out: Path,
    per_query_out: Path,
    predictions_out: Optional[Path] = None,
    run_fn: Optional[Callable] = None,
) -> None:
    """Drive ``run_rag.run`` once per retriever over the SAME NIAH ``BenchmarkData`` and
    the SAME ``llm`` (shared -> no duplicate model loads). ``append`` accumulates all arms
    into one per-metric per-query CSV (column per retriever) for paired significance.
    ``run_fn`` is injected in tests; production uses ``eval.run_rag.run``."""
    from eval.run_rag import RAGEvalConfig

    if run_fn is None:
        from eval.run_rag import run as run_fn

    data = niah_to_benchmark_data(task)
    for name in retrievers:
        config = RAGEvalConfig(
            dataset="niah",
            retriever=name,
            top_k=top_k,
            results_path=out,
            append=True,
            per_query_out=per_query_out,
            predictions_out=predictions_out,
        )
        run_fn(config, data=data, llm=llm)


def main(argv: Optional[List[str]] = None) -> None:
    from eval.build_niah_task import load_niah_task
    from src.llm_client import LLMClient

    args = _parse_args(argv)
    task = load_niah_task(args.task, max_docs=args.max_docs)
    llm = LLMClient()  # one client: q2d transform + corroboration reranking + generation
    run_niah_rag(
        task, args.retrievers, llm, args.top_k, args.out,
        args.per_query_out, args.predictions_out,
    )
    cover = f"{args.per_query_out}_answer_cover.csv"
    print(f"wrote {cover} -> significance:")
    print(f"  python -m eval.significance --per-query-csv {cover} --reference granite_dense")


if __name__ == "__main__":
    main()
