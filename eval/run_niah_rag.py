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
