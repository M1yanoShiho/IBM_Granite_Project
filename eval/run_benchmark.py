"""Primary evaluation entry point: system vs. baselines on standard benchmarks.

Orchestrates the headline experiment of the project:

1. Load a standard benchmark (corpus, queries, qrels) via ``eval.benchmarks``.
2. Index the corpus once (``src.ingestion.indexer``).
3. Run each retriever over all queries:
   - the **Granite dense retriever** (the delivered system), and
   - the **BM25 baseline** (and optionally a sentence-transformers dense baseline).
4. Score every run with ``eval.ir_metrics`` (precision/recall/nDCG/MRR).
5. Save a comparison table for the report.

Run from the project root once the retrievers are wired:

    python -m eval.run_benchmark

The orchestration, scoring, CSV output, and the BM25 baseline path are
implemented and unit-tested. The dense retrievers (``granite_dense`` /
``st_dense``) are pending P5's vector index — see :func:`_build_retrievers`
and contract 5 in ``docs/interfaces.md``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from eval.benchmarks.loader import BenchmarkData, load_benchmark
from eval.ir_metrics import Run, evaluate_run
from src.retrieval.base import Retriever
from src.retrieval.bm25_baseline import BM25Retriever


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark evaluation run.

    Attributes
    ----------
    dataset:
        Benchmark name to evaluate on (e.g. ``"scifact"``, ``"nq"``).
    k_values:
        Cut-offs at which to report precision/recall/nDCG.
    retrievers:
        Which retrievers to compare (system + baselines).
    results_path:
        Where to write the comparison table (CSV).
    """

    dataset: str = "scifact"
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    retrievers: List[str] = field(
        default_factory=lambda: ["granite_dense", "bm25", "st_dense"]
    )
    results_path: Path = Path("results/benchmark_results.csv")


def build_run(retriever: Retriever, queries: Dict[str, str]) -> Run:
    """Aggregate a retriever's chunk-level results into a doc-level ``Run``.

    For each query, call ``retriever.retrieve`` and collapse the returned chunks
    to one score per ``doc_id`` by **max-pool** (contract 3): a document's score
    is the highest score among its retrieved chunks. On unchunked corpora
    (e.g. SciFact) this is a pass-through; it only bites once documents are split
    into chunks (e.g. NQ). The result is keyed by query id, ready to hand to
    ``eval.ir_metrics.evaluate_run`` (contract 2).
    """
    run: Run = {}
    for query_id, query_text in queries.items():
        doc_scores: Dict[str, float] = {}
        for chunk in retriever.retrieve(query_text):
            if chunk.score > doc_scores.get(chunk.doc_id, float("-inf")):
                doc_scores[chunk.doc_id] = chunk.score
        run[query_id] = doc_scores
    return run


def evaluate_one(
    retriever: Retriever,
    data: BenchmarkData,
    k_values: List[int] | None = None,
) -> Dict[str, float]:
    """Score one retriever on a benchmark.

    Builds the retriever's doc-level ``Run`` (:func:`build_run`, contract 3),
    then scores it against the benchmark's qrels with
    ``eval.ir_metrics.evaluate_run`` (contract 2). Returns the metric suite
    (precision/recall/nDCG at each k, plus MRR).
    """
    run = build_run(retriever, data.queries)
    return evaluate_run(run, data.qrels, k_values)


def write_results_csv(results: Dict[str, Dict[str, float]], path: Path) -> None:
    """Write the system-vs-baseline comparison table to ``path`` as CSV.

    ``results`` maps each retriever name to its metric suite (the output of
    :func:`evaluate_one`). Emits one row per retriever: a leading ``retriever``
    column followed by the metric columns — the table the report and the P7
    plots consume. All retrievers are scored on the same ``k_values``, so they
    share metric keys and line up into a rectangular table.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["retriever"] + list(next(iter(results.values())).keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, metrics in results.items():
            writer.writerow({"retriever": name, **metrics})


def _build_retrievers(
    config: BenchmarkConfig,
    data: BenchmarkData,
) -> Dict[str, Retriever]:
    """Construct the retrievers named in ``config`` over ``data.corpus``.

    The BM25 baseline builds its own term index from the corpus, so it is wired
    here directly. The dense retrievers (``granite_dense`` / ``st_dense``) need
    P5's vector index and are not available yet — see contract 5 in
    ``docs/interfaces.md``.
    """
    doc_ids = list(data.corpus.keys())
    corpus = list(data.corpus.values())
    top_k = max(config.k_values)

    retrievers: Dict[str, Retriever] = {}
    for name in config.retrievers:
        if name == "bm25":
            retrievers[name] = BM25Retriever(corpus, doc_ids, top_k=top_k)
        else:
            raise NotImplementedError(
                f"Retriever '{name}' needs P5's vector index (contract 5 in "
                "docs/interfaces.md). Run with retrievers=['bm25'] until it lands."
            )
    return retrievers


def run(
    config: BenchmarkConfig,
    retrievers: Dict[str, Retriever] | None = None,
    data: BenchmarkData | None = None,
) -> None:
    """Run the system-vs-baseline benchmark and write the comparison CSV.

    Loads the benchmark, scores every retriever with :func:`evaluate_one`, and
    writes the table with :func:`write_results_csv`.

    ``data`` and ``retrievers`` are injectable so the orchestration is testable
    without the real loader or real retrievers; in normal use both are left
    ``None`` and built from ``config``. BM25 builds directly today; the dense
    retrievers are pending P5 (see :func:`_build_retrievers`).
    """
    if data is None:
        data = load_benchmark(config.dataset)
    if retrievers is None:
        retrievers = _build_retrievers(config, data)

    results = {
        name: evaluate_one(retriever, data, config.k_values)
        for name, retriever in retrievers.items()
    }
    write_results_csv(results, config.results_path)


def main() -> None:
    run(BenchmarkConfig())


if __name__ == "__main__":
    main()
