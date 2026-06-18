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

The full pipeline is implemented and unit-tested: BM25 and the dense retrievers
(``granite_dense`` / ``st_dense``, over P5's FAISS index) all run end-to-end.
Constructing a dense retriever downloads its embedding model on first use — see
:func:`_build_retrievers`.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from eval.benchmarks.loader import BenchmarkData, load_benchmark
from eval.ir_metrics import Run, evaluate_run
from src.ingestion.chunker import chunk_document
from src.ingestion.indexer import VectorIndexer
from src.retrieval.base import Retriever
from src.retrieval.bm25_baseline import BM25Retriever
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import DenseRetriever


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

    - ``bm25`` builds its own term index from the corpus.
    - ``granite_dense`` / ``st_dense`` chunk the corpus, embed it with the
      matching :class:`~src.retrieval.embedder.Embedder` backend, build a FAISS
      index (:class:`~src.ingestion.indexer.VectorIndexer`, contract 5), and wrap
      it in a :class:`~src.retrieval.retriever.DenseRetriever`.

    Building a dense retriever loads its embedding model (downloaded from Hugging
    Face on first use), so that is the one path needing network/models.
    """
    doc_ids = list(data.corpus.keys())
    corpus = list(data.corpus.values())
    top_k = max(config.k_values)

    retrievers: Dict[str, Retriever] = {}
    for name in config.retrievers:
        if name == "bm25":
            retrievers[name] = BM25Retriever(corpus, doc_ids, top_k=top_k)
        elif name in ("granite_dense", "st_dense"):
            backend = "granite" if name == "granite_dense" else "sentence-transformers"
            embedder = Embedder(backend=backend)
            chunks = [
                chunk
                for did, text in data.corpus.items()
                for chunk in chunk_document(did, text)
            ]
            index = VectorIndexer(embedder).build(chunks)
            retrievers[name] = DenseRetriever(embedder, index, top_k=top_k)
        else:
            raise ValueError(
                f"Unknown retriever {name!r}; expected granite_dense, bm25, or st_dense."
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
    ``None`` and built from ``config`` via :func:`_build_retrievers` (BM25 and
    the dense retrievers are all wired; dense construction downloads its model).
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


def _parse_args(argv: List[str] | None = None) -> BenchmarkConfig:
    """Parse command-line arguments into a :class:`BenchmarkConfig`.

    Example::

        python -m eval.run_benchmark --dataset scifact --retrievers bm25 st_dense
    """
    defaults = BenchmarkConfig()
    parser = argparse.ArgumentParser(
        prog="python -m eval.run_benchmark",
        description="Run the system-vs-baselines retrieval benchmark and write a CSV.",
    )
    parser.add_argument(
        "--dataset",
        default=defaults.dataset,
        help="Benchmark name to evaluate on (default: %(default)s).",
    )
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=defaults.retrievers,
        help="Retrievers to compare (default: %(default)s).",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=defaults.k_values,
        dest="k_values",
        help="Cut-offs for precision/recall/nDCG (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=defaults.results_path,
        dest="results_path",
        help="Where to write the comparison CSV (default: %(default)s).",
    )
    args = parser.parse_args(argv)
    return BenchmarkConfig(
        dataset=args.dataset,
        retrievers=args.retrievers,
        k_values=args.k_values,
        results_path=args.results_path,
    )


def main(argv: List[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
