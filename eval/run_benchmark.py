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
from src.ingestion.chunker import Chunk, chunk_document
from src.ingestion.indexer import FaissIndex, VectorIndexer
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
    index_cache_dir:
        If set, dense retrievers cache their FAISS index here (keyed by dataset
        and retriever name) so reruns skip re-embedding. ``None`` = no caching.
    chunk_size:
        Token window size for splitting documents into chunks. SciFact abstracts
        are short (≈150 words), so 512 yields ~1 chunk/doc; chunk ablations are
        more informative on long-document corpora (e.g. NQ).
    chunk_overlap:
        Number of tokens shared between consecutive chunks (sliding window).
    pool_strategy:
        How to collapse per-chunk scores to a single doc score: ``"max"``
        (contract 3 default) or ``"mean"``.
    """

    dataset: str = "scifact"
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    retrievers: List[str] = field(
        default_factory=lambda: ["granite_dense", "bm25", "st_dense"]
    )
    results_path: Path = Path("results/benchmark_results.csv")
    index_cache_dir: Path | None = None
    chunk_size: int = 512
    chunk_overlap: int = 50
    pool_strategy: str = "max"


def build_run(
    retriever: Retriever,
    queries: Dict[str, str],
    pool_strategy: str = "max",
) -> Run:
    """Aggregate a retriever's chunk-level results into a doc-level ``Run``.

    For each query, call ``retriever.retrieve`` and collapse the returned chunks
    to one score per ``doc_id``. Pooling strategy is controlled by
    ``pool_strategy``:

        - ``"max"`` (default, contract 3): doc score = max chunk score.
        - ``"mean"``: doc score = average chunk score.

    On unchunked corpora (e.g. SciFact at chunk_size 512) this is a pass-through;
    pooling matters once a document produces multiple chunks.
    """
    if pool_strategy not in ("max", "mean"):
        raise ValueError(
            f"Unknown pool_strategy {pool_strategy!r}; expected 'max' or 'mean'."
        )
    run: Run = {}
    for query_id, query_text in queries.items():
        chunks_by_doc: Dict[str, List[float]] = {}
        for chunk in retriever.retrieve(query_text):
            chunks_by_doc.setdefault(chunk.doc_id, []).append(chunk.score)
        if pool_strategy == "max":
            run[query_id] = {did: max(scores) for did, scores in chunks_by_doc.items()}
        else:
            run[query_id] = {
                did: sum(scores) / len(scores) for did, scores in chunks_by_doc.items()
            }
    return run


def evaluate_one(
    retriever: Retriever,
    data: BenchmarkData,
    k_values: List[int] | None = None,
    pool_strategy: str = "max",
) -> Dict[str, float]:
    """Score one retriever on a benchmark.

    Builds the retriever's doc-level ``Run`` (:func:`build_run`, contract 3),
    then scores it against the benchmark's qrels with
    ``eval.ir_metrics.evaluate_run`` (contract 2). Returns the metric suite
    (precision/recall/nDCG at each k, plus MRR).
    """
    run = build_run(retriever, data.queries, pool_strategy=pool_strategy)
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


def _load_or_build_index(
    indexer: VectorIndexer,
    chunks: List[Chunk],
    cache_path: Path,
) -> FaissIndex:
    """Load a persisted FAISS index from ``cache_path`` if present, otherwise
    build it from ``chunks`` and save it there — so repeat runs skip the
    expensive re-embedding step. The ``.faiss`` suffix is what
    ``VectorIndexer.save`` writes.
    """
    if Path(f"{cache_path}.faiss").exists():
        return indexer.load(cache_path)
    index = indexer.build(chunks)
    indexer.save(index, cache_path)
    return index


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
                for chunk in chunk_document(
                    did, text,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                )
            ]
            indexer = VectorIndexer(embedder)
            if config.index_cache_dir is not None:
                cache_path = config.index_cache_dir / f"{config.dataset}__{name}"
                index = _load_or_build_index(indexer, chunks, cache_path)
            else:
                index = indexer.build(chunks)
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
        name: evaluate_one(retriever, data, config.k_values, pool_strategy=config.pool_strategy)
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
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=defaults.index_cache_dir,
        dest="index_cache_dir",
        help="Cache dense indexes here to skip re-embedding on reruns (default: off).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=defaults.chunk_size,
        dest="chunk_size",
        help="Token window for chunking documents (default: %(default)s). "
             "Note: SciFact abstracts are short — chunk ablations are more "
             "informative on long-document corpora such as NQ.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=defaults.chunk_overlap,
        dest="chunk_overlap",
        help="Overlap tokens between consecutive chunks (default: %(default)s).",
    )
    parser.add_argument(
        "--pool",
        default=defaults.pool_strategy,
        dest="pool_strategy",
        choices=["max", "mean"],
        help="Chunk-to-doc score aggregation: max (default) or mean.",
    )
    args = parser.parse_args(argv)
    return BenchmarkConfig(
        dataset=args.dataset,
        retrievers=args.retrievers,
        k_values=args.k_values,
        results_path=args.results_path,
        index_cache_dir=args.index_cache_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        pool_strategy=args.pool_strategy,
    )


def main(argv: List[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
