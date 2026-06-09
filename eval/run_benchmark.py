"""Primary evaluation entry point: system vs. baselines on standard benchmarks.

Orchestrates the headline experiment of the project:

1. Load a standard benchmark (corpus, queries, qrels) via ``eval.benchmarks``.
2. Index the corpus once (``src.ingestion.indexer``).
3. Run each retriever over all queries:
   - the **Granite dense retriever** (the delivered system), and
   - the **BM25 baseline** (and optionally a sentence-transformers dense baseline).
4. Score every run with ``eval.ir_metrics`` (precision/recall/nDCG/MRR).
5. Save a comparison table for the report.

Run from the project root:

    python -m eval.run_benchmark --dataset scifact

The implementation is left as a documented skeleton.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


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


def run(config: BenchmarkConfig) -> None:
    """Load data, index, run all retrievers, score, and persist the comparison."""
    # Ensure the results directory exists before any retriever writes to it.
    config.results_path.parent.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        "TODO: load benchmark, index corpus, run retrievers, score with "
        "eval.ir_metrics, and write the system-vs-baseline table."
    )


def main() -> None:
    run(BenchmarkConfig())


if __name__ == "__main__":
    main()
