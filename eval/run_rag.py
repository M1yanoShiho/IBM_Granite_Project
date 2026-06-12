"""RAG evaluation entry point: retrieve-then-generate, scored on answer quality.

The generation counterpart to ``eval/run_benchmark.py``. Where ``run_benchmark``
scores the retrieval layer with precision/recall/nDCG/MRR against qrels, this
runner scores the full RAG system on a QA benchmark:

1. Load a QA benchmark with **gold answers** (e.g. Natural Questions, MS MARCO QA)
   via ``eval.benchmarks`` — note retrieval-only sets like SciFact ship qrels but
   no free-text answers, so answer-quality scoring needs an answer-bearing set
   (see ``BenchmarkData.answers`` and meeting question Q5).
2. Index the corpus once (``src.ingestion.indexer``) — the *same* index used by
   the retrieval benchmark.
3. For each query, run the :class:`~src.rag_pipeline.RAGPipeline` (the delivered
   retriever + the Granite generative model).
4. Score answers with ``eval.rag_metrics`` (answer correctness, context
   precision, faithfulness).
5. Save a results table for the report.

Run from the project root:

    python -m eval.run_rag --dataset nq

The implementation is left as a documented skeleton.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class RAGEvalConfig:
    """Configuration for a RAG evaluation run.

    Attributes
    ----------
    dataset:
        QA benchmark to evaluate on (must provide gold answers, e.g. ``"nq"``).
    retriever:
        Which retriever the RAG pipeline reads from (the delivered system or a
        baseline) — reuses the ``Retriever`` contract from ``run_benchmark``.
    top_k:
        Number of chunks passed to the generator as context.
    metrics:
        Which answer-quality metrics to report.
    results_path:
        Where to write the results table (CSV).
    """

    dataset: str = "nq"
    retriever: str = "granite_dense"
    top_k: int = 4
    metrics: List[str] = field(
        default_factory=lambda: ["answer_correctness", "context_precision", "faithfulness"]
    )
    results_path: Path = Path("results/rag_results.csv")


def run(config: RAGEvalConfig) -> None:
    """Load QA data, index, run the RAG pipeline per query, score, and persist."""
    # Ensure the results directory exists before writing.
    config.results_path.parent.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        "TODO: load answer-bearing QA benchmark, index corpus, run RAGPipeline "
        "over all queries, score with eval.rag_metrics, and write the table."
    )


def main() -> None:
    run(RAGEvalConfig())


if __name__ == "__main__":
    main()
