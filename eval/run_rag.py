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
4. Score answers with ``eval.rag_metrics`` (Exact Match, token-F1, qrels-based
   context precision, faithfulness).
5. Save a results table for the report.

Run from the project root:

    python -m eval.run_rag --dataset nq
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from eval.benchmarks.loader import BenchmarkData, load_benchmark
from eval.rag_metrics import evaluate_rag
from src.llm_client import LLMClient
from src.rag_pipeline import RAGPipeline
from src.retrieval.base import Retriever


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
    append:
        When ``True`` and ``results_path`` already exists, append this run's row
        under the existing header instead of overwriting — so comparing several
        retrievers accumulates one table rather than clobbering the previous run.
    """

    dataset: str = "nq"
    retriever: str = "granite_dense"
    top_k: int = 4
    metrics: List[str] = field(
        default_factory=lambda: [
            "answer_em",
            "answer_f1",
            "context_precision",
            "faithfulness",
        ]
    )
    results_path: Path = Path("results/rag_results.csv")
    append: bool = False


def run(
    config: RAGEvalConfig,
    data: Optional[BenchmarkData] = None,
    retriever: Optional[Retriever] = None,
    llm: Optional[LLMClient] = None,
) -> Dict[str, float]:
    """Load QA data, run RAGPipeline per query, score, and write CSV."""
    config.results_path.parent.mkdir(parents=True, exist_ok=True)

    if data is None:
        data = load_benchmark(config.dataset)

    if data.answers is None:
        raise ValueError(
            f"Dataset '{config.dataset}' has no gold answers — "
            "RAG answer-quality scoring requires a QA benchmark with answers."
        )

    if llm is None:
        llm = LLMClient()

    if retriever is None:
        from eval.run_benchmark import BenchmarkConfig, _build_retrievers

        bench_cfg = BenchmarkConfig(
            dataset=config.dataset, retrievers=[config.retriever]
        )
        retriever = _build_retrievers(bench_cfg, data)[config.retriever]

    pipeline = RAGPipeline(retriever=retriever, llm=llm, top_k=config.top_k)

    predictions: Dict[str, str] = {}
    references: Dict[str, List[str]] = {}
    contexts: Dict[str, List[str]] = {}
    retrieved_doc_ids: Dict[str, List[str]] = {}

    for qid, question in data.queries.items():
        if qid not in data.answers:
            continue
        result = pipeline.query(question)
        predictions[qid] = result.answer
        references[qid] = data.answers[qid]
        contexts[qid] = [chunk.text for chunk in result.retrieved_chunks]
        retrieved_doc_ids[qid] = [chunk.doc_id for chunk in result.retrieved_chunks]

    metrics = evaluate_rag(
        predictions, references, contexts, retrieved_doc_ids, data.qrels
    )

    _write_results(config, metrics)
    return metrics


def _write_results(config: RAGEvalConfig, metrics: Dict[str, float]) -> None:
    """Write one ``retriever``-tagged row of metrics, honouring ``append``."""
    fieldnames = ["retriever", *metrics.keys()]
    appending = config.append and config.results_path.exists()
    with open(config.results_path, "a" if appending else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not appending:
            writer.writeheader()
        writer.writerow({"retriever": config.retriever, **metrics})


def main() -> None:
    run(RAGEvalConfig())


if __name__ == "__main__":
    main()
