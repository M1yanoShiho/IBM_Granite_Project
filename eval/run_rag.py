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

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from eval.benchmarks.loader import BenchmarkData, load_benchmark
from eval.rag_metrics import METRIC_NAMES, evaluate_rag, score_rag_per_query
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
    max_queries, max_docs:
        Subset knobs forwarded to :func:`~eval.benchmarks.loader.load_benchmark`
        so a huge answer-bearing set (NQ) can be run on a small, valid subset
        first (gold docs are always kept; distractors capped at ``max_docs``).
        ``None`` = use the full set.
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
    max_queries: Optional[int] = None
    max_docs: Optional[int] = None
    per_query_out: Optional[Path] = None
    predictions_out: Optional[Path] = None
    pipeline: str = "vanilla"


def run(
    config: RAGEvalConfig,
    data: Optional[BenchmarkData] = None,
    retriever: Optional[Retriever] = None,
    llm: Optional[LLMClient] = None,
    judge=None,
) -> Dict[str, float]:
    """Load QA data, run RAGPipeline per query, score, and write CSV.

    ``judge`` (optional): a callable ``(premise, hypothesis) -> bool`` enabling the
    claim-level ``answer_claims`` metric (:func:`~eval.rag_metrics.score_answer_claims`);
    ``None`` keeps the model-free CPU metric suite.
    """
    config.results_path.parent.mkdir(parents=True, exist_ok=True)

    if data is None:
        data = load_benchmark(
            config.dataset,
            max_queries=config.max_queries,
            max_docs=config.max_docs,
        )

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
        retriever = _build_retrievers(bench_cfg, data, llm=llm)[config.retriever]

    if config.pipeline == "corrective":
        from src.rag_pipeline import CorrectiveRAGPipeline
        from src.retrieval.query_transform import HyDETransform

        # Only the pipeline varies vs the vanilla run (same retriever/generator/
        # prompt), so the cover-EM delta isolates the adaptive re-retrieval loop.
        pipeline = CorrectiveRAGPipeline(
            retriever=retriever,
            llm=llm,
            top_k=config.top_k,
            query_rewriter=HyDETransform(llm),
        )
    else:
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
        predictions, references, contexts, retrieved_doc_ids, data.qrels, judge=judge
    )

    _write_results(config, metrics)
    if config.per_query_out is not None:
        per_query = score_rag_per_query(
            predictions, references, contexts, retrieved_doc_ids, data.qrels, judge=judge
        )
        _write_per_query(config.per_query_out, config.retriever, per_query)
    if config.predictions_out is not None:
        _write_predictions(
            config.predictions_out, config.retriever, data.queries, predictions, references
        )
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


def _write_per_query(
    prefix: Path, retriever: str, per_query: Dict[str, Dict[str, float]]
) -> None:
    """Merge this retriever's per-query scores into one wide CSV per metric.

    Writes ``<prefix>_<metric>.csv`` (``qid`` + one column per retriever) for each
    metric, merging the column in if the file already exists. So calling
    ``run_rag`` once per retriever (granite/gte/bm25) accumulates a wide table
    per metric that :mod:`eval.significance` can compare directly (paired test
    between two retriever columns).
    """
    stem = prefix.with_suffix("")  # drop any extension; the suffix is per-metric
    stem.parent.mkdir(parents=True, exist_ok=True)
    # Persist whatever metrics are present (METRIC_NAMES first, then extras like
    # answer_claims when a judge was used), so the significance-ready per-query CSVs
    # cover the judge metric too. Falls back to METRIC_NAMES when there are no rows.
    present = next(iter(per_query.values())).keys() if per_query else METRIC_NAMES
    metric_list = [m for m in METRIC_NAMES if m in present] + [
        m for m in present if m not in METRIC_NAMES
    ]
    for metric in metric_list:
        path = stem.with_name(f"{stem.name}_{metric}.csv")
        table: Dict[str, Dict[str, str]] = {}
        columns: List[str] = []
        if path.exists():
            with open(path, newline="") as f:
                reader = csv.reader(f)
                header = next(reader, ["qid"])
                columns = header[1:]
                for row in reader:
                    table[row[0]] = {columns[i]: row[i + 1] for i in range(len(columns))}
        if retriever not in columns:
            columns.append(retriever)
        for qid, scores in per_query.items():
            table.setdefault(qid, {})[retriever] = scores[metric]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["qid"] + columns)
            for qid in sorted(table):
                writer.writerow([qid] + [table[qid].get(col, "") for col in columns])


def _write_predictions(
    prefix: Path,
    retriever: str,
    questions: Dict[str, str],
    predictions: Dict[str, str],
    references: Dict[str, object],
) -> None:
    """Dump per-question (question, gold, generated answer) as JSONL for eyeballing.

    Writes ``<prefix>_<retriever>.jsonl`` — one record per answered question — so
    you can see *what the model actually said* vs the gold (e.g. confirm that a low
    EM/F1 is verbosity rather than wrong answers). JSONL, not CSV, because answers
    contain commas and newlines.
    """
    stem = prefix.with_suffix("")
    path = stem.with_name(f"{stem.name}_{retriever}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, prediction in predictions.items():
            record = {
                "qid": qid,
                "question": questions.get(qid, ""),
                "gold": references.get(qid),
                "prediction": prediction,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _parse_args(argv: Optional[List[str]] = None) -> RAGEvalConfig:
    """Parse command-line arguments into a :class:`RAGEvalConfig`.

    Example (NQ subset, comparing two retrievers into one table)::

        python -m eval.run_rag --dataset nq --retriever granite_dense \\
            --max-queries 300 --max-docs 50000 --out results/rag_nq_subset.csv
        python -m eval.run_rag --dataset nq --retriever bm25 \\
            --max-queries 300 --max-docs 50000 --out results/rag_nq_subset.csv --append
    """
    defaults = RAGEvalConfig()
    parser = argparse.ArgumentParser(
        prog="python -m eval.run_rag",
        description="Run the RAG evaluation (retrieve-then-generate) and write a CSV.",
    )
    parser.add_argument("--dataset", default=defaults.dataset,
                        help="Answer-bearing QA benchmark (default: %(default)s).")
    parser.add_argument("--retriever", default=defaults.retriever,
                        help="Retriever the pipeline reads from (default: %(default)s).")
    parser.add_argument("--top-k", type=int, default=defaults.top_k, dest="top_k",
                        help="Chunks passed to the generator as context (default: %(default)s).")
    parser.add_argument("--out", type=Path, default=defaults.results_path,
                        dest="results_path", help="Where to write the results CSV.")
    parser.add_argument("--append", action="store_true", default=defaults.append,
                        help="Append a row instead of overwriting (compare retrievers).")
    parser.add_argument("--max-queries", type=int, default=defaults.max_queries,
                        dest="max_queries",
                        help="Run only the first N queries (subset mode; default: all).")
    parser.add_argument("--max-docs", type=int, default=defaults.max_docs,
                        dest="max_docs",
                        help="Cap distractor docs in the corpus; gold docs always kept "
                        "(subset mode; default: full corpus).")
    parser.add_argument("--per-query-out", type=Path, default=defaults.per_query_out,
                        dest="per_query_out",
                        help="Also write per-query scores as <prefix>_<metric>.csv "
                        "(qid x retriever), merged across retriever runs, for "
                        "eval.significance. Default: off.")
    parser.add_argument("--predictions-out", type=Path, default=defaults.predictions_out,
                        dest="predictions_out",
                        help="Also dump per-question question/gold/prediction JSONL to "
                        "<prefix>_<retriever>.jsonl for inspection. Default: off.")
    parser.add_argument("--pipeline", default=defaults.pipeline,
                        choices=["vanilla", "corrective"],
                        help="RAG pipeline: 'vanilla' single-shot (default) or "
                        "'corrective' confidence-gated re-retrieval with a rewritten "
                        "query.")
    args = parser.parse_args(argv)
    return RAGEvalConfig(
        dataset=args.dataset,
        retriever=args.retriever,
        top_k=args.top_k,
        results_path=args.results_path,
        append=args.append,
        max_queries=args.max_queries,
        max_docs=args.max_docs,
        per_query_out=args.per_query_out,
        predictions_out=args.predictions_out,
        pipeline=args.pipeline,
    )


def main(argv: Optional[List[str]] = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
