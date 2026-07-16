"""Evaluate retrievers on MS MARCO Passage Ranking.

Aligned with 项目完整说明: Retriever is evaluated on its ability to find
all required evidence from a large corpus ("找全"). MS MARCO provides:
  - A large passage corpus (8.8M passages via BEIR)
  - Dev queries (6,980 questions)
  - Relevance judgments (qrels)

Metrics:
  - MRR@10   : MS MARCO standard ranking metric
  - Recall@20: fraction of relevant passages found in top-20 (找全)
  - ms/query : retrieval latency (efficiency)

This script is intended to run on the presenter's server (not locally).
Send this file along with the project repo to the server and run:

    python eval/run_msmarco.py --output results/msmarco.csv

For a quick smoke test with fewer docs:

    python eval/run_msmarco.py --max-queries 100 --max-docs 10000 --output results/msmarco_small.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.benchmarks.loader import load_benchmark
from src.retrieval.bm25_baseline import BM25Retriever
from src.retrieval.strong_bm25 import StrongBM25Retriever

_RETRIEVER_CLASSES = {
    "bm25": BM25Retriever,
    "strong_bm25": StrongBM25Retriever,
}


def mrr_at_k(retrieved_ids: List[str], relevant: Set[str], k: int) -> float:
    for rank, doc_id in enumerate(retrieved_ids[:k], 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_ids: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    found = len(relevant & set(retrieved_ids[:k]))
    return found / len(relevant)


def evaluate_retriever(retriever, queries, qrels, top_k: int = 20):
    mrr_scores = []
    recall_scores = []
    latencies = []

    for qid, query in queries.items():
        relevant = set(qrels.get(qid, {}).keys())
        if not relevant:
            continue

        t0 = time.time()
        results = retriever.retrieve(query)
        latencies.append((time.time() - t0) * 1000)

        retrieved_ids = [r.doc_id for r in results]
        mrr_scores.append(mrr_at_k(retrieved_ids, relevant, 10))
        recall_scores.append(recall_at_k(retrieved_ids, relevant, top_k))

    n = len(mrr_scores)
    return {
        "MRR@10": sum(mrr_scores) / n if n else 0.0,
        f"Recall@{top_k}": sum(recall_scores) / n if n else 0.0,
        "ms_per_query": sum(latencies) / len(latencies) if latencies else 0.0,
        "n_queries": n,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrievers on MS MARCO (Retriever 找全 evaluation)"
    )
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Limit number of queries (default: all 6980 dev queries)")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Limit corpus size (default: full 8.8M passages)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Candidate pool size for Recall@k (default: 20)")
    parser.add_argument("--output", default="results/msmarco.csv")
    parser.add_argument("--retriever", default="strong_bm25",
                        choices=list(_RETRIEVER_CLASSES),
                        help="Retriever variant (default: strong_bm25)")
    args = parser.parse_args()

    print("Loading MS MARCO (BEIR)...")
    data = load_benchmark(
        "msmarco",
        split="dev",
        max_queries=args.max_queries,
        max_docs=args.max_docs,
    )

    corpus_texts = list(data.corpus.values())
    corpus_ids = list(data.corpus.keys())
    print(f"Corpus: {len(corpus_texts)} passages | Queries: {len(data.queries)}")

    rows = []

    # BM25
    retriever_cls = _RETRIEVER_CLASSES[args.retriever]
    print(f"\nBuilding {args.retriever} index...")
    bm25 = retriever_cls(corpus=corpus_texts, doc_ids=corpus_ids, top_k=args.top_k)
    print(f"Evaluating {args.retriever}...")
    bm25_metrics = evaluate_retriever(bm25, data.queries, data.qrels, top_k=args.top_k)
    bm25_metrics["retriever"] = args.retriever
    rows.append(bm25_metrics)
    print(f"  BM25  MRR@10={bm25_metrics['MRR@10']:.4f}  "
          f"Recall@{args.top_k}={bm25_metrics[f'Recall@{args.top_k}']:.4f}  "
          f"ms/q={bm25_metrics['ms_per_query']:.1f}")

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
