"""Retriever stage evaluation on the FEVER benchmark.

Aligned with 项目完整说明:
  - FEVER is used for SELECTOR evaluation (区分 SUPPORTS / REFUTES / NOT ENOUGH INFO).
  - Retriever's job here: retrieve candidate Wikipedia passages for each claim,
    so the Selector can decide whether the evidence supports or refutes it.
  - Retriever metric: Recall@k (找全 — did we pull in at least one gold passage?).

Data source: BEIR/FEVER via ir_datasets
  - Corpus: ~5.4M Wikipedia passages (2017 dump)
  - Queries: 6,666 claims (test split)
  - Qrels: binary (1 = passage is evidence for the claim)

Local smoke test (fast, ~30s):
    python -m eval.run_fever --max-queries 100 --max-docs 20000 --out results/fever_candidates_small.jsonl

Full run (needs server, full 5.4M corpus):
    python -m eval.run_fever --out results/fever_candidates.jsonl

Output format per record (Retriever → Selector interface, 项目完整说明 §6.2):
    {
      "query_id"  : str,
      "claim"     : str,        # the FEVER claim text
      "candidates": [
        {
          "candidate_id"    : str,
          "candidate_text"  : str,
          "retrieval_score" : float,
          "retrieval_rank"  : int,
          "source_parent_id": str,
        }, ...
      ]
    }

Selector group: consume results/fever_candidates.jsonl — each record gives you
the retriever's top-k passages for a claim. Your task is to classify each claim
as SUPPORTS / REFUTES / NOT ENOUGH INFO based on those passages.
"""

from __future__ import annotations

import argparse
import json
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    found = len(relevant & set(retrieved_ids[:k]))
    return found / len(relevant)


def evaluate_and_collect(
    retriever: BM25Retriever,
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    top_k: int,
) -> tuple[List[Dict], Dict[str, float]]:
    """Retrieve for all claims; return (records_for_selector, metric_summary)."""
    records: List[Dict] = []
    recall_scores: List[float] = []
    latencies: List[float] = []

    for qid, claim in queries.items():
        relevant = {doc_id for doc_id, rel in qrels.get(qid, {}).items() if rel > 0}

        t0 = time.time()
        results = retriever.retrieve(claim)
        latencies.append((time.time() - t0) * 1000)

        retrieved_ids = [r.doc_id for r in results]
        recall_scores.append(recall_at_k(retrieved_ids, relevant, top_k))

        records.append({
            "query_id": qid,
            "claim": claim,
            # Gold passage IDs — used by eval/analyze_fever.py to compute Recall@k
            "positive_ids": sorted(relevant),
            # Retriever → Selector interface (项目完整说明 §6.2)
            "candidates": [
                {
                    "candidate_id": r.doc_id,
                    "candidate_text": r.text,
                    "retrieval_score": r.score,
                    "retrieval_rank": r.rank,
                    "source_parent_id": r.metadata.get("source_parent_id", r.doc_id),
                    # §6.2 metadata fields — empty for Wikipedia benchmarks,
                    # populated from real enterprise documents via VectorIndexer
                    "company": r.metadata.get("company", ""),
                    "date": r.metadata.get("date", ""),
                    "document_type": r.metadata.get("document_type", ""),
                    "version": r.metadata.get("version", ""),
                }
                for r in results
            ],
        })

    n = len(recall_scores)
    metrics = {
        f"Recall@{top_k}": sum(recall_scores) / n if n else 0.0,
        "ms_per_query": sum(latencies) / len(latencies) if latencies else 0.0,
        "n_queries": n,
    }
    return records, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FEVER Retriever stage — retrieve candidates, save for Selector"
    )
    parser.add_argument(
        "--max-queries", type=int, default=None,
        help="Limit number of claims (default: all 6666 test claims)"
    )
    parser.add_argument(
        "--max-docs", type=int, default=None,
        help="Limit corpus size (default: full ~5.4M passages). Use 20000 for local smoke test."
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Number of candidate passages to retrieve per claim (default: 20)"
    )
    parser.add_argument(
        "--out", default="results/fever_candidates.jsonl",
        help="Output JSONL file (one record per claim)"
    )
    parser.add_argument(
        "--retriever", default="strong_bm25",
        choices=list(_RETRIEVER_CLASSES),
        help="Retriever variant (default: strong_bm25)"
    )
    args = parser.parse_args()

    # Load FEVER via BEIR
    print("Loading FEVER (BEIR)...")
    if args.max_docs:
        print(f"  [local mode] corpus capped at {args.max_docs} docs — gold passages always included")
    data = load_benchmark(
        "fever",
        split="test",
        max_queries=args.max_queries,
        max_docs=args.max_docs,
    )

    corpus_texts = list(data.corpus.values())
    corpus_ids = list(data.corpus.keys())
    print(f"Corpus: {len(corpus_texts)} passages | Claims: {len(data.queries)}")

    # Build index
    print(f"\nBuilding {args.retriever} index...")
    retriever_cls = _RETRIEVER_CLASSES[args.retriever]
    retriever = retriever_cls(corpus=corpus_texts, doc_ids=corpus_ids, top_k=args.top_k)

    # Retrieve + evaluate
    print(f"Retrieving top-{args.top_k} candidates for each claim...")
    records, metrics = evaluate_and_collect(
        retriever, data.queries, data.qrels, top_k=args.top_k
    )

    # Print metrics
    print("\n--- Retriever metrics (FEVER) ---")
    print(f"  Recall@{args.top_k}  : {metrics[f'Recall@{args.top_k}']:.4f}")
    print(f"  ms/query     : {metrics['ms_per_query']:.1f}")
    print(f"  n_queries    : {metrics['n_queries']}")

    # Save candidates for Selector group
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nCandidates saved to {args.out}")
    print("Next step: pass this file to the Selector group — they classify each claim")
    print("           as SUPPORTS / REFUTES / NOT ENOUGH INFO based on these passages.")


if __name__ == "__main__":
    main()
