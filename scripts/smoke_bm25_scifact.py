"""Smoke test for P1 loader + P3 BM25 retriever on SciFact.

This script is not a unit test because it may download/load benchmark data via
ir_datasets. Use it when the development environment is ready and network/data
cache access is available.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.benchmarks.loader import load_benchmark
from src.retrieval.bm25_baseline import BM25Retriever


def main() -> None:
    data = load_benchmark("scifact", split="test")

    doc_ids = list(data.corpus.keys())
    corpus = list(data.corpus.values())
    query_id, query = next(iter(data.queries.items()))

    retriever = BM25Retriever(corpus=corpus, doc_ids=doc_ids, top_k=5)
    results = retriever.retrieve(query)

    print(f"query_id={query_id}")
    print(f"query={query}")
    print("top results:")
    for rank, item in enumerate(results, start=1):
        preview = item.text.replace("\n", " ")[:160]
        print(f"{rank}. doc_id={item.doc_id} score={item.score:.4f} text={preview}")


if __name__ == "__main__":
    main()
