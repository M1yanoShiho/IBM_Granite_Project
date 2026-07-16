"""Analyze Retriever recall on RGB candidates output.

Reads results/rgb_candidates.jsonl (produced by run_rgb.py) and computes
Recall@k per RGB dimension, showing how many gold passages the Retriever
successfully retrieved into the candidate pool.

Recall@k = fraction of samples where at least one gold passage is in top-k.

Usage:
    python -m eval.analyze_rgb --candidates results/rgb_candidates.jsonl
    python -m eval.analyze_rgb --candidates results/rgb_candidates.jsonl --k 5 10 20
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def recall_at_k(candidate_ids: List[str], positive_ids: List[str], k: int) -> float:
    """1.0 if any gold passage appears in the top-k candidates, else 0.0."""
    gold = set(positive_ids)
    return 1.0 if gold & set(candidate_ids[:k]) else 0.0


def analyze(candidates_file: str, k_values: List[int]) -> None:
    path = Path(candidates_file)
    if not path.exists():
        print(f"File not found: {candidates_file}")
        print("Run:  python -m eval.run_rgb --data-dir RGB/data --out results/rgb_candidates.jsonl")
        return

    # Per-dimension, per-k raw hits
    dim_hits: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    total_hits: Dict[int, List[float]] = defaultdict(list)
    n_no_gold = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            positive_ids: List[str] = rec.get("positive_ids", [])
            candidates: List[Dict] = rec.get("candidates", [])
            dimension: str = rec.get("dimension", "unknown")

            if not positive_ids:
                n_no_gold += 1
                continue

            retrieved_ids = [c["candidate_id"] for c in candidates]
            for k in k_values:
                r = recall_at_k(retrieved_ids, positive_ids, k)
                dim_hits[dimension][k].append(r)
                total_hits[k].append(r)

    if not total_hits:
        print("No records found.")
        return

    k_headers = "".join(f"  {'R@'+str(k):>7}" for k in k_values)
    header = f"{'Dimension':<30}{k_headers}  {'N':>6}"
    sep = "-" * len(header)

    print(f"\n=== RGB Retriever Recall@k (StrongBM25) ===")
    print(header)
    print(sep)

    for dim in sorted(dim_hits):
        row = f"{dim:<30}"
        n_dim = len(next(iter(dim_hits[dim].values())))
        for k in k_values:
            scores = dim_hits[dim][k]
            row += f"  {sum(scores)/len(scores):>7.4f}"
        row += f"  {n_dim:>6}"
        print(row)

    print(sep)
    overall_row = f"{'Overall':<30}"
    n_total = len(next(iter(total_hits.values())))
    for k in k_values:
        scores = total_hits[k]
        overall_row += f"  {sum(scores)/len(scores):>7.4f}"
    overall_row += f"  {n_total:>6}"
    print(overall_row)

    if n_no_gold:
        print(f"\n  (skipped {n_no_gold} samples with no gold passages)")

    print(f"\nInterpretation: Recall@k = fraction of claims where at least")
    print("one gold passage appeared in the Retriever's top-k candidates.")
    print("Target: ≥ 0.80 for the Retriever to give Selector enough signal.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="results/rgb_candidates.jsonl")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20],
                        help="Candidate pool sizes to evaluate (default: 5 10 20)")
    args = parser.parse_args()
    analyze(args.candidates, sorted(set(args.k)))


if __name__ == "__main__":
    main()
