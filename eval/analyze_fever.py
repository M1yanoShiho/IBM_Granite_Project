"""Analyze Retriever recall on FEVER candidates output.

Reads results/fever_candidates_strong.jsonl (produced by run_fever.py) and
computes Recall@k, showing how many gold passages the Retriever found.

Recall@k = fraction of claims where at least one gold passage is in top-k.

Usage:
    python -m eval.analyze_fever
    python -m eval.analyze_fever --candidates results/fever_candidates.jsonl --k 5 10 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def recall_at_k(candidate_ids: List[str], relevant: List[str], k: int) -> float:
    """1.0 if any gold passage appears in top-k, else 0.0 (binary hit@k)."""
    gold = set(relevant)
    return 1.0 if gold & set(candidate_ids[:k]) else 0.0


def analyze(candidates_file: str, k_values: List[int]) -> None:
    path = Path(candidates_file)
    if not path.exists():
        print(f"File not found: {candidates_file}")
        print("Run:  python -m eval.run_fever --max-queries 200 --max-docs 30000 "
              "--out results/fever_candidates_strong.jsonl")
        return

    hits: Dict[int, List[float]] = {k: [] for k in k_values}
    n_no_gold = 0
    n_total = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            candidates: List[Dict] = rec.get("candidates", [])
            relevant: List[str] = rec.get("positive_ids", [])

            # run_fever.py doesn't embed positive_ids; fall back to empty
            if not relevant:
                n_no_gold += 1
                continue

            n_total += 1
            retrieved_ids = [c["candidate_id"] for c in candidates]
            for k in k_values:
                hits[k].append(recall_at_k(retrieved_ids, relevant, k))

    if n_total == 0:
        if n_no_gold > 0:
            print(f"All {n_no_gold} records lack 'positive_ids' — "
                  "this file was produced by run_fever.py which omits gold labels.")
            print("Recall@k cannot be computed from this file alone.")
            print("Use eval/run_fever.py output metrics printed during the run.")
        else:
            print("No records found.")
        return

    k_headers = "".join(f"  {'R@'+str(k):>7}" for k in k_values)
    header = f"{'Metric':<20}{k_headers}  {'N':>6}"
    sep = "-" * len(header)

    print(f"\n=== FEVER Retriever Recall@k (StrongBM25) ===")
    print(header)
    print(sep)

    row = f"{'Recall':<20}"
    for k in k_values:
        scores = hits[k]
        row += f"  {sum(scores)/len(scores):>7.4f}"
    row += f"  {n_total:>6}"
    print(row)
    print(sep)

    if n_no_gold:
        print(f"\n  (skipped {n_no_gold} records without positive_ids)")

    print("\nInterpretation: Recall@k = fraction of claims where ≥1 gold")
    print("Wikipedia passage appeared in top-k candidates.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="results/fever_candidates_strong.jsonl")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20],
                        help="Candidate pool sizes (default: 5 10 20)")
    args = parser.parse_args()
    analyze(args.candidates, sorted(set(args.k)))


if __name__ == "__main__":
    main()
