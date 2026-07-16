"""Full pipeline evaluation on the RGB benchmark.

Aligned with 项目完整说明 §4.3: RGB is used for FULL PIPELINE evaluation
(Retriever → Selector → Generator), not Retriever-only evaluation.

RGB tests 4 dimensions of robustness:
  - noise_robustness        : can the system answer correctly despite noisy docs?
  - counterfactual_robustness: can it avoid being misled by wrong info?
  - information_integration : can it combine evidence from multiple passages?
  - negative_rejection      : can it refuse when no correct evidence exists?

Each dimension measures answer correctness, not just retrieval recall.

Current status:
  - Stage 1 (Retriever): DONE — retrieves top-k candidates, saves to file
  - Stage 2 (Selector) : TODO — wire in when Selector group is ready
  - Stage 3 (Generator): TODO — wire in when Generator group is ready

Usage:
    # Run Retriever stage only (saves candidates for Selector):
    python -m eval.run_rgb --data-dir RGB/data --stage retriever --out results/rgb_candidates.jsonl

    # Run full pipeline (once all 3 modules are ready):
    python -m eval.run_rgb --data-dir RGB/data --stage full --out results/rgb_full.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.benchmarks.rgb_loader import RGBDataset, RGBSample, load_rgb
from src.retrieval.bm25_baseline import BM25Retriever
from src.retrieval.strong_bm25 import StrongBM25Retriever


# ---------------------------------------------------------------------------
# Stage 1: Retriever — build pooled corpus per dimension and retrieve top-k
# ---------------------------------------------------------------------------

def build_sample_corpus(sample: RGBSample):
    texts: List[str] = []
    doc_ids: List[str] = []
    positive_ids: Set[str] = set()

    flat_positive = []
    for item in sample.positive:
        if isinstance(item, list):
            flat_positive.extend(item)
        else:
            flat_positive.append(item)

    for i, text in enumerate(flat_positive):
        doc_id = f"pos_{i}"
        texts.append(text)
        doc_ids.append(doc_id)
        positive_ids.add(doc_id)

    for i, text in enumerate(sample.negative):
        doc_id = f"neg_{i}"
        texts.append(text)
        doc_ids.append(doc_id)

    return texts, doc_ids, positive_ids


_RETRIEVER_CLASSES = {
    "bm25": BM25Retriever,
    "strong_bm25": StrongBM25Retriever,
}


def run_retriever_stage(
    datasets: List[RGBDataset], top_k: int, retriever_name: str = "strong_bm25"
) -> List[Dict]:
    """Retrieve top-k candidates for every sample. Returns records for Selector."""
    retriever_cls = _RETRIEVER_CLASSES[retriever_name]
    records = []
    for dataset in datasets:
        print(f"\n[Retriever:{retriever_name}] {dataset.dimension} ({len(dataset.samples)} samples)")
        for sample in dataset.samples:
            texts, doc_ids, positive_ids = build_sample_corpus(sample)
            if not texts:
                continue

            retriever = retriever_cls(corpus=texts, doc_ids=doc_ids, top_k=top_k)
            results = retriever.retrieve(sample.question)

            records.append({
                "sample_id": sample.sample_id,
                "dimension": sample.dimension,
                "question": sample.question,
                "answer": sample.answer,
                "positive_ids": list(positive_ids),
                # Retriever → Selector interface (项目完整说明 §6.2)
                "candidates": [
                    {
                        "candidate_id": r.doc_id,
                        "candidate_text": r.text,
                        "retrieval_score": r.score,
                        "retrieval_rank": r.rank,
                        "source_parent_id": r.metadata.get("source_parent_id", ""),
                        # §6.2 metadata fields — empty for synthetic benchmarks,
                        # populated from real enterprise documents via VectorIndexer
                        "company": r.metadata.get("company", ""),
                        "date": r.metadata.get("date", ""),
                        "document_type": r.metadata.get("document_type", ""),
                        "version": r.metadata.get("version", ""),
                    }
                    for r in results
                ],
            })
    return records


# ---------------------------------------------------------------------------
# Stage 2: Selector — TODO (wire in when Selector group delivers their module)
# ---------------------------------------------------------------------------

def run_selector_stage(retriever_records: List[Dict]) -> List[Dict]:
    """
    TODO: Replace this stub with the real Selector when ready.

    Expected input per record: candidates (list of dicts with candidate_id,
      candidate_text, retrieval_score, retrieval_rank)
    Expected output per record: selected_evidence (subset of candidates),
      evidence_completeness, missing_facts, detected_conflicts
    """
    for rec in retriever_records:
        # Stub: pass all candidates through unchanged
        rec["selected_evidence"] = rec["candidates"]
        rec["evidence_completeness"] = "unknown"
        rec["missing_facts"] = []
        rec["detected_conflicts"] = []
    return retriever_records


# ---------------------------------------------------------------------------
# Stage 3: Generator — TODO (wire in when Generator group delivers their module)
# ---------------------------------------------------------------------------

def run_generator_stage(selector_records: List[Dict]) -> List[Dict]:
    """
    TODO: Replace this stub with the real Generator when ready.

    Expected input per record: selected_evidence, question
    Expected output per record: answer, citations, abstained
    """
    for rec in selector_records:
        # Stub: mark as not yet generated
        rec["generated_answer"] = None
        rec["abstained"] = None
        rec["correct"] = None
    return selector_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Path to RGB/data/ directory")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Retriever candidate pool size (default: 20)")
    parser.add_argument("--dimension", default=None,
                        choices=["noise_robustness", "negative_rejection",
                                 "information_integration", "counterfactual_robustness"])
    parser.add_argument("--stage", default="retriever",
                        choices=["retriever", "full"],
                        help="retriever: save candidates only | full: run all 3 stages")
    parser.add_argument("--out", default="results/rgb_candidates.jsonl")
    parser.add_argument("--retriever", default="strong_bm25",
                        choices=list(_RETRIEVER_CLASSES),
                        help="Retriever variant (default: strong_bm25)")
    args = parser.parse_args()

    datasets = load_rgb(args.data_dir, dimension=args.dimension)

    # Stage 1
    records = run_retriever_stage(datasets, top_k=args.top_k, retriever_name=args.retriever)
    print(f"\n[Retriever] Done. {len(records)} samples processed.")

    if args.stage == "full":
        # Stage 2
        print("[Selector] Running... (stub)")
        records = run_selector_stage(records)

        # Stage 3
        print("[Generator] Running... (stub)")
        records = run_generator_stage(records)

    # Save output
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nOutput saved to {args.out}")
    if args.stage == "retriever":
        print("Next step: pass this file to the Selector group when their module is ready.")


if __name__ == "__main__":
    main()
