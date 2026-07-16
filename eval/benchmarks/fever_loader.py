"""Loader for the FEVER (Fact Extraction and VERification) benchmark.

Source: https://huggingface.co/datasets/fever/fever
Usage:
    from eval.benchmarks.fever_loader import load_fever
    data = load_fever(split="labelled_dev")

Each sample has:
  - claim      : the statement to verify
  - label      : SUPPORTS / REFUTES / NOT ENOUGH INFO
  - evidence   : list of (wiki_page, sentence_id) gold evidence pointers

Retriever stage: given a claim, retrieve relevant Wikipedia sentences.
Simplified setup (as noted in datasets_wzn.md): use gold article sentences
as the candidate corpus — no need to build a full Wikipedia index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from datasets import load_dataset


@dataclass
class FEVERSample:
    sample_id: str
    claim: str
    label: str                    # SUPPORTS / REFUTES / NOT ENOUGH INFO
    evidence_pages: List[str]     # gold Wikipedia page titles
    evidence_lines: List[int]     # gold sentence IDs within those pages
    has_evidence: bool            # False when label == NOT ENOUGH INFO


@dataclass
class FEVERDataset:
    split: str
    samples: List[FEVERSample] = field(default_factory=list)


def load_fever(
    split: str = "labelled_dev",
    labels: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
) -> FEVERDataset:
    """Load FEVER from HuggingFace.

    Args:
        split: HuggingFace split name. Use "labelled_dev" for dev evaluation.
        labels: Filter to specific labels, e.g. ["SUPPORTS", "REFUTES"].
                Defaults to all three labels.
        max_samples: Cap the number of samples returned.

    Returns:
        FEVERDataset with filtered samples.
    """
    ds = load_dataset("fever", "v1.0", split=split, trust_remote_code=True)

    if labels is None:
        labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    label_set = set(labels)

    samples = []
    for row in ds:
        if row["label"] not in label_set:
            continue

        pages = []
        lines = []
        for evidence_group in row.get("evidence", []):
            for ev in evidence_group:
                if ev.get("Wikipedia URL") or ev.get("wikipedia_url"):
                    page = ev.get("Wikipedia URL") or ev.get("wikipedia_url") or ""
                    line = ev.get("evidence_id", -1)
                else:
                    page = ev[2] if len(ev) > 2 else ""
                    line = ev[3] if len(ev) > 3 else -1
                if page:
                    pages.append(str(page))
                    lines.append(int(line) if line is not None else -1)

        samples.append(FEVERSample(
            sample_id=str(row["id"]),
            claim=row["claim"],
            label=row["label"],
            evidence_pages=pages,
            evidence_lines=lines,
            has_evidence=(row["label"] != "NOT ENOUGH INFO"),
        ))

        if max_samples and len(samples) >= max_samples:
            break

    return FEVERDataset(split=split, samples=samples)
