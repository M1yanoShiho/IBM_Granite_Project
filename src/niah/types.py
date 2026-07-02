# src/niah/types.py
"""Dataclasses for a constructed Needle-In-A-Haystack retrieval task.

A NIAH task is an ordinary (corpus, queries, qrels) benchmark — so it runs on the
existing eval harness unchanged — plus the provenance of the injected distractors.
Invariant: only *needles* (the gold docs from the source qrels) are relevant;
every injected ``Distractor`` is non-relevant by construction and must never be
written into ``qrels`` (see src/niah/filters.py for how that is enforced).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

# Distractor provenance labels (also the three §5.1 sources).
SOURCE_COUNTERFACTUAL = "counterfactual"
SOURCE_GENERATIVE = "generative"
SOURCE_MINED = "mined"


@dataclass(frozen=True)
class Distractor:
    """One injected non-relevant passage and where it came from."""

    doc_id: str
    text: str
    source: str  # one of SOURCE_*
    parent_needle_id: str


@dataclass
class NiahExample:
    """One query with its needle(s) and the distractors built for it."""

    query_id: str
    query: str
    needle_ids: List[str]
    distractors: List[Distractor] = field(default_factory=list)


@dataclass
class NiahTask:
    """A built task: a runnable benchmark + distractor provenance."""

    corpus: Dict[str, str]
    queries: Dict[str, str]
    qrels: Dict[str, Dict[str, int]]
    examples: List[NiahExample] = field(default_factory=list)
