# src/niah/assembly.py
"""Assemble the haystack: inject distractors and cap corpus size for the scale axis.

``cap_haystack`` mirrors ``eval.benchmarks.loader.load_benchmark(max_docs=...)``:
needles and injected distractors are ALWAYS kept; the remainder is filled with a
deterministic random sample of background docs, so recall stays well-defined while
the corpus size is swept (Phase 1).
"""
from __future__ import annotations

import random
from typing import Dict, Iterable, Optional, Set

from src.niah.types import Distractor


def inject(corpus: Dict[str, str], distractors: Iterable[Distractor]) -> Dict[str, str]:
    """Return a new corpus with the distractor docs added (input not mutated)."""
    out = dict(corpus)
    for d in distractors:
        out[d.doc_id] = d.text
    return out


def cap_haystack(
    corpus: Dict[str, str],
    keep_ids: Set[str],
    max_docs: Optional[int],
    seed: int,
) -> Dict[str, str]:
    """Keep all ``keep_ids`` + up to ``max_docs`` deterministically-sampled others."""
    if max_docs is None:
        return dict(corpus)
    kept = {doc_id: corpus[doc_id] for doc_id in keep_ids if doc_id in corpus}
    background = sorted(doc_id for doc_id in corpus if doc_id not in keep_ids)
    rng = random.Random(seed)
    rng.shuffle(background)
    for doc_id in background[:max_docs]:
        kept[doc_id] = corpus[doc_id]
    return kept
