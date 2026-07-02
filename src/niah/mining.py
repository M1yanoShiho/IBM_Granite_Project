# src/niah/mining.py
"""Source C — topical hard-negative mining (spec §5.1).

Union of the top-``k`` doc ids from a dense and a sparse retriever (covering both
retriever families), minus the needle ids. These "natural" distractors still pass
through Filter 1 (answerability) before injection, since a mined neighbour may in
fact be relevant (NV-Retriever: ~70% of naive top negatives are false negatives).
"""
from __future__ import annotations

from typing import List, Set


def mine_topical(query: str, dense, sparse, k: int, exclude_ids: Set[str]) -> List[str]:
    """Top-``k`` doc ids from dense ∪ sparse, excluding needles, order-preserving."""
    seen: Set[str] = set()
    out: List[str] = []
    for retriever in (dense, sparse):
        for chunk in retriever.retrieve(query)[:k]:
            if chunk.doc_id in exclude_ids or chunk.doc_id in seen:
                continue
            seen.add(chunk.doc_id)
            out.append(chunk.doc_id)
    return out
