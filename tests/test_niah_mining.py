# tests/test_niah_mining.py
"""Tests for src/niah/mining.py — Source C (topical hard-negative mining)."""
from __future__ import annotations

from src.retrieval.base import RetrievedChunk
from src.niah.mining import mine_topical


class _FakeRetriever:
    def __init__(self, ids):
        self._ids = ids

    def retrieve(self, query: str):
        return [RetrievedChunk(doc_id=i, text=f"t{i}", score=1.0) for i in self._ids]


def test_mine_topical_unions_both_retrievers_minus_needles() -> None:
    dense = _FakeRetriever(["d1", "n1", "d2"])   # n1 is a needle -> excluded
    sparse = _FakeRetriever(["d2", "d3"])
    out = mine_topical("q", dense, sparse, k=3, exclude_ids={"n1"})
    assert set(out) == {"d1", "d2", "d3"}


def test_mine_topical_respects_k_per_retriever() -> None:
    dense = _FakeRetriever(["d1", "d2", "d3", "d4"])
    sparse = _FakeRetriever([])
    out = mine_topical("q", dense, sparse, k=2, exclude_ids=set())
    assert set(out) == {"d1", "d2"}
