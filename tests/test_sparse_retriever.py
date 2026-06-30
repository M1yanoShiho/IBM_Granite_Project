"""Tests for src/retrieval/sparse_retriever.py — the SPLADE (sparse) retriever.

Ties a fake SPLADE encoder to a real SparseIndex; no model downloaded. Pins that
results are ranked by the index's dot product and carry the right doc_id / text.
"""
from __future__ import annotations

from typing import Dict

import pytest

from src.retrieval.base import RetrievedChunk, Retriever
from src.retrieval.sparse_index import SparseIndex
from src.retrieval.sparse_retriever import SparseRetriever


class _FakeEncoder:
    """Maps each query string to a canned ``{term_id: weight}`` (no model)."""

    def __init__(self, mapping: Dict[str, Dict[int, float]]) -> None:
        self._mapping = mapping

    def encode(self, texts):
        return [self._mapping.get(t, {}) for t in texts]


def _index() -> SparseIndex:
    return SparseIndex.build(
        [{0: 1.0, 1: 2.0}, {1: 1.0, 2: 3.0}, {3: 5.0}],
        ["d0", "d1", "d2"],
        vocab_size=5,
    )


def test_sparse_retriever_satisfies_retriever_protocol() -> None:
    r = SparseRetriever(_FakeEncoder({}), _index(), ["t0", "t1", "t2"])
    assert isinstance(r, Retriever)


def test_sparse_retriever_ranks_by_sparse_dot_product() -> None:
    enc = _FakeEncoder({"q": {1: 1.0, 2: 1.0}})  # d1 -> 4, d0 -> 2, d2 -> 0 (filtered)
    r = SparseRetriever(enc, _index(), ["text d0", "text d1", "text d2"], top_k=10)
    out = r.retrieve("q")
    assert [(c.doc_id, c.text) for c in out] == [("d1", "text d1"), ("d0", "text d0")]
    assert out[0].score == pytest.approx(4.0)
    assert all(isinstance(c, RetrievedChunk) for c in out)


def test_sparse_retriever_respects_top_k() -> None:
    enc = _FakeEncoder({"q": {0: 1.0, 1: 1.0, 2: 1.0}})  # d1 -> 4, d0 -> 3, d2 -> 0
    r = SparseRetriever(enc, _index(), ["t0", "t1", "t2"], top_k=1)
    assert [c.doc_id for c in r.retrieve("q")] == ["d1"]


def test_sparse_retriever_empty_query_result_is_empty() -> None:
    enc = _FakeEncoder({"q": {}})  # encoder returns no terms -> index returns []
    r = SparseRetriever(enc, _index(), ["t0", "t1", "t2"])
    assert r.retrieve("q") == []


def test_sparse_retriever_rejects_bad_top_k_and_mismatched_texts() -> None:
    with pytest.raises(ValueError, match="top_k"):
        SparseRetriever(_FakeEncoder({}), _index(), ["t0", "t1", "t2"], top_k=0)
    with pytest.raises(ValueError, match="parallel"):
        SparseRetriever(_FakeEncoder({}), _index(), ["t0", "t1"])  # 2 texts, 3 docs
