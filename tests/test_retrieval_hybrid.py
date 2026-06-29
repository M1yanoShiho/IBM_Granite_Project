"""Tests for src/retrieval/hybrid.py — Reciprocal Rank Fusion (RRF) retriever.

The failure analysis showed granite (dense) and BM25 (lexical) are complementary on
SciFact (they win different queries). RRF fuses their *rankings* — score-scale-free,
the standard dense+sparse hybrid — so a query a single retriever misses can still
surface if the other ranked it. These tests pin the RRF maths on hand-checkable
rankings; no models needed.
"""
from __future__ import annotations

from typing import List

import pytest

from src.retrieval.base import RetrievedChunk, Retriever
from src.retrieval.hybrid import HybridRetriever


class FixedRetriever:
    """Returns a canned, pre-ranked result list (satisfies the Retriever Protocol)."""

    def __init__(self, results: List[RetrievedChunk]) -> None:
        self._results = results

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        return list(self._results)


def ranked(*doc_ids: str) -> List[RetrievedChunk]:
    """Build a best-first result list; RRF uses position (rank), not the score."""
    n = len(doc_ids)
    return [
        RetrievedChunk(doc_id=d, text=f"text-{d}", score=float(n - i))
        for i, d in enumerate(doc_ids)
    ]


def test_hybrid_satisfies_retriever_protocol() -> None:
    assert isinstance(HybridRetriever([FixedRetriever([])]), Retriever)


def test_rrf_fuses_two_rankings_by_reciprocal_rank() -> None:
    # a: d1 d2 d3   b: d3 d1 d4   (k_rrf=1 for clean hand-maths)
    # RRF: d1 = 1/2+1/3, d3 = 1/4+1/2, d2 = 1/3, d4 = 1/4
    # -> d1 (0.833) > d3 (0.75) > d2 (0.333) > d4 (0.25).
    a = FixedRetriever(ranked("d1", "d2", "d3"))
    b = FixedRetriever(ranked("d3", "d1", "d4"))

    out = HybridRetriever([a, b], top_k=10, k_rrf=1).retrieve("q")

    assert [c.doc_id for c in out] == ["d1", "d3", "d2", "d4"]
    assert out[0].score == pytest.approx(1 / 2 + 1 / 3)  # d1: rank1 in a, rank2 in b


def test_rrf_counts_each_doc_once_at_its_best_rank() -> None:
    # Dense returns two chunks of d1 (ranks 1 and 3); d1 must count once, at rank 1.
    a = FixedRetriever(ranked("d1", "d2", "d1"))

    out = HybridRetriever([a], top_k=10, k_rrf=1).retrieve("q")

    assert [c.doc_id for c in out] == ["d1", "d2"]
    assert out[0].score == pytest.approx(1 / 2)  # d1 at rank 1 only, not 1/2 + 1/4
    assert out[1].score == pytest.approx(1 / 3)  # d2 at rank 2


def test_rrf_respects_top_k() -> None:
    a = FixedRetriever(ranked("d1", "d2", "d3", "d4"))

    out = HybridRetriever([a], top_k=2, k_rrf=1).retrieve("q")

    assert [c.doc_id for c in out] == ["d1", "d2"]


def test_rrf_single_component_preserves_doc_order() -> None:
    # One retriever: RRF score is monotonic in rank, so order is unchanged.
    a = FixedRetriever(ranked("d3", "d1", "d2"))

    out = HybridRetriever([a], top_k=10).retrieve("q")

    assert [c.doc_id for c in out] == ["d3", "d1", "d2"]


def test_rrf_carries_doc_text_through() -> None:
    out = HybridRetriever([FixedRetriever(ranked("d1"))], top_k=10).retrieve("q")
    assert out[0].text == "text-d1"


def test_hybrid_rejects_empty_components() -> None:
    with pytest.raises(ValueError, match="at least one"):
        HybridRetriever([])


def test_hybrid_rejects_bad_top_k_and_k_rrf() -> None:
    with pytest.raises(ValueError, match="top_k"):
        HybridRetriever([FixedRetriever([])], top_k=0)
    with pytest.raises(ValueError, match="k_rrf"):
        HybridRetriever([FixedRetriever([])], k_rrf=0)
