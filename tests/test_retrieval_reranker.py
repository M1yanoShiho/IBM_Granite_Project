"""Tests for src/retrieval/reranker.py — cross-encoder reranking (two-stage retrieval).

Standard production retrieval: a fast first stage (Granite dense) returns a wide
candidate pool, then a cross-encoder (the Granite reranker) re-scores each
(query, doc) pair directly and keeps the top-k. The cross-encoder sees the query
and document together, so it ranks better than separate-embedding cosine — and it
can pull a complementary BM25 find out of a hybrid candidate pool. The real model
is injected here so tests need no download.
"""
from __future__ import annotations

from typing import Dict, List

import pytest

from src.retrieval.base import RetrievedChunk, Retriever
from src.retrieval.reranker import Reranker, TwoStageRetriever


class FakeCrossEncoder:
    """Stand-in for a sentence-transformers CrossEncoder, scoring by doc text."""

    def __init__(self, scores_by_text: Dict[str, float]) -> None:
        self._scores = scores_by_text

    def predict(self, pairs):
        return [self._scores[text] for _query, text in pairs]


class FixedRetriever:
    """First-stage retriever returning a canned, pre-ranked list."""

    def __init__(self, results: List[RetrievedChunk]) -> None:
        self._results = results

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        return list(self._results)


def test_reranker_resorts_candidates_by_cross_encoder_score() -> None:
    # First stage ordered d2,d1,d3; the cross-encoder prefers d1 > d3 > d2.
    candidates = [
        RetrievedChunk("d2", "text-d2", 0.9),
        RetrievedChunk("d1", "text-d1", 0.8),
        RetrievedChunk("d3", "text-d3", 0.7),
    ]
    reranker = Reranker(model=FakeCrossEncoder({"text-d2": 0.1, "text-d1": 0.9, "text-d3": 0.5}))

    out = reranker.rerank("q", candidates, top_k=3)

    assert [c.doc_id for c in out] == ["d1", "d3", "d2"]
    assert out[0].score == pytest.approx(0.9)  # the reranker score replaces the first-stage score


def test_reranker_truncates_to_top_k() -> None:
    candidates = [RetrievedChunk(f"d{i}", f"text-d{i}", 0.5) for i in range(5)]
    reranker = Reranker(model=FakeCrossEncoder({f"text-d{i}": float(i) for i in range(5)}))

    out = reranker.rerank("q", candidates, top_k=2)

    assert [c.doc_id for c in out] == ["d4", "d3"]  # the two highest cross-encoder scores


def test_reranker_handles_empty_candidates() -> None:
    assert Reranker(model=FakeCrossEncoder({})).rerank("q", [], top_k=5) == []


def test_two_stage_reranks_first_stage_output() -> None:
    # The first stage ranks d2 above d1; the reranker flips it.
    first = FixedRetriever(
        [RetrievedChunk("d2", "text-d2", 0.9), RetrievedChunk("d1", "text-d1", 0.8)]
    )
    reranker = Reranker(model=FakeCrossEncoder({"text-d2": 0.1, "text-d1": 0.9}))

    out = TwoStageRetriever(first, reranker, top_k=10).retrieve("q")

    assert [c.doc_id for c in out] == ["d1", "d2"]


def test_two_stage_reranks_only_the_top_candidates() -> None:
    # candidates=3: only the first 3 first-stage results are reranked; d3/d4 dropped.
    first = FixedRetriever(
        [RetrievedChunk(f"d{i}", f"text-d{i}", float(10 - i)) for i in range(5)]
    )
    reranker = Reranker(model=FakeCrossEncoder({f"text-d{i}": float(i) for i in range(5)}))

    out = TwoStageRetriever(first, reranker, top_k=10, candidates=3).retrieve("q")

    assert [c.doc_id for c in out] == ["d2", "d1", "d0"]


def test_two_stage_satisfies_retriever_protocol() -> None:
    ts = TwoStageRetriever(FixedRetriever([]), Reranker(model=FakeCrossEncoder({})))
    assert isinstance(ts, Retriever)


def test_two_stage_rejects_bad_top_k_and_candidates() -> None:
    reranker = Reranker(model=FakeCrossEncoder({}))
    with pytest.raises(ValueError, match="top_k"):
        TwoStageRetriever(FixedRetriever([]), reranker, top_k=0)
    with pytest.raises(ValueError, match="candidates"):
        TwoStageRetriever(FixedRetriever([]), reranker, candidates=0)
