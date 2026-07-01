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
from src.retrieval.reranker import (
    LLMListwiseReranker,
    Reranker,
    TwoStageRetriever,
)


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


# ---------------------------------------------------------------------------
# LLMListwiseReranker — RankGPT-style listwise reranking with the project's LLM
# ---------------------------------------------------------------------------
class ScriptedLLM:
    """Returns a scripted permutation string per call and records prompts."""

    def __init__(self, responses) -> None:
        self._responses = list(responses)
        self.calls: List[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._responses.pop(0) if self._responses else ""


def test_listwise_parse_permutation_reorders_by_identifiers() -> None:
    # "[3] > [1] > [2]" (1-indexed) -> zero-indexed order [2, 0, 1].
    assert LLMListwiseReranker._parse_permutation("[3] > [1] > [2]", 3) == [2, 0, 1]


def test_listwise_parse_permutation_fills_missing_and_ignores_invalid() -> None:
    # Only [2] named; out-of-range [9] and the duplicate are dropped, then the
    # unranked positions (0, 2) are appended in their original order.
    assert LLMListwiseReranker._parse_permutation("[9] > [2] > [2]", 3) == [1, 0, 2]


def test_listwise_rerank_applies_ranking_with_descending_scores() -> None:
    candidates = [
        RetrievedChunk("d1", "t1", 0.3),
        RetrievedChunk("d2", "t2", 0.2),
        RetrievedChunk("d3", "t3", 0.1),
    ]
    llm = ScriptedLLM(["[3] > [1] > [2]"])

    out = LLMListwiseReranker(llm, window=10, step=10).rerank("q", candidates, top_k=2)

    assert [c.doc_id for c in out] == ["d3", "d1"]
    assert out[0].score > out[1].score  # carries a rank-derived descending score
    assert len(llm.calls) == 1  # one window covers all candidates


def test_listwise_rerank_sliding_window_bubbles_best_to_front() -> None:
    # window=2, step=1 over 3 candidates -> windows [d2,d3] then [d1, winner].
    # Each response ranks the 2nd shown passage first, bubbling d3 to the front.
    candidates = [
        RetrievedChunk("d1", "t1", 0.9),
        RetrievedChunk("d2", "t2", 0.8),
        RetrievedChunk("d3", "t3", 0.7),
    ]
    llm = ScriptedLLM(["[2] > [1]", "[2] > [1]"])

    out = LLMListwiseReranker(llm, window=2, step=1).rerank("q", candidates, top_k=3)

    assert [c.doc_id for c in out] == ["d3", "d1", "d2"]
    assert len(llm.calls) == 2  # two overlapping windows


def test_listwise_rerank_empty_candidates_returns_empty_without_calling_llm() -> None:
    llm = ScriptedLLM([])

    out = LLMListwiseReranker(llm).rerank("q", [], top_k=5)

    assert out == []
    assert llm.calls == []  # no LLM call when there is nothing to rank


def test_listwise_reranker_works_inside_two_stage() -> None:
    first = FixedRetriever(
        [RetrievedChunk("d2", "t2", 0.9), RetrievedChunk("d1", "t1", 0.8)]
    )
    llm = ScriptedLLM(["[2] > [1]"])  # ranks the 2nd shown (d1) above d2

    out = TwoStageRetriever(first, LLMListwiseReranker(llm, window=10), top_k=10).retrieve("q")

    assert [c.doc_id for c in out] == ["d1", "d2"]

