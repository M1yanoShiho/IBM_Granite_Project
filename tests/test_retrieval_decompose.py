"""Tests for QueryDecomposeTransform, _rrf_merge, and DecomposingRetriever."""

from __future__ import annotations

import pytest

from src.retrieval.base import RetrievedChunk
from src.retrieval.bm25_baseline import BM25Retriever
from src.retrieval.query_transform import (
    DecomposingRetriever,
    QueryDecomposeTransform,
    _rrf_merge,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeLLM:
    """Returns a scripted list of sub-questions (numbered, one per line)."""

    def __init__(self, sub_questions: list[str]) -> None:
        self._sub_qs = sub_questions

    def generate(self, prompt: str) -> str:
        return "\n".join(
            f"{i + 1}. {q}" for i, q in enumerate(self._sub_qs)
        )


def _chunk(doc_id: str, text: str, score: float, rank: int) -> RetrievedChunk:
    return RetrievedChunk(doc_id=doc_id, text=text, score=score, rank=rank)


# ---------------------------------------------------------------------------
# _rrf_merge
# ---------------------------------------------------------------------------

class TestRrfMerge:
    def test_single_list_preserves_order(self) -> None:
        lst = [_chunk("a", "A", 0.9, 1), _chunk("b", "B", 0.8, 2)]
        result = _rrf_merge([lst], k=60, top_k=5)
        assert [r.doc_id for r in result] == ["a", "b"]

    def test_document_in_both_lists_ranks_higher(self) -> None:
        list1 = [_chunk("shared", "X", 0.9, 1), _chunk("only1", "Y", 0.8, 2)]
        list2 = [_chunk("shared", "X", 0.7, 1), _chunk("only2", "Z", 0.6, 2)]
        result = _rrf_merge([list1, list2], k=60, top_k=3)
        assert result[0].doc_id == "shared"

    def test_rrf_score_is_sum_of_reciprocal_ranks(self) -> None:
        lst = [_chunk("doc", "T", 1.0, 1)]
        result = _rrf_merge([lst, lst], k=60, top_k=1)
        expected = 2 * (1.0 / (60 + 1))
        assert abs(result[0].score - expected) < 1e-9

    def test_top_k_limits_output(self) -> None:
        lst = [_chunk(f"d{i}", "t", 1.0 - i * 0.1, i + 1) for i in range(10)]
        result = _rrf_merge([lst], k=60, top_k=3)
        assert len(result) == 3

    def test_output_ranks_are_consecutive_from_one(self) -> None:
        lst = [_chunk("a", "A", 0.9, 1), _chunk("b", "B", 0.8, 2), _chunk("c", "C", 0.7, 3)]
        result = _rrf_merge([lst], k=60, top_k=3)
        assert [r.rank for r in result] == [1, 2, 3]

    def test_empty_lists_return_empty(self) -> None:
        assert _rrf_merge([], k=60, top_k=5) == []

    def test_deduplicates_doc_ids(self) -> None:
        list1 = [_chunk("x", "X", 0.9, 1)]
        list2 = [_chunk("x", "X", 0.9, 1)]
        result = _rrf_merge([list1, list2], k=60, top_k=5)
        assert len(result) == 1
        assert result[0].doc_id == "x"


# ---------------------------------------------------------------------------
# QueryDecomposeTransform
# ---------------------------------------------------------------------------

class TestQueryDecomposeTransform:
    def test_parses_numbered_list(self) -> None:
        llm = FakeLLM(["Who is A?", "When did B happen?", "Where is C?"])
        transform = QueryDecomposeTransform(llm, n_subqueries=3)
        result = transform("complex multi-fact question")
        assert result == ["Who is A?", "When did B happen?", "Where is C?"]

    def test_respects_n_subqueries_cap(self) -> None:
        llm = FakeLLM(["Q1", "Q2", "Q3", "Q4"])
        transform = QueryDecomposeTransform(llm, n_subqueries=2)
        result = transform("anything")
        assert len(result) == 2

    def test_falls_back_to_original_on_empty_llm_output(self) -> None:
        class EmptyLLM:
            def generate(self, prompt: str) -> str:
                return ""

        transform = QueryDecomposeTransform(EmptyLLM(), n_subqueries=3)
        result = transform("original query")
        assert result == ["original query"]

    def test_strips_leading_numbering(self) -> None:
        class BulletLLM:
            def generate(self, prompt: str) -> str:
                return "1) First\n2) Second"

        transform = QueryDecomposeTransform(BulletLLM(), n_subqueries=2)
        result = transform("q")
        assert result == ["First", "Second"]


# ---------------------------------------------------------------------------
# DecomposingRetriever (integration with BM25)
# ---------------------------------------------------------------------------

CORPUS = [
    "IBM founded in 1911 as Computing-Tabulating-Recording Company",
    "IBM revenue in 2023 was 61.9 billion dollars",
    "Granite is IBM's family of AI models for enterprise use",
    "Thomas Watson Sr led IBM for nearly four decades",
    "Python is a high-level programming language",
]
DOC_IDS = ["doc_founding", "doc_revenue", "doc_granite", "doc_watson", "doc_python"]


class TestDecomposingRetriever:
    def _make_retriever(self, sub_questions: list[str]) -> DecomposingRetriever:
        base = BM25Retriever(corpus=CORPUS, doc_ids=DOC_IDS, top_k=5)
        llm = FakeLLM(sub_questions)
        decompose = QueryDecomposeTransform(llm, n_subqueries=len(sub_questions))
        return DecomposingRetriever(base, decompose, include_original=True, top_k=5)

    def test_returns_list_of_retrieved_chunks(self) -> None:
        retriever = self._make_retriever(["when was IBM founded?", "IBM revenue?"])
        results = retriever.retrieve("Tell me about IBM history and financials")
        assert isinstance(results, list)
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_output_size_bounded_by_top_k(self) -> None:
        retriever = self._make_retriever(["IBM AI models?", "IBM leadership?"])
        results = retriever.retrieve("IBM overview")
        assert len(results) <= 5

    def test_ranks_are_consecutive_from_one(self) -> None:
        retriever = self._make_retriever(["IBM founding?"])
        results = retriever.retrieve("IBM history")
        for expected_rank, chunk in enumerate(results, 1):
            assert chunk.rank == expected_rank

    def test_multi_subquery_finds_more_diverse_docs(self) -> None:
        """Documents covering sub-topics should both appear when sub-questions target each."""
        retriever = self._make_retriever([
            "when was IBM founded?",
            "what is IBM revenue?",
        ])
        results = retriever.retrieve("IBM founding and revenue")
        doc_ids = {r.doc_id for r in results}
        assert "doc_founding" in doc_ids
        assert "doc_revenue" in doc_ids

    def test_include_original_false_still_works(self) -> None:
        base = BM25Retriever(corpus=CORPUS, doc_ids=DOC_IDS, top_k=5)
        llm = FakeLLM(["IBM AI models?"])
        decompose = QueryDecomposeTransform(llm, n_subqueries=1)
        retriever = DecomposingRetriever(
            base, decompose, include_original=False, top_k=3
        )
        results = retriever.retrieve("IBM Granite")
        assert len(results) <= 3
        assert all(isinstance(r, RetrievedChunk) for r in results)
