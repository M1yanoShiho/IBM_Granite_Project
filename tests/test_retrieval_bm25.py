from __future__ import annotations

import pytest

from src.retrieval.base import RetrievedChunk, Retriever
from src.retrieval.bm25_baseline import BM25Retriever


def test_bm25_retriever_satisfies_shared_retriever_contract() -> None:
    retriever = BM25Retriever(
        corpus=["granite retrieval benchmark"],
        doc_ids=["doc-1"],
        top_k=1,
    )

    assert isinstance(retriever, Retriever)


def test_bm25_returns_ranked_retrieved_chunks() -> None:
    retriever = BM25Retriever(
        corpus=[
            "ibm granite retrieval improves enterprise search",
            "banana cake recipe with sugar and butter",
            "scifact benchmark contains scientific claims",
        ],
        doc_ids=["doc-granite", "doc-recipe", "doc-scifact"],
        top_k=2,
    )

    results = retriever.retrieve("granite enterprise retrieval")

    assert len(results) == 2
    assert all(isinstance(item, RetrievedChunk) for item in results)
    assert results[0].doc_id == "doc-granite"
    assert results[0].text == "ibm granite retrieval improves enterprise search"
    assert results[0].score >= results[1].score


def test_bm25_respects_top_k() -> None:
    retriever = BM25Retriever(
        corpus=[
            "retrieval alpha",
            "retrieval beta",
            "retrieval gamma",
            "retrieval delta",
        ],
        doc_ids=["d1", "d2", "d3", "d4"],
        top_k=3,
    )

    results = retriever.retrieve("retrieval")

    assert len(results) == 3


def test_bm25_returns_empty_list_for_empty_query() -> None:
    retriever = BM25Retriever(
        corpus=["granite retrieval"],
        doc_ids=["doc-1"],
        top_k=5,
    )

    assert retriever.retrieve("   !!!   ") == []


def test_bm25_handles_empty_corpus() -> None:
    retriever = BM25Retriever(corpus=[], doc_ids=[], top_k=5)

    assert retriever.retrieve("anything") == []


def test_bm25_rejects_mismatched_corpus_and_doc_ids() -> None:
    with pytest.raises(ValueError, match="corpus and doc_ids"):
        BM25Retriever(corpus=["one document"], doc_ids=["d1", "d2"], top_k=1)

