from __future__ import annotations

import pytest

from src.retrieval.base import RetrievedChunk, Retriever
from src.retrieval.retriever import DenseRetriever


class FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0, 0.5]


class FakeIndex:
    def __init__(self) -> None:
        self.calls: list[tuple[list[float], int]] = []

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        self.calls.append((query_vector, top_k))
        return [
            RetrievedChunk(doc_id="doc-1", text="granite retrieval", score=0.9),
            RetrievedChunk(doc_id="doc-2", text="bm25 baseline", score=0.4),
            RetrievedChunk(doc_id="doc-3", text="irrelevant", score=0.1),
        ]


def test_dense_retriever_satisfies_shared_retriever_contract() -> None:
    retriever = DenseRetriever(
        embedder=FakeEmbedder(),
        index=FakeIndex(),
        top_k=2,
    )

    assert isinstance(retriever, Retriever)


def test_dense_retriever_embeds_query_and_searches_index() -> None:
    embedder = FakeEmbedder()
    index = FakeIndex()
    retriever = DenseRetriever(embedder=embedder, index=index, top_k=2)

    results = retriever.retrieve("enterprise retrieval")

    assert embedder.queries == ["enterprise retrieval"]
    assert index.calls == [([1.0, 0.0, 0.5], 2)]
    assert [item.doc_id for item in results] == ["doc-1", "doc-2"]
    assert results[0].score >= results[1].score


def test_dense_retriever_rejects_missing_index() -> None:
    retriever = DenseRetriever(embedder=FakeEmbedder(), index=None, top_k=2)

    with pytest.raises(ValueError, match="requires an index"):
        retriever.retrieve("query")


def test_dense_retriever_rejects_index_without_search_method() -> None:
    retriever = DenseRetriever(embedder=FakeEmbedder(), index=object(), top_k=2)

    with pytest.raises(TypeError, match="search"):
        retriever.retrieve("query")

