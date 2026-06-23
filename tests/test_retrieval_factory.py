"""Tests for the product-side retriever factory (``src/retrieval/factory.py``).

The factory turns raw documents (what a user pastes/uploads in the demo) into a
ready-to-query ``DenseRetriever`` by composing the existing chunker, embedder,
FAISS indexer, and retriever. The embedding model is monkeypatched with a
deterministic bag-of-words fake so nothing is downloaded (same approach as the
``run_benchmark`` dense tests).
"""

from __future__ import annotations

import pytest

from src.ingestion.chunker import Chunk
from src.retrieval.base import RetrievedChunk, Retriever
from src.retrieval.factory import (
    build_dense_retriever,
    build_dense_retriever_from_text,
)


class FakeSentenceTransformer:
    """Deterministic SentenceTransformer stand-in (no model download).

    Embeds text as bag-of-words counts over a tiny vocabulary, so a query lands
    closest to the document sharing its words. Records the model ids it is
    constructed with, so tests can assert the override is threaded through.
    """

    _VOCAB = ("granite", "retrieval", "banana", "cake")
    instances: list[str] = []

    def __init__(self, model_id, cache_folder=None) -> None:
        self.model_id = model_id
        FakeSentenceTransformer.instances.append(model_id)

    def encode(
        self,
        texts,
        convert_to_numpy: bool = False,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
    ):
        if isinstance(texts, str):
            texts = [texts]
        return [[float(t.lower().split().count(w)) for w in self._VOCAB] for t in texts]


@pytest.fixture(autouse=True)
def _fake_embedding_backend(monkeypatch):
    """Swap the real SentenceTransformer for the fake in every test here."""
    FakeSentenceTransformer.instances = []
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSentenceTransformer
    )


def test_builds_retriever_that_finds_matching_document() -> None:
    # Two documents; the query shares all its words with d1. The factory should
    # return a Retriever-contract object whose top hit is d1 — and the result's
    # doc_id is the PARENT doc id (contract 5), not a chunk id like "d1::0",
    # even though d1 is split into multiple chunks here.
    retriever = build_dense_retriever(
        {"d1": "granite retrieval granite retrieval", "d2": "banana cake banana cake"},
        chunk_size=2,
        chunk_overlap=0,
    )

    assert isinstance(retriever, Retriever)
    results = retriever.retrieve("granite retrieval")
    assert all(isinstance(r, RetrievedChunk) for r in results)
    assert results[0].doc_id == "d1"


def test_respects_top_k() -> None:
    retriever = build_dense_retriever(
        {"d1": "granite retrieval", "d2": "banana", "d3": "cake"},
        top_k=1,
    )

    results = retriever.retrieve("granite retrieval")

    assert len(results) == 1
    assert results[0].doc_id == "d1"


def test_threads_chunk_params_to_chunk_document(monkeypatch) -> None:
    # The chunk size/overlap must reach chunk_document, else the demo silently
    # re-chunks at the default 512/50 regardless of what the caller asked for.
    calls = []

    def spy_chunk_document(doc_id, text, chunk_size=512, chunk_overlap=50):
        calls.append((chunk_size, chunk_overlap))
        return [Chunk(chunk_id=f"{doc_id}::0", doc_id=doc_id, text=text)]

    monkeypatch.setattr("src.retrieval.factory.chunk_document", spy_chunk_document)

    build_dense_retriever({"d1": "granite retrieval"}, chunk_size=256, chunk_overlap=32)

    assert calls == [(256, 32)]


def test_threads_backend_and_embedding_model_id_to_embedder() -> None:
    # A model override must reach the embedding model, so the demo's model picker
    # actually swaps encoders.
    build_dense_retriever(
        {"d1": "granite retrieval"},
        backend="granite",
        embedding_model_id="ibm-granite/granite-embedding-small-english-r2",
    )

    assert (
        "ibm-granite/granite-embedding-small-english-r2"
        in FakeSentenceTransformer.instances
    )


def test_from_text_wraps_a_single_pasted_document() -> None:
    # The demo's main path: one pasted document, given an id for citations.
    retriever = build_dense_retriever_from_text("granite retrieval", doc_id="pasted")

    results = retriever.retrieve("granite retrieval")

    assert results[0].doc_id == "pasted"


def test_rejects_empty_document_set() -> None:
    with pytest.raises(ValueError, match="at least one document"):
        build_dense_retriever({})


def test_rejects_documents_with_no_chunkable_text() -> None:
    # Whitespace-only paste -> no chunks -> a clear error, not a cryptic numpy
    # crash deep inside index construction.
    with pytest.raises(ValueError, match="no chunkable text"):
        build_dense_retriever({"d1": "   "})
