"""Build a ready-to-query retriever from raw documents (product-side glue).

The evaluation harness builds its retrievers from a *labelled benchmark* corpus
(``eval/run_benchmark._build_retrievers``). This is the product-side mirror: it
turns the raw documents a user pastes or uploads in the demo into a
``DenseRetriever``, by composing the parts that already exist —

    chunk the documents  (``src.ingestion.chunker.chunk_document``)
    embed the chunks     (``src.retrieval.embedder.Embedder``)
    build a FAISS index  (``src.ingestion.indexer.VectorIndexer``, CONTRACT 5)
    wrap it              (``src.retrieval.retriever.DenseRetriever``, CONTRACT 1)

so the RAG demo runs over real documents instead of mock data. The index is
in-memory only (no persistence): the demo rebuilds it per document set.
"""

from __future__ import annotations

from typing import Mapping

from src.ingestion.chunker import chunk_document
from src.ingestion.indexer import VectorIndexer
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import DenseRetriever


def build_dense_retriever(
    documents: Mapping[str, str],
    *,
    backend: str = "granite",
    embedding_model_id: str | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    top_k: int = 10,
) -> DenseRetriever:
    """Build a ``DenseRetriever`` over ``documents`` (``{doc_id: text}``).

    Chunks each document, embeds the chunks with the chosen ``backend``
    (``"granite"`` or ``"sentence-transformers"``; ``embedding_model_id``
    overrides the backend default), builds an in-memory FAISS index, and returns
    a retriever ready for ``retrieve(query)``. Retrieved chunks carry their
    **parent** ``doc_id`` (CONTRACT 5), so answers can be cited back to the
    source document.

    Raises
    ------
    ValueError
        If ``documents`` is empty, or none of the documents yield any chunkable
        text (e.g. all whitespace) — surfaced as a clear error rather than a
        cryptic failure deep inside index construction.
    """
    if not documents:
        raise ValueError("build_dense_retriever requires at least one document.")

    embedder = Embedder(backend=backend, model_id=embedding_model_id)
    chunks = [
        chunk
        for doc_id, text in documents.items()
        for chunk in chunk_document(
            doc_id, text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    ]
    if not chunks:
        raise ValueError("Documents produced no chunkable text to index.")

    index = VectorIndexer(embedder).build(chunks)
    return DenseRetriever(embedder, index, top_k=top_k)


def build_dense_retriever_from_text(
    text: str,
    *,
    doc_id: str = "document",
    **kwargs,
) -> DenseRetriever:
    """Convenience wrapper for a single pasted document (the demo's main path).

    Wraps ``text`` as a one-document corpus keyed by ``doc_id`` and forwards the
    remaining keyword arguments to :func:`build_dense_retriever`.
    """
    return build_dense_retriever({doc_id: text}, **kwargs)
