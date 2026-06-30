"""Sparse (learned-sparse / SPLADE) retriever — the strong lexical arm.

Mirrors :class:`~src.retrieval.retriever.DenseRetriever` but over a
:class:`~src.retrieval.sparse_index.SparseIndex`: it encodes the query into SPLADE
term weights, searches the CSR index by sparse dot product, and returns
``RetrievedChunk``s. Doc-level (the index rows are whole documents), so each result's
``doc_id`` is the parent document id straight away (contract 5) — a drop-in lexical arm
for the convex hybrid (Phase 2), interchangeable with ``BM25Retriever``.
"""

from __future__ import annotations

from typing import List, Sequence

from src.retrieval.base import RetrievedChunk
from src.retrieval.sparse_index import SparseIndex
from src.retrieval.splade_encoder import SpladeEncoder


class SparseRetriever:
    """Encode a query with SPLADE, search a ``SparseIndex``, return ranked chunks.

    Parameters
    ----------
    encoder:
        A :class:`SpladeEncoder` (or anything with ``encode([text]) -> [TermWeights]``).
    index:
        The :class:`SparseIndex` over the corpus; its ``doc_ids`` are parallel to ``texts``.
    texts:
        Document texts in the same row order as ``index.doc_ids``.
    top_k:
        Number of documents to return per query.
    """

    def __init__(
        self,
        encoder: SpladeEncoder,
        index: SparseIndex,
        texts: Sequence[str],
        top_k: int = 10,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if len(texts) != len(index.doc_ids):
            raise ValueError("texts must be parallel to index.doc_ids (same length).")
        self.encoder = encoder
        self.index = index
        self.texts = list(texts)
        self.top_k = top_k

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """Return the top-k documents most relevant to ``query`` by sparse dot product."""
        query_vector = self.encoder.encode([query])[0]
        hits = self.index.search(query_vector, self.top_k)
        return [
            RetrievedChunk(doc_id=self.index.doc_ids[i], text=self.texts[i], score=score)
            for i, score in hits
        ]
