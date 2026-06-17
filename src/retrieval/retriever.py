"""Dense (semantic) retriever — the core of the delivered system.

Given an indexed corpus, returns the top-k most relevant chunks for a query
using vector similarity. This is the component whose retrieval quality is
measured with precision/recall, nDCG, and MRR on standard benchmarks and
compared against the BM25 baseline.
"""

from __future__ import annotations

from typing import List

# RetrievedChunk's canonical home is src.retrieval.base (the shared contract);
# re-exported here so existing imports keep working. See docs/interfaces.md.
from src.retrieval.base import RetrievedChunk
from src.retrieval.embedder import Embedder


class DenseRetriever:
    """Semantic retriever over a vector index wrapper.

    Parameters
    ----------
    embedder:
        The :class:`~src.retrieval.embedder.Embedder` used for queries.
    index:
        Object exposing ``search(query_vector, top_k) -> List[RetrievedChunk]``.
        P5's FAISS indexer should return or load such a wrapper.
    top_k:
        Number of chunks to return per query.
    """

    def __init__(self, embedder: Embedder, index=None, top_k: int = 10) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        self.embedder = embedder
        self.index = index
        self.top_k = top_k

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """Return the top-k chunks most relevant to ``query``."""
        if self.index is None:
            raise ValueError("DenseRetriever requires an index before retrieval.")
        if not hasattr(self.index, "search"):
            raise TypeError(
                "DenseRetriever index must expose search(query_vector, top_k)."
            )

        query_vector = self.embedder.embed_query(query)
        results = self.index.search(query_vector, self.top_k)
        return list(results)[: self.top_k]
