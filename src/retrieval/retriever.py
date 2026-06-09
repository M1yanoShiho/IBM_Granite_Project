"""Dense (semantic) retriever — the core of the delivered system.

Given an indexed corpus, returns the top-k most relevant chunks for a query
using vector similarity. This is the component whose retrieval quality is
measured with precision/recall, nDCG, and MRR on standard benchmarks and
compared against the BM25 baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.retrieval.embedder import Embedder


@dataclass
class RetrievedChunk:
    """A single retrieval result.

    Attributes
    ----------
    doc_id:
        Identifier of the source chunk/passage (used to score against qrels).
    text:
        The chunk text.
    score:
        Similarity score (higher = more relevant).
    """

    doc_id: str
    text: str
    score: float


class DenseRetriever:
    """Semantic retriever over a persisted vector index.

    Parameters
    ----------
    embedder:
        The :class:`~src.retrieval.embedder.Embedder` used for queries.
    index:
        A vector index (e.g. a FAISS index produced by ``src.ingestion.indexer``).
    top_k:
        Number of chunks to return per query.
    """

    def __init__(self, embedder: Embedder, index=None, top_k: int = 10) -> None:
        self.embedder = embedder
        self.index = index
        self.top_k = top_k

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """Return the top-k chunks most relevant to ``query``."""
        raise NotImplementedError(
            "TODO: embed query, search the vector index, return ranked chunks."
        )
