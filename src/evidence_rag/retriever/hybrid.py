"""Hybrid retrieval that fuses several retrievers into one ranked CandidateSet.

Two fusion strategies are provided:

- :class:`HybridRetriever` — Reciprocal Rank Fusion over any number of arms. Fuses
  rankings, so the arms' scores need not be comparable (the robust default).
- :class:`ConvexHybridRetriever` — a convex combination of two arms' per-query
  min-max normalised scores, weighted by ``alpha`` (the dense weight).

Both are pure compositions over the ``Retriever`` protocol and touch no
infrastructure, so any concrete retrievers can be swapped in.
"""

from collections.abc import Sequence

from evidence_rag.contracts.models import CandidateSet, Query
from evidence_rag.contracts.protocols import Retriever
from evidence_rag.retriever.fusion import (
    DEFAULT_RRF_K,
    convex_fusion,
    reciprocal_rank_fusion,
)


def _retrieve_checked(retriever: Retriever, query: Query, top_k: int) -> CandidateSet:
    result = retriever.retrieve(query, top_k)
    if result.query_id != query.query_id:
        raise ValueError("component retriever returned the wrong query ID")
    return result


class HybridRetriever:
    """Fuse any number of retrievers with Reciprocal Rank Fusion.

    Each arm is queried for ``pool_size`` candidates (defaults to the requested
    ``top_k``); deeper pools give RRF more overlap to work with.
    """

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        k: int = DEFAULT_RRF_K,
        pool_size: int | None = None,
    ) -> None:
        self.retrievers = tuple(retrievers)
        if not self.retrievers:
            raise ValueError("HybridRetriever requires at least one retriever")
        if k <= 0:
            raise ValueError("RRF k must be positive")
        if pool_size is not None and pool_size <= 0:
            raise ValueError("pool_size must be positive")
        self.k = k
        self.pool_size = pool_size

    def retrieve(self, query: Query, top_k: int) -> CandidateSet:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        pool = self.pool_size or top_k
        results = [_retrieve_checked(retriever, query, pool) for retriever in self.retrievers]
        return reciprocal_rank_fusion(
            results,
            query_id=query.query_id,
            top_k=top_k,
            k=self.k,
        )


class ConvexHybridRetriever:
    """Fuse a sparse and a dense arm by a convex combination of min-max scores."""

    def __init__(
        self,
        sparse: Retriever,
        dense: Retriever,
        *,
        alpha: float = 0.5,
        pool_size: int | None = None,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("convex alpha must satisfy 0 <= alpha <= 1")
        if pool_size is not None and pool_size <= 0:
            raise ValueError("pool_size must be positive")
        self.sparse = sparse
        self.dense = dense
        self.alpha = alpha
        self.pool_size = pool_size

    def retrieve(self, query: Query, top_k: int) -> CandidateSet:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        pool = self.pool_size or top_k
        return convex_fusion(
            _retrieve_checked(self.sparse, query, pool),
            _retrieve_checked(self.dense, query, pool),
            query_id=query.query_id,
            top_k=top_k,
            alpha=self.alpha,
        )
