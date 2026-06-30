"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Combines several retrievers (in practice the Granite **dense** retriever and the
**BM25** lexical baseline) into one. The failure analysis
(``eval/failure_analysis.py``) showed the two are *complementary* on SciFact — they
win different queries — so fusing them should recover documents either misses alone.

RRF (Cormack et al., 2009) fuses *rankings*, not scores, so it needs no score
normalisation across the very different BM25 and cosine-similarity scales::

    rrf(doc) = sum over retrievers of   1 / (k_rrf + rank_of_doc_in_that_retriever)

where ``rank`` is 1-based and a document absent from a retriever contributes 0.
``k_rrf`` (default 60, the value from the original paper) damps the influence of
top ranks so many moderate agreements can outweigh a single first place.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from src.retrieval.base import RetrievedChunk, Retriever
from src.retrieval.fusion import Scores, fuse_one


class HybridRetriever:
    """Fuse several retrievers' rankings with Reciprocal Rank Fusion.

    Parameters
    ----------
    retrievers:
        The component retrievers to fuse (each satisfies the ``Retriever``
        Protocol). Typically ``[DenseRetriever, BM25Retriever]``.
    top_k:
        Number of fused documents to return per query.
    k_rrf:
        The RRF damping constant (default 60). Larger flattens the rank weighting.

    Notes
    -----
    A component may return several chunks of the same document (the dense retriever
    does); each document is counted **once**, at its best (smallest) rank within
    that component, so multi-chunk documents are not double-credited.
    """

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        top_k: int = 10,
        k_rrf: int = 60,
    ) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one component retriever.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if k_rrf < 1:
            raise ValueError("k_rrf must be at least 1.")
        self.retrievers = list(retrievers)
        self.top_k = top_k
        self.k_rrf = k_rrf

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """Return the top-k documents by fused RRF score, ranked best-first."""
        rrf_scores: Dict[str, float] = {}
        texts: Dict[str, str] = {}
        for retriever in self.retrievers:
            rank = 0
            seen: set[str] = set()
            for chunk in retriever.retrieve(query):
                if chunk.doc_id in seen:
                    continue  # same doc via another chunk -> keep only its best rank
                seen.add(chunk.doc_id)
                rank += 1
                rrf_scores[chunk.doc_id] = (
                    rrf_scores.get(chunk.doc_id, 0.0) + 1.0 / (self.k_rrf + rank)
                )
                texts.setdefault(chunk.doc_id, chunk.text)
        # Sort by fused score desc; doc_id as a deterministic tie-breaker.
        ranked = sorted(rrf_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [
            RetrievedChunk(doc_id=doc_id, text=texts[doc_id], score=score)
            for doc_id, score in ranked[: self.top_k]
        ]


class ConvexHybridRetriever:
    """Fuse a dense and a lexical retriever by convex combination of normalised scores.

    Unlike RRF (:class:`HybridRetriever`), this uses the arms' *scores* — per-query
    min-max normalised — with a tunable weight ``alpha`` (the dense weight).
    ``alpha = 1`` reproduces the dense ranking, ``alpha = 0`` the lexical ranking. The
    fusion maths live in :mod:`src.retrieval.fusion`, shared with the offline alpha
    sweep (``eval/tune_alpha.py``).

    Parameters
    ----------
    dense, lexical:
        The two component retrievers. They must already return a deep candidate pool
        (~100 each) — the caller builds them that way; this class has no pool knob, it
        fuses whatever the arms return and keeps the top ``top_k``.
    alpha:
        Dense weight in [0, 1].
    top_k:
        Number of fused documents to return.
    """

    def __init__(
        self,
        dense: Retriever,
        lexical: Retriever,
        alpha: float,
        top_k: int = 10,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {alpha}.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        self.dense = dense
        self.lexical = lexical
        self.alpha = alpha
        self.top_k = top_k

    def _doc_scores(self, retriever: Retriever, query: str) -> tuple[Scores, Dict[str, str]]:
        """Max-pool one arm's chunks to ``{doc_id: score}`` (+ ``{doc_id: text}``).

        A document hit via several chunks is scored at its best (max) chunk, matching
        contract 3, so a multi-chunk document is not under-credited before fusion.
        """
        scores: Scores = {}
        texts: Dict[str, str] = {}
        for chunk in retriever.retrieve(query):
            if chunk.doc_id not in scores or chunk.score > scores[chunk.doc_id]:
                scores[chunk.doc_id] = chunk.score
            texts.setdefault(chunk.doc_id, chunk.text)
        return scores, texts

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """Return the top-k documents by convex-fused score, ranked best-first."""
        dense_scores, dense_texts = self._doc_scores(self.dense, query)
        lex_scores, lex_texts = self._doc_scores(self.lexical, query)
        fused = fuse_one(dense_scores, lex_scores, self.alpha)
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
        out: List[RetrievedChunk] = []
        for doc_id, score in ranked[: self.top_k]:
            text = dense_texts.get(doc_id) or lex_texts.get(doc_id, "")
            out.append(RetrievedChunk(doc_id=doc_id, text=text, score=score))
        return out
