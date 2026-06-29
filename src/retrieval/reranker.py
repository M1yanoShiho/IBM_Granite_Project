"""Cross-encoder reranking — second-stage retrieval.

A bi-encoder retriever (the Granite dense retriever) embeds query and document
*separately* and compares vectors — fast, but it never lets the two texts interact.
A **cross-encoder reranker** scores a ``(query, document)`` pair jointly (full
attention over both), so it ranks a candidate pool more accurately. The standard
two-stage architecture, and the usual way to actually improve a retrieval system:

    dense retrieve top-N  ->  cross-encoder rerank  ->  top-k

Here both stages are IBM Granite (``granite-embedding-*`` bi-encoder +
``granite-embedding-reranker-english-r2``), so the system stays in-family. Reranking
a *hybrid* (granite + BM25) candidate pool can also surface the complementary BM25
finds that rank fusion alone could not.
"""

from __future__ import annotations

import os
from typing import List, Sequence

from src.retrieval.base import RetrievedChunk, Retriever

DEFAULT_RERANKER_MODEL_ID = "ibm-granite/granite-embedding-reranker-english-r2"


class Reranker:
    """A cross-encoder that re-scores ``(query, document)`` pairs.

    Parameters
    ----------
    model_id:
        The reranker model (default: the Granite English r2 reranker).
    model:
        An already-loaded model exposing ``predict(pairs) -> scores`` (injected in
        tests). ``None`` loads ``model_id`` via sentence-transformers ``CrossEncoder``.

    Notes
    -----
    Loading uses sentence-transformers ``CrossEncoder``; if the Granite reranker's
    model card prescribes a different entry point (e.g. ``AutoModelForSequence
    Classification`` or ``trust_remote_code=True``), adjust :meth:`_load_model` —
    the rest of the class only depends on ``predict``.
    """

    def __init__(self, model_id: str = DEFAULT_RERANKER_MODEL_ID, model=None) -> None:
        self.model_id = model_id
        self._model = model if model is not None else self._load_model()

    def _load_model(self):
        from sentence_transformers import CrossEncoder

        return CrossEncoder(self.model_id, cache_folder=os.getenv("MODEL_CACHE_DIR") or None)

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        """Cross-encoder relevance score for ``query`` against each text."""
        if not texts:
            return []
        return [float(s) for s in self._model.predict([(query, t) for t in texts])]

    def rerank(
        self, query: str, candidates: List[RetrievedChunk], top_k: int
    ) -> List[RetrievedChunk]:
        """Re-rank ``candidates`` by cross-encoder score, best-first, keep ``top_k``.

        The returned chunks carry the **reranker** score (not the first-stage one).
        Stable: equal scores keep their first-stage order.
        """
        if not candidates:
            return []
        scored = list(zip(candidates, self.score(query, [c.text for c in candidates])))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [
            RetrievedChunk(doc_id=c.doc_id, text=c.text, score=float(s))
            for c, s in scored[:top_k]
        ]


class TwoStageRetriever:
    """First-stage retriever + cross-encoder reranker, as one ``Retriever``.

    Parameters
    ----------
    retriever:
        The first stage (any ``Retriever`` — the Granite dense retriever or a
        hybrid) producing the candidate pool.
    reranker:
        The :class:`Reranker` that re-scores the pool.
    top_k:
        Number of reranked results to return.
    candidates:
        How many of the first stage's results to rerank (the rest are dropped). A
        wider pool gives the reranker more to work with at more cost.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        top_k: int = 10,
        candidates: int = 100,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if candidates < 1:
            raise ValueError("candidates must be at least 1.")
        self.retriever = retriever
        self.reranker = reranker
        self.top_k = top_k
        self.candidates = candidates

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """Retrieve a candidate pool, then return the reranked top-k."""
        pool = self.retriever.retrieve(query)[: self.candidates]
        return self.reranker.rerank(query, pool, self.top_k)
