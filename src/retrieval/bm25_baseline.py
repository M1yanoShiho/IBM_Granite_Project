"""BM25 lexical retrieval — the classical baseline.

BM25 (Okapi BM25) is the standard sparse, term-matching baseline that any
neural retrieval system must beat to justify itself. We use a pure-Python
implementation (``rank_bm25``) so the baseline is reproducible and dependency-light.

The system's dense retriever (``src.retrieval.retriever.DenseRetriever``) is
compared head-to-head against this baseline on the same benchmark queries and
metrics.
"""

from __future__ import annotations

from typing import List, Sequence

from src.retrieval.retriever import RetrievedChunk


class BM25Retriever:
    """Okapi BM25 baseline retriever.

    Parameters
    ----------
    corpus:
        The collection of chunks/passages to search over.
    doc_ids:
        Parallel list of identifiers for ``corpus`` (to score against qrels).
    top_k:
        Number of chunks to return per query.
    """

    def __init__(
        self,
        corpus: Sequence[str],
        doc_ids: Sequence[str],
        top_k: int = 10,
    ) -> None:
        self.corpus = list(corpus)
        self.doc_ids = list(doc_ids)
        self.top_k = top_k
        self._bm25 = self._build_index()

    def _build_index(self):
        """Tokenise the corpus and build the BM25 index (e.g. via rank_bm25)."""
        raise NotImplementedError("TODO: tokenise corpus and build BM25 index.")

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """Return the top-k chunks most relevant to ``query`` by BM25 score."""
        raise NotImplementedError("TODO: score query against BM25 index.")
