"""BM25 lexical retrieval — the classical baseline.

BM25 (Okapi BM25) is the standard sparse, term-matching baseline that any
neural retrieval system must beat to justify itself. We use a pure-Python
implementation (``rank_bm25``) so the baseline is reproducible and dependency-light.

The system's dense retriever (``src.retrieval.retriever.DenseRetriever``) is
compared head-to-head against this baseline on the same benchmark queries and
metrics.
"""

from __future__ import annotations

import re
from typing import List, Sequence

from rank_bm25 import BM25Okapi

from src.retrieval.base import RetrievedChunk

_TOKEN_RE = re.compile(r"\b\w+\b")


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
        if len(corpus) != len(doc_ids):
            raise ValueError("corpus and doc_ids must have the same length.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        self.corpus = list(corpus)
        self.doc_ids = list(doc_ids)
        self.top_k = top_k
        self._tokenized_corpus = [self._tokenize(text) for text in self.corpus]
        self._bm25 = self._build_index()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercase word-tokenisation used consistently for docs and queries."""
        return _TOKEN_RE.findall(text.lower())

    def _build_index(self) -> BM25Okapi | None:
        """Tokenise the corpus and build the BM25 index."""
        if not self._tokenized_corpus:
            return None
        return BM25Okapi(self._tokenized_corpus)

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """Return the top-k chunks most relevant to ``query`` by BM25 score."""
        query_tokens = self._tokenize(query)
        if self._bm25 is None or not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: float(scores[i]),
            reverse=True,
        )[: self.top_k]

        return [
            RetrievedChunk(
                doc_id=self.doc_ids[i],
                text=self.corpus[i],
                score=float(scores[i]),
            )
            for i in ranked_indices
        ]
