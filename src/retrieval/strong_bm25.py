"""A tuned, analyzed BM25 — the *fair* lexical baseline.

The naive :class:`~src.retrieval.bm25_baseline.BM25Retriever` uses ``\\b\\w+\\b``
tokens, no stopword removal, no stemming, and ``rank_bm25``'s default
``k1=1.5, b=0.75`` — it sits *below* the Anserini/Lucene BM25 that published work
reports, so beating it overstates the neural gap (and "SPLADE is N× faster than
BM25" flatters a pure-Python baseline). This retriever keeps the same
``Retriever`` contract but closes most of that gap with the two changes that
matter most and need **no new dependency**:

- **Tuned parameters** — ``k1=0.9, b=0.4`` (the Anserini BEIR defaults), instead
  of ``rank_bm25``'s generic ``1.5 / 0.75``.
- **A real analyzer** — lower-case, stopword removal (a vendored standard English
  list, overridable), and an *optional injected* stemmer applied to both corpus
  and queries.

Stemming is left injectable rather than bundled so the module stays
dependency-light (the project pins ``rank_bm25`` but not PyStemmer/nltk); pass a
Snowball/Porter ``stemmer`` for full Lucene-``EnglishAnalyzer`` parity. It is a
drop-in for :class:`BM25Retriever` in ``_build_component`` / ``tune_alpha`` /
convex fusion.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, List, Optional, Sequence

from rank_bm25 import BM25Okapi

from src.retrieval.base import RetrievedChunk

_TOKEN_RE = re.compile(r"\b\w+\b")

# Vendored standard English stopword list (the classic ~127-word set) so the
# analyzer needs no nltk/sklearn dependency. Override via the ``stopwords`` arg.
ENGLISH_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can't cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he he'd he'll he's
    her here here's hers herself him himself his how how's i i'd i'll i'm i've
    if in into is isn't it it's its itself let's me more most mustn't my myself
    no nor not of off on once only or other ought our ours ourselves out over
    own same shan't she she'd she'll she's should shouldn't so some such than
    that that's the their theirs them themselves then there there's these they
    they'd they'll they're they've this those through to too under until up very
    was wasn't we we'd we'll we're we've were weren't what what's when when's
    where where's which while who who's whom why why's with won't would wouldn't
    you you'd you'll you're you've your yours yourself yourselves
    """.split()
)


class StrongBM25Retriever:
    """Okapi BM25 with tuned parameters and a real analyzer.

    Parameters
    ----------
    corpus:
        The chunks/passages to search over.
    doc_ids:
        Parallel identifiers for ``corpus`` (scored against qrels).
    top_k:
        Number of chunks to return per query.
    k1, b:
        BM25 term-saturation / length-normalisation parameters. Default to the
        Anserini BEIR values (``0.9 / 0.4``) rather than ``rank_bm25``'s generic
        ``1.5 / 0.75``.
    stopwords:
        Words removed during analysis. ``None`` uses :data:`ENGLISH_STOPWORDS`;
        pass an empty iterable to disable stopword removal.
    stemmer:
        Optional ``token -> token`` callable applied to both corpus and query
        tokens (e.g. a Snowball stemmer). ``None`` = no stemming.
    """

    def __init__(
        self,
        corpus: Sequence[str],
        doc_ids: Sequence[str],
        top_k: int = 10,
        k1: float = 0.9,
        b: float = 0.4,
        stopwords: Optional[Iterable[str]] = None,
        stemmer: Optional[Callable[[str], str]] = None,
    ) -> None:
        if len(corpus) != len(doc_ids):
            raise ValueError("corpus and doc_ids must have the same length.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        self.corpus = list(corpus)
        self.doc_ids = list(doc_ids)
        self.top_k = top_k
        self.k1 = k1
        self.b = b
        self.stopwords = (
            ENGLISH_STOPWORDS if stopwords is None else frozenset(stopwords)
        )
        self.stemmer = stemmer
        self._tokenized_corpus = [self._analyze(text) for text in self.corpus]
        self._bm25 = self._build_index()

    def _analyze(self, text: str) -> List[str]:
        """Lower-case, tokenise, drop stopwords, then optionally stem."""
        tokens = [
            token
            for token in _TOKEN_RE.findall(text.lower())
            if token not in self.stopwords
        ]
        if self.stemmer is not None:
            tokens = [self.stemmer(token) for token in tokens]
        return tokens

    def _build_index(self) -> BM25Okapi | None:
        if not self._tokenized_corpus:
            return None
        return BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """Return the top-k chunks most relevant to ``query`` by BM25 score."""
        query_tokens = self._analyze(query)
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
