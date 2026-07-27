"""A stronger BM25 baseline: real analyzer plus Anserini/BEIR-tuned parameters.

The default :class:`~evidence_rag.retriever.bm25.BM25Retriever` scores over plain
lower-cased tokens with ``rank_bm25``-style ``k1=1.5, b=0.75`` and keeps every
term. Published BM25 baselines (Anserini/Lucene over BEIR) instead:

- tune the parameters to ``k1=0.9, b=0.4``; and
- run a real analyzer that drops a standard English stopword list, so common
  function words stop dominating the length normalisation.

This retriever reuses the exact BM25 scoring engine of ``BM25Retriever`` (it only
swaps in a stopword-filtering analyzer and different defaults), so rankings stay
deterministic and the ``from_corpus`` fast path still avoids re-chunking.
"""

from collections.abc import Iterable

from evidence_rag.contracts.models import CandidateSet, Document, Query
from evidence_rag.infrastructure.corpus import CorpusSnapshot
from evidence_rag.retriever.bm25 import Analyzer, BM25Retriever, tokenize
from evidence_rag.retriever.chunking import Chunk, Chunker

# The classic ~127-word English stopword set (the Lucene/Snowball baseline), so
# the analyzer needs no nltk/sklearn dependency. Override via the ``stopwords``
# argument, or pass an empty iterable to disable stopword removal entirely.
ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "all", "am", "an",
        "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
        "before", "being", "below", "between", "both", "but", "by", "can't",
        "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
        "doing", "don't", "down", "during", "each", "few", "for", "from",
        "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
        "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
        "i", "if", "in", "into", "is", "isn't", "it", "its", "itself", "let's",
        "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of",
        "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
        "ourselves", "out", "over", "own", "same", "shan't", "she", "should",
        "shouldn't", "so", "some", "such", "than", "that", "the", "their",
        "theirs", "them", "themselves", "then", "there", "these", "they", "this",
        "those", "through", "to", "too", "under", "until", "up", "very", "was",
        "wasn't", "we", "were", "weren't", "what", "when", "where", "which",
        "while", "who", "whom", "why", "with", "won't", "would", "wouldn't",
        "you", "your", "yours", "yourself", "yourselves",
    }
)


def build_analyzer(stopwords: Iterable[str] | None = None) -> Analyzer:
    """Return an analyzer that lower-case tokenises then drops stopwords.

    Contraction stopwords (``don't``) are matched against the raw token text
    before it is split on non-alphanumerics, so both the raw form and its
    tokenised pieces (``don``/``t``) are filtered.
    """

    active = ENGLISH_STOPWORDS if stopwords is None else frozenset(stopwords)

    def analyze(text: str) -> tuple[str, ...]:
        return tuple(token for token in tokenize(text) if token not in active)

    return analyze


class StrongBM25Retriever:
    """BM25 with a stopword-filtering analyzer and Anserini/BEIR-tuned defaults.

    Wraps an inner :class:`BM25Retriever` configured with the strong analyzer, so
    the scoring engine is reused verbatim rather than reimplemented.
    """

    def __init__(
        self,
        documents: Iterable[Document],
        chunker: Chunker | None = None,
        k1: float = 0.9,
        b: float = 0.4,
        *,
        stopwords: Iterable[str] | None = None,
    ) -> None:
        self._bm25 = BM25Retriever(
            documents,
            chunker=chunker,
            k1=k1,
            b=b,
            analyzer=build_analyzer(stopwords),
        )

    @classmethod
    def from_corpus(
        cls,
        corpus: CorpusSnapshot,
        *,
        k1: float = 0.9,
        b: float = 0.4,
        stopwords: Iterable[str] | None = None,
    ) -> "StrongBM25Retriever":
        retriever = cls.__new__(cls)
        retriever._bm25 = BM25Retriever.from_corpus(
            corpus,
            k1=k1,
            b=b,
            analyzer=build_analyzer(stopwords),
        )
        return retriever

    @property
    def k1(self) -> float:
        return self._bm25.k1

    @property
    def b(self) -> float:
        return self._bm25.b

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return self._bm25.chunks

    def retrieve(self, query: Query, top_k: int) -> CandidateSet:
        return self._bm25.retrieve(query, top_k)
