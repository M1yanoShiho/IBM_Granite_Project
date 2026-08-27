import math
import re
from collections import Counter
from collections.abc import Callable, Iterable

from evidence_rag.contracts.models import CandidateSet, Document, EvidenceCandidate, Query
from evidence_rag.infrastructure.corpus import CorpusSnapshot
from evidence_rag.retriever.chunking import Chunk, Chunker, WordChunker

TOKEN = re.compile(r"[A-Za-z0-9]+")

# An analyzer turns raw text into the term sequence BM25 scores over. The default
# is plain lower-cased tokenisation; StrongBM25 supplies a stopword-filtering one.
Analyzer = Callable[[str], tuple[str, ...]]


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).lower() for match in TOKEN.finditer(text))


def validate_bm25_parameters(k1: object, b: object) -> tuple[float, float]:
    values: dict[str, float] = {}
    for name, value in (("k1", k1), ("b", b)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"BM25 parameter {name!r} must be numeric and finite")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"BM25 parameter {name!r} must be numeric and finite")
        values[name] = numeric
    if values["k1"] < 0:
        raise ValueError("BM25 parameter 'k1' must satisfy k1 >= 0")
    if not 0 <= values["b"] <= 1:
        raise ValueError("BM25 parameter 'b' must satisfy 0 <= b <= 1")
    return values["k1"], values["b"]


class BM25Retriever:
    def __init__(
        self,
        documents: Iterable[Document],
        chunker: Chunker | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        *,
        analyzer: Analyzer | None = None,
    ) -> None:
        self.chunker: Chunker | None = chunker or WordChunker()
        self.k1, self.b = validate_bm25_parameters(k1, b)
        self.analyzer: Analyzer = analyzer or tokenize
        self._set_chunks(chunk for document in documents for chunk in self.chunker.chunk(document))

    @classmethod
    def from_corpus(
        cls,
        corpus: CorpusSnapshot,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        analyzer: Analyzer | None = None,
    ) -> "BM25Retriever":
        retriever = cls.__new__(cls)
        retriever.chunker = None
        retriever.k1, retriever.b = validate_bm25_parameters(k1, b)
        retriever.analyzer = analyzer or tokenize
        retriever._set_chunks(corpus.chunks)
        return retriever

    def _set_chunks(self, chunks: Iterable[Chunk]) -> None:
        self.chunks = tuple(chunks)
        analyzed = [self.analyzer(chunk.text) for chunk in self.chunks]
        self.average_length = (
            sum(len(tokens) for tokens in analyzed) / len(analyzed) if analyzed else 0.0
        )
        self.document_frequency = Counter(token for tokens in analyzed for token in set(tokens))

        # The inverted index. R5 measured that the remaining cost was the *linearity* --
        # every query touched every chunk regardless of what it asked for, and the constant
        # had already been cut as far as it goes (5.27-5.40x, bit-for-bit identical). Only
        # this changes the asymptotics: a query now touches the chunks containing its terms
        # and no others.
        #
        # This is a transpose of the per-chunk Counters it replaces, not an addition to
        # them, so it holds the same postings in the other orientation. Its memory cost
        # relative to the old forward index is unmeasured -- see the limitations in R5.
        postings: dict[str, list[tuple[int, int]]] = {}
        for index, tokens in enumerate(analyzed):
            for term, frequency in Counter(tokens).items():
                postings.setdefault(term, []).append((index, frequency))
        self.postings = {term: tuple(entries) for term, entries in postings.items()}

        # Length normalisation depends only on the chunk and on (k1, b), both fixed for the
        # life of the retriever, so it is computed once here. The expression is grouped
        # exactly as it was inside the scan, so the float is identical rather than close.
        self.normalisation = tuple(
            self.k1
            * (
                1.0
                - self.b
                + self.b * (len(tokens) / self.average_length if self.average_length else 0.0)
            )
            for tokens in analyzed
        )

    def retrieve(self, query: Query, top_k: int) -> CandidateSet:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        total = len(self.chunks)
        # Analysing the query, and each term's IDF, depend only on the query and the
        # corpus — never on the chunk being scored — so both are lifted out of the scan
        # rather than recomputed once per chunk. Duplicate query terms are preserved
        # (the tuple is iterated, not a set) because a term repeated in the query
        # contributes its score once per occurrence, which is the existing behaviour.
        query_terms = self.analyzer(query.text)
        inverse_document_frequency: dict[str, float] = {}
        for term in query_terms:
            if term not in inverse_document_frequency:
                df = self.document_frequency[term]
                inverse_document_frequency[term] = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
        # Walking the postings term by term, rather than the corpus chunk by chunk, is what
        # makes the cost proportional to the query's own postings instead of to the corpus.
        # It must not move a score, and the order of the additions is what decides that:
        # a chunk accumulates one contribution per query term, in query-term order, exactly
        # as the old inner loop did. Float addition is not associative, so iterating terms
        # in any other order would change the last bits of the sum.
        accumulated: dict[int, float] = {}
        k1_plus_one = self.k1 + 1.0
        for term in query_terms:
            entries = self.postings.get(term)
            if entries is None:
                continue
            weight = inverse_document_frequency[term]
            for index, frequency in entries:
                accumulated[index] = accumulated.get(index, 0.0) + weight * (
                    frequency * k1_plus_one / (frequency + self.normalisation[index])
                )
        # Every contribution is strictly positive (IDF > 0 for any df, and the term factor
        # is positive whenever frequency > 0), so a chunk absent from the postings is
        # exactly a chunk the old scan would have left at 0.0 and dropped. The filter is
        # kept anyway rather than assumed away.
        scored: list[tuple[float, Chunk]] = [
            (score, self.chunks[index]) for index, score in accumulated.items() if score > 0.0
        ]
        scored.sort(key=lambda item: (-item[0], item[1].evidence_id))
        candidates = tuple(
            EvidenceCandidate(
                evidence_id=chunk.evidence_id,
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source_uri=chunk.source_uri,
                retrieval_score=score,
                retrieval_rank=rank,
                metadata=chunk.metadata,
            )
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        )
        return CandidateSet(query_id=query.query_id, candidates=candidates)
