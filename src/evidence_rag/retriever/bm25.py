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
        self.tokens = tuple(self.analyzer(chunk.text) for chunk in self.chunks)
        # Term counts are a function of the corpus alone, so they are built once here
        # instead of being rebuilt for every chunk on every query. Chunks hold ~180
        # tokens against ~6 effective query terms, so that rebuild was most of the
        # per-query constant (measured in R5, docs/hpc-run-log.md).
        self.term_frequencies = tuple(Counter(tokens) for tokens in self.tokens)
        self.average_length = (
            sum(len(tokens) for tokens in self.tokens) / len(self.tokens) if self.tokens else 0.0
        )
        self.document_frequency = Counter(token for tokens in self.tokens for token in set(tokens))

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
        scored: list[tuple[float, Chunk]] = []
        for chunk, tokens, counts in zip(
            self.chunks, self.tokens, self.term_frequencies, strict=True
        ):
            # Length normalisation is constant across the terms of a single chunk. The
            # arithmetic below is grouped exactly as it was when computed per term, so
            # scores stay bit-for-bit identical rather than merely close.
            length_ratio = len(tokens) / self.average_length if self.average_length else 0.0
            normalisation = self.k1 * (1.0 - self.b + self.b * length_ratio)
            score = 0.0
            for term in query_terms:
                frequency = counts[term]
                if frequency == 0:
                    continue
                score += inverse_document_frequency[term] * (
                    frequency * (self.k1 + 1.0) / (frequency + normalisation)
                )
            if score > 0.0:
                scored.append((score, chunk))
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
