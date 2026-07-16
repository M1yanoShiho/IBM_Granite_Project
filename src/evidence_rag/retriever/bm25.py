import math
import re
from collections import Counter
from collections.abc import Iterable

from evidence_rag.contracts.models import CandidateSet, Document, EvidenceCandidate, Query
from evidence_rag.infrastructure.corpus import CorpusSnapshot
from evidence_rag.retriever.chunking import Chunk, Chunker, WordChunker

TOKEN = re.compile(r"[A-Za-z0-9]+")


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
    ) -> None:
        self.chunker: Chunker | None = chunker or WordChunker()
        self.k1, self.b = validate_bm25_parameters(k1, b)
        self._set_chunks(chunk for document in documents for chunk in self.chunker.chunk(document))

    @classmethod
    def from_corpus(
        cls,
        corpus: CorpusSnapshot,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Retriever":
        retriever = cls.__new__(cls)
        retriever.chunker = None
        retriever.k1, retriever.b = validate_bm25_parameters(k1, b)
        retriever._set_chunks(corpus.chunks)
        return retriever

    def _set_chunks(self, chunks: Iterable[Chunk]) -> None:
        self.chunks = tuple(chunks)
        self.tokens = tuple(tokenize(chunk.text) for chunk in self.chunks)
        self.average_length = (
            sum(len(tokens) for tokens in self.tokens) / len(self.tokens) if self.tokens else 0.0
        )
        self.document_frequency = Counter(token for tokens in self.tokens for token in set(tokens))

    def retrieve(self, query: Query, top_k: int) -> CandidateSet:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scored: list[tuple[float, Chunk]] = []
        total = len(self.chunks)
        for chunk, tokens in zip(self.chunks, self.tokens, strict=True):
            counts = Counter(tokens)
            score = 0.0
            for term in tokenize(query.text):
                frequency = counts[term]
                if frequency == 0:
                    continue
                df = self.document_frequency[term]
                inverse_document_frequency = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
                length_ratio = len(tokens) / self.average_length if self.average_length else 0.0
                denominator = frequency + self.k1 * (1.0 - self.b + self.b * length_ratio)
                score += inverse_document_frequency * (frequency * (self.k1 + 1.0) / denominator)
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
