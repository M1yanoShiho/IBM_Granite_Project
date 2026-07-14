import importlib
import math
import os
from collections.abc import Iterable, Sequence
from typing import Any, Protocol

from evidence_rag.contracts.models import CandidateSet, Document, EvidenceCandidate, Query
from evidence_rag.contracts.protocols import Retriever
from evidence_rag.retriever.chunking import Chunk, WordChunker

DEFAULT_GRANITE_EMBEDDING_MODEL_ID = "ibm-granite/granite-embedding-english-r2"
QUERY2DOC_PROMPT = (
    "Write a short, factual passage that answers the question.\n"
    "Question: {question}\n"
    "Passage:"
)


class TextEmbedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

    def embed_query(self, text: str) -> Sequence[float]: ...


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


class GraniteEmbedder:
    """Lazy sentence-transformers wrapper for IBM Granite embeddings."""

    def __init__(
        self,
        model_id: str | None = None,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> None:
        self.model_id = (
            model_id
            or os.getenv("GRANITE_EMBEDDING_MODEL_ID")
            or DEFAULT_GRANITE_EMBEDDING_MODEL_ID
        )
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self._model: Any = self._load_model()

    def _load_model(self) -> Any:
        try:
            sentence_transformers = importlib.import_module("sentence_transformers")
        except ImportError as exc:  # pragma: no cover - depends on optional runtime deps
            raise RuntimeError(
                "GraniteEmbedder requires the optional 'sentence-transformers' package."
            ) from exc
        cache_folder = os.getenv("MODEL_CACHE_DIR") or None
        return sentence_transformers.SentenceTransformer(
            self.model_id,
            cache_folder=cache_folder,
        )

    @staticmethod
    def _as_vectors(raw: Any) -> tuple[tuple[float, ...], ...]:
        values = raw.tolist() if hasattr(raw, "tolist") else raw
        return tuple(tuple(float(item) for item in row) for row in values)

    def _encode(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        raw = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return self._as_vectors(raw)

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        prefixed = (
            [self.document_prefix + text for text in texts]
            if self.document_prefix
            else list(texts)
        )
        return self._encode(prefixed)

    def embed_query(self, text: str) -> tuple[float, ...]:
        prefixed = self.query_prefix + text if self.query_prefix else text
        encoded = self._encode((prefixed,))
        return encoded[0]


def _normalise(vector: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return tuple(0.0 for _ in vector)
    return tuple(value / norm for value in vector)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions differ")
    return sum(a * b for a, b in zip(left, right, strict=True))


class GraniteDenseRetriever:
    """Retriever implementation backed by IBM Granite text embeddings."""

    def __init__(
        self,
        documents: Iterable[Document],
        *,
        embedder: TextEmbedder | None = None,
        chunker: WordChunker | None = None,
    ) -> None:
        self.chunker = chunker or WordChunker()
        self.embedder = embedder or GraniteEmbedder()
        self.chunks: tuple[Chunk, ...] = tuple(
            chunk for document in documents for chunk in self.chunker.chunk(document)
        )
        self.chunk_vectors: tuple[tuple[float, ...], ...] = tuple(
            _normalise(vector)
            for vector in self.embedder.embed_documents(
                tuple(chunk.text for chunk in self.chunks)
            )
        )
        if len(self.chunk_vectors) != len(self.chunks):
            raise ValueError("embedder returned a different number of document vectors")

    def retrieve(self, query: Query, top_k: int) -> CandidateSet:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self.chunks:
            return CandidateSet(query_id=query.query_id, candidates=())

        query_vector = _normalise(self.embedder.embed_query(query.text))
        scored = sorted(
            (
                (_dot(query_vector, chunk_vector), chunk)
                for chunk, chunk_vector in zip(
                    self.chunks,
                    self.chunk_vectors,
                    strict=True,
                )
            ),
            key=lambda item: (-item[0], item[1].evidence_id),
        )
        candidates = tuple(
            EvidenceCandidate(
                evidence_id=chunk.evidence_id,
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source_uri=chunk.source_uri,
                retrieval_score=score,
                retrieval_rank=rank,
            )
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        )
        return CandidateSet(query_id=query.query_id, candidates=candidates)


class Query2DocRetriever:
    """Retriever wrapper that prepends an LLM-generated pseudo document to the query."""

    def __init__(
        self,
        base: Retriever,
        generator: TextGenerator,
        prompt_template: str = QUERY2DOC_PROMPT,
    ) -> None:
        self.base = base
        self.generator = generator
        self.prompt_template = prompt_template

    def _transform(self, query: Query) -> Query:
        pseudo_document = self.generator.generate(
            self.prompt_template.format(question=query.text)
        ).strip()
        if not pseudo_document:
            return query
        return Query(
            query_id=query.query_id,
            text=f"{query.text} {pseudo_document}",
        )

    def retrieve(self, query: Query, top_k: int) -> CandidateSet:
        return self.base.retrieve(self._transform(query), top_k)
