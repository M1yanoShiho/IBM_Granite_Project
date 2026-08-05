import importlib
import math
import os
from collections.abc import Iterable, Sequence
from typing import Any, Protocol

from evidence_rag.contracts.models import CandidateSet, Document, EvidenceCandidate, Query
from evidence_rag.contracts.protocols import Retriever
from evidence_rag.infrastructure.corpus import CorpusSnapshot
from evidence_rag.retriever.chunking import Chunk, Chunker, WordChunker
from evidence_rag.retriever.fusion import (
    DEFAULT_RRF_K,
    best_rank_fusion,
    reciprocal_rank_fusion,
)

DEFAULT_GRANITE_EMBEDDING_MODEL_ID = "ibm-granite/granite-embedding-english-r2"
QUERY2DOC_PROMPT = (
    "Write a short, factual passage that answers the question.\n"
    "Question: {question}\n"
    "Passage:"
)
HYDE_PROMPT = (
    "Write a short, factual passage that plausibly answers the question, as if it "
    "were excerpted from a relevant document.\n"
    "Question: {question}\n"
    "Passage:"
)
DECOMPOSE_PROMPT = (
    "Break the question into a few simpler, self-contained sub-questions, one per "
    "line. If it is already simple, return it unchanged.\n"
    "Question: {question}\n"
    "Sub-questions:"
)

# Fusion strategies selectable by DecomposingRetriever. "rrf" sums each arm's
# 1/(k+rank) and is the default that every recorded result used; "best-rank" takes the
# maximum instead (see fusion.best_rank_fusion for why that suits multi-hop).
_FUSIONS = {
    "rrf": reciprocal_rank_fusion,
    "best-rank": best_rank_fusion,
}


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
        chunker: Chunker | None = None,
    ) -> None:
        self.chunker: Chunker | None = chunker or WordChunker()
        self.embedder = embedder or GraniteEmbedder()
        self._embed_chunks(
            chunk for document in documents for chunk in self.chunker.chunk(document)
        )

    @classmethod
    def from_corpus(
        cls,
        corpus: CorpusSnapshot,
        *,
        embedder: TextEmbedder | None = None,
    ) -> "GraniteDenseRetriever":
        """Embed a pre-chunked corpus snapshot directly, without re-chunking."""

        retriever = cls.__new__(cls)
        retriever.chunker = None
        retriever.embedder = embedder or GraniteEmbedder()
        retriever._embed_chunks(corpus.chunks)
        return retriever

    def _embed_chunks(self, chunks: Iterable[Chunk]) -> None:
        self.chunks: tuple[Chunk, ...] = tuple(chunks)
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
                metadata=chunk.metadata,
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


class HyDERetriever:
    """Retriever wrapper that *replaces* the query with an LLM hypothetical document.

    Unlike :class:`Query2DocRetriever` (which appends the pseudo-document to the
    original query), HyDE (Gao et al., 2022) retrieves against the generated
    hypothetical document alone. Falls back to the original query if generation is
    empty.
    """

    def __init__(
        self,
        base: Retriever,
        generator: TextGenerator,
        prompt_template: str = HYDE_PROMPT,
    ) -> None:
        self.base = base
        self.generator = generator
        self.prompt_template = prompt_template

    def _transform(self, query: Query) -> Query:
        hypothesis = self.generator.generate(
            self.prompt_template.format(question=query.text)
        ).strip()
        if not hypothesis:
            return query
        return Query(query_id=query.query_id, text=hypothesis)

    def retrieve(self, query: Query, top_k: int) -> CandidateSet:
        return self.base.retrieve(self._transform(query), top_k)


class DecomposingRetriever:
    """Decompose a complex query into sub-questions, retrieve each, and RRF-merge.

    Improves recall on multi-hop queries: each sub-question is retrieved
    independently against the base retriever and the ranked lists are fused with
    Reciprocal Rank Fusion. If the LLM returns nothing usable, falls back to
    retrieving the original query unchanged.

    ``include_original`` adds the *unmodified* query as one more fusion arm. It is
    off by default because every recorded result predates it (see
    ``docs/retriever/eval-results.md``), but on multi-hop corpora the default is
    known to lose ranking quality: on 2Wiki, decompose scores MRR .570 against
    strong-BM25's .958 while top-50 recall is untouched (−0.7pp), i.e. the gold
    document stays in the pool and is merely demoted — RRF sums ``1/(k+rank)``
    across arms, so a document ranking highly for exactly one sub-question (which
    is what a multi-hop gold *is*) is overtaken by documents ranking mediocrely
    across all of them. Fusing the original query re-injects the ranking that put
    gold first in 92.7% of those cases. See the R1 finding in
    ``docs/results-summary.md``.
    """

    def __init__(
        self,
        base: Retriever,
        generator: TextGenerator,
        prompt_template: str = DECOMPOSE_PROMPT,
        *,
        k: int = DEFAULT_RRF_K,
        pool_size: int | None = None,
        include_original: bool = False,
        fusion: str = "rrf",
    ) -> None:
        if pool_size is not None and pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if fusion not in _FUSIONS:
            raise ValueError(
                f"decompose 'fusion' must be one of {sorted(_FUSIONS)}, got {fusion!r}"
            )
        self.base = base
        self.generator = generator
        self.prompt_template = prompt_template
        self.k = k
        self.pool_size = pool_size
        self.include_original = include_original
        self.fusion = fusion

    def _subqueries(self, query: Query) -> tuple[str, ...]:
        raw = self.generator.generate(self.prompt_template.format(question=query.text))
        subqueries = tuple(
            cleaned
            for line in raw.splitlines()
            if (cleaned := line.strip().lstrip("-*0123456789.()[] ").strip())
        )
        if not subqueries:
            return (query.text,)
        if self.include_original and query.text not in subqueries:
            return (query.text, *subqueries)
        return subqueries

    def retrieve(self, query: Query, top_k: int) -> CandidateSet:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        pool = self.pool_size or top_k
        results = []
        for text in self._subqueries(query):
            result = self.base.retrieve(Query(query_id=query.query_id, text=text), pool)
            if result.query_id != query.query_id:
                raise ValueError("base retriever returned the wrong query ID")
            results.append(result)
        return _FUSIONS[self.fusion](
            results,
            query_id=query.query_id,
            top_k=top_k,
            k=self.k,
        )
