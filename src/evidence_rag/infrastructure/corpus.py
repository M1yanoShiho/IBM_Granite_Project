import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from hashlib import sha256
from typing import Annotated, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from evidence_rag.contracts.models import Document, SourceMetadata

NonEmpty = Annotated[str, Field(min_length=1)]
NonNegativeCount = Annotated[int, Field(ge=0)]
PositiveInteger = Annotated[int, Field(gt=0)]
NonNegativeInteger = Annotated[int, Field(ge=0)]
RecordT = TypeVar("RecordT")


@dataclass(frozen=True)
class Chunk:
    document_id: str
    chunk_id: str
    evidence_id: str
    text: str
    source_uri: str
    metadata: SourceMetadata | None = None


class Chunker(Protocol):
    version: str

    def chunk(self, document: Document) -> tuple[Chunk, ...]: ...


def _derive_chunk(document: Document, version: str, start: int, end: int, text: str) -> Chunk:
    chunk_hash = sha256(
        f"{document.document_id}|{version}|{start}|{end}|{text}".encode()
    ).hexdigest()[:16]
    chunk_id = f"chunk-{chunk_hash}"
    evidence_hash = sha256(f"{document.source_uri}|{chunk_id}".encode()).hexdigest()[:16]
    return Chunk(
        document_id=document.document_id,
        chunk_id=chunk_id,
        evidence_id=f"ev-{evidence_hash}",
        text=text,
        source_uri=document.source_uri,
        metadata=document.metadata,
    )


class WordChunker:
    version = "word-v1"

    def __init__(self, chunk_size: int = 120, overlap: int = 20) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> tuple[Chunk, ...]:
        words = document.text.split()
        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        for start in range(0, len(words), step):
            end = min(start + self.chunk_size, len(words))
            text = " ".join(words[start:end])
            chunks.append(_derive_chunk(document, self.version, start, end, text))
            if end == len(words):
                break
        return tuple(chunks)


class PrechunkedChunker:
    """Chunker for corpora already chunked at ingestion (e.g. Docling HybridChunker).

    Emits each document as exactly one chunk, so structure-aware units produced
    by a loader (tables, cross-page sections) are never split again downstream.
    """

    version = "prechunked-v1"

    def chunk(self, document: Document) -> tuple[Chunk, ...]:
        text = document.text.strip()
        if not text:
            return ()
        # The hashed (start, end) span is in word indices like WordChunker's;
        # stripping only removes surrounding whitespace, so the word count of
        # the unstripped text is identical and the ids stay deterministic.
        return (_derive_chunk(document, self.version, 0, len(document.text.split()), text),)


def build_chunker(name: str, *, chunk_size: int, overlap: int) -> Chunker:
    """Construct the chunker a config's ``[chunker] name`` selects.

    ``chunk_size``/``overlap`` are ignored by window-less chunkers; ``ChunkerConfig``
    already refuses to let a config set them there, so they are never silently dropped.
    """

    if name == "word":
        return WordChunker(chunk_size=chunk_size, overlap=overlap)
    if name == "prechunked":
        return PrechunkedChunker()
    raise ValueError(f"unknown chunker name: {name}")


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class CorpusManifest(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset_signature: NonEmpty
    chunker_name: NonEmpty
    chunker_version: NonEmpty
    # ``None`` for chunkers that do not split by a fixed window (``PrechunkedChunker``),
    # where a number here would be an invented one. Word chunking still records both, so
    # every manifest written before this was optional parses and re-hashes identically.
    chunk_size: PositiveInteger | None
    overlap: NonNegativeInteger | None
    document_count: NonNegativeCount
    chunk_count: NonNegativeCount
    corpus_signature: NonEmpty


class CorpusSnapshot(FrozenModel):
    manifest: CorpusManifest
    documents: tuple[Document, ...]
    chunks: tuple[Chunk, ...]


def _corpus_signature(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


class CorpusBuilder:
    def __init__(self, chunker: Chunker | None = None) -> None:
        self.chunker: Chunker = chunker or WordChunker()

    def build(
        self,
        documents: Iterable[Document],
        dataset_signature: str,
    ) -> CorpusSnapshot:
        document_tuple = tuple(documents)
        self._validate_unique(document_tuple, lambda document: document.document_id, "document")

        chunks = tuple(
            chunk
            for document in document_tuple
            for chunk in self.chunker.chunk(document)
        )
        self._validate_unique(chunks, lambda chunk: chunk.chunk_id, "chunk")
        self._validate_unique(chunks, lambda chunk: chunk.evidence_id, "evidence")

        # A window-less chunker exposes neither attribute; the keys stay in the hashed
        # manifest so the word-chunker signature is byte-identical to before.
        chunk_size: int | None = getattr(self.chunker, "chunk_size", None)
        overlap: int | None = getattr(self.chunker, "overlap", None)

        manifest_data = {
            "schema_version": "1.0",
            "dataset_signature": dataset_signature,
            "chunker_name": type(self.chunker).__name__,
            "chunker_version": self.chunker.version,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "document_count": len(document_tuple),
            "chunk_count": len(chunks),
        }
        corpus_signature = _corpus_signature(
            {
                "manifest": manifest_data,
                "documents": [document.model_dump(mode="json") for document in document_tuple],
                "chunks": [
                    {
                        "document_id": chunk.document_id,
                        "chunk_id": chunk.chunk_id,
                        "evidence_id": chunk.evidence_id,
                        "text": chunk.text,
                        "source_uri": chunk.source_uri,
                        "metadata": (
                            chunk.metadata.model_dump(mode="json")
                            if chunk.metadata is not None
                            else None
                        ),
                    }
                    for chunk in chunks
                ],
            }
        )
        manifest = CorpusManifest(
            dataset_signature=dataset_signature,
            chunker_name=type(self.chunker).__name__,
            chunker_version=self.chunker.version,
            chunk_size=chunk_size,
            overlap=overlap,
            document_count=len(document_tuple),
            chunk_count=len(chunks),
            corpus_signature=corpus_signature,
        )
        return CorpusSnapshot(manifest=manifest, documents=document_tuple, chunks=chunks)

    @staticmethod
    def _validate_unique(
        records: Iterable[RecordT],
        identifier: Callable[[RecordT], str],
        label: str,
    ) -> None:
        seen: set[str] = set()
        for record in records:
            value = identifier(record)
            if value in seen:
                raise ValueError(f"duplicate {label} ID: {value}")
            seen.add(value)
