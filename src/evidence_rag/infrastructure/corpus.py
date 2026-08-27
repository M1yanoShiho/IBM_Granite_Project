import json
import re
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


_HEADING = re.compile(r"^#{1,6}\s+\S")
_TABLE_ROW = re.compile(r"^\s*\|")


@dataclass(frozen=True)
class _Block:
    """One structural unit of a document, with its span in word indices."""

    kind: str  # "heading" | "table" | "paragraph"
    text: str
    start: int
    end: int


def _parse_blocks(text: str) -> tuple[_Block, ...]:
    """Split text into headings, tables and paragraphs, keeping word offsets.

    The offsets are indices into ``text.split()`` — the same coordinate system
    ``WordChunker`` hashes into chunk ids, so ids stay comparable across chunkers.
    """

    blocks: list[_Block] = []
    buffered: list[str] = []
    buffered_kind = "paragraph"
    buffered_start = 0
    consumed = 0

    def flush(end: int) -> None:
        nonlocal buffered, buffered_kind
        body = "\n".join(buffered).strip()
        if body:
            blocks.append(_Block(buffered_kind, body, buffered_start, end))
        buffered = []
        buffered_kind = "paragraph"

    for line in text.splitlines():
        words_here = len(line.split())
        if _HEADING.match(line):
            flush(consumed)
            blocks.append(_Block("heading", line.strip(), consumed, consumed + words_here))
            consumed += words_here
            buffered_start = consumed
            continue

        kind = "table" if _TABLE_ROW.match(line) else "paragraph"
        if not line.strip():
            # A blank line ends whatever was accumulating; it belongs to no block.
            flush(consumed)
            buffered_start = consumed
            continue
        if buffered and kind != buffered_kind:
            # A table starting mid-paragraph (or prose resuming after one) is a boundary:
            # a table row swept into a prose chunk is exactly what this chunker exists to
            # prevent.
            flush(consumed)
            buffered_start = consumed
        if not buffered:
            buffered_start = consumed
        buffered_kind = kind
        buffered.append(line)
        consumed += words_here

    flush(consumed)
    return tuple(blocks)


class SectionChunker:
    """Structure-aware chunking for text corpora: cut at structure, not at a word count.

    ``WordChunker`` slices every ``chunk_size`` words regardless of what it cuts through,
    which splits tables down the middle and severs a heading from the section it titles.
    Both hurt retrieval directly — half a table is not evidence, and a section whose title
    landed in the previous chunk cannot be matched on its own subject.

    The rules, in order:

    - A heading starts a new chunk and stays *with* the section it introduces, so every
      chunk carries its own title.
    - A table is never split, even when it alone exceeds ``chunk_size``. An oversized
      table becomes one oversized chunk; that is a deliberate trade (see limitations).
    - Otherwise blocks accumulate until adding the next would exceed ``chunk_size``.
    - A single prose paragraph longer than ``chunk_size`` falls back to ``WordChunker``'s
      sliding window, with ``overlap``, since nothing structural is left to cut on.

    On a corpus with no markup at all (SciFact, 2Wiki) there are no headings or tables, so
    this degrades to paragraph-aware windowing: still never cutting mid-paragraph unless a
    paragraph is itself too long. That is a real behavioural difference from ``WordChunker``
    and therefore a different ``corpus_signature`` — the two cannot silently share an index.

    ### Limitations

    - Markdown only (ATX ``#`` headings, ``|`` table rows). reStructuredText, HTML tables
      and setext headings are not detected and simply read as prose.
    - An oversized table yields an oversized chunk. Splitting it would produce rows with no
      header, which is worse than a long chunk; a table-aware splitter that repeats the
      header row on each piece would be the fix, and is not implemented.
    - ``overlap`` applies only inside the oversized-paragraph fallback. Structural chunks do
      not overlap, so a fact spanning a section boundary is not duplicated into both.
    """

    version = "section-v1"

    def __init__(self, chunk_size: int = 120, overlap: int = 20) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> tuple[Chunk, ...]:
        blocks = _parse_blocks(document.text)
        if not blocks:
            return ()

        chunks: list[Chunk] = []
        pending: list[_Block] = []

        def emit() -> None:
            if not pending:
                return
            text = "\n\n".join(block.text for block in pending)
            chunks.append(
                _derive_chunk(document, self.version, pending[0].start, pending[-1].end, text)
            )
            pending.clear()

        for block in blocks:
            length = block.end - block.start
            pending_length = (pending[-1].end - pending[0].start) if pending else 0

            if block.kind == "heading":
                emit()
                pending.append(block)
                continue

            # A chunk that is nothing but a heading is worthless on its own and leaves the
            # section it titles anonymous, so a pending heading never triggers a split —
            # it rides along even when that overflows chunk_size.
            titles_only = all(held.kind == "heading" for held in pending)

            if length > self.chunk_size and block.kind == "paragraph":
                # Nothing structural left to cut on; fall back to the sliding window,
                # carrying any pending heading into the first piece.
                prefix = list(pending) if titles_only else []
                if not titles_only:
                    emit()
                pending.clear()
                chunks.extend(self._window(document, block, prefix))
                continue

            if pending and not titles_only and pending_length + length > self.chunk_size:
                emit()
            pending.append(block)

        emit()
        return tuple(chunks)

    def _window(
        self,
        document: Document,
        block: _Block,
        prefix: list[_Block],
    ) -> list[Chunk]:
        words = block.text.split()
        step = self.chunk_size - self.overlap
        heading = "\n\n".join(held.text for held in prefix)
        start_of_span = prefix[0].start if prefix else block.start
        pieces: list[Chunk] = []
        for offset in range(0, len(words), step):
            end = min(offset + self.chunk_size, len(words))
            body = " ".join(words[offset:end])
            # Only the first piece carries the heading; repeating it on every piece would
            # inflate every chunk's term counts with the same words.
            text = f"{heading}\n\n{body}" if heading and offset == 0 else body
            pieces.append(
                _derive_chunk(
                    document,
                    self.version,
                    start_of_span if offset == 0 else block.start + offset,
                    block.start + end,
                    text,
                )
            )
            if end == len(words):
                break
        return pieces


def build_chunker(name: str, *, chunk_size: int, overlap: int) -> Chunker:
    """Construct the chunker a config's ``[chunker] name`` selects.

    ``chunk_size``/``overlap`` are ignored by window-less chunkers; ``ChunkerConfig``
    already refuses to let a config set them there, so they are never silently dropped.
    """

    if name == "word":
        return WordChunker(chunk_size=chunk_size, overlap=overlap)
    if name == "prechunked":
        return PrechunkedChunker()
    if name == "section":
        return SectionChunker(chunk_size=chunk_size, overlap=overlap)
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
