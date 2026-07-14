from dataclasses import dataclass
from hashlib import sha256

from evidence_rag.contracts.models import Document


@dataclass(frozen=True)
class Chunk:
    document_id: str
    chunk_id: str
    evidence_id: str
    text: str
    source_uri: str


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
            chunk_hash = sha256(
                f"{document.document_id}|{self.version}|{start}|{end}|{text}".encode()
            ).hexdigest()[:16]
            chunk_id = f"chunk-{chunk_hash}"
            evidence_hash = sha256(
                f"{document.source_uri}|{chunk_id}".encode()
            ).hexdigest()[:16]
            chunks.append(
                Chunk(
                    document_id=document.document_id,
                    chunk_id=chunk_id,
                    evidence_id=f"ev-{evidence_hash}",
                    text=text,
                    source_uri=document.source_uri,
                )
            )
            if end == len(words):
                break
        return tuple(chunks)
