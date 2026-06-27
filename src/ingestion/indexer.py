"""Vector index construction and persistence.

Embeds chunks and builds a **persistent** FAISS index that the retriever loads
at query time. Persistence (save/load) is what distinguishes a real system from
the throwaway, rebuilt-every-run index used in early experiments.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Sequence

import faiss
import numpy as np

from src.ingestion.chunker import Chunk
from src.retrieval.base import RetrievedChunk
from src.retrieval.embedder import Embedder


class FaissIndex:
    """Wraps a FAISS index + chunk metadata.

    DenseRetriever calls index.search(query_vector, top_k) and expects
    List[RetrievedChunk] back — this class provides that interface.
    """

    def __init__(self, faiss_index, chunks: List[Chunk]) -> None:
        self._index = faiss_index
        self._chunks = chunks

    def search(self, query_vector: List[float], top_k: int) -> List[RetrievedChunk]:
        q = np.array([query_vector], dtype="float32")
        scores, ids = self._index.search(q, top_k)
        results = []
        for rank, i in enumerate(ids[0]):
            if i < 0:
                continue
            chunk = self._chunks[i]
            results.append(RetrievedChunk(
                doc_id=chunk.doc_id,
                text=chunk.text,
                score=float(scores[0][rank]),
            ))
        return results


class VectorIndexer:
    """Builds and persists a FAISS vector index over chunks."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def build(self, chunks: Sequence[Chunk]) -> FaissIndex:
        """Embed chunks and build an in-memory FAISS index.

        Raises
        ------
        ValueError
            If ``chunks`` is empty — surfaced as a clear error rather than the
            cryptic ``IndexError: tuple index out of range`` from reading the
            embedding dimension off a zero-row array.
        """
        chunks = list(chunks)
        if not chunks:
            raise ValueError(
                "Cannot build an index from zero chunks; provide at least one "
                "chunk (check that the corpus is non-empty and produced chunkable text)."
            )
        texts = [c.text for c in chunks]
        vectors = np.array(self.embedder.embed_documents(texts), dtype="float32")
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        return FaissIndex(index, chunks)

    def save(self, faiss_index_obj: FaissIndex, path: str | Path) -> None:
        """Persist the index and chunk metadata to disk.

        Serialises the FAISS index in memory and writes the bytes with Python's
        unicode-safe I/O, instead of handing a path to ``faiss.write_index``:
        FAISS's C++ narrow-char file API cannot open non-ASCII paths on Windows.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Path(str(path) + ".faiss").write_bytes(
            faiss.serialize_index(faiss_index_obj._index).tobytes()
        )
        with open(str(path) + ".meta", "wb") as f:
            pickle.dump(faiss_index_obj._chunks, f)

    def load(self, path: str | Path) -> FaissIndex:
        """Load a previously persisted index from disk (unicode-safe; see save)."""
        path = Path(path)
        data = Path(str(path) + ".faiss").read_bytes()
        index = faiss.deserialize_index(np.frombuffer(data, dtype="uint8"))
        with open(str(path) + ".meta", "rb") as f:
            chunks = pickle.load(f)
        return FaissIndex(index, chunks)