"""Vector index construction and persistence.

Embeds chunks and builds a **persistent** FAISS index that the retriever loads
at query time. Persistence (save/load) is what distinguishes a real system from
the throwaway, rebuilt-every-run index used in early experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.ingestion.chunker import Chunk
from src.retrieval.embedder import Embedder


class VectorIndexer:
    """Builds and persists a FAISS vector index over chunks.

    Parameters
    ----------
    embedder:
        The :class:`~src.retrieval.embedder.Embedder` used to vectorise chunks.
    """

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def build(self, chunks: Sequence[Chunk]):
        """Embed ``chunks`` and build an in-memory FAISS index."""
        raise NotImplementedError("TODO: embed chunks, add to a FAISS index.")

    def save(self, index, path: str | Path) -> None:
        """Persist the index (and chunk metadata) to disk."""
        raise NotImplementedError("TODO: write FAISS index + metadata to disk.")

    def load(self, path: str | Path):
        """Load a previously persisted index from disk."""
        raise NotImplementedError("TODO: read FAISS index + metadata from disk.")
