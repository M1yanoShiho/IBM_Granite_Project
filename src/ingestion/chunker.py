"""Chunking: split documents into retrievable passages.

Retrieval quality depends heavily on chunking. Chunks must be small enough to
be precise units of relevance, but large enough to stay self-contained. Default
of 512 tokens with 50-token overlap follows common RAG practice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    """A retrievable passage derived from a source document.

    Attributes
    ----------
    chunk_id:
        Unique identifier (e.g. ``"<doc_id>::<n>"``) — used for citations and
        for scoring against benchmark relevance judgments.
    doc_id:
        Identifier of the parent document.
    text:
        The chunk text.
    """

    chunk_id: str
    doc_id: str
    text: str


def chunk_document(
    doc_id: str,
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[Chunk]:
    """Split a document into overlapping :class:`Chunk` objects."""
    raise NotImplementedError("TODO: token-aware splitting with overlap.")
