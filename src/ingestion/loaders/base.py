"""Shared record type for all loaders.

Every loader — plain text, Docling PDF, Granite Vision captions — normalises
its source into :class:`LoadedDocument`: plain text plus provenance metadata.
Downstream (chunker, embedder, retrievers, generation) consumes only the text;
the metadata rides along on chunks so RAG answers can be traced back to the
exact page or image they came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class LoadedDocument:
    """A source document normalised to plain text + provenance metadata.

    Attributes
    ----------
    doc_id:
        Unique identifier for this record (e.g. ``"report::p3"`` for page 3 of
        ``report.pdf``). Chunk ids are derived from it (``"<doc_id>::<n>"``),
        so it must be unique per record, not just per source file.
    text:
        The plain text the rest of the pipeline sees (Markdown for PDFs, a
        caption for images, the raw content for text files).
    metadata:
        Provenance keys, by source:

        - all: ``source_type`` (``"pdf"`` | ``"image"`` | ``"txt"``), ``file_name``
        - pdf: ``page_number`` (1-based; ``None`` if per-page export unavailable)
        - image: ``image_path`` (absolute path, for click-through in the demo)
    """

    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
