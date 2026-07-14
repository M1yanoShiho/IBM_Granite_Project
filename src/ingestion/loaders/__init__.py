"""Document loaders: every source format is normalised to plain text here.

This package is the system's only multimodal boundary. Each loader turns its
source into text + provenance metadata (:class:`LoadedDocument`); everything
downstream — chunker, embedder, BM25/dense retrieval, Granite instruct
generation — consumes that text unchanged, exactly as if it had been typed in.

- ``text_loader``  — ``.txt`` / ``.md`` passthrough.
- ``pdf_loader``   — Docling: PDF -> per-page Markdown.
- ``image_loader`` — Granite Vision: image -> caption (model loaded on demand,
  VRAM released immediately after).
- ``dispatch``     — ``load_directory``: one entry point for a mixed
  directory, routing each file by extension.

Heavy dependencies (docling, torch, transformers) are imported lazily inside
the loader functions, so importing this package is always cheap.
"""

from src.ingestion.loaders.base import LoadedDocument
from src.ingestion.loaders.dispatch import TEXT_SUFFIXES, load_directory
from src.ingestion.loaders.image_loader import (
    DEFAULT_CAPTION_PROMPT,
    DEFAULT_VISION_MODEL_ID,
    IMAGE_SUFFIXES,
    caption_images,
)
from src.ingestion.loaders.pdf_loader import build_converter, load_pdf
from src.ingestion.loaders.text_loader import load_documents, load_text_file

__all__ = [
    "LoadedDocument",
    "TEXT_SUFFIXES",
    "load_directory",
    "DEFAULT_CAPTION_PROMPT",
    "DEFAULT_VISION_MODEL_ID",
    "IMAGE_SUFFIXES",
    "caption_images",
    "build_converter",
    "load_pdf",
    "load_documents",
    "load_text_file",
]
