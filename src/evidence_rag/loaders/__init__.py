"""Ingestion loaders: normalise txt/md, PDF, and image sources into ``Document``s.

Everything downstream (chunker, retrievers, selector, generator) consumes the
same plain-text ``Document`` contract regardless of the original modality.
"""

from evidence_rag.loaders.cache import DocumentCache
from evidence_rag.loaders.dispatch import (
    IMAGE_EXTENSIONS,
    load_directory,
    load_directory_from_config,
)
from evidence_rag.loaders.image_loader import (
    DEFAULT_CAPTION_PROMPT,
    DEFAULT_VISION_MODEL_ID,
    VisionCaptioner,
    caption_images,
)
from evidence_rag.loaders.pdf_loader import (
    PdfPicture,
    build_converter,
    convert_pdf,
    extract_pictures,
    load_pdf,
)
from evidence_rag.loaders.text_loader import TEXT_EXTENSIONS, load_text_file

__all__ = [
    "DEFAULT_CAPTION_PROMPT",
    "DEFAULT_VISION_MODEL_ID",
    "IMAGE_EXTENSIONS",
    "TEXT_EXTENSIONS",
    "DocumentCache",
    "PdfPicture",
    "VisionCaptioner",
    "build_converter",
    "caption_images",
    "convert_pdf",
    "extract_pictures",
    "load_directory",
    "load_directory_from_config",
    "load_pdf",
    "load_text_file",
]
