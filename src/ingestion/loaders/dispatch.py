"""Directory dispatcher: one entry point for mixed txt/md/PDF/image corpora.

Walks a directory, routes each file to its loader by extension, and returns a
flat list of :class:`LoadedDocument` records — plain text + provenance
metadata — ready for ``chunk_document(rec.doc_id, rec.text, metadata=rec.metadata)``.

The legacy ``load_documents`` (``(doc_id, text)`` pairs, txt/md only) is kept
untouched in ``text_loader`` for existing callers; new multimodal ingestion
should use :func:`load_directory`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from src.ingestion.loaders.base import LoadedDocument
from src.ingestion.loaders.image_loader import (
    DEFAULT_CAPTION_PROMPT,
    IMAGE_SUFFIXES,
    caption_images,
)
from src.ingestion.loaders.pdf_loader import build_converter, load_pdf

TEXT_SUFFIXES = {".txt", ".md"}


def load_directory(
    directory: str | Path,
    *,
    converter=None,
    caption_prompt: str = DEFAULT_CAPTION_PROMPT,
    vision_model_id: Optional[str] = None,
    vision_device: Optional[str] = None,
) -> List[LoadedDocument]:
    """Load every supported file in ``directory`` into text records.

    Routing (unsupported extensions are silently skipped, matching the legacy
    loader's behaviour):

    - ``.txt`` / ``.md``  -> read as-is, ``source_type="txt"``
    - ``.pdf``            -> Docling, one Markdown record per page
    - image suffixes      -> Granite Vision captions

    Heavy resources are paid for at most once per call: the Docling converter
    is built only if a PDF is actually present (or pass a pre-built one via
    ``converter``), and all images are captioned in a **single**
    ``caption_images`` batch — one vision-model load + VRAM release for the
    whole directory. Records therefore come back text/PDF first (in sorted
    file order), then images (in sorted file order).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    records: List[LoadedDocument] = []
    image_paths: List[Path] = []
    for path in sorted(directory.iterdir()):
        suffix = path.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            records.append(
                LoadedDocument(
                    doc_id=path.stem,
                    text=path.read_text(encoding="utf-8"),
                    metadata={"source_type": "txt", "file_name": path.name},
                )
            )
        elif suffix == ".pdf":
            if converter is None:
                converter = build_converter()
            records.extend(load_pdf(path, converter=converter))
        elif suffix in IMAGE_SUFFIXES:
            image_paths.append(path)

    if image_paths:
        records.extend(
            caption_images(
                image_paths,
                prompt=caption_prompt,
                model_id=vision_model_id,
                device=vision_device,
            )
        )
    return records
