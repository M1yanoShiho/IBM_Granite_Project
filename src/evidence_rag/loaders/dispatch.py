"""Directory dispatcher: one entry point for mixed txt/md/PDF/image corpora.

Walks a directory, routes each file to its loader by extension, and returns a
flat list of ``Document`` records ready for any ``build_*`` pipeline factory
(pair PDF ``chunks`` mode with ``PrechunkedChunker`` downstream).

Two-phase orchestration keeps resources tight: phase 1 parses text and PDFs
(consulting the ingestion cache) and collects every image that needs a
caption — standalone files and, when ``caption_pdf_pictures`` is enabled,
pictures embedded in PDFs; phase 2 captions all of them inside a single
``VisionCaptioner`` batch, so the Granite Vision weights are loaded and
released exactly once per call.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evidence_rag.contracts.models import Document, SourceMetadata
from evidence_rag.loaders.cache import DocumentCache
from evidence_rag.loaders.image_loader import (
    DEFAULT_CAPTION_PROMPT,
    DEFAULT_VISION_MODEL_ID,
    IMAGE_LOADER_VERSION,
    OnError,
    VisionCaptioner,
    build_image_document,
    caption_image_paths,
)
from evidence_rag.loaders.pdf_loader import (
    PDF_LOADER_VERSION,
    PdfMode,
    PdfPicture,
    build_converter,
    convert_pdf,
    documents_from_docling,
    extract_pictures,
)
from evidence_rag.loaders.text_loader import TEXT_EXTENSIONS, load_text_file

logger = logging.getLogger("evidence_rag.loaders")

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"})


@dataclass
class _PdfJob:
    """A cache-miss PDF awaiting embedded-picture captions before finalising."""

    path: Path
    text_documents: list[Document]
    pictures: list[PdfPicture]
    caption_documents: list[Document] = field(default_factory=list)
    caption_failed: bool = False


def load_directory(
    directory: str | Path,
    *,
    converter: Any | None = None,
    caption_prompt: str = DEFAULT_CAPTION_PROMPT,
    vision_model_id: str | None = None,
    vision_device: str | None = None,
    pdf_mode: PdfMode = "chunks",
    caption_pdf_pictures: bool = False,
    image_ocr: bool = True,
    on_error: OnError = "skip",
    cache_dir: str | Path | None = None,
) -> list[Document]:
    """Load every supported file in ``directory`` into plain-text ``Document``s.

    ``cache_dir`` (or env ``INGESTION_CACHE_DIR``) enables the file-level parse
    cache; note that when passing a custom ``converter`` its pipeline options
    are not part of the cache fingerprint, so clear the cache after changing
    them. Unsupported extensions are skipped with a debug log.
    """
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    resolved_cache_dir = cache_dir or os.getenv("INGESTION_CACHE_DIR") or None
    cache = DocumentCache(resolved_cache_dir) if resolved_cache_dir else None
    resolved_model_id = (
        vision_model_id or os.getenv("GRANITE_VISION_MODEL_ID") or DEFAULT_VISION_MODEL_ID
    )
    pdf_fingerprint = "|".join(
        ["pdf", PDF_LOADER_VERSION, f"mode={pdf_mode}", f"pictures={caption_pdf_pictures}"]
        + (
            [f"prompt={caption_prompt}", f"vision={resolved_model_id}"]
            if caption_pdf_pictures
            else []
        )
    )
    image_fingerprint = "|".join(
        [
            "image",
            IMAGE_LOADER_VERSION,
            f"prompt={caption_prompt}",
            f"vision={resolved_model_id}",
            f"ocr={image_ocr}",
        ]
    )

    # Phase 1: parse text/PDFs, collect caption work.
    ordered_files: list[Path] = []
    results: dict[Path, list[Document]] = {}
    pending_images: list[Path] = []
    pdf_jobs: list[_PdfJob] = []

    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            ordered_files.append(path)
            results[path] = [load_text_file(path)]
        elif suffix == ".pdf":
            ordered_files.append(path)
            cached = cache.get(path, pdf_fingerprint) if cache else None
            if cached is not None:
                results[path] = cached
                continue
            if converter is None:
                converter = build_converter(generate_picture_images=caption_pdf_pictures)
            resolved_path = path.resolve()
            docling_document = convert_pdf(resolved_path, converter)
            job = _PdfJob(
                path=path,
                text_documents=documents_from_docling(
                    docling_document, resolved_path, mode=pdf_mode
                ),
                pictures=(
                    extract_pictures(docling_document, resolved_path)
                    if caption_pdf_pictures
                    else []
                ),
            )
            pdf_jobs.append(job)
        elif suffix in IMAGE_EXTENSIONS:
            ordered_files.append(path)
            cached = cache.get(path, image_fingerprint) if cache else None
            if cached is not None:
                results[path] = cached
            else:
                pending_images.append(path)
        else:
            logger.debug("Skipping unsupported file %s", path)

    # Phase 2: caption standalone images and PDF pictures in one model batch.
    image_captions: list[tuple[Path, str]] = []
    caption_errors: dict[Path, bool] = dict.fromkeys(pending_images, False)
    if pending_images or any(job.pictures for job in pdf_jobs):
        # Pass the resolved id so the model actually used is the one in the
        # cache fingerprints above, even if VisionCaptioner's own env fallback
        # ever changes.
        captioner = VisionCaptioner(
            prompt=caption_prompt, model_id=resolved_model_id, device=vision_device
        )
        with captioner:
            image_captions = caption_image_paths(captioner, pending_images, on_error=on_error)
            captioned = {path for path, _ in image_captions}
            for path in pending_images:
                caption_errors[path] = path not in captioned
            for job in pdf_jobs:
                _caption_pdf_pictures(job, captioner, on_error)

    # Assemble: OCR (vision model already released), documents, cache writes.
    ocr_converter = _resolve_ocr_converter(converter, enabled=image_ocr and bool(image_captions))
    for path, caption in image_captions:
        document = build_image_document(
            path.resolve(),
            caption,
            include_ocr=image_ocr and ocr_converter is not None,
            converter=ocr_converter,
        )
        results[path] = [document]
        if cache and not caption_errors[path]:
            cache.put(path, image_fingerprint, results[path])

    for job in pdf_jobs:
        results[job.path] = job.text_documents + job.caption_documents
        if cache and not job.caption_failed:
            cache.put(job.path, pdf_fingerprint, results[job.path])

    return [document for path in ordered_files for document in results.get(path, [])]


def _caption_pdf_pictures(job: _PdfJob, captioner: VisionCaptioner, on_error: OnError) -> None:
    name = job.path.name
    resolved = job.path.resolve()
    for picture in job.pictures:
        try:
            caption = captioner.caption(picture.image)
        except Exception:
            if on_error == "raise":
                raise
            job.caption_failed = True
            logger.warning(
                "Captioning failed for picture %d of %s; skipping", picture.index, name,
                exc_info=True,
            )
            continue
        if not caption:
            logger.warning("Empty caption for picture %d of %s; skipping", picture.index, name)
            continue
        page = picture.page_number
        document_id = (
            f"{name}::p{page}::img{picture.index}" if page is not None else f"{name}::img{picture.index}"
        )
        job.caption_documents.append(
            Document(
                document_id=document_id,
                text=caption,
                source_uri=f"{resolved}#page={page}" if page is not None else str(resolved),
                metadata=SourceMetadata(
                    source_type="image",
                    file_name=name,
                    page_number=page,
                ),
            )
        )


def _resolve_ocr_converter(converter: Any | None, *, enabled: bool) -> Any | None:
    """Reuse the shared Docling converter for image OCR, building one if needed."""
    if not enabled:
        return None
    if converter is not None:
        return converter
    try:
        return build_converter()
    except RuntimeError:
        logger.warning("docling unavailable; image OCR disabled (captions only)")
        return None
