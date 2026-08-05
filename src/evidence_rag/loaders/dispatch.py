"""Directory dispatcher: one entry point for mixed txt/md/PDF/image corpora.

Walks a directory, routes each file to its loader by extension, and returns a
flat list of ``Document`` records ready for any ``build_*`` pipeline factory
(pair PDF ``chunks`` mode with ``PrechunkedChunker`` downstream).

Three-phase orchestration keeps resources tight: phase 1 parses text and PDFs
(consulting the ingestion cache) and collects every image that needs a
caption — standalone files and, when ``caption_pdf_pictures`` is enabled,
pictures embedded in PDFs; phase 2 captions all of them inside a single
``VisionCaptioner`` batch, so the Granite Vision weights are loaded and
released exactly once per call; phase 3 runs Docling OCR (when ``image_ocr``
is set) over standalone images and embedded PDF pictures alike, appending any
recognised text so exact figures survive the lossy caption. OCR runs only
after the vision weights are freed, so at most one model is resident at a time.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evidence_rag.contracts.models import Document, SourceMetadata
from evidence_rag.infrastructure.config import IngestionConfig
from evidence_rag.loaders.cache import DocumentCache
from evidence_rag.loaders.image_loader import (
    DEFAULT_CAPTION_PROMPT,
    DEFAULT_VISION_MODEL_ID,
    IMAGE_LOADER_VERSION,
    OnError,
    VisionCaptioner,
    build_image_document,
    caption_image_paths,
    extract_ocr_text_from_image,
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
class _CaptionedPicture:
    """A PDF picture that has been captioned and still awaits OCR of its text."""

    caption: str
    image: Any  # PIL.Image.Image
    page_number: int | None
    index: int  # 1-based position within the source PDF


@dataclass
class _PdfJob:
    """A cache-miss PDF awaiting embedded-picture captions before finalising."""

    path: Path
    text_documents: list[Document]
    pictures: list[PdfPicture]
    captioned: list[_CaptionedPicture] = field(default_factory=list)
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
    recursive: bool = True,
) -> list[Document]:
    """Load every supported file under ``directory`` into plain-text ``Document``s.

    ``cache_dir`` (or env ``INGESTION_CACHE_DIR``) enables the file-level parse
    cache; note that when passing a custom ``converter`` its pipeline options
    are not part of the cache fingerprint, so clear the cache after changing
    them. Unsupported extensions are skipped with a debug log.

    ``recursive`` (default) walks sub-directories. A flat scan drops nested files
    with no signal whatsoever — the corpus just silently comes out smaller than the
    directory — so recursion is the default and ``recursive=False`` is the opt-out.
    Because two sub-directories may hold the same file name, ``document_id`` is the
    path *relative to* ``directory`` (``reports/q1.pdf``) rather than the bare name;
    for a flat directory the two are identical, so ids are unchanged for corpora
    that predate this.

    ``on_error`` now governs parsing as well as captioning: with the default
    ``"skip"`` a file that fails to parse is logged at warning level and left out,
    instead of aborting the whole ingest partway through; ``"raise"`` restores the
    fail-fast behaviour. Every skip and failure is counted in a summary log line —
    silence is what made a short corpus indistinguishable from a complete one.
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
            [f"prompt={caption_prompt}", f"vision={resolved_model_id}", f"ocr={image_ocr}"]
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

    unsupported: list[Path] = []
    parse_failures: list[Path] = []

    for path in _scan(root, recursive=recursive):
        suffix = path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            ordered_files.append(path)
            try:
                results[path] = [load_text_file(path)]
            except Exception:
                if on_error == "raise":
                    raise
                logger.warning("Failed to read %s; skipping", path, exc_info=True)
                parse_failures.append(path)
        elif suffix == ".pdf":
            ordered_files.append(path)
            cached = cache.get(path, pdf_fingerprint) if cache else None
            if cached is not None:
                results[path] = cached
                continue
            if converter is None:
                converter = build_converter(generate_picture_images=caption_pdf_pictures)
            resolved_path = path.resolve()
            try:
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
            except Exception:
                if on_error == "raise":
                    raise
                logger.warning("Failed to parse %s; skipping", path, exc_info=True)
                parse_failures.append(path)
                continue
            pdf_jobs.append(job)
        elif suffix in IMAGE_EXTENSIONS:
            ordered_files.append(path)
            cached = cache.get(path, image_fingerprint) if cache else None
            if cached is not None:
                results[path] = cached
            else:
                pending_images.append(path)
        else:
            unsupported.append(path)
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
    needs_ocr = bool(image_captions) or any(job.captioned for job in pdf_jobs)
    ocr_converter = _resolve_ocr_converter(converter, enabled=image_ocr and needs_ocr)
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
        _finalize_pdf_pictures(
            job, include_ocr=image_ocr and ocr_converter is not None, converter=ocr_converter
        )
        results[job.path] = job.text_documents + job.caption_documents
        if cache and not job.caption_failed:
            cache.put(job.path, pdf_fingerprint, results[job.path])

    caption_failures = sorted(
        [path for path, failed in caption_errors.items() if failed]
        + [job.path for job in pdf_jobs if job.caption_failed]
    )
    documents = [
        _restamped(document, name=path.name, document_id=_relative_id(path, root))
        for path in ordered_files
        for document in results.get(path, [])
    ]
    _log_summary(
        root,
        documents=len(documents),
        ingested=sum(1 for path in ordered_files if results.get(path)),
        unsupported=unsupported,
        parse_failures=parse_failures,
        caption_failures=caption_failures,
    )
    return documents


def _scan(root: Path, *, recursive: bool) -> list[Path]:
    """Files under ``root``, ordered by their path relative to it.

    Sorting on the relative path (not the absolute one) keeps ingestion order
    independent of where the corpus happens to live on disk.
    """
    walker = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        (path for path in walker if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _relative_id(path: Path, root: Path) -> str:
    """Collision-free document id: ``path`` relative to the scan root, posix-style.

    A bare file name is not unique once sub-directories are walked — two folders
    can each hold ``report.pdf`` — and duplicate ``document_id``s break the corpus
    contract downstream. For a flat directory this returns exactly the file name,
    so ids are byte-identical to what a pre-recursion scan produced.
    """
    return path.relative_to(root).as_posix()


def _restamped(document: Document, *, name: str, document_id: str) -> Document:
    """Re-key ``document`` onto its relative id, preserving any id suffix.

    PDF chunk documents carry ids like ``report.pdf::c1``; only the file-name part
    is replaced, so the suffix that distinguishes chunks survives. A no-op when the
    relative id already equals the file name (the flat-directory case).
    """
    if document_id == name or not document.document_id.startswith(name):
        return document
    return document.model_copy(
        update={"document_id": document_id + document.document_id[len(name) :]}
    )


def _log_summary(
    root: Path,
    *,
    documents: int,
    ingested: int,
    unsupported: list[Path],
    parse_failures: list[Path],
    caption_failures: list[Path],
) -> None:
    """Report what the scan actually consumed, loudly enough to be noticed.

    A short corpus and a complete one look identical unless the skips are stated,
    so anything dropped is logged at warning level with the paths named.
    """
    logger.info(
        "Ingested %d file(s) from %s into %d document(s)", ingested, root, documents
    )
    for label, paths in (
        ("failed to parse", parse_failures),
        ("failed to caption", caption_failures),
    ):
        if paths:
            logger.warning(
                "%d file(s) %s and were skipped: %s",
                len(paths),
                label,
                ", ".join(str(path) for path in paths[:10])
                + (" ..." if len(paths) > 10 else ""),
            )
    if unsupported:
        logger.info(
            "%d file(s) had unsupported extensions and were skipped: %s",
            len(unsupported),
            ", ".join(sorted({path.suffix or "<none>" for path in unsupported})),
        )


def load_directory_from_config(
    directory: str | Path,
    config: IngestionConfig,
    *,
    converter: Any | None = None,
) -> list[Document]:
    """Ingest ``directory`` driven by a config-file :class:`IngestionConfig`.

    Maps the config's switches onto :func:`load_directory`, passing the optional
    string fields only when set so ``load_directory``'s own defaults / env-var
    fallbacks still apply.
    """
    kwargs: dict[str, Any] = {
        "pdf_mode": config.pdf_mode,
        "caption_pdf_pictures": config.caption_pdf_pictures,
        "image_ocr": config.image_ocr,
        "on_error": config.on_error,
        "recursive": config.recursive,
    }
    if config.caption_prompt is not None:
        kwargs["caption_prompt"] = config.caption_prompt
    if config.vision_model_id is not None:
        kwargs["vision_model_id"] = config.vision_model_id
    if config.vision_device is not None:
        kwargs["vision_device"] = config.vision_device
    if config.cache_dir is not None:
        kwargs["cache_dir"] = config.cache_dir
    return load_directory(directory, converter=converter, **kwargs)


def _caption_pdf_pictures(job: _PdfJob, captioner: VisionCaptioner, on_error: OnError) -> None:
    """Caption each embedded picture (phase 2); OCR is deferred to phase 3.

    Only captioning happens here, inside the vision-model context. The PIL image
    is retained on the job so OCR can run in :func:`_finalize_pdf_pictures` once
    the vision weights are released, preserving the one-model-resident invariant.
    """
    name = job.path.name
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
        job.captioned.append(
            _CaptionedPicture(
                caption=caption,
                image=picture.image,
                page_number=picture.page_number,
                index=picture.index,
            )
        )


def _finalize_pdf_pictures(
    job: _PdfJob, *, include_ocr: bool, converter: Any | None
) -> None:
    """Build the picture documents (phase 3), appending OCR text when enabled.

    Mirrors :func:`build_image_document` for standalone images: OCR runs after
    the vision model is released, and any recognised text is appended so exact
    figures in charts survive the lossy caption.
    """
    name = job.path.name
    resolved = job.path.resolve()
    for item in job.captioned:
        text = item.caption
        if include_ocr:
            ocr_text = extract_ocr_text_from_image(item.image, converter=converter)
            if ocr_text:
                text = f"{item.caption}\n\nText in image:\n{ocr_text}"
        page = item.page_number
        document_id = (
            f"{name}::p{page}::img{item.index}" if page is not None else f"{name}::img{item.index}"
        )
        job.caption_documents.append(
            Document(
                document_id=document_id,
                text=text,
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
