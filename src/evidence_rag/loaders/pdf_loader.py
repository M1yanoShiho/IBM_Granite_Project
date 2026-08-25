"""PDF loader: parse PDFs with Docling and emit plain-text ``Document`` records.

Docling runs layout analysis (and OCR where needed) locally. The default
``chunks`` mode feeds the whole parsed document through Docling's structure-aware
HybridChunker, so tables stay atomic and sections remain coherent across page
breaks; pair the resulting documents with ``PrechunkedChunker`` downstream so
they are never re-split. The ``pages`` mode keeps the older one-Markdown-record
per-page behaviour. Requires the optional 'ingestion' extra
(``pip install evidence-rag[ingestion]``).
"""

import importlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from evidence_rag.contracts.models import Document, SourceMetadata

logger = logging.getLogger("evidence_rag.loaders")

PDF_LOADER_VERSION = "pdf-loader-v2"

PdfMode = Literal["chunks", "pages"]


def build_converter(
    *,
    do_ocr: bool = True,
    table_mode: Literal["accurate", "fast"] = "accurate",
    artifacts_path: str | Path | None = None,
    generate_picture_images: bool = False,
) -> Any:
    """Build a Docling ``DocumentConverter`` with an explicit PDF pipeline.

    Constructing the converter loads Docling's layout models, so batch callers
    (directory ingestion) should build one and pass it to every ``load_pdf``.
    ``artifacts_path`` (or env ``DOCLING_ARTIFACTS_PATH``) points at pre-downloaded
    model artifacts for offline cluster nodes. Set ``generate_picture_images``
    when embedded pictures will be extracted for captioning.
    """
    try:
        document_converter = importlib.import_module("docling.document_converter")
        base_models = importlib.import_module("docling.datamodel.base_models")
        pipeline_options_module = importlib.import_module("docling.datamodel.pipeline_options")
    except ImportError as exc:  # pragma: no cover - depends on optional runtime deps
        raise RuntimeError(
            "PDF loading requires the optional 'docling' package "
            "(install the 'ingestion' extra)."
        ) from exc

    resolved_artifacts = artifacts_path or os.getenv("DOCLING_ARTIFACTS_PATH") or None
    pipeline_options = pipeline_options_module.PdfPipelineOptions(
        artifacts_path=resolved_artifacts,
        do_ocr=do_ocr,
        generate_picture_images=generate_picture_images,
    )
    if generate_picture_images:
        pipeline_options.images_scale = 2.0
    table_former_mode = pipeline_options_module.TableFormerMode
    pipeline_options.table_structure_options.mode = (
        table_former_mode.ACCURATE if table_mode == "accurate" else table_former_mode.FAST
    )
    return document_converter.DocumentConverter(
        format_options={
            base_models.InputFormat.PDF: document_converter.PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )


def convert_pdf(path: str | Path, converter: Any | None = None) -> Any:
    """Convert ``path`` once and return the parsed ``DoclingDocument``.

    Exposed separately so directory ingestion can reuse a single conversion for
    both text extraction and embedded-picture captioning.
    """
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"PDF not found: {resolved}")
    active_converter = converter if converter is not None else build_converter()
    return active_converter.convert(resolved).document


def load_pdf(
    path: str | Path,
    converter: Any | None = None,
    *,
    mode: PdfMode = "chunks",
    hybrid_chunker: Any | None = None,
) -> list[Document]:
    """Parse ``path`` with Docling and return plain-text ``Document`` records."""
    resolved = Path(path).resolve()
    docling_document = convert_pdf(resolved, converter)
    return documents_from_docling(
        docling_document, resolved, mode=mode, hybrid_chunker=hybrid_chunker
    )


def documents_from_docling(
    docling_document: Any,
    path: Path,
    *,
    mode: PdfMode = "chunks",
    hybrid_chunker: Any | None = None,
) -> list[Document]:
    """Turn an already-converted ``DoclingDocument`` into ``Document`` records."""
    if mode == "chunks":
        return _chunk_documents(docling_document, path, hybrid_chunker)
    if mode == "pages":
        return _page_documents(docling_document, path)
    raise ValueError(f"Unknown PDF mode: {mode!r}")


def _build_hybrid_chunker() -> Any:
    try:
        chunking = importlib.import_module("docling.chunking")
    except ImportError as exc:  # pragma: no cover - depends on optional runtime deps
        raise RuntimeError(
            "PDF chunk mode requires 'docling-core[chunking]' "
            "(install the 'ingestion' extra)."
        ) from exc
    return chunking.HybridChunker()


def _chunk_documents(docling_document: Any, path: Path, hybrid_chunker: Any | None) -> list[Document]:
    chunker = hybrid_chunker if hybrid_chunker is not None else _build_hybrid_chunker()
    documents: list[Document] = []
    for index, chunk in enumerate(chunker.chunk(docling_document), start=1):
        text = str(chunker.contextualize(chunk)).strip()
        if not text:
            logger.info("Skipping empty chunk %d of %s", index, path.name)
            continue
        page_number = _first_page_number(chunk)
        source_uri = f"{path}#page={page_number}" if page_number is not None else str(path)
        documents.append(
            Document(
                document_id=f"{path.name}::c{index}",
                text=text,
                source_uri=source_uri,
                metadata=SourceMetadata(
                    source_type="pdf",
                    file_name=path.name,
                    page_number=page_number,
                ),
            )
        )
    if not documents:
        logger.warning("PDF %s produced no non-empty chunks", path)
    return documents


def _page_documents(docling_document: Any, path: Path) -> list[Document]:
    documents: list[Document] = []
    for page_number in sorted(docling_document.pages):
        markdown = docling_document.export_to_markdown(page_no=page_number).strip()
        if not markdown:
            logger.info("Skipping blank page %d of %s", page_number, path.name)
            continue
        documents.append(
            Document(
                document_id=f"{path.name}::p{page_number}",
                text=markdown,
                source_uri=f"{path}#page={page_number}",
                metadata=SourceMetadata(
                    source_type="pdf",
                    file_name=path.name,
                    page_number=page_number,
                ),
            )
        )
    if not documents:
        logger.warning("PDF %s produced no non-empty pages", path)
    return documents


def _first_page_number(chunk: Any) -> int | None:
    meta = getattr(chunk, "meta", None)
    for item in getattr(meta, "doc_items", None) or ():
        for prov in getattr(item, "prov", None) or ():
            page_no = getattr(prov, "page_no", None)
            if isinstance(page_no, int) and page_no >= 1:
                return page_no
    return None


@dataclass(frozen=True)
class PdfPicture:
    """An embedded PDF picture extracted for captioning."""

    image: Any  # PIL.Image.Image
    page_number: int | None
    index: int  # 1-based position within the source PDF


def extract_pictures(docling_document: Any, path: Path) -> list[PdfPicture]:
    """Extract embedded pictures from a converted PDF for vision captioning.

    Returns an empty list (with a warning) when the converter was not built
    with ``generate_picture_images=True``, since picture data is then absent.
    """
    pictures: list[PdfPicture] = []
    skipped = 0
    for index, picture in enumerate(getattr(docling_document, "pictures", None) or (), start=1):
        image = picture.get_image(docling_document)
        if image is None:
            skipped += 1
            continue
        page_number = None
        for prov in getattr(picture, "prov", None) or ():
            page_no = getattr(prov, "page_no", None)
            if isinstance(page_no, int) and page_no >= 1:
                page_number = page_no
                break
        pictures.append(PdfPicture(image=image, page_number=page_number, index=index))
    if skipped:
        logger.warning(
            "%d picture(s) in %s had no image data; build the converter with "
            "generate_picture_images=True to caption them",
            skipped,
            path.name,
        )
    return pictures
