"""Office/web loader: parse DOCX, PPTX and HTML with Docling into ``Document`` records.

These formats are where document *structure* actually lives — headings, tables, slide
titles — and the corpus so far has had none of it: SciFact and 2Wiki are plain prose, and
the only PDF in the tree is a synthetic OCR smoke file. That matters beyond format coverage,
because ``SectionChunker`` (``[chunker] name = "section"``) cannot be shown to help or hurt
on a corpus with no sections to cut on. This loader is what makes that experiment possible.

Two modes, and the choice decides *who* does the chunking:

- ``markdown`` (default) exports the whole file as one Markdown ``Document`` and leaves the
  splitting to the corpus builder's chunker. Pair with ``name = "section"`` to cut on the
  headings and tables Docling recovered — and with ``name = "word"`` for the comparison arm,
  which is the point of having both.
- ``chunks`` uses Docling's own structure-aware HybridChunker, one ``Document`` per unit.
  Pair with ``PrechunkedChunker`` downstream so the units are never re-split. Same contract
  as the PDF loader's ``chunks`` mode.

Requires the optional 'ingestion' extra (``pip install evidence-rag[ingestion]``).
"""

import importlib
import logging
from pathlib import Path
from typing import Any, Literal

from evidence_rag.contracts.models import Document, SourceMetadata

logger = logging.getLogger("evidence_rag.loaders")

OFFICE_LOADER_VERSION = "office-loader-v1"

OfficeMode = Literal["markdown", "chunks"]

# Extension -> the SourceMetadata.source_type it records. ``.htm`` and ``.html`` are the
# same format and deliberately collapse to one type, so provenance does not depend on
# which spelling a file happened to use.
OFFICE_EXTENSIONS: dict[str, str] = {
    ".docx": "docx",
    ".pptx": "pptx",
    ".html": "html",
    ".htm": "html",
}


def build_office_converter() -> Any:
    """Build a Docling ``DocumentConverter`` for the office/web formats.

    Separate from ``pdf_loader.build_converter`` because that one configures a PDF
    pipeline (OCR, table-former mode, picture images) that means nothing here — DOCX and
    HTML carry their structure explicitly and need no layout analysis.
    """
    try:
        document_converter = importlib.import_module("docling.document_converter")
        base_models = importlib.import_module("docling.datamodel.base_models")
    except ImportError as exc:  # pragma: no cover - depends on optional runtime deps
        raise RuntimeError(
            "Office/HTML loading requires the optional 'docling' package "
            "(install the 'ingestion' extra)."
        ) from exc

    input_format = base_models.InputFormat
    return document_converter.DocumentConverter(
        allowed_formats=[input_format.DOCX, input_format.PPTX, input_format.HTML]
    )


def load_office(
    path: str | Path,
    converter: Any | None = None,
    *,
    mode: OfficeMode = "markdown",
    hybrid_chunker: Any | None = None,
) -> list[Document]:
    """Parse ``path`` with Docling and return ``Document`` records."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Document not found: {resolved}")
    source_type = OFFICE_EXTENSIONS.get(resolved.suffix.lower())
    if source_type is None:
        raise ValueError(f"Unsupported office/web extension: {resolved.suffix!r}")

    active = converter if converter is not None else build_office_converter()
    docling_document = active.convert(resolved).document
    return documents_from_office(
        docling_document,
        resolved,
        source_type=source_type,
        mode=mode,
        hybrid_chunker=hybrid_chunker,
    )


def documents_from_office(
    docling_document: Any,
    path: Path,
    *,
    source_type: str,
    mode: OfficeMode = "markdown",
    hybrid_chunker: Any | None = None,
) -> list[Document]:
    """Turn an already-converted Docling document into ``Document`` records."""
    if mode == "markdown":
        return _markdown_document(docling_document, path, source_type)
    if mode == "chunks":
        return _chunk_documents(docling_document, path, source_type, hybrid_chunker)
    raise ValueError(f"Unknown office mode: {mode!r}")


def _metadata(path: Path, source_type: str) -> SourceMetadata:
    # No page_number: DOCX and HTML have no pages, and a PPTX slide is a section rather
    # than a page. Inventing one would put a number in provenance that means nothing.
    return SourceMetadata(source_type=source_type, file_name=path.name)  # type: ignore[arg-type]


def _markdown_document(docling_document: Any, path: Path, source_type: str) -> list[Document]:
    markdown = docling_document.export_to_markdown().strip()
    if not markdown:
        logger.warning("%s produced no text", path)
        return []
    return [
        Document(
            document_id=path.name,
            text=markdown,
            source_uri=str(path),
            metadata=_metadata(path, source_type),
        )
    ]


def _build_hybrid_chunker() -> Any:
    try:
        chunking = importlib.import_module("docling.chunking")
    except ImportError as exc:  # pragma: no cover - depends on optional runtime deps
        raise RuntimeError(
            "Office chunk mode requires 'docling-core[chunking]' "
            "(install the 'ingestion' extra)."
        ) from exc
    return chunking.HybridChunker()


def _chunk_documents(
    docling_document: Any,
    path: Path,
    source_type: str,
    hybrid_chunker: Any | None,
) -> list[Document]:
    chunker = hybrid_chunker if hybrid_chunker is not None else _build_hybrid_chunker()
    documents: list[Document] = []
    for index, chunk in enumerate(chunker.chunk(docling_document), start=1):
        text = str(chunker.contextualize(chunk)).strip()
        if not text:
            logger.info("Skipping empty chunk %d of %s", index, path.name)
            continue
        documents.append(
            Document(
                document_id=f"{path.name}::c{index}",
                text=text,
                source_uri=str(path),
                metadata=_metadata(path, source_type),
            )
        )
    if not documents:
        logger.warning("%s produced no non-empty chunks", path)
    return documents
