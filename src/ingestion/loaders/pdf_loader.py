"""PDF loader: parse PDFs with Docling and emit Markdown text records.

Docling runs layout analysis (and OCR where needed) locally and exports clean
Markdown. From there the text flows through the *unchanged* pipeline —
chunker, embedder, retrievers, Granite instruct — exactly like any pasted
document: the multimodal boundary lives entirely inside ingestion.

Records are emitted **per page** so each chunk's ``page_number`` metadata is
exact for citation back-links; a whole-document record is the fallback when
per-page export is unavailable.

Docling (which drags in torch) is imported lazily inside the functions, so
importing this module stays cheap on machines without the ML stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from src.ingestion.loaders.base import LoadedDocument


def build_converter():
    """Build a Docling ``DocumentConverter``.

    Constructing the converter loads Docling's layout models, so batch callers
    (the smoke script, directory ingestion) should build one and pass it to
    every :func:`load_pdf` call instead of paying that cost per file.
    """
    from docling.document_converter import DocumentConverter

    return DocumentConverter()


def load_pdf(path: str | Path, *, converter=None) -> List[LoadedDocument]:
    """Parse ``path`` with Docling and return one Markdown record per page.

    Each record carries ``source_type="pdf"``, ``file_name`` and a 1-based
    ``page_number``. Pages that export to empty Markdown (blank or
    figure-only) are skipped. If the installed Docling cannot export per page,
    the whole document becomes a single record with ``page_number=None``.

    Parameters
    ----------
    converter:
        Optional pre-built Docling converter (see :func:`build_converter`).
        Defaults to building a fresh one per call.
    """
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if converter is None:
        converter = build_converter()
    doc = converter.convert(str(pdf_path)).document

    stem, name = pdf_path.stem, pdf_path.name
    records: List[LoadedDocument] = []
    for page_no in sorted(getattr(doc, "pages", None) or {}):
        try:
            markdown = doc.export_to_markdown(page_no=page_no)
        except TypeError:  # older docling-core without per-page export
            records = []
            break
        markdown = markdown.strip()
        if not markdown:
            continue
        records.append(
            LoadedDocument(
                doc_id=f"{stem}::p{page_no}",
                text=markdown,
                metadata={
                    "source_type": "pdf",
                    "file_name": name,
                    "page_number": page_no,
                },
            )
        )
    if records:
        return records

    markdown = doc.export_to_markdown().strip()
    if not markdown:
        return []
    return [
        LoadedDocument(
            doc_id=stem,
            text=markdown,
            metadata={"source_type": "pdf", "file_name": name, "page_number": None},
        )
    ]
