"""DOCX/PPTX/HTML ingestion, and the reason it exists.

These formats carry document structure explicitly — headings, tables, slide titles — and
the corpus has had none of it: SciFact and 2Wiki are plain prose and the only PDF in the
tree is a synthetic smoke file. Without a structured corpus, ``SectionChunker`` cannot be
shown to help or hurt, so the last test here is the one that matters most: a Markdown
document from this loader must actually reach the section chunker and be cut on its
structure.

Docling is an optional dependency, so everything is driven through fakes, exactly as
``test_pdf_loader.py`` does.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from evidence_rag.contracts.models import Document
from evidence_rag.infrastructure.corpus import CorpusBuilder, build_chunker
from evidence_rag.loaders.office_loader import (
    OFFICE_EXTENSIONS,
    documents_from_office,
    load_office,
)

MARKDOWN = (
    "# Quarterly report\n\n"
    "Revenue grew across every region this quarter.\n\n"
    "## Revenue by quarter\n\n"
    "| quarter | revenue |\n| --- | --- |\n| Q1 | 100 |\n| Q2 | 200 |\n"
)


class FakeDoclingDocument:
    def __init__(self, markdown: str = MARKDOWN) -> None:
        self._markdown = markdown

    def export_to_markdown(self) -> str:
        return self._markdown


class FakeConverter:
    def __init__(self, document: FakeDoclingDocument) -> None:
        self.document = document
        self.converted: list[Path] = []

    def convert(self, path: Path) -> SimpleNamespace:
        self.converted.append(path)
        return SimpleNamespace(document=self.document)


class FakeChunk:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeHybridChunker:
    def __init__(self, chunks: list[FakeChunk]) -> None:
        self._chunks = chunks

    def chunk(self, docling_document: object) -> list[FakeChunk]:
        return self._chunks

    def contextualize(self, chunk: FakeChunk) -> str:
        return f"Section > {chunk.text}" if chunk.text else ""


@pytest.fixture
def docx(tmp_path: Path) -> Path:
    path = tmp_path / "report.docx"
    path.write_bytes(b"not really a docx; the converter is faked")
    return path


def test_markdown_mode_emits_one_document_carrying_the_structure(docx: Path) -> None:
    # The whole point of this mode: leave the splitting to the corpus chunker, and hand it
    # text that still has headings and tables in it.
    documents = load_office(docx, FakeConverter(FakeDoclingDocument()), mode="markdown")
    assert len(documents) == 1
    assert documents[0].document_id == "report.docx"
    assert "## Revenue by quarter" in documents[0].text
    assert "| Q1 | 100 |" in documents[0].text


def test_provenance_records_the_real_format_not_pdf(docx: Path, tmp_path: Path) -> None:
    documents = load_office(docx, FakeConverter(FakeDoclingDocument()), mode="markdown")
    assert documents[0].metadata is not None
    assert documents[0].metadata.source_type == "docx"
    assert documents[0].metadata.file_name == "report.docx"
    # No page number is invented: DOCX and HTML have no pages, and a PPTX slide is a
    # section rather than a page.
    assert documents[0].metadata.page_number is None

    html = tmp_path / "page.html"
    html.write_text("<h1>x</h1>", encoding="utf-8")
    from_html = load_office(html, FakeConverter(FakeDoclingDocument()), mode="markdown")
    assert from_html[0].metadata is not None
    assert from_html[0].metadata.source_type == "html"


def test_htm_and_html_collapse_to_the_same_source_type() -> None:
    # Provenance must not depend on which spelling a file happened to use.
    assert OFFICE_EXTENSIONS[".htm"] == OFFICE_EXTENSIONS[".html"] == "html"


def test_chunks_mode_matches_the_pdf_loader_contract(docx: Path) -> None:
    chunker = FakeHybridChunker([FakeChunk("first"), FakeChunk(""), FakeChunk("second")])
    documents = documents_from_office(
        FakeDoclingDocument(),
        docx,
        source_type="docx",
        mode="chunks",
        hybrid_chunker=chunker,
    )
    # The empty chunk is dropped and its number goes with it, leaving a gap. That is
    # deliberate and matches the PDF loader: the id names the chunk's position in the
    # source document, so renumbering to close the gap would make ::c2 point at something
    # that was never the second chunk.
    assert [document.document_id for document in documents] == [
        "report.docx::c1",
        "report.docx::c3",
    ]
    assert documents[0].text == "Section > first"


def test_an_empty_export_yields_no_documents_rather_than_an_invalid_one(docx: Path) -> None:
    # Document.text is NonEmpty, so emitting a blank one would fail validation deep in the
    # pipeline instead of here.
    assert load_office(docx, FakeConverter(FakeDoclingDocument("   \n  ")), mode="markdown") == []


def test_an_unsupported_extension_is_rejected_by_name(tmp_path: Path) -> None:
    path = tmp_path / "notes.rtf"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported office/web extension"):
        load_office(path, FakeConverter(FakeDoclingDocument()))


def test_an_unknown_mode_fails_loudly(docx: Path) -> None:
    with pytest.raises(ValueError, match="Unknown office mode"):
        documents_from_office(
            FakeDoclingDocument(), docx, source_type="docx", mode="sections"  # type: ignore[arg-type]
        )


def test_the_loaded_markdown_is_actually_chunkable_on_its_structure(docx: Path) -> None:
    """The reason this loader was written, end to end.

    A DOCX becomes Markdown, the section chunker cuts it on headings and keeps the table
    whole, and the word chunker — on the same document — does not. Without a loader like
    this there is no corpus in the tree on which that difference can be shown at all.
    """
    documents = load_office(docx, FakeConverter(FakeDoclingDocument()), mode="markdown")

    section = CorpusBuilder(build_chunker("section", chunk_size=20, overlap=4)).build(
        documents, "office-signature"
    )
    holding_table = [chunk for chunk in section.chunks if "| Q1 | 100 |" in chunk.text]
    assert len(holding_table) == 1
    assert "| Q2 | 200 |" in holding_table[0].text
    assert "## Revenue by quarter" in holding_table[0].text

    word = CorpusBuilder(build_chunker("word", chunk_size=20, overlap=4)).build(
        documents, "office-signature"
    )
    assert word.manifest.corpus_signature != section.manifest.corpus_signature


def test_documents_stay_within_the_public_contract(docx: Path) -> None:
    documents = load_office(docx, FakeConverter(FakeDoclingDocument()), mode="markdown")
    assert all(isinstance(document, Document) for document in documents)
