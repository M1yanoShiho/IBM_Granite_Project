from pathlib import Path
from types import SimpleNamespace

import pytest

from evidence_rag.loaders.pdf_loader import extract_pictures, load_pdf


class FakeDoclingDocument:
    def __init__(
        self,
        pages: dict[int, str] | None = None,
        pictures: list["FakePicture"] | None = None,
    ) -> None:
        self.pages = pages or {}
        self.pictures = pictures or []

    def export_to_markdown(self, page_no: int) -> str:
        return self.pages[page_no]


class FakeConverter:
    def __init__(self, document: FakeDoclingDocument) -> None:
        self.document = document
        self.converted: list[Path] = []

    def convert(self, path: Path) -> SimpleNamespace:
        self.converted.append(path)
        return SimpleNamespace(document=self.document)


class FakeChunk:
    def __init__(self, text: str, page: int | None) -> None:
        self.text = text
        prov = [SimpleNamespace(page_no=page)] if page is not None else []
        self.meta = SimpleNamespace(doc_items=[SimpleNamespace(prov=prov)])


class FakeHybridChunker:
    def __init__(self, chunks: list[FakeChunk]) -> None:
        self.chunks_seen: list[object] = []
        self._chunks = chunks

    def chunk(self, docling_document: object) -> list[FakeChunk]:
        self.chunks_seen.append(docling_document)
        return self._chunks

    def contextualize(self, chunk: FakeChunk) -> str:
        return f"Section > {chunk.text}" if chunk.text else ""


class FakePicture:
    def __init__(self, image: object, page: int | None) -> None:
        self._image = image
        self.prov = [SimpleNamespace(page_no=page)] if page is not None else []

    def get_image(self, docling_document: object) -> object:
        return self._image


def _write_pdf(tmp_path: Path) -> Path:
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-fake")
    return pdf


def test_chunks_mode_emits_contextualized_cross_page_chunks(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path)
    converter = FakeConverter(FakeDoclingDocument())
    hybrid = FakeHybridChunker(
        [
            FakeChunk("Revenue table spanning pages.", page=1),
            FakeChunk("", page=2),  # empty chunk: skipped, numbering preserved
            FakeChunk("Outlook section.", page=3),
        ]
    )

    documents = load_pdf(pdf, converter=converter, hybrid_chunker=hybrid)

    assert [document.document_id for document in documents] == [
        "report.pdf::c1",
        "report.pdf::c3",
    ]
    assert documents[0].text == "Section > Revenue table spanning pages."
    assert documents[0].source_uri == f"{pdf.resolve()}#page=1"
    assert documents[0].metadata is not None
    assert documents[0].metadata.source_type == "pdf"
    assert documents[0].metadata.file_name == "report.pdf"
    assert documents[0].metadata.page_number == 1
    assert documents[1].metadata is not None
    assert documents[1].metadata.page_number == 3


def test_chunks_mode_without_page_provenance_falls_back_to_plain_uri(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path)
    hybrid = FakeHybridChunker([FakeChunk("No provenance chunk.", page=None)])

    documents = load_pdf(pdf, converter=FakeConverter(FakeDoclingDocument()), hybrid_chunker=hybrid)

    assert documents[0].source_uri == str(pdf.resolve())
    assert documents[0].metadata is not None
    assert documents[0].metadata.page_number is None


def test_pages_mode_emits_one_document_per_nonempty_page(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path)
    converter = FakeConverter(
        FakeDoclingDocument(
            pages={
                1: "# Executive Summary\n\nRevenue grew.",
                2: "   ",  # blank page: skipped
                3: "## Outlook\n\nProfit is stable.",
            }
        )
    )

    documents = load_pdf(pdf, converter=converter, mode="pages")

    assert converter.converted == [pdf.resolve()]
    assert [document.document_id for document in documents] == [
        "report.pdf::p1",
        "report.pdf::p3",
    ]
    assert documents[0].source_uri == f"{pdf.resolve()}#page=1"
    for document, page_number in zip(documents, (1, 3), strict=True):
        assert document.metadata is not None
        assert document.metadata.page_number == page_number


def test_load_pdf_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "missing.pdf", converter=FakeConverter(FakeDoclingDocument()))


def test_load_pdf_rejects_unknown_mode(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path)
    with pytest.raises(ValueError, match="Unknown PDF mode"):
        load_pdf(pdf, converter=FakeConverter(FakeDoclingDocument()), mode="bogus")  # type: ignore[arg-type]


def test_extract_pictures_returns_images_with_page_provenance(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pdf = _write_pdf(tmp_path)
    document = FakeDoclingDocument(
        pictures=[
            FakePicture("pil-image-1", page=2),
            FakePicture(None, page=3),  # no image data: skipped with warning
            FakePicture("pil-image-2", page=None),
        ]
    )

    with caplog.at_level("WARNING", logger="evidence_rag.loaders"):
        pictures = extract_pictures(document, pdf.resolve())

    assert [(picture.image, picture.page_number, picture.index) for picture in pictures] == [
        ("pil-image-1", 2, 1),
        ("pil-image-2", None, 3),
    ]
    assert "generate_picture_images=True" in caplog.text
