from pathlib import Path

import pytest

from evidence_rag.contracts.models import Document, SourceMetadata
from evidence_rag.loaders.cache import DocumentCache


def _documents() -> list[Document]:
    return [
        Document(
            document_id="report.pdf::c1",
            text="Revenue grew ten percent.",
            source_uri="/data/report.pdf#page=1",
            metadata=SourceMetadata(source_type="pdf", file_name="report.pdf", page_number=1),
        )
    ]


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-original")
    return path


def test_roundtrip_preserves_documents_and_metadata(tmp_path: Path, source: Path) -> None:
    cache = DocumentCache(tmp_path / "cache")
    documents = _documents()

    cache.put(source, "pdf|v1", documents)

    assert cache.get(source, "pdf|v1") == documents


def test_content_change_invalidates(tmp_path: Path, source: Path) -> None:
    cache = DocumentCache(tmp_path / "cache")
    cache.put(source, "pdf|v1", _documents())

    source.write_bytes(b"%PDF-modified")

    assert cache.get(source, "pdf|v1") is None


def test_fingerprint_change_invalidates(tmp_path: Path, source: Path) -> None:
    cache = DocumentCache(tmp_path / "cache")
    cache.put(source, "pdf|v1|prompt=a", _documents())

    assert cache.get(source, "pdf|v1|prompt=b") is None


def test_corrupt_entry_treated_as_miss(
    tmp_path: Path, source: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cache_dir = tmp_path / "cache"
    cache = DocumentCache(cache_dir)
    cache.put(source, "pdf|v1", _documents())
    (entry,) = cache_dir.iterdir()
    entry.write_text("not json", encoding="utf-8")

    with caplog.at_level("WARNING", logger="evidence_rag.loaders"):
        assert cache.get(source, "pdf|v1") is None
    assert "corrupt ingestion cache" in caplog.text


def test_empty_document_list_is_a_valid_entry(tmp_path: Path, source: Path) -> None:
    cache = DocumentCache(tmp_path / "cache")

    cache.put(source, "pdf|v1", [])

    assert cache.get(source, "pdf|v1") == []
