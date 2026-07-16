from pathlib import Path

import pytest

from evidence_rag.loaders.text_loader import load_text_file


def test_load_text_file_builds_document_with_txt_metadata(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("Granite embeddings are compact.", encoding="utf-8")

    document = load_text_file(source)

    assert document.document_id == "notes.txt"
    assert document.text == "Granite embeddings are compact."
    assert document.source_uri == str(source.resolve())
    assert document.metadata is not None
    assert document.metadata.source_type == "txt"
    assert document.metadata.file_name == "notes.txt"
    assert document.metadata.page_number is None
    assert document.metadata.image_path is None


def test_load_text_file_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_text_file(tmp_path / "missing.txt")
