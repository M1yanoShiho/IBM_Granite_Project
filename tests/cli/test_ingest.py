import json
from pathlib import Path

import pytest

from evidence_rag.cli.ingest import main
from evidence_rag.contracts.models import Document


def test_ingest_writes_documents_jsonl_and_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "notes.txt").write_text("plain notes", encoding="utf-8")
    output = tmp_path / "out"

    exit_code = main(["--input", str(source), "--output", str(output)])

    assert exit_code == 0
    documents_path = output / "documents.jsonl"
    lines = documents_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    document = Document.model_validate_json(lines[0])
    assert document.text == "plain notes"

    summary = json.loads(capsys.readouterr().out)
    assert summary["document_count"] == 1
    assert summary["by_source_type"] == {"txt": 1}
    assert summary["caption_pdf_pictures"] is False
    assert summary["documents"] == str(documents_path)


def test_ingest_reads_switches_from_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "notes.txt").write_text("plain notes", encoding="utf-8")
    config_path = tmp_path / "ingestion.toml"
    config_path.write_text(
        "[ingestion]\ncaption_pdf_pictures = true\npdf_mode = \"pages\"\n", encoding="utf-8"
    )
    output = tmp_path / "out"

    exit_code = main(
        ["--input", str(source), "--output", str(output), "--config", str(config_path)]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["caption_pdf_pictures"] is True
    assert summary["pdf_mode"] == "pages"
