import json
from pathlib import Path

import pytest

from evidence_rag.cli.build_source_parent import main


def _documents(tmp_path: Path) -> Path:
    documents = tmp_path / "documents.jsonl"
    documents.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"schema_version": "1.0", "document_id": "d1",
                 "text": "John F. Kennedy\n\npart one", "source_uri": "s://1"},
                {"schema_version": "1.0", "document_id": "d2",
                 "text": "John F. Kennedy\n\npart two", "source_uri": "s://2"},
                {"schema_version": "1.0", "document_id": "d3",
                 "text": "Richard Nixon\n\nonly part", "source_uri": "s://3"},
                {"schema_version": "1.0", "document_id": "d4",
                 "text": "no title paragraph here", "source_uri": "s://4"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return documents


def test_builds_index_and_reports_counts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "source_parent.jsonl"
    assert main(["--documents", str(_documents(tmp_path)), "--output", str(output)]) == 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    assert {row["document_id"]: row["source_parent_id"] for row in rows} == {
        "d1": "john f. kennedy",
        "d2": "john f. kennedy",
        "d3": "richard nixon",
    }
    report = json.loads(capsys.readouterr().out)
    assert report["n_documents"] == 4
    assert report["n_resolved"] == 3
    assert report["n_unresolved"] == 1
    assert report["n_parents"] == 2


def test_unresolved_documents_are_omitted_not_self_mapped(tmp_path: Path) -> None:
    """The sidecar records only what the rule could determine; ParentIndex supplies the
    self-parent fallback at read time, so the artifact stays an honest record of the rule."""
    output = tmp_path / "source_parent.jsonl"
    main(["--documents", str(_documents(tmp_path)), "--output", str(output)])
    ids = {
        json.loads(line)["document_id"]
        for line in output.read_text(encoding="utf-8").splitlines()
        if line
    }
    assert "d4" not in ids


def test_output_is_round_trippable(tmp_path: Path) -> None:
    from evidence_rag.materializer.source_parent import read_parent_index

    output = tmp_path / "nested" / "source_parent.jsonl"
    main(["--documents", str(_documents(tmp_path)), "--output", str(output)])
    index = read_parent_index(output)
    assert index.parent_of("d1") == index.parent_of("d2")
    assert index.parent_of("d4") == "d4"
