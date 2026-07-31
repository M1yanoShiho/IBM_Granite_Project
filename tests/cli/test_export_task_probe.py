import json
from pathlib import Path

import pytest

from evidence_rag.cli.export_task_probe import main
from evidence_rag.materializer.provenance import MutationRecord, write_provenance


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, query_ids: tuple[str, ...] = ("q1",)) -> tuple[Path, Path]:
    _write_jsonl(
        tmp_path / "documents.jsonl",
        [
            {"schema_version": "1.0", "document_id": "needle",
             "text": "JFK\n\nKennedy won", "source_uri": "s://n"},
            {"schema_version": "1.0", "document_id": "cf::needle",
             "text": "JFK\n\nNixon won", "source_uri": "s://c"},
        ],
    )
    _write_jsonl(
        tmp_path / "queries.jsonl",
        [{"schema_version": "1.0", "query_id": qid, "text": "who won?"} for qid in query_ids],
    )
    _write_jsonl(
        tmp_path / "gold_cases.jsonl",
        [
            {"query_id": qid, "relevant_document_ids": ["needle"],
             "reference_answers": ["Kennedy"]}
            for qid in query_ids
        ],
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0", "dataset_id": "niah", "dataset_version": "test+cf42",
                "split": "train", "documents_file": "documents.jsonl",
                "queries_file": "queries.jsonl", "gold_cases_file": "gold_cases.jsonl",
            }
        ),
        encoding="utf-8",
    )
    provenance = tmp_path / "provenance.jsonl"
    write_provenance(
        provenance,
        [
            MutationRecord(
                query_id=qid, needle_document_id="needle",
                counterfactual_document_id="cf::needle", gold_value="Kennedy",
                gold_alias_used="Kennedy", replacement_value="Nixon",
                string_class="proper_name_1", seed=42, char_span=(0, 7),
                text_hash_before="a" * 8, text_hash_after="b" * 8, answer_bank_hash="c" * 8,
            )
            for qid in query_ids
        ],
    )
    return tmp_path / "manifest.json", provenance


def test_exports_four_pairs_per_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest, provenance = _fixture(tmp_path)
    output = tmp_path / "task_pairs.jsonl"
    assert (
        main([
            "--manifest", str(manifest),
            "--provenance", str(provenance),
            "--output", str(output),
        ])
        == 0
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 4
    assert {row["kind"] for row in rows} == {
        "needle_gold", "cf_replacement", "cf_gold", "needle_replacement"
    }
    assert set(rows[0]) == {"premise", "hypothesis", "label", "group", "kind", "query_id"}
    report = json.loads(capsys.readouterr().out)
    assert report["n_records"] == 1
    assert report["n_pairs"] == 4
    assert report["n_skipped_records"] == 0


def test_premise_is_the_full_document_text_including_the_title_line(tmp_path: Path) -> None:
    """The probe scores the same text the extractor sees, so it must not silently strip the
    title paragraph that base_loader prepends."""
    manifest, provenance = _fixture(tmp_path)
    output = tmp_path / "task_pairs.jsonl"
    main([
        "--manifest", str(manifest),
        "--provenance", str(provenance),
        "--output", str(output),
    ])
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    assert all(row["premise"].startswith("JFK\n\n") for row in rows)


def test_reports_skipped_records(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest, provenance = _fixture(tmp_path, query_ids=("q1", "q2"))
    # q2 stays in the mutation log but is no longer in the manifest, so its record is unusable
    _write_jsonl(
        tmp_path / "queries.jsonl",
        [{"schema_version": "1.0", "query_id": "q1", "text": "who won?"}],
    )
    _write_jsonl(
        tmp_path / "gold_cases.jsonl",
        [{"query_id": "q1", "relevant_document_ids": ["needle"],
          "reference_answers": ["Kennedy"]}],
    )
    output = tmp_path / "task_pairs.jsonl"
    main([
        "--manifest", str(manifest),
        "--provenance", str(provenance),
        "--output", str(output),
    ])
    report = json.loads(capsys.readouterr().out)
    assert report["n_records"] == 2
    assert report["n_pairs"] == 4
    assert report["n_skipped_records"] == 1
