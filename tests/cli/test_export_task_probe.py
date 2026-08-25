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
                counterfactual_document_id="cf::needle", gold_value="kennedy",
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


def _run(tmp_path: Path, output: Path, *extra: str) -> int:
    manifest, provenance = _fixture(tmp_path)
    return main([
        "--manifest", str(manifest),
        "--provenance", str(provenance),
        "--output", str(output),
        *extra,
    ])


def _write_qa2d_cache(path: Path) -> Path:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in (
                # keyed on the canonical answer, which is what the default arm asks for
                {"question": "who won?", "answer": "kennedy",
                 "declarative": "Kennedy won."},
                {"question": "who won?", "answer": "Nixon", "declarative": "Nixon won."},
            )
        ),
        encoding="utf-8",
    )
    return path


def test_default_output_is_byte_identical_to_the_frozen_template_arm(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Adding the ablation switch must not move the pre-registered main arm by one byte."""
    default_output = tmp_path / "default.jsonl"
    explicit_output = tmp_path / "explicit.jsonl"
    _run(tmp_path, default_output)
    _run(tmp_path, explicit_output, "--hypothesis-form", "template")
    capsys.readouterr()
    assert default_output.read_bytes() == explicit_output.read_bytes()
    assert b'The answer to the question \\"who won?\\" is kennedy.' in default_output.read_bytes()


def test_report_records_which_form_produced_the_pairs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The pairs file carries no marker of its own arm, so the protocol record depends on the
    report line saying which one ran."""
    _run(tmp_path, tmp_path / "pairs.jsonl")
    assert json.loads(capsys.readouterr().out)["hypothesis_form"] == "template"
    _run(tmp_path, tmp_path / "qa.jsonl", "--hypothesis-form", "question_answer")
    assert json.loads(capsys.readouterr().out)["hypothesis_form"] == "question_answer"


def test_question_answer_form_changes_the_hypothesis_surface(tmp_path: Path) -> None:
    output = tmp_path / "pairs.jsonl"
    _run(tmp_path, output, "--hypothesis-form", "question_answer")
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    assert {row["hypothesis"] for row in rows} == {"who won? kennedy.", "who won? Nixon."}


def test_qa2d_form_reads_the_pre_generated_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = _write_qa2d_cache(tmp_path / "qa2d.jsonl")
    output = tmp_path / "pairs.jsonl"
    assert _run(tmp_path, output, "--hypothesis-form", "qa2d", "--qa2d-cache", str(cache)) == 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    assert {row["hypothesis"] for row in rows} == {"Kennedy won.", "Nixon won."}
    assert json.loads(capsys.readouterr().out)["hypothesis_form"] == "qa2d"


def test_qa2d_form_without_a_cache_exits_instead_of_using_the_template(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Falling through to the template here would produce a file labelled qa2d that is actually
    the main arm — the one failure mode that survives every downstream check."""
    output = tmp_path / "pairs.jsonl"
    with pytest.raises(SystemExit) as error:
        _run(tmp_path, output, "--hypothesis-form", "qa2d")
    assert error.value.code == 2
    assert "--qa2d-cache" in capsys.readouterr().err
    assert not output.exists()


def test_qa2d_form_with_a_nonexistent_cache_exits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "pairs.jsonl"
    with pytest.raises(SystemExit) as error:
        _run(
            tmp_path, output,
            "--hypothesis-form", "qa2d",
            "--qa2d-cache", str(tmp_path / "missing.jsonl"),
        )
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "missing.jsonl" in stderr
    assert "not found" in stderr
    assert not output.exists()


def test_an_unknown_form_is_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        _run(tmp_path, tmp_path / "pairs.jsonl", "--hypothesis-form", "freeform")
    assert "invalid choice" in capsys.readouterr().err


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


def _export(tmp_path: Path, *extra: str) -> tuple[list[dict[str, object]], Path]:
    manifest, provenance = _fixture(tmp_path)
    output = tmp_path / f"pairs{len(extra)}.jsonl"
    main([
        "--manifest", str(manifest),
        "--provenance", str(provenance),
        "--output", str(output),
        *extra,
    ])
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    return rows, output


def test_gold_answer_source_defaults_to_canonical(tmp_path: Path) -> None:
    """R012 and R012b all ran on the canonical string. An invocation predating the flag must
    still produce the same file, or those results stop being reproducible."""
    rows, _ = _export(tmp_path)
    gold = {r["hypothesis"] for r in rows if r["kind"] in ("needle_gold", "cf_gold")}
    assert gold == {'The answer to the question "who won?" is kennedy.'}


def test_surface_gold_answer_changes_the_gold_claim_and_nothing_else(tmp_path: Path) -> None:
    """R012d switches one variable: the gold claim carries the string actually present in the
    needle document instead of its lowercased canonical form. The replacement claim must not
    move — it was already the raw surface string."""
    canonical, _ = _export(tmp_path)
    surface, _ = _export(tmp_path, "--gold-answer", "surface")
    gold = {r["hypothesis"] for r in surface if r["kind"] in ("needle_gold", "cf_gold")}
    assert gold == {'The answer to the question "who won?" is Kennedy.'}

    def by_kind(rows: list[dict[str, object]], k: str) -> set[object]:
        return {r["hypothesis"] for r in rows if r["kind"] == k}

    assert by_kind(surface, "cf_replacement") == by_kind(canonical, "cf_replacement")
    assert by_kind(surface, "needle_replacement") == by_kind(canonical, "needle_replacement")
    assert [r["premise"] for r in surface] == [r["premise"] for r in canonical]


def test_report_records_the_gold_answer_source(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The pairs file carries no marker of which probe variant produced it, so the report line
    is the protocol record — same reason `hypothesis_form` is echoed there."""
    _export(tmp_path, "--gold-answer", "surface")
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report["gold_answer_source"] == "surface"
