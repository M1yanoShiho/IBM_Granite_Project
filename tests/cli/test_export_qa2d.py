import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from evidence_rag.cli.export_qa2d import collect_pairs, main
from evidence_rag.materializer.provenance import MutationRecord, write_provenance
from evidence_rag.relations.qa2d import load_qa2d_cache


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _record(query_id: str, replacement: str = "Nixon") -> MutationRecord:
    return MutationRecord(
        query_id=query_id,
        needle_document_id="needle",
        counterfactual_document_id="cf::needle",
        gold_value="Kennedy",
        gold_alias_used="Kennedy",
        replacement_value=replacement,
        string_class="proper_name_1",
        seed=42,
        char_span=(0, 7),
        text_hash_before="a" * 8,
        text_hash_after="b" * 8,
        answer_bank_hash="c" * 8,
    )


def _fixture(
    tmp_path: Path,
    questions: dict[str, str] | None = None,
    records: Sequence[MutationRecord] | None = None,
) -> tuple[Path, Path]:
    questions = questions or {"q1": "who won the 1960 election"}
    records = records if records is not None else [_record("q1")]
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
        [
            {"schema_version": "1.0", "query_id": qid, "text": text}
            for qid, text in questions.items()
        ],
    )
    _write_jsonl(
        tmp_path / "gold_cases.jsonl",
        [
            {"query_id": qid, "relevant_document_ids": ["needle"],
             "reference_answers": ["Kennedy"]}
            for qid in questions
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
    write_provenance(provenance, records)
    return tmp_path / "manifest.json", provenance


_CALLS: list[list[tuple[str, str]]] = []


def _fake_loader(model_id: str):  # type: ignore[no-untyped-def]
    """Stands in for the seq2seq converter so the unit tests never import transformers."""

    def generate(pairs):  # type: ignore[no-untyped-def]
        _CALLS.append(list(pairs))
        return [f"{answer} :: {question}" for question, answer in pairs]

    return generate


@pytest.fixture(autouse=True)
def _reset_calls() -> None:
    _CALLS.clear()


def test_writes_one_row_per_unique_question_answer_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("evidence_rag.cli.export_qa2d.load_generator", _fake_loader)
    manifest, provenance = _fixture(tmp_path)
    output = tmp_path / "cache" / "qa2d.jsonl"
    assert (
        main([
            "--manifest", str(manifest),
            "--provenance", str(provenance),
            "--output", str(output),
            "--model", "fake/qa2d",
        ])
        == 0
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    assert [(row["question"], row["answer"], row["declarative"]) for row in rows] == [
        ("who won the 1960 election", "Kennedy", "Kennedy :: who won the 1960 election"),
        ("who won the 1960 election", "Nixon", "Nixon :: who won the 1960 election"),
    ]
    report = json.loads(capsys.readouterr().out)
    assert report == {"model": "fake/qa2d", "n_pairs": 2, "n_unique": 2}


def test_duplicate_pairs_are_generated_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two queries in the same synthetic family ask for the same sentence. Generating it twice
    would burn login-node time and risk two different strings for one key."""
    monkeypatch.setattr("evidence_rag.cli.export_qa2d.load_generator", _fake_loader)
    manifest, provenance = _fixture(
        tmp_path,
        questions={"q1": "who won the 1960 election", "q2": "who won the 1960 election"},
        records=[_record("q1"), _record("q2")],
    )
    output = tmp_path / "qa2d.jsonl"
    main([
        "--manifest", str(manifest),
        "--provenance", str(provenance),
        "--output", str(output),
        "--model", "fake/qa2d",
    ])
    report = json.loads(capsys.readouterr().out)
    assert report["n_pairs"] == 4
    assert report["n_unique"] == 2
    assert _CALLS == [
        [("who won the 1960 election", "Kennedy"), ("who won the 1960 election", "Nixon")]
    ]


def test_cache_covers_every_hypothesis_the_probe_will_ask_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gold AND replacement, for every record — a partial cache turns into a Qa2dCacheMiss on
    the cluster, hours after the login-node step is over."""
    monkeypatch.setattr("evidence_rag.cli.export_qa2d.load_generator", _fake_loader)
    manifest, provenance = _fixture(
        tmp_path,
        questions={"q1": "who won the 1960 election", "q2": "who wrote Dune"},
        records=[_record("q1"), _record("q2", replacement="Bradbury")],
    )
    output = tmp_path / "qa2d.jsonl"
    main([
        "--manifest", str(manifest),
        "--provenance", str(provenance),
        "--output", str(output),
        "--model", "fake/qa2d",
    ])
    lookup = load_qa2d_cache(output)
    assert lookup("who won the 1960 election", "Kennedy") == "Kennedy :: who won the 1960 election"
    assert lookup("who won the 1960 election", "Nixon") == "Nixon :: who won the 1960 election"
    assert lookup("who wrote Dune", "Kennedy") == "Kennedy :: who wrote Dune"
    assert lookup("who wrote Dune", "Bradbury") == "Bradbury :: who wrote Dune"


def test_model_is_required_and_has_no_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This repo does not ship guessed checkpoint ids — see LABEL_ORDER in cli/gate0b.py. A
    default here would silently decide which converter the pre-registered arm ran on."""
    monkeypatch.setattr("evidence_rag.cli.export_qa2d.load_generator", _fake_loader)
    manifest, provenance = _fixture(tmp_path)
    with pytest.raises(SystemExit) as error:
        main([
            "--manifest", str(manifest),
            "--provenance", str(provenance),
            "--output", str(tmp_path / "qa2d.jsonl"),
        ])
    assert error.value.code == 2


def test_records_whose_query_is_missing_are_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe skips them too, so generating for them would only waste login-node time."""
    monkeypatch.setattr("evidence_rag.cli.export_qa2d.load_generator", _fake_loader)
    manifest, provenance = _fixture(
        tmp_path,
        questions={"q1": "who won the 1960 election"},
        records=[_record("q1"), _record("gone")],
    )
    output = tmp_path / "qa2d.jsonl"
    main([
        "--manifest", str(manifest),
        "--provenance", str(provenance),
        "--output", str(output),
        "--model", "fake/qa2d",
    ])
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 2


def test_collect_pairs_normalises_keys_the_way_the_lookup_does() -> None:
    """The written key and the looked-up key must be produced by the same normalisation."""
    pairs = collect_pairs(
        records=(_record("q1"),),
        question_by_query={"q1": "  who won the 1960 election \n"},
    )
    assert pairs == (
        ("who won the 1960 election", "Kennedy"),
        ("who won the 1960 election", "Nixon"),
    )
