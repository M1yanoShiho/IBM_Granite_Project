import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from evidence_rag.cli.export_qa2d import (
    INPUT_TEMPLATE,
    collect_pairs,
    load_generator,
    main,
    normalise_terminal_space,
    qa2d_input,
)
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
            "--model", "MarkS/bart-base-qa2d",
        ])
        == 0
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    assert [(row["question"], row["answer"], row["declarative"]) for row in rows] == [
        ("who won the 1960 election", "Kennedy", "Kennedy :: who won the 1960 election"),
        ("who won the 1960 election", "Nixon", "Nixon :: who won the 1960 election"),
    ]
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "model": "MarkS/bart-base-qa2d",
        "n_pairs": 2,
        "n_unique": 2,
        "non_initial_uppercase_rate": 0.0,
    }


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
        "--model", "MarkS/bart-base-qa2d",
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
        "--model", "MarkS/bart-base-qa2d",
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
        "--model", "MarkS/bart-base-qa2d",
    ])
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 2


def _bart_shaped_loader(model_id: str):  # type: ignore[no-untyped-def]
    """Reproduces the checkpoint's detokenisation, which puts a space before the full stop."""

    def generate(pairs):  # type: ignore[no-untyped-def]
        return [f"{answer} won the 1960 election ." for _, answer in pairs]

    return generate


def test_the_written_cache_carries_normalised_sentences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache file IS the arm's data, so the normalisation has to land in it rather than
    somewhere downstream; a reviewer diffs this file, not an in-memory string."""
    monkeypatch.setattr("evidence_rag.cli.export_qa2d.load_generator", _bart_shaped_loader)
    manifest, provenance = _fixture(tmp_path)
    output = tmp_path / "qa2d.jsonl"
    main([
        "--manifest", str(manifest),
        "--provenance", str(provenance),
        "--output", str(output),
        "--model", "MarkS/bart-base-qa2d",
    ])
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    assert [row["declarative"] for row in rows] == [
        "Kennedy won the 1960 election.",
        "Nixon won the 1960 election.",
    ]


def test_qa2d_input_uses_the_verified_prefix_format() -> None:
    """The MarkS/bart-base-qa2d card's own usage example is
    "question: what day is it today? answer: Tuesday", so that string is asserted verbatim.

    Asserting only that both halves appear would still pass on the plain space join this
    replaced, and a space join does not crash — it yields fluent sentences that are not the
    QA2D transform, which nothing downstream can distinguish from the real thing.
    """
    assert (
        qa2d_input("what day is it today?", "Tuesday", INPUT_TEMPLATE["MarkS/bart-base-qa2d"])
        == "question: what day is it today? answer: Tuesday"
    )


def test_input_template_matches_the_verified_card_format_of_each_checkpoint() -> None:
    """Pins the TEMPLATE, not just that the id is present.

    Verified against the real checkpoint on 2026-08-03: MarkS/bart-base-qa2d's card gives
    "question: what day is it today? answer: Tuesday", and 30 pairs from this project's own
    probe were run through it on the cluster in that format.

    A test that only checked the id had an entry would let a future wrong template ship a
    converter that emits fluent sentences which are not the QA2D transform - and unlike a
    crash, nothing downstream can tell.
    """
    assert INPUT_TEMPLATE["MarkS/bart-base-qa2d"] == "question: {question} answer: {answer}"
    for template in INPUT_TEMPLATE.values():
        assert "{question}" in template
        assert "{answer}" in template


def test_qa2d_input_applies_the_template_it_is_given() -> None:
    """The registry has to be load-bearing, not decorative.

    If qa2d_input hardcoded one format, registering a second checkpoint would record its format
    in INPUT_TEMPLATE and still feed it the first one. The format below is the rejected
    question_converter-3b's, used here only because it is unmistakably not the registered one.
    """
    assert qa2d_input("who won", "Kennedy", "{question} </s> {answer}") == "who won </s> Kennedy"


def test_whitespace_before_terminal_punctuation_is_collapsed() -> None:
    """BART's detokenisation emits "... in 2002 ." where the other two rungs of this ablation
    emit "... is 2002.". That is a surface difference with nothing to do with the variable under
    test (sentence form), so it is normalised out rather than left uncontrolled between arms."""
    assert (
        normalise_terminal_space("They stopped making the half dollar in 2002 .")
        == "They stopped making the half dollar in 2002."
    )


def test_normalisation_leaves_everything_but_the_terminal_punctuation_alone() -> None:
    """The generated text is experimental data, so every edit to it has to be declarable in one
    sentence. Only the whitespace before the FINAL mark goes: detokenised abbreviations keep
    their spacing, casing is untouched, and a sentence with no terminal mark is returned as is.
    """
    assert (
        normalise_terminal_space("He was born in Washington , D . C . and died in 2002 .")
        == "He was born in Washington , D . C . and died in 2002."
    )
    assert normalise_terminal_space("Kennedy won the 1960 election.") == (
        "Kennedy won the 1960 election."
    )
    assert normalise_terminal_space("Howard jones sings no one ever is to blame") == (
        "Howard jones sings no one ever is to blame"
    )


def test_load_generator_rejects_an_unverified_checkpoint() -> None:
    """The named rejection: domenicrosati/question_converter-3b joins with "{question} </s>
    {answer}", so running it under this repo's format would produce a cache that looks fine and
    is not rung 3. It must fail before any weights are touched, not silently generate."""
    with pytest.raises(ValueError, match="no verified input template"):
        load_generator("domenicrosati/question_converter-3b")


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


def test_written_rows_carry_the_checkpoint_and_template_that_produced_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provenance on the row, for the same reason `cli/gate0b.py` stamps a weight-derived
    model_version on every edge: a cache regenerated with a different converter is
    shape-identical to a correct one, and `relations/qa2d.py` refuses a mixed file only if the
    rows say who made them."""
    monkeypatch.setattr("evidence_rag.cli.export_qa2d.load_generator", _bart_shaped_loader)
    manifest, provenance = _fixture(tmp_path)
    output = tmp_path / "qa2d.jsonl"
    main([
        "--manifest", str(manifest),
        "--provenance", str(provenance),
        "--output", str(output),
        "--model", "MarkS/bart-base-qa2d",
    ])
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    assert rows, "export produced no rows"
    assert {row["model"] for row in rows} == {"MarkS/bart-base-qa2d"}
    assert {row["template"] for row in rows} == {"question: {question} answer: {answer}"}


def test_report_measures_how_much_the_converter_recases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R012b 判读纪律 5 forbids reading the rung-3 delta until the case-change rate of the
    generated strings is reported. Questions and answers both arrive lowercased, so any capital
    after position 0 is introduced by the converter — if that rate is high, rung 3 varies casing
    as well as sentence form and its increment cannot be attributed to form alone. The obligation
    is computed here rather than left to whoever remembers it."""
    monkeypatch.setattr("evidence_rag.cli.export_qa2d.load_generator", _bart_shaped_loader)
    manifest, provenance = _fixture(tmp_path)
    main([
        "--manifest", str(manifest),
        "--provenance", str(provenance),
        "--output", str(tmp_path / "qa2d.jsonl"),
        "--model", "MarkS/bart-base-qa2d",
    ])
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert "non_initial_uppercase_rate" in report
    assert 0.0 <= report["non_initial_uppercase_rate"] <= 1.0
