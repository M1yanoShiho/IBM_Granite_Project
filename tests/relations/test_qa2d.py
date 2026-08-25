import json
from pathlib import Path

import pytest

from evidence_rag.materializer.provenance import MutationRecord
from evidence_rag.relations.qa2d import (
    Qa2dCacheMiss,
    Qa2dLookup,
    cache_key,
    load_qa2d_cache,
)
from evidence_rag.relations.task_probe import build_probe_pairs


def _write_cache(path: Path, rows: list[dict[str, str]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    return path


_ROW = {
    "question": "who won the 1960 election",
    "answer": "Kennedy",
    "declarative": "Kennedy won the 1960 election.",
}


def test_lookup_returns_the_pre_generated_declarative_sentence() -> None:
    """The rung-3 surface is the QA2D fusion, not a frame wrapped round the question."""
    lookup = Qa2dLookup({cache_key("who won the 1960 election", "Kennedy"): _ROW["declarative"]})
    assert lookup("who won the 1960 election", "Kennedy") == "Kennedy won the 1960 election."


def test_a_missing_pair_raises_and_names_it_instead_of_falling_back() -> None:
    """A silent template fallback would mix rung-0 rows into the QA2D arm and quietly corrupt
    the comparison the ablation exists to make, so a miss must stop the export."""
    lookup = Qa2dLookup({cache_key("who won the 1960 election", "Kennedy"): _ROW["declarative"]})
    with pytest.raises(Qa2dCacheMiss) as error:
        lookup("who won the 1960 election", "Nixon")
    message = str(error.value)
    assert "who won the 1960 election" in message
    assert "Nixon" in message
    assert "The answer to the question" not in message


def test_keys_are_normalised_by_stripping_whitespace() -> None:
    """The generator writes stripped keys; the probe may hand back padded strings. If the two
    normalisations disagree every lookup misses and the arm cannot run at all."""
    lookup = Qa2dLookup({cache_key("who won the 1960 election", "Kennedy"): _ROW["declarative"]})
    assert lookup("  who won the 1960 election\n", "\tKennedy ") == "Kennedy won the 1960 election."


def test_loader_reads_the_cache_jsonl(tmp_path: Path) -> None:
    lookup = load_qa2d_cache(_write_cache(tmp_path / "qa2d.jsonl", [_ROW]))
    assert lookup("who won the 1960 election", "Kennedy") == "Kennedy won the 1960 election."


def test_loader_normalises_the_keys_it_reads(tmp_path: Path) -> None:
    """Pins the loader to the same normalisation as `cache_key`, so a cache written with stray
    whitespace still resolves rather than producing an unexplainable miss on the cluster."""
    padded = dict(_ROW, question="  who won the 1960 election ", answer=" Kennedy\n")
    lookup = load_qa2d_cache(_write_cache(tmp_path / "qa2d.jsonl", [padded]))
    assert lookup("who won the 1960 election", "Kennedy") == "Kennedy won the 1960 election."


def test_loader_rejects_two_rows_disagreeing_on_the_same_pair(tmp_path: Path) -> None:
    """Silently keeping the last row would make the arm depend on cache line order, which the
    provenance discipline does not allow."""
    clash = dict(_ROW, declarative="Kennedy was the winner in 1960.")
    path = _write_cache(tmp_path / "qa2d.jsonl", [_ROW, clash])
    with pytest.raises(ValueError, match="conflicting"):
        load_qa2d_cache(path)


def test_lookup_is_usable_as_the_probe_hypothesis_form(tmp_path: Path) -> None:
    """The whole point of the cache is that `build_probe_pairs` can take it where it takes the
    two deterministic rungs, with no model on the compute node."""
    record = MutationRecord(
        query_id="q1",
        needle_document_id="needle",
        counterfactual_document_id="cf::needle",
        gold_value="Kennedy",
        gold_alias_used="Kennedy",
        replacement_value="Nixon",
        string_class="proper_name_1",
        seed=42,
        char_span=(0, 7),
        text_hash_before="a" * 8,
        text_hash_after="b" * 8,
        answer_bank_hash="c" * 8,
    )
    lookup = load_qa2d_cache(
        _write_cache(
            tmp_path / "qa2d.jsonl",
            [
                _ROW,
                dict(_ROW, answer="Nixon", declarative="Nixon won the 1960 election."),
            ],
        )
    )
    pairs = build_probe_pairs(
        records=(record,),
        question_by_query={"q1": "who won the 1960 election"},
        text_by_document={"needle": "Kennedy won", "cf::needle": "Nixon won"},
        hypothesis_form=lookup,
    )
    assert {pair.hypothesis for pair in pairs} == {
        "Kennedy won the 1960 election.",
        "Nixon won the 1960 election.",
    }


def test_a_cache_mixing_two_checkpoints_is_rejected(tmp_path: Path) -> None:
    """Same hazard class as the edge cache's `model_version`: the cache is the artefact rung 3
    consumes, and a file regenerated with a different checkpoint or template is shape-identical
    to a correct one. Provenance on every row makes a partial regeneration loud instead of
    silent — the arm would otherwise measure two transforms averaged together."""
    path = tmp_path / "qa2d.jsonl"
    path.write_text(
        json.dumps({"question": "q1", "answer": "a1", "declarative": "A1 is q1.",
                    "model": "MarkS/bart-base-qa2d", "template": "question: {question} answer: {answer}"})
        + "\n"
        + json.dumps({"question": "q2", "answer": "a2", "declarative": "A2 is q2.",
                      "model": "some/other-converter", "template": "question: {question} answer: {answer}"})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="two checkpoints"):
        load_qa2d_cache(path)


def test_a_cache_mixing_two_templates_is_rejected(tmp_path: Path) -> None:
    """One checkpoint fed two formats produces two different transforms under one name."""
    path = tmp_path / "qa2d.jsonl"
    path.write_text(
        json.dumps({"question": "q1", "answer": "a1", "declarative": "A1 is q1.",
                    "model": "MarkS/bart-base-qa2d", "template": "question: {question} answer: {answer}"})
        + "\n"
        + json.dumps({"question": "q2", "answer": "a2", "declarative": "A2 is q2.",
                      "model": "MarkS/bart-base-qa2d", "template": "{question} </s> {answer}"})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="two templates"):
        load_qa2d_cache(path)
