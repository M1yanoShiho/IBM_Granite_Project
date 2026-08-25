from pathlib import Path

import pytest
from pydantic import ValidationError

from evidence_rag.materializer.provenance import MutationRecord, read_provenance, write_provenance


def record(query_id: str = "q1") -> MutationRecord:
    return MutationRecord(
        query_id=query_id,
        needle_document_id="doc-9",
        counterfactual_document_id="cf::q1::doc-9",
        gold_value="18",
        gold_alias_used="18%",
        replacement_value="23",
        string_class="integer",
        seed=42,
        char_span=(10, 13),
        text_hash_before="a" * 64,
        text_hash_after="b" * 64,
        answer_bank_hash="c" * 64,
    )


def test_record_round_trips_through_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "provenance.jsonl"
    write_provenance(path, (record("q1"), record("q2")))
    loaded = read_provenance(path)
    assert tuple(r.query_id for r in loaded) == ("q1", "q2")
    assert loaded[0].char_span == (10, 13)


def test_record_is_frozen_and_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        MutationRecord(query_id="q", extra="x")  # type: ignore[call-arg]
