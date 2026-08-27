from __future__ import annotations

import json
from pathlib import Path

import pytest

from evidence_rag.evaluation.sealed_runtime import (
    RUNTIME_SCHEMA_VERSION,
    RuntimeContractError,
    read_runtime_bundle,
    runtime_audit,
    validate_runtime_record,
)


def _runtime(query_id: str, text: str) -> dict[str, object]:
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "dataset": "hotpotqa",
        "query_id": query_id,
        "question": f"Question for {query_id}?",
        "candidates": [
            {
                "source_id": "p000",
                "title": "Local source",
                "text": text,
                "units": [{"unit_id": "p000:u000", "text": text}],
            }
        ],
    }


def test_runtime_rejects_gold_even_when_nested() -> None:
    record = _runtime("q1", "Revealed development text.")
    record["candidates"][0]["support_units"] = []  # type: ignore[index]

    with pytest.raises(RuntimeContractError, match="scorer-only fields"):
        validate_runtime_record(record)


def test_candidate_ids_are_local_to_each_query_not_a_global_pool() -> None:
    first = _runtime("q1", "First query's revealed candidate.")
    second = _runtime("q2", "Second query's revealed candidate.")

    audit = runtime_audit(
        [first, second],
        dataset="hotpotqa",
        expected_query_ids=["q1", "q2"],
    )

    assert audit["candidate_pool_scope"] == "nested_per_query_only"
    assert audit["candidate_pool_isolated"] is True
    assert first["candidates"] is not second["candidates"]
    assert first["candidates"][0]["text"] != second["candidates"][0]["text"]  # type: ignore[index]


def test_system_reader_accepts_only_the_gold_free_runtime_file(tmp_path: Path) -> None:
    runtime_path = tmp_path / "runtime" / "hotpotqa.jsonl"
    scorer_path = tmp_path / "scorer_only" / "hotpotqa.jsonl"
    runtime_path.parent.mkdir()
    scorer_path.parent.mkdir()
    runtime_path.write_text(json.dumps(_runtime("q1", "Revealed candidate.")) + "\n")
    scorer_path.write_text('{"query_id":"q1","gold_answer_aliases":["secret"]}\n')

    loaded = read_runtime_bundle(runtime_path, dataset="hotpotqa")

    assert len(loaded) == 1
    assert runtime_path.parent != scorer_path.parent
    assert "gold" not in json.dumps(loaded)
