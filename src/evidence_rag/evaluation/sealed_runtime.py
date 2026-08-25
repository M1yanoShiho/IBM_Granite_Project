"""Gold-free runtime bundle contract for Experiment 04.

The runtime reader in this module deliberately knows nothing about scorer-only
records.  A runtime query owns its candidate pool directly; there is no global
corpus lookup through which one query could retrieve another query's candidates.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

RUNTIME_SCHEMA_VERSION = "experiment04.runtime.v1"
RUNTIME_KEYS = {"schema_version", "dataset", "query_id", "question", "candidates"}
CANDIDATE_KEYS = {"source_id", "title", "text", "units"}
UNIT_KEYS = {"unit_id", "text"}
FORBIDDEN_RUNTIME_KEYS = {
    "answer",
    "answers",
    "answer_aliases",
    "answerable",
    "component_id",
    "gold",
    "gold_answer",
    "gold_answers",
    "gold_answer_aliases",
    "is_supporting",
    "support",
    "support_labels",
    "support_units",
    "supporting_facts",
}


class RuntimeContractError(ValueError):
    """Raised when a runtime bundle violates the sealed, gold-free contract."""


def ordered_id_sha256(query_ids: Iterable[str]) -> str:
    """Hash an ordered ID sequence using the frozen pre-registration convention."""

    return hashlib.sha256("\n".join(query_ids).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_keys(value: object) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list | tuple):
        for child in value:
            yield from _walk_keys(child)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise RuntimeContractError(
            f"{label} keys differ: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def validate_runtime_record(record: Mapping[str, Any], *, dataset: str | None = None) -> None:
    """Validate one nested, per-query runtime record without inspecting its prose."""

    _require_exact_keys(record, RUNTIME_KEYS, "runtime record")
    forbidden = sorted({key for key in _walk_keys(record) if key.casefold() in FORBIDDEN_RUNTIME_KEYS})
    if forbidden:
        raise RuntimeContractError(f"runtime record contains scorer-only fields: {forbidden}")
    if record["schema_version"] != RUNTIME_SCHEMA_VERSION:
        raise RuntimeContractError("runtime schema version is not frozen v1")
    if dataset is not None and record["dataset"] != dataset:
        raise RuntimeContractError("runtime dataset does not match its bundle")
    if not isinstance(record["query_id"], str) or not record["query_id"].strip():
        raise RuntimeContractError("runtime query_id must be a non-blank string")
    if not isinstance(record["question"], str) or not record["question"].strip():
        raise RuntimeContractError("runtime question must be a non-blank string")
    candidates = record["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeContractError("runtime candidates must be a non-empty list")

    source_ids: set[str] = set()
    unit_ids: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise RuntimeContractError(f"candidate {index} is not an object")
        _require_exact_keys(candidate, CANDIDATE_KEYS, f"candidate {index}")
        source_id = candidate["source_id"]
        if not isinstance(source_id, str) or not source_id:
            raise RuntimeContractError(f"candidate {index} has no source_id")
        if source_id in source_ids:
            raise RuntimeContractError(f"duplicate source_id within query: {source_id}")
        source_ids.add(source_id)
        if not isinstance(candidate["title"], str) or not isinstance(candidate["text"], str):
            raise RuntimeContractError(f"candidate {index} title/text must be strings")
        if not candidate["text"].strip():
            raise RuntimeContractError(f"candidate {index} has empty text")
        units = candidate["units"]
        if not isinstance(units, list) or not units:
            raise RuntimeContractError(f"candidate {index} has no scoreable units")
        for unit_index, unit in enumerate(units):
            if not isinstance(unit, Mapping):
                raise RuntimeContractError(f"candidate {index} unit {unit_index} is not an object")
            _require_exact_keys(unit, UNIT_KEYS, f"candidate {index} unit {unit_index}")
            unit_id = unit["unit_id"]
            if not isinstance(unit_id, str) or not unit_id.startswith(f"{source_id}:u"):
                raise RuntimeContractError(f"unit_id is not local to source {source_id}")
            if unit_id in unit_ids:
                raise RuntimeContractError(f"duplicate unit_id within query: {unit_id}")
            unit_ids.add(unit_id)
            if not isinstance(unit["text"], str) or not unit["text"].strip():
                raise RuntimeContractError(f"candidate {index} unit {unit_index} has empty text")


def read_runtime_bundle(path: Path, *, dataset: str | None = None) -> tuple[dict[str, Any], ...]:
    """The system-facing reader: it accepts one runtime JSONL file and no sidecar."""

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeContractError(f"runtime line {line_number} is not an object")
            validate_runtime_record(value, dataset=dataset)
            records.append(value)
    query_ids = [str(record["query_id"]) for record in records]
    if len(query_ids) != len(set(query_ids)):
        raise RuntimeContractError("runtime bundle contains duplicate query_id values")
    return tuple(records)


def runtime_audit(
    records: Iterable[Mapping[str, Any]],
    *,
    dataset: str,
    expected_query_ids: list[str],
) -> dict[str, Any]:
    """Return a content-free audit suitable for the committed Goal 1 manifest."""

    materialized = list(records)
    for record in materialized:
        validate_runtime_record(record, dataset=dataset)
    observed_ids = [str(record["query_id"]) for record in materialized]
    if observed_ids != expected_query_ids:
        raise RuntimeContractError("runtime ordered IDs differ from the frozen manifest")
    candidate_counts = [len(record["candidates"]) for record in materialized]
    unit_counts = [
        sum(len(candidate["units"]) for candidate in record["candidates"])
        for record in materialized
    ]
    return {
        "count": len(materialized),
        "ordered_ids_sha256": ordered_id_sha256(observed_ids),
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "top_level_fields": sorted(RUNTIME_KEYS),
        "candidate_count_min": min(candidate_counts),
        "candidate_count_max": max(candidate_counts),
        "unit_count_min": min(unit_counts),
        "unit_count_max": max(unit_counts),
        "forbidden_gold_field_count": 0,
        "candidate_pool_scope": "nested_per_query_only",
        "candidate_pool_isolated": True,
    }
