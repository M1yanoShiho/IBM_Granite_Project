"""Experiment 05 Goal 1 deterministic data-selection contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUNTIME_SCHEMA_VERSION = "experiment05.runtime.v1"
SIDECAR_SCHEMA_VERSION = "experiment05.scorer.v1"
RUNTIME_KEYS = {
    "schema_version",
    "dataset",
    "query_id",
    "question",
    "corpus_snapshot_id",
    "bm25_index_id",
    "dense_index_id",
}
SIDECAR_KEYS = {
    "schema_version",
    "dataset",
    "query_id",
    "reference_fact_groups",
    "gold_provenance",
}
FACT_GROUP_KEYS = {"fact_id", "fact_question", "aliases"}
PROVENANCE_KEYS = {
    "fact_id",
    "source_kind",
    "source_id",
    "start_unit",
    "end_unit",
}


class Goal1DataError(RuntimeError):
    """Raised when frozen Goal 1 data gates cannot be satisfied."""


class RuntimeIsolationError(ValueError):
    """Raised when runtime/scorer physical or schema isolation is violated."""


@dataclass(frozen=True)
class FrozenIdSelection:
    development_ids: tuple[str, ...]
    formal_ids: tuple[str, ...]
    development_ordered_ids_sha256: str
    formal_ordered_ids_sha256: str


@dataclass(frozen=True)
class ExposureRegistry:
    exposure_ids: tuple[str, ...]
    sources: tuple[dict[str, object], ...]


def read_triviaqa_question_lookup(paths: Sequence[Path]) -> dict[str, str]:
    """Read the pinned TriviaQA parquet conversion without loading answer fields."""

    try:
        import pyarrow.parquet as parquet  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - optional Goal 1 dependency
        raise Goal1DataError("reading TriviaQA parquet requires pyarrow") from exc

    questions: dict[str, str] = {}
    for path in paths:
        table = parquet.read_table(path, columns=["question_id", "question"])
        for raw_id, raw_question in zip(
            table.column("question_id").to_pylist(),
            table.column("question").to_pylist(),
            strict=True,
        ):
            query_id = _require_nonblank_string(raw_id, f"TriviaQA ID in {path}")
            question = _require_nonblank_string(raw_question, f"TriviaQA question {query_id}")
            prior = questions.get(query_id)
            if prior is not None and prior != question:
                raise Goal1DataError(
                    f"TriviaQA {query_id} has conflicting question text across parquet files"
                )
            questions[query_id] = question
    return questions


def read_kilt_records(
    path: Path,
    *,
    question_lookup: Mapping[str, str] | None = None,
    allowed_ids: Collection[str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Read an official KILT JSONL split and restore TriviaQA inputs by ID."""

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise Goal1DataError(f"KILT line {line_number} in {path} is not an object")
            query_id = _require_nonblank_string(
                value.get("id"), f"KILT canonical ID at line {line_number}"
            )
            if query_id in seen:
                raise Goal1DataError(f"KILT split contains duplicate canonical ID {query_id}")
            seen.add(query_id)
            if allowed_ids is not None and query_id not in allowed_ids:
                continue
            record = dict(value)
            raw_question = record.get("input")
            if not isinstance(raw_question, str) or not raw_question.strip():
                question = question_lookup.get(query_id) if question_lookup is not None else None
                if not isinstance(question, str) or not question.strip():
                    raise Goal1DataError(f"KILT {query_id} is missing a TriviaQA question")
                record["input"] = question
            records.append(record)
    return tuple(records)


def read_asqa_records(path: Path) -> tuple[dict[str, Any], ...]:
    """Read ALCE ASQA data while enforcing stable, unique canonical IDs."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise Goal1DataError("ASQA source must be a JSON list")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise Goal1DataError(f"ASQA record {index} is not an object")
        query_id = _require_nonblank_string(raw.get("sample_id"), f"ASQA record {index} ID")
        if query_id in seen:
            raise Goal1DataError(f"ASQA source contains duplicate canonical ID {query_id}")
        seen.add(query_id)
        records.append(raw)
    return tuple(records)


def _validate_ids(label: str, values: Sequence[str]) -> tuple[str, ...]:
    materialized = tuple(values)
    if any(not isinstance(value, str) or not value for value in materialized):
        raise Goal1DataError(f"{label} contains a blank canonical ID")
    if len(materialized) != len(set(materialized)):
        raise Goal1DataError(f"{label} contains duplicate canonical IDs")
    return materialized


def _rank(kind: str, dataset: str, query_id: str) -> str:
    payload = f"experiment05-{kind}|{dataset}|{query_id}|20260822"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ordered_hash(query_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(query_ids).encode("utf-8")).hexdigest()


def _selection(development_ids: Sequence[str], formal_ids: Sequence[str]) -> FrozenIdSelection:
    development = tuple(development_ids)
    formal = tuple(formal_ids)
    if not set(development).isdisjoint(formal):
        raise Goal1DataError("development and formal canonical IDs overlap")
    return FrozenIdSelection(
        development_ids=development,
        formal_ids=formal,
        development_ordered_ids_sha256=_ordered_hash(development),
        formal_ordered_ids_sha256=_ordered_hash(formal),
    )


def select_kilt_ids(
    *,
    dataset: str,
    train_ids: Sequence[str],
    formal_pool_ids: Sequence[str],
    exposure_ids: Collection[str],
    development_count: int = 120,
    formal_count: int = 400,
) -> FrozenIdSelection:
    train = _validate_ids(f"{dataset} train pool", train_ids)
    formal_pool = _validate_ids(f"{dataset} formal pool", formal_pool_ids)
    if development_count <= 0 or formal_count <= 0:
        raise Goal1DataError("development/formal counts must be positive")
    if len(train) < development_count:
        raise Goal1DataError(
            f"{dataset} requires {development_count} development IDs; found {len(train)}"
        )
    development = sorted(train, key=lambda query_id: _rank("dev", dataset, query_id))[
        :development_count
    ]
    exposure = set(exposure_ids)
    eligible_formal = [query_id for query_id in formal_pool if query_id not in exposure]
    if len(eligible_formal) < formal_count:
        raise Goal1DataError(
            f"{dataset} requires {formal_count} unexposed formal IDs; "
            f"found {len(eligible_formal)}"
        )
    formal = sorted(
        eligible_formal,
        key=lambda query_id: _rank("formal", dataset, query_id),
    )[:formal_count]
    return _selection(development, formal)


def select_asqa_ids(
    *,
    dataset: str,
    all_ids: Sequence[str],
    exposure_ids: Collection[str],
    development_count: int = 120,
    formal_count: int = 400,
) -> FrozenIdSelection:
    canonical_ids = _validate_ids(f"{dataset} pool", all_ids)
    if development_count <= 0 or formal_count <= 0:
        raise Goal1DataError("development/formal counts must be positive")
    canonical_set = set(canonical_ids)
    exposure = canonical_set.intersection(exposure_ids)
    exposed_development = sorted(
        exposure,
        key=lambda query_id: _rank("dev", dataset, query_id),
    )[:development_count]
    needed = development_count - len(exposed_development)
    fresh_development = sorted(
        canonical_set - exposure,
        key=lambda query_id: _rank("dev", dataset, query_id),
    )[:needed]
    development = exposed_development + fresh_development
    if len(development) < development_count:
        raise Goal1DataError(
            f"{dataset} requires {development_count} development IDs; found {len(development)}"
        )
    eligible_formal = canonical_set - exposure - set(development)
    if len(eligible_formal) < formal_count:
        raise Goal1DataError(
            f"{dataset} requires {formal_count} unexposed formal IDs; "
            f"found {len(eligible_formal)}"
        )
    formal = sorted(
        eligible_formal,
        key=lambda query_id: _rank("formal", dataset, query_id),
    )[:formal_count]
    return _selection(development, formal)


def _require_keys(record: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(record)
    if observed != expected:
        raise RuntimeIsolationError(
            f"{label} keys differ: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _require_nonblank_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeIsolationError(f"{label} must be a non-blank string")
    return value


def validate_runtime_record(
    record: Mapping[str, Any], *, dataset: str | None = None
) -> None:
    _require_keys(record, RUNTIME_KEYS, "runtime record")
    if record["schema_version"] != RUNTIME_SCHEMA_VERSION:
        raise RuntimeIsolationError("runtime schema version differs from frozen v1")
    observed_dataset = _require_nonblank_string(record["dataset"], "runtime dataset")
    if dataset is not None and observed_dataset != dataset:
        raise RuntimeIsolationError("runtime dataset differs from its bundle")
    for key in (
        "query_id",
        "question",
        "corpus_snapshot_id",
        "bm25_index_id",
        "dense_index_id",
    ):
        _require_nonblank_string(record[key], f"runtime {key}")


def validate_sidecar_record(
    record: Mapping[str, Any], *, dataset: str | None = None
) -> None:
    _require_keys(record, SIDECAR_KEYS, "sidecar record")
    if record["schema_version"] != SIDECAR_SCHEMA_VERSION:
        raise RuntimeIsolationError("sidecar schema version differs from frozen v1")
    observed_dataset = _require_nonblank_string(record["dataset"], "sidecar dataset")
    if dataset is not None and observed_dataset != dataset:
        raise RuntimeIsolationError("sidecar dataset differs from its bundle")
    _require_nonblank_string(record["query_id"], "sidecar query_id")

    fact_groups = record["reference_fact_groups"]
    if not isinstance(fact_groups, list) or not fact_groups:
        raise RuntimeIsolationError("reference_fact_groups must be a non-empty list")
    fact_ids: set[str] = set()
    for index, group in enumerate(fact_groups):
        if not isinstance(group, Mapping):
            raise RuntimeIsolationError(f"reference_fact_groups[{index}] must be an object")
        _require_keys(group, FACT_GROUP_KEYS, f"reference_fact_groups[{index}]")
        fact_id = _require_nonblank_string(group["fact_id"], f"fact group {index} fact_id")
        _require_nonblank_string(group["fact_question"], f"fact group {index} fact_question")
        aliases = group["aliases"]
        if not isinstance(aliases, list) or not aliases:
            raise RuntimeIsolationError(f"reference_fact_groups[{index}] aliases must be non-empty")
        if any(not isinstance(alias, str) or not alias.strip() for alias in aliases):
            raise RuntimeIsolationError(f"reference_fact_groups[{index}] contains a blank alias")
        if fact_id in fact_ids:
            raise RuntimeIsolationError("reference_fact_groups contain duplicate fact_id values")
        fact_ids.add(fact_id)

    provenance = record["gold_provenance"]
    if not isinstance(provenance, list):
        raise RuntimeIsolationError("gold_provenance must be a list")
    for index, item in enumerate(provenance):
        if not isinstance(item, Mapping):
            raise RuntimeIsolationError(f"gold_provenance[{index}] must be an object")
        _require_keys(item, PROVENANCE_KEYS, f"gold_provenance[{index}]")
        if item["fact_id"] not in fact_ids:
            raise RuntimeIsolationError("gold_provenance refers to an unknown fact_id")
        source_kind = _require_nonblank_string(
            item["source_kind"], f"gold_provenance[{index}] source_kind"
        )
        if source_kind not in {"kilt-paragraph", "wikipedia-title"}:
            raise RuntimeIsolationError("gold_provenance source_kind is unsupported")
        _require_nonblank_string(item["source_id"], f"gold_provenance[{index}] source_id")
        start = item["start_unit"]
        end = item["end_unit"]
        if start is None or end is None:
            if start is not None or end is not None or source_kind != "wikipedia-title":
                raise RuntimeIsolationError("gold_provenance unit span is invalid")
        elif (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 0
            or end < start
        ):
            raise RuntimeIsolationError("gold_provenance unit span is invalid")


def read_runtime_bundle(path: Path, *, dataset: str | None = None) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeIsolationError(f"runtime line {line_number} is not an object")
            validate_runtime_record(value, dataset=dataset)
            records.append(value)
    query_ids = [str(record["query_id"]) for record in records]
    if len(query_ids) != len(set(query_ids)):
        raise RuntimeIsolationError("runtime bundle contains duplicate query IDs")
    return tuple(records)


def _read_sidecar_bundle(path: Path, *, dataset: str) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeIsolationError(f"sidecar line {line_number} is not an object")
            validate_sidecar_record(value, dataset=dataset)
            records.append(value)
    query_ids = [str(record["query_id"]) for record in records]
    if len(query_ids) != len(set(query_ids)):
        raise RuntimeIsolationError("sidecar bundle contains duplicate query IDs")
    return tuple(records)


def audit_bundle_pair(
    runtime_path: Path,
    sidecar_path: Path,
    *,
    dataset: str,
    expected_query_ids: Sequence[str],
) -> dict[str, object]:
    if runtime_path.resolve().parent == sidecar_path.resolve().parent:
        raise RuntimeIsolationError("runtime and scorer-only files share a parent directory")
    runtime = read_runtime_bundle(runtime_path, dataset=dataset)
    sidecar = _read_sidecar_bundle(sidecar_path, dataset=dataset)
    runtime_ids = tuple(str(record["query_id"]) for record in runtime)
    sidecar_ids = tuple(str(record["query_id"]) for record in sidecar)
    expected = tuple(expected_query_ids)
    if runtime_ids != expected:
        raise RuntimeIsolationError("runtime ordered IDs differ from the frozen selection")
    if sidecar_ids != expected:
        raise RuntimeIsolationError("sidecar ordered IDs differ from the frozen selection")
    return {
        "count": len(expected),
        "ordered_ids_sha256": _ordered_hash(expected),
        "runtime_gold_field_count": 0,
        "physical_parent_directories_separate": True,
        "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
        "sidecar_schema_version": SIDECAR_SCHEMA_VERSION,
    }


def materialize_kilt_record(
    raw: Mapping[str, Any],
    *,
    dataset: str,
    corpus_snapshot_id: str,
    bm25_index_id: str,
    dense_index_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    query_id = _require_nonblank_string(raw.get("id"), f"{dataset} canonical ID")
    question = _require_nonblank_string(raw.get("input"), f"{dataset} question")
    outputs = raw.get("output")
    if not isinstance(outputs, list) or not outputs:
        raise Goal1DataError(f"{dataset} {query_id} has no scorer outputs")

    aliases: list[str] = []
    provenance: list[dict[str, Any]] = []
    seen_provenance: set[tuple[str, int, int]] = set()
    for output in outputs:
        if not isinstance(output, Mapping):
            raise Goal1DataError(f"{dataset} {query_id} has a malformed output")
        answer = output.get("answer")
        if isinstance(answer, str) and answer.strip() and answer.strip() not in aliases:
            aliases.append(answer.strip())
        raw_provenance = output.get("provenance", [])
        if raw_provenance is None:
            raw_provenance = []
        if not isinstance(raw_provenance, list):
            raise Goal1DataError(f"{dataset} {query_id} provenance must be a list")
        for item in raw_provenance:
            if not isinstance(item, Mapping):
                raise Goal1DataError(f"{dataset} {query_id} has malformed provenance")
            source_id = str(item.get("wikipedia_id", "")).strip()
            start = item.get("start_paragraph_id")
            end = item.get("end_paragraph_id")
            if (
                not source_id
                or not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
            ):
                raise Goal1DataError(f"{dataset} {query_id} has incomplete provenance")
            key = (source_id, start, end)
            if key not in seen_provenance:
                provenance.append(
                    {
                        "fact_id": "fact-0001",
                        "source_kind": "kilt-paragraph",
                        "source_id": source_id,
                        "start_unit": start,
                        "end_unit": end,
                    }
                )
                seen_provenance.add(key)
    if not aliases:
        raise Goal1DataError(f"{dataset} {query_id} has no answer aliases")

    runtime = _runtime_record(
        dataset=dataset,
        query_id=query_id,
        question=question,
        corpus_snapshot_id=corpus_snapshot_id,
        bm25_index_id=bm25_index_id,
        dense_index_id=dense_index_id,
    )
    sidecar = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "dataset": dataset,
        "query_id": query_id,
        "reference_fact_groups": [
            {
                "fact_id": "fact-0001",
                "fact_question": question,
                "aliases": aliases,
            }
        ],
        "gold_provenance": provenance,
    }
    validate_runtime_record(runtime, dataset=dataset)
    validate_sidecar_record(sidecar, dataset=dataset)
    return runtime, sidecar


def materialize_asqa_record(
    raw: Mapping[str, Any],
    *,
    corpus_snapshot_id: str,
    bm25_index_id: str,
    dense_index_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = "alce-asqa"
    query_id = _require_nonblank_string(raw.get("sample_id"), "ASQA canonical ID")
    question = _require_nonblank_string(raw.get("question"), "ASQA question")
    qa_pairs = raw.get("qa_pairs")
    if not isinstance(qa_pairs, list) or not qa_pairs:
        raise Goal1DataError(f"ASQA {query_id} has no qa_pairs")

    fact_groups: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for index, pair in enumerate(qa_pairs, start=1):
        if not isinstance(pair, Mapping):
            raise Goal1DataError(f"ASQA {query_id} has a malformed qa_pair")
        fact_id = f"fact-{index:04d}"
        fact_question = _require_nonblank_string(
            pair.get("question"), f"ASQA {query_id} qa_pair question"
        )
        raw_aliases = pair.get("short_answers")
        if not isinstance(raw_aliases, list):
            raise Goal1DataError(f"ASQA {query_id} qa_pair aliases must be a list")
        aliases = list(
            dict.fromkeys(
                alias.strip()
                for alias in raw_aliases
                if isinstance(alias, str) and alias.strip()
            )
        )
        if not aliases:
            raise Goal1DataError(f"ASQA {query_id} qa_pair has no answer aliases")
        fact_groups.append(
            {
                "fact_id": fact_id,
                "fact_question": fact_question,
                "aliases": aliases,
            }
        )
        wikipage = pair.get("wikipage")
        if isinstance(wikipage, str) and wikipage.strip():
            provenance.append(
                {
                    "fact_id": fact_id,
                    "source_kind": "wikipedia-title",
                    "source_id": wikipage.strip(),
                    "start_unit": None,
                    "end_unit": None,
                }
            )

    runtime = _runtime_record(
        dataset=dataset,
        query_id=query_id,
        question=question,
        corpus_snapshot_id=corpus_snapshot_id,
        bm25_index_id=bm25_index_id,
        dense_index_id=dense_index_id,
    )
    sidecar = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "dataset": dataset,
        "query_id": query_id,
        "reference_fact_groups": fact_groups,
        "gold_provenance": provenance,
    }
    validate_runtime_record(runtime, dataset=dataset)
    validate_sidecar_record(sidecar, dataset=dataset)
    return runtime, sidecar


def _runtime_record(
    *,
    dataset: str,
    query_id: str,
    question: str,
    corpus_snapshot_id: str,
    bm25_index_id: str,
    dense_index_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "dataset": dataset,
        "query_id": query_id,
        "question": question,
        "corpus_snapshot_id": _require_nonblank_string(
            corpus_snapshot_id, "corpus_snapshot_id"
        ),
        "bm25_index_id": _require_nonblank_string(bm25_index_id, "bm25_index_id"),
        "dense_index_id": _require_nonblank_string(dense_index_id, "dense_index_id"),
    }


def scan_exposure_paths(
    paths: Sequence[Path], *, canonical_ids: Collection[str]
) -> ExposureRegistry:
    canonical = set(canonical_ids)
    query_keys = {
        "base_query_id",
        "case_id",
        "case_ids",
        "ordered_ids",
        "ordered_query_ids",
        "query_id",
        "query_ids",
        "sample_id",
        "sample_ids",
        "source_query_id",
        "source_query_ids",
        "task_id",
        "task_ids",
    }

    def values_for_keys(value: object) -> set[str]:
        found: set[str] = set()
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key) in query_keys:
                    if isinstance(child, str):
                        found.add(child)
                    elif isinstance(child, list):
                        found.update(item for item in child if isinstance(item, str))
                found.update(values_for_keys(child))
        elif isinstance(value, list):
            for child in value:
                found.update(values_for_keys(child))
        return found

    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix in {".json", ".jsonl"}
            )
        elif path.is_file() and path.suffix in {".json", ".jsonl"}:
            files.append(path)
    unique_files = sorted(set(files), key=lambda path: str(path))

    all_matches: set[str] = set()
    source_rows: list[dict[str, object]] = []
    for path in unique_files:
        matches: set[str] = set()
        try:
            if path.suffix == ".jsonl":
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            matches.update(values_for_keys(json.loads(line)))
            else:
                raw_text = path.read_text(encoding="utf-8")
                try:
                    matches.update(values_for_keys(json.loads(raw_text)))
                except json.JSONDecodeError as whole_file_error:
                    if "Extra data" not in str(whole_file_error):
                        raise
                    for line in raw_text.splitlines():
                        if line.strip():
                            matches.update(values_for_keys(json.loads(line)))
        except (OSError, json.JSONDecodeError) as exc:
            raise Goal1DataError(f"cannot audit exposure source {path}: {exc}") from exc
        matches.intersection_update(canonical)
        all_matches.update(matches)
        source_rows.append(
            {
                "path": str(path),
                "sha256": _file_sha256(path),
                "matched_count": len(matches),
                "matched_ids_sha256": _ordered_hash(sorted(matches)),
            }
        )
    return ExposureRegistry(
        exposure_ids=tuple(sorted(all_matches)),
        sources=tuple(source_rows),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()
