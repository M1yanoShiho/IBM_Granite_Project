"""Selector-only experiment helpers over one frozen candidate-pool artifact.

The Retriever is deliberately absent from this module.  A caller supplies one
``candidate_sets.jsonl`` file, both Selector arms consume those exact records,
and every output manifest records the file's SHA-256.  This keeps the
experiment's only method variable at the Selector stage.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from evidence_rag.contracts.models import (
    CandidateSet,
    Query,
    SelectedEvidenceSet,
    SelectionResult,
)
from evidence_rag.contracts.protocols import Selector
from evidence_rag.contracts.validation import resolve_selection
from evidence_rag.evaluation.paired_metric import PairedComparison, compare_paired
from evidence_rag.evaluation.stage_evaluators import evaluate_selector_stage
from evidence_rag.infrastructure.datasets import DatasetBundle, GoldCase
from evidence_rag.materializer.source_parent import ParentIndex

ModelT = TypeVar("ModelT", bound=BaseModel)
RecordT = TypeVar("RecordT")
EventLookup = Callable[[str], object | None]


def sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path, model_type: type[ModelT]) -> tuple[ModelT, ...]:
    records: list[ModelT] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(model_type.model_validate_json(line))
        except ValueError as error:
            raise ValueError(f"invalid JSONL record at {path}:{line_number}: {error}") from error
    return tuple(records)


def _canonical_json(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(_canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, records: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_canonical_json(record) + "\n")
    temporary.replace(path)


def _git_state(root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ("git", "-C", str(root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "-C", str(root), "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "git-unavailable", False
    return commit or "git-unavailable", bool(status.strip())


def _unique_by_query(
    records: Sequence[RecordT],
    label: str,
    query_id: Callable[[RecordT], str],
) -> dict[str, RecordT]:
    indexed: dict[str, RecordT] = {}
    for record in records:
        identifier = query_id(record)
        if identifier in indexed:
            raise ValueError(f"duplicate {label} query ID: {identifier}")
        indexed[identifier] = record
    return indexed


def align_inputs(
    bundle: DatasetBundle,
    candidate_sets: Sequence[CandidateSet],
) -> tuple[tuple[Query, CandidateSet, GoldCase], ...]:
    candidates = _unique_by_query(candidate_sets, "candidate set", lambda item: item.query_id)
    gold = _unique_by_query(bundle.gold_cases, "gold case", lambda item: item.query_id)
    expected = tuple(query.query_id for query in bundle.queries)
    if set(candidates) != set(expected):
        missing = sorted(set(expected) - set(candidates))
        extra = sorted(set(candidates) - set(expected))
        raise ValueError(f"candidate/query IDs differ: missing={missing[:1]}, extra={extra[:1]}")
    if set(gold) != set(expected):
        raise ValueError("gold/query IDs differ")
    return tuple(
        (query, candidates[query.query_id], gold[query.query_id])
        for query in bundle.queries
    )


def _sole_retriever(candidate_sets: Sequence[CandidateSet]) -> dict[str, object]:
    provenances = {candidate.retriever for candidate in candidate_sets}
    if None in provenances:
        raise ValueError("candidate pool has missing Retriever provenance")
    if len(provenances) != 1:
        raise ValueError("candidate pool contains more than one Retriever provenance")
    provenance = next(iter(provenances))
    if provenance is None:  # narrowed above; keeps type checkers honest
        raise AssertionError("unreachable missing Retriever provenance")
    return provenance.model_dump(mode="json")


def audit_candidate_pool(
    bundle: DatasetBundle,
    candidate_sets: Sequence[CandidateSet],
    parent_index: ParentIndex,
    *,
    candidate_pool_path: Path,
    dataset_manifest_path: Path,
    retriever_config_path: Path,
    top_n: int,
    seed: int,
    harm_by_query: Mapping[str, str] | None = None,
    embedding_model_id: str,
    embedding_revision: str,
    git_root: Path,
) -> dict[str, object]:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    aligned = align_inputs(bundle, candidate_sets)
    exact_count = sum(len(candidates.candidates) == top_n for _, candidates, _ in aligned)
    required_values: list[float] = []
    unresolved_ids: set[str] = set()
    harmful_hits = 0
    harmful_total = 0
    for _query, candidates, gold in aligned:
        candidate_documents = {item.document_id for item in candidates.candidates}
        relevant = set(gold.relevant_document_ids or ())
        if relevant:
            required_values.append(len(candidate_documents & relevant) / len(relevant))
        unresolved_ids.update(
            item.document_id
            for item in candidates.candidates
            if item.document_id not in parent_index.parent_by_document
        )
        if harm_by_query is not None and candidates.query_id in harm_by_query:
            harmful_total += 1
            harmful_hits += int(harm_by_query[candidates.query_id] in candidate_documents)

    commit, dirty = _git_state(git_root)
    candidate_sha = sha256_file(candidate_pool_path)
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "dataset": {
            "id": bundle.manifest.dataset_id,
            "version": bundle.manifest.dataset_version,
            "split": bundle.manifest.split,
            "signature": bundle.dataset_signature,
            "manifest_path": str(dataset_manifest_path.resolve()),
            "manifest_sha256": sha256_file(dataset_manifest_path),
        },
        "candidate_pool": {
            "path": str(candidate_pool_path.resolve()),
            "sha256": candidate_sha,
            "query_count": len(aligned),
            "top_n": top_n,
            "exact_top_n_count": exact_count,
            "exact_top_n_rate": exact_count / len(aligned),
        },
        "audit": {
            "required_recall_at_top_n": (
                None if not required_values else sum(required_values) / len(required_values)
            ),
            "required_recall_n": len(required_values),
            "harmful_pool_hit_rate": (
                None if harmful_total == 0 else harmful_hits / harmful_total
            ),
            "harmful_pool_hit_count": harmful_hits,
            "harmful_label_count": harmful_total,
            "unresolved_parent_count": len(unresolved_ids),
            "unresolved_parent_ids": sorted(unresolved_ids),
        },
        "retriever": _sole_retriever(candidate_sets),
        "retriever_config": {
            "path": str(retriever_config_path.resolve()),
            "sha256": sha256_file(retriever_config_path),
        },
        "embedding": {
            "model_id": embedding_model_id,
            "revision": embedding_revision,
        },
        "seed": seed,
        "git_commit": commit,
        "git_dirty": dirty,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    return manifest


def validate_pool_manifest(
    manifest: Mapping[str, object],
    *,
    candidate_pool_path: Path,
    expected_top_n: int,
) -> str:
    pool = manifest.get("candidate_pool")
    if not isinstance(pool, Mapping):
        raise ValueError("candidate pool manifest has no candidate_pool section")
    recorded_sha = pool.get("sha256")
    actual_sha = sha256_file(candidate_pool_path)
    if recorded_sha != actual_sha:
        raise ValueError("candidate pool SHA-256 differs from the frozen manifest")
    if pool.get("top_n") != expected_top_n:
        raise ValueError("candidate pool top_n differs from the requested comparison")
    if pool.get("exact_top_n_rate") != 1.0:
        raise ValueError("candidate pool does not contain exactly top_n items for every query")
    audit = manifest.get("audit")
    if not isinstance(audit, Mapping) or audit.get("unresolved_parent_count") != 0:
        raise ValueError("candidate pool contains unresolved source parents")
    return actual_sha


def _event_json(event: object | None) -> object | None:
    if event is None:
        return None
    if is_dataclass(event) and not isinstance(event, type):
        return asdict(event)
    if isinstance(event, BaseModel):
        return event.model_dump(mode="json")
    if isinstance(event, Mapping):
        return dict(event)
    raise TypeError("Selector diagnostic event must be a dataclass, Pydantic model, or mapping")


def _case_metrics(
    selected: SelectedEvidenceSet,
    candidates: CandidateSet,
    gold: GoldCase,
    harmful_document_id: str | None,
) -> dict[str, object]:
    selected_documents = {item.document_id for item in selected.evidence}
    candidate_documents = {item.document_id for item in candidates.candidates}
    relevant = set(gold.relevant_document_ids or ())
    required_recall = (
        None if not relevant else len(selected_documents & relevant) / len(relevant)
    )
    evidence_precision = (
        None
        if not relevant
        else (0.0 if not selected_documents else len(selected_documents & relevant) / len(selected_documents))
    )
    harmful_pool_hit = (
        None if harmful_document_id is None else harmful_document_id in candidate_documents
    )
    harmful_selected = (
        None if harmful_document_id is None else harmful_document_id in selected_documents
    )
    return {
        "candidate_ids": [item.evidence_id for item in candidates.candidates],
        "candidate_ranks": [item.retrieval_rank for item in candidates.candidates],
        "selected_ids": [item.evidence_id for item in selected.evidence],
        "required_document_ids": sorted(relevant),
        "harmful_document_id": harmful_document_id,
        "harmful_pool_hit": harmful_pool_hit,
        "harmful_selected": harmful_selected,
        "required_evidence_recall": required_recall,
        "evidence_precision": evidence_precision,
        "selected_evidence_count": len(selected.evidence),
    }


def _load_checkpoint(
    path: Path,
    expected: Sequence[tuple[Query, CandidateSet, GoldCase]],
    *,
    method: str,
    candidate_pool_sha256: str,
) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid checkpoint line {path}:{line_number}") from error
        index = len(records)
        if index >= len(expected) or row.get("query_id") != expected[index][0].query_id:
            raise ValueError("checkpoint query order does not match the frozen input")
        if row.get("method") != method or row.get("candidate_pool_sha256") != candidate_pool_sha256:
            raise ValueError("checkpoint identity differs from the requested run")
        SelectionResult.model_validate(row.get("selection"))
        SelectedEvidenceSet.model_validate(row.get("selected"))
        records.append(row)
    return records


def _append_checkpoint(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(dict(record)) + "\n")
        handle.flush()


def _numeric(value: object) -> float:
    if not isinstance(value, (bool, int, float)):
        raise TypeError(f"metric value must be numeric, found {type(value).__name__}")
    return float(value)


def _mean(rows: Sequence[Mapping[str, object]], key: str) -> float | None:
    values = [_numeric(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else sum(values) / len(values)


def _arm_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    conditional_harm = [
        float(bool(row["harmful_selected"]))
        for row in rows
        if row.get("harmful_pool_hit") is True
    ]
    unconditional_harm = [
        float(bool(row["harmful_selected"]))
        for row in rows
        if row.get("harmful_selected") is not None
    ]
    return {
        "harmful_in_context_conditional": (
            None if not conditional_harm else sum(conditional_harm) / len(conditional_harm)
        ),
        "harmful_in_context_conditional_n": len(conditional_harm),
        "harmful_in_context_unconditional": (
            None if not unconditional_harm else sum(unconditional_harm) / len(unconditional_harm)
        ),
        "harmful_in_context_unconditional_n": len(unconditional_harm),
        "required_evidence_recall": _mean(rows, "required_evidence_recall"),
        "evidence_precision": _mean(rows, "evidence_precision"),
        "selected_evidence_count": _mean(rows, "selected_evidence_count"),
    }


def _comparison_json(comparison: PairedComparison) -> dict[str, object]:
    return {
        "mean_mis": comparison.mean_on,
        "mean_top_k": comparison.mean_off,
        "delta_mis_minus_top_k": comparison.delta,
        "p_value": comparison.p_value,
        "ci_low": comparison.ci_low,
        "ci_high": comparison.ci_high,
        "n_paired": comparison.n_paired,
    }


def _paired_values(
    mis_rows: Sequence[Mapping[str, object]],
    top_k_rows: Sequence[Mapping[str, object]],
    key: str,
    *,
    condition: Callable[[Mapping[str, object]], bool] | None = None,
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    top_by_query = {str(row["query_id"]): row for row in top_k_rows}
    mis_values: dict[str, float | None] = {}
    top_values: dict[str, float | None] = {}
    for row in mis_rows:
        query_id = str(row["query_id"])
        top = top_by_query.get(query_id)
        if top is None:
            raise ValueError(f"TopK output is missing query {query_id}")
        if condition is not None and (not condition(row) or not condition(top)):
            mis_values[query_id] = None
            top_values[query_id] = None
            continue
        mis_value = row.get(key)
        top_value = top.get(key)
        mis_values[query_id] = None if mis_value is None else _numeric(mis_value)
        top_values[query_id] = None if top_value is None else _numeric(top_value)
    return mis_values, top_values


def run_selector_arm(
    method: str,
    selector: Selector,
    aligned: Sequence[tuple[Query, CandidateSet, GoldCase]],
    *,
    output_directory: Path,
    candidate_pool_sha256: str,
    source_parent_sha256: str,
    max_selected: int,
    run_seed: int,
    dataset_signature: str,
    harm_by_query: Mapping[str, str] | None = None,
    event_lookup: EventLookup | None = None,
    resume: bool = False,
    git_root: Path,
) -> tuple[list[dict[str, object]], tuple[SelectionResult, ...], tuple[SelectedEvidenceSet, ...]]:
    if max_selected <= 0:
        raise ValueError("max_selected must be positive")
    output_directory = Path(output_directory)
    checkpoint = output_directory / "checkpoint.jsonl"
    if checkpoint.exists() and not resume:
        raise ValueError(f"checkpoint already exists for {method}; use --resume")
    completed = _load_checkpoint(
        checkpoint,
        aligned,
        method=method,
        candidate_pool_sha256=candidate_pool_sha256,
    ) if resume else []

    commit, dirty = _git_state(git_root)
    started = datetime.now(UTC).isoformat()
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "status": "running",
        "method": method,
        "candidate_pool_sha256": candidate_pool_sha256,
        "source_parent_sha256": source_parent_sha256,
        "query_count": len(aligned),
        "max_selected": max_selected,
        "seed": run_seed,
        "git_commit": commit,
        "git_dirty": dirty,
        "started_at_utc": started,
    }
    _write_json(output_directory / "run_manifest.json", manifest)

    for index in range(len(completed), len(aligned)):
        query, candidates, gold = aligned[index]
        selection = selector.select(query, candidates, max_selected)
        if selection.query_id != query.query_id:
            raise ValueError("Selector returned the wrong query ID")
        if len(selection.items) > max_selected:
            raise ValueError("Selector returned more than max_selected evidence items")
        selected = resolve_selection(candidates, selection)
        selected_ids = {item.evidence_id for item in selected.evidence}
        candidate_ids = {item.evidence_id for item in candidates.candidates}
        if not selected_ids <= candidate_ids:
            raise ValueError("Selector chose evidence outside the frozen candidate pool")
        metrics = _case_metrics(
            selected,
            candidates,
            gold,
            None if harm_by_query is None else harm_by_query.get(query.query_id),
        )
        row: dict[str, object] = {
            "schema_version": "1.0",
            "query_id": query.query_id,
            "method": method,
            "candidate_pool_sha256": candidate_pool_sha256,
            "selection": selection.model_dump(mode="json"),
            "selected": selected.model_dump(mode="json"),
            "metrics": metrics,
            "diagnostic": _event_json(
                None if event_lookup is None else event_lookup(query.query_id)
            ),
        }
        _append_checkpoint(checkpoint, row)
        completed.append(row)
        if (index + 1) % 10 == 0 or index + 1 == len(aligned):
            print(f"[{method}] {index + 1}/{len(aligned)}", flush=True)

    selections = tuple(SelectionResult.model_validate(row["selection"]) for row in completed)
    selected_sets = tuple(
        SelectedEvidenceSet.model_validate(row["selected"]) for row in completed
    )
    report = evaluate_selector_stage(
        tuple(item[1] for item in aligned),
        selected_sets,
        tuple(item[2] for item in aligned),
        dataset_signature=dataset_signature,
    )
    per_query: list[dict[str, object]] = []
    for row in completed:
        checkpoint_metrics = row.get("metrics")
        if not isinstance(checkpoint_metrics, Mapping):
            raise ValueError("checkpoint record has invalid metrics")
        per_query.append({"query_id": str(row["query_id"]), **dict(checkpoint_metrics)})
    _write_jsonl(output_directory / "selection_results.jsonl", selections)
    _write_jsonl(output_directory / "selected_evidence_sets.jsonl", selected_sets)
    _write_json(output_directory / "selector_report.json", report)
    _write_jsonl(output_directory / "per_query.jsonl", per_query)
    diagnostics = [
        {"query_id": row["query_id"], "diagnostic": row["diagnostic"]}
        for row in completed
        if row.get("diagnostic") is not None
    ]
    if diagnostics:
        _write_jsonl(output_directory / "selection_events.jsonl", diagnostics)

    files = (
        "selection_results.jsonl",
        "selected_evidence_sets.jsonl",
        "selector_report.json",
        "per_query.jsonl",
    )
    manifest.update(
        {
            "status": "complete",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "output_sha256": {
                name: sha256_file(output_directory / name) for name in files
            },
        }
    )
    _write_json(output_directory / "run_manifest.json", manifest)
    return per_query, selections, selected_sets


def compare_arm_results(
    top_k_rows: Sequence[Mapping[str, object]],
    mis_rows: Sequence[Mapping[str, object]],
    *,
    stats_seed: int,
    iterations: int,
) -> dict[str, object]:
    required_mis, required_top = _paired_values(
        mis_rows, top_k_rows, "required_evidence_recall"
    )
    required = compare_paired(
        required_mis,
        required_top,
        seed=stats_seed,
        iterations=iterations,
    )
    harm: PairedComparison | None = None
    if any(row.get("harmful_selected") is not None for row in mis_rows):
        harm_mis, harm_top = _paired_values(
            mis_rows,
            top_k_rows,
            "harmful_selected",
            condition=lambda row: row.get("harmful_pool_hit") is True,
        )
        harm = compare_paired(harm_mis, harm_top, seed=stats_seed, iterations=iterations)
    return {
        "schema_version": "1.0",
        "statistical_unit": "query",
        "bootstrap_iterations": iterations,
        "stats_seed": stats_seed,
        "top_k": _arm_summary(top_k_rows),
        "reliability_mis": _arm_summary(mis_rows),
        "paired_harm": None if harm is None else _comparison_json(harm),
        "paired_required_recall": _comparison_json(required),
        "decision": {
            "harm_reduction_pass": None if harm is None else harm.ci_high < 0.0,
            "required_recall_noninferiority_pass": required.ci_low > -0.01,
            "noninferiority_margin": -0.01,
        },
    }


def write_comparison_outputs(
    output_directory: Path,
    top_k_rows: Sequence[Mapping[str, object]],
    mis_rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
) -> None:
    top_by_query = {str(row["query_id"]): row for row in top_k_rows}
    combined: list[dict[str, object]] = []
    for mis in mis_rows:
        query_id = str(mis["query_id"])
        top = top_by_query.get(query_id)
        if top is None:
            raise ValueError(f"TopK output is missing query {query_id}")
        combined.append(
            {
                "schema_version": "1.0",
                "query_id": query_id,
                "top_k": dict(top),
                "reliability_mis": dict(mis),
            }
        )
    _write_jsonl(Path(output_directory) / "per_query_comparison.jsonl", combined)
    _write_json(Path(output_directory) / "comparison_summary.json", dict(summary))
