"""Frozen TopK quantity baselines and future count-matched control protocol.

R003 has two deliberately separate products:

* TopK10/9/8/7 are evaluated in one pass over the same frozen Hybrid Top20 pool.
  Every smaller context is a prefix of TopK10 (``S0``); nothing from ranks 11--20
  may be promoted as a replacement.
* The count-matched random/bottom-rank *generator protocol* is frozen before a
  learned Selector trace exists.  It can choose drops for a supplied deletion
  count, but this module does not invent a trace or publish fake control results.

All reported recall and chain metrics operate on document IDs.  Candidate IDs are
retained in the trace so candidate-level deletion counts remain auditable.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate
from evidence_rag.evaluation.paired_metric import PairedComparison, compare_paired
from evidence_rag.evaluation.selector_components import (
    MANIFEST_FILE as COMPONENT_MANIFEST_FILE,
)
from evidence_rag.evaluation.selector_components import (
    OUTPUT_FILES as COMPONENT_OUTPUT_FILES,
)
from evidence_rag.evaluation.selector_components import (
    ROLE_ASSIGNMENTS_FILE,
    DatasetKind,
    NiahSelectorAssignment,
    SourceSplit,
    verify_selector_component_artifacts,
)
from evidence_rag.evaluation.selector_risk import (
    conditional_chain_loss,
    deletion_precision,
    document_recall,
    relative_recall_loss,
    required_deletion_rate,
)
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.selector.top_k import TopKSelector

TOP_K_VALUES = (10, 9, 8, 7)
TOP_K_BASELINE = 10
EXPECTED_POOL_SIZE = 20
TOPK_PROTOCOL_VERSION = "selector-topk-controls-v1"
PAIRED_INFERENCE_SEED = 13
PAIRED_INFERENCE_ITERATIONS = 10_000
TOPK_ROWS_FILE = "topk_controls.jsonl"
TOPK_REPORT_FILE = "topk_controls_report.json"
TOPK_MANIFEST_FILE = "topk_controls_manifest.json"
TOPK_OUTPUT_FILES = (TOPK_ROWS_FILE, TOPK_REPORT_FILE, TOPK_MANIFEST_FILE)

COUNT_MATCHED_PROTOCOL_VERSION = "selector-count-matched-v1"
COUNT_MATCHED_MASTER_SEED = 20260811
COUNT_MATCHED_REPEATS = 100
COUNT_MATCHED_MAX_DELETIONS = 3
COUNT_MATCHED_PROTOCOL_FILE = "count_matched_protocol.json"
COUNT_MATCHED_MANIFEST_FILE = "count_matched_protocol_manifest.json"
COUNT_MATCHED_OUTPUT_FILES = (
    COUNT_MATCHED_PROTOCOL_FILE,
    COUNT_MATCHED_MANIFEST_FILE,
)


@dataclass(frozen=True)
class TopKControlArtifacts:
    """Canonical TopK trace/report bytes and their parsed summaries."""

    files: Mapping[str, bytes]
    report: Mapping[str, object]
    manifest: Mapping[str, object]


@dataclass(frozen=True)
class CountMatchedProtocolArtifacts:
    """Canonical generator protocol bytes; no Selector outcomes are included."""

    files: Mapping[str, bytes]
    protocol: Mapping[str, object]
    manifest: Mapping[str, object]


@dataclass(frozen=True)
class CountMatchedDrops:
    """One deterministic count-matched choice for a future real trace row."""

    deletion_count: int
    random_dropped_evidence_ids: tuple[str, ...]
    bottom_rank_dropped_evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class _RoleBinding:
    dataset_id: str
    component_id: str
    role: str
    chain_eligible_topk10: bool


@dataclass(frozen=True)
class _Observation:
    query_id: str
    component_id: str
    role: str
    k: int
    selected_evidence_ids: tuple[str, ...]
    selected_document_ids: tuple[str, ...]
    dropped_evidence_ids: tuple[str, ...]
    dropped_document_ids: tuple[str, ...]
    required_document_ids: tuple[str, ...]
    recall: float
    relative_recall_loss: float
    chain_eligible_topk10: bool
    conditional_chain_loss: float | None
    required_deletion_rate: float | None
    harmful_document_id: str | None = None
    harmful_in_pool: bool | None = None
    harmful_in_topk10: bool | None = None
    pool_conditional_harmful_exposure: float | None = None
    pool_conditional_harmful_reduction: float | None = None
    baseline_exposed_harmful_deletion: float | None = None
    unconditional_harmful_exposure: float | None = None
    deletion_precision: float | None = None

    def as_row(self, *, dataset_id: str, dataset_kind: DatasetKind) -> dict[str, object]:
        row: dict[str, object] = {
            "schema_version": "1.0",
            "protocol_version": TOPK_PROTOCOL_VERSION,
            "dataset_id": dataset_id,
            "dataset_kind": dataset_kind,
            "query_id": self.query_id,
            "component_id": self.component_id,
            "role": self.role,
            "system": f"topk{self.k}",
            "k": self.k,
            "selected_evidence_ids": list(self.selected_evidence_ids),
            "selected_document_ids": list(self.selected_document_ids),
            "dropped_evidence_ids": list(self.dropped_evidence_ids),
            "dropped_document_ids": list(self.dropped_document_ids),
            "required_document_ids": list(self.required_document_ids),
            "n_selected_candidates": len(self.selected_evidence_ids),
            "n_selected_documents": len(self.selected_document_ids),
            "n_dropped_candidates": len(self.dropped_evidence_ids),
            "document_recall": self.recall,
            "topk10_relative_recall_loss": self.relative_recall_loss,
            "chain_eligible_topk10": self.chain_eligible_topk10,
            "conditional_chain_loss": self.conditional_chain_loss,
            "required_deletion_rate": self.required_deletion_rate,
        }
        if dataset_kind == "niah":
            row.update(
                {
                    "harmful_document_id": self.harmful_document_id,
                    "harmful_in_top20_pool": self.harmful_in_pool,
                    "harmful_in_topk10": self.harmful_in_topk10,
                    "pool_conditional_harmful_exposure": (self.pool_conditional_harmful_exposure),
                    "pool_conditional_harmful_reduction": (self.pool_conditional_harmful_reduction),
                    "baseline_exposed_harmful_deletion": (self.baseline_exposed_harmful_deletion),
                    "unconditional_harmful_exposure": (self.unconditional_harmful_exposure),
                    "deletion_precision": self.deletion_precision,
                }
            )
        return row


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _bytes_pin(payload: bytes, *, records: int | None = None) -> dict[str, object]:
    pin: dict[str, object] = {"sha256": _sha256_bytes(payload), "bytes": len(payload)}
    if records is not None:
        pin["records"] = records
    return pin


def _file_pin(path: Path) -> dict[str, object]:
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise ValueError(f"unable to read input artifact {path}: {error}") from error
    return _bytes_pin(payload)


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    lines = [
        json.dumps(
            row,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        for row in rows
    ]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _read_jsonl_bytes(payload: bytes, *, label: str) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8: {error}") from error
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank {label} record at line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid {label} JSON at line {line_number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{label} record at line {line_number} must be an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{label} is empty")
    return tuple(rows)


def _nonblank(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-blank string")
    return value


def _read_role_bindings(payload: bytes) -> dict[str, _RoleBinding]:
    bindings: dict[str, _RoleBinding] = {}
    for row in _read_jsonl_bytes(payload, label="R002 role assignment"):
        query_id = _nonblank(row.get("query_id"), label="role query_id")
        dataset_id = _nonblank(row.get("dataset_id"), label=f"role dataset_id for {query_id}")
        component_id = _nonblank(row.get("component_id"), label=f"role component_id for {query_id}")
        role = _nonblank(row.get("role"), label=f"role name for {query_id}")
        eligible = row.get("chain_eligible_topk10")
        if not isinstance(eligible, bool):
            raise ValueError(f"chain_eligible_topk10 for {query_id} must be boolean")
        if query_id in bindings:
            raise ValueError(f"duplicate R002 role query ID: {query_id}")
        bindings[query_id] = _RoleBinding(dataset_id, component_id, role, eligible)
    if len({binding.dataset_id for binding in bindings.values()}) != 1:
        raise ValueError("R002 role assignments contain more than one dataset_id")
    return bindings


def _candidate_file(candidate_pool_path: Path) -> Path:
    path = Path(candidate_pool_path)
    return path / "candidate_sets.jsonl" if path.is_dir() else path


def _read_candidate_sets(path: Path) -> dict[str, CandidateSet]:
    values: dict[str, CandidateSet] = {}
    try:
        lines = Path(path).read_bytes().splitlines()
    except OSError as error:
        raise ValueError(f"unable to read candidate pool {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"blank candidate-pool record at {path}:{line_number}")
        try:
            candidate_set = CandidateSet.model_validate_json(line)
        except (ValidationError, ValueError) as error:
            raise ValueError(f"invalid candidate set at {path}:{line_number}: {error}") from error
        if candidate_set.query_id in values:
            raise ValueError(f"duplicate candidate query ID: {candidate_set.query_id}")
        values[candidate_set.query_id] = candidate_set
    if not values:
        raise ValueError(f"candidate pool {path} is empty")
    return values


def _read_niah_assignments(path: Path) -> dict[str, NiahSelectorAssignment]:
    assignments: dict[str, NiahSelectorAssignment] = {}
    try:
        lines = Path(path).read_bytes().splitlines()
    except OSError as error:
        raise ValueError(f"unable to read NIAH assignment {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"blank NIAH assignment record at {path}:{line_number}")
        try:
            assignment = NiahSelectorAssignment.model_validate_json(line)
        except (ValidationError, ValueError) as error:
            raise ValueError(f"invalid NIAH assignment at {path}:{line_number}: {error}") from error
        if assignment.query_id in assignments:
            raise ValueError(f"duplicate NIAH assignment query ID: {assignment.query_id}")
        assignments[assignment.query_id] = assignment
    if not assignments:
        raise ValueError(f"NIAH assignment {path} is empty")
    return assignments


def _same_keys(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    left_label: str,
    right_label: str,
) -> None:
    left_keys = set(left)
    right_keys = set(right)
    if left_keys != right_keys:
        raise ValueError(
            f"{left_label}/{right_label} keys differ: "
            f"missing from {right_label}={sorted(left_keys - right_keys)[:5]}, "
            f"missing from {left_label}={sorted(right_keys - left_keys)[:5]}"
        )


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    result = sum(values) / len(values)
    if not math.isfinite(result):
        raise AssertionError("aggregate metric unexpectedly became non-finite")
    return result


def _aggregate(
    observations: Sequence[_Observation], *, dataset_kind: DatasetKind
) -> dict[str, object]:
    if not observations:
        raise ValueError("cannot aggregate an empty TopK observation set")
    k_values = {observation.k for observation in observations}
    if len(k_values) != 1:
        raise ValueError("one TopK aggregate must contain exactly one k")
    chain_values = [
        observation.conditional_chain_loss
        for observation in observations
        if observation.conditional_chain_loss is not None
    ]
    required_rates = [
        observation.required_deletion_rate
        for observation in observations
        if observation.required_deletion_rate is not None
    ]
    result: dict[str, object] = {
        "k": next(iter(k_values)),
        "n_queries": len(observations),
        "n_components": len({observation.component_id for observation in observations}),
        "mean_selected_candidates": _mean(
            [float(len(observation.selected_evidence_ids)) for observation in observations]
        ),
        "mean_selected_documents": _mean(
            [float(len(observation.selected_document_ids)) for observation in observations]
        ),
        "mean_dropped_candidates": _mean(
            [float(len(observation.dropped_evidence_ids)) for observation in observations]
        ),
        "document_recall": _mean([observation.recall for observation in observations]),
        "topk10_relative_recall_loss": _mean(
            [observation.relative_recall_loss for observation in observations]
        ),
        "conditional_chain": {
            "n_topk10_chain_eligible": len(chain_values),
            "n_failures": int(sum(chain_values)),
            "loss": _mean(chain_values),
            "survival": (None if not chain_values else 1.0 - cast(float, _mean(chain_values))),
        },
        "required_deletion_rate": {
            "n_queries_with_deletions": len(required_rates),
            "mean_query_rate": _mean(required_rates),
            "candidate_action_rate": (
                None
                if not required_rates
                else sum(
                    document_id in set(observation.required_document_ids)
                    for observation in observations
                    for document_id in observation.dropped_document_ids
                )
                / sum(len(observation.dropped_document_ids) for observation in observations)
            ),
        },
    }
    if dataset_kind == "niah":
        pool_eligible = [observation for observation in observations if observation.harmful_in_pool]
        baseline_exposed = [
            observation for observation in observations if observation.harmful_in_topk10
        ]
        pool_exposure = [
            cast(float, observation.pool_conditional_harmful_exposure)
            for observation in pool_eligible
        ]
        pool_reduction = [
            cast(float, observation.pool_conditional_harmful_reduction)
            for observation in pool_eligible
        ]
        baseline_deletion = [
            cast(float, observation.baseline_exposed_harmful_deletion)
            for observation in baseline_exposed
        ]
        unconditional = [
            cast(float, observation.unconditional_harmful_exposure) for observation in observations
        ]
        dropped_count = sum(len(observation.dropped_document_ids) for observation in observations)
        harmful_drop_count = sum(
            document_id == observation.harmful_document_id
            for observation in observations
            for document_id in observation.dropped_document_ids
        )
        result["harmful"] = {
            "denominators": {
                "pool_conditional": len(pool_eligible),
                "baseline_exposed": len(baseline_exposed),
                "unconditional": len(observations),
            },
            "pool_conditional_exposure": _mean(pool_exposure),
            "pool_conditional_reduction_from_topk10": _mean(pool_reduction),
            "baseline_exposed_deletion": _mean(baseline_deletion),
            "unconditional_exposure": _mean(unconditional),
            "deletion_precision": {
                "n_dropped_candidate_actions": dropped_count,
                "n_harmful_dropped_candidate_actions": harmful_drop_count,
                "candidate_action_precision": (
                    None if dropped_count == 0 else harmful_drop_count / dropped_count
                ),
            },
        }
    return result


def _aggregates_by_system(
    observations: Sequence[_Observation], *, dataset_kind: DatasetKind
) -> dict[str, object]:
    return {
        f"topk{k}": _aggregate(
            [observation for observation in observations if observation.k == k],
            dataset_kind=dataset_kind,
        )
        for k in TOP_K_VALUES
    }


def _paired_result(
    on: Mapping[str, float | None],
    off: Mapping[str, float | None],
    *,
    component_ids: Mapping[str, str],
    contrast: str,
    positive_direction: str,
) -> dict[str, object]:
    """Run the frozen strict paired test, or state why an empty mask is not estimable."""

    if set(on) != set(off) or set(on) != set(component_ids):
        raise ValueError("paired metric, baseline and component query keys must match exactly")
    if any((on[query_id] is None) != (off[query_id] is None) for query_id in on):
        raise ValueError("paired metric baseline/system masks must match exactly")
    eligible = [query_id for query_id in on if on[query_id] is not None]
    if not eligible:
        return {
            "status": "NOT_ESTIMABLE_NO_ELIGIBLE_QUERIES",
            "contrast": contrast,
            "positive_direction": positive_direction,
            "seed": PAIRED_INFERENCE_SEED,
            "iterations": PAIRED_INFERENCE_ITERATIONS,
            "n_queries": 0,
            "n_clusters": 0,
            "n_total": len(on),
            "n_unscored": len(on),
        }
    comparison: PairedComparison = compare_paired(
        on,
        off,
        component_ids=component_ids,
        seed=PAIRED_INFERENCE_SEED,
        iterations=PAIRED_INFERENCE_ITERATIONS,
    )
    return {
        "status": "ESTIMATED",
        "contrast": contrast,
        "positive_direction": positive_direction,
        "seed": PAIRED_INFERENCE_SEED,
        "iterations": PAIRED_INFERENCE_ITERATIONS,
        **asdict(comparison),
    }


def _paired_comparisons(
    observations: Sequence[_Observation], *, dataset_kind: DatasetKind
) -> dict[str, object]:
    """TopK10-vs-smaller-K inference using complete components as resampling units."""

    by_query_and_k: dict[tuple[str, int], _Observation] = {}
    for observation in observations:
        key = (observation.query_id, observation.k)
        if key in by_query_and_k:
            raise ValueError(f"duplicate TopK observation: {key}")
        by_query_and_k[key] = observation
    query_ids = tuple(sorted({observation.query_id for observation in observations}))
    expected = {(query_id, k) for query_id in query_ids for k in TOP_K_VALUES}
    if set(by_query_and_k) != expected:
        raise ValueError("paired TopK trace does not contain one row per query and frozen k")
    component_ids = {
        query_id: by_query_and_k[(query_id, TOP_K_BASELINE)].component_id for query_id in query_ids
    }
    comparisons: dict[str, object] = {}
    for k in TOP_K_VALUES[1:]:
        baseline = {query_id: by_query_and_k[(query_id, TOP_K_BASELINE)] for query_id in query_ids}
        selected = {query_id: by_query_and_k[(query_id, k)] for query_id in query_ids}
        metrics: dict[str, object] = {
            "recall_loss": _paired_result(
                {query_id: value.recall for query_id, value in baseline.items()},
                {query_id: value.recall for query_id, value in selected.items()},
                component_ids=component_ids,
                contrast=f"topk10_document_recall - topk{k}_document_recall",
                positive_direction="loss (worse)",
            ),
            "conditional_chain_loss": _paired_result(
                {
                    query_id: (
                        None
                        if value.conditional_chain_loss is None
                        else 1.0 - value.conditional_chain_loss
                    )
                    for query_id, value in baseline.items()
                },
                {
                    query_id: (
                        None
                        if value.conditional_chain_loss is None
                        else 1.0 - value.conditional_chain_loss
                    )
                    for query_id, value in selected.items()
                },
                component_ids=component_ids,
                contrast=f"topk10_chain_survival - topk{k}_chain_survival",
                positive_direction="loss (worse)",
            ),
        }
        if dataset_kind == "niah":
            metrics["pool_conditional_harmful_reduction"] = _paired_result(
                {
                    query_id: (
                        (1.0 if value.harmful_in_topk10 else 0.0) if value.harmful_in_pool else None
                    )
                    for query_id, value in baseline.items()
                },
                {
                    query_id: value.pool_conditional_harmful_exposure
                    for query_id, value in selected.items()
                },
                component_ids=component_ids,
                contrast=f"topk10_harm_exposure - topk{k}_harm_exposure | harmful in Top20",
                positive_direction="harm reduction (better)",
            )
            metrics["baseline_exposed_harmful_deletion"] = _paired_result(
                {
                    query_id: 1.0 if value.harmful_in_topk10 else None
                    for query_id, value in baseline.items()
                },
                {
                    query_id: (
                        value.unconditional_harmful_exposure if value.harmful_in_topk10 else None
                    )
                    for query_id, value in selected.items()
                },
                component_ids=component_ids,
                contrast=f"topk10_harm_exposure - topk{k}_harm_exposure | harmful in TopK10",
                positive_direction="harm deletion (better)",
            )
            metrics["unconditional_harmful_reduction"] = _paired_result(
                {
                    query_id: 1.0 if value.harmful_in_topk10 else 0.0
                    for query_id, value in baseline.items()
                },
                {
                    query_id: value.unconditional_harmful_exposure
                    for query_id, value in selected.items()
                },
                component_ids=component_ids,
                contrast=f"topk10_harm_exposure - topk{k}_harm_exposure | all injected queries",
                positive_direction="harm reduction (better)",
            )
        comparisons[f"topk10_vs_topk{k}"] = metrics
    return comparisons


def build_topk_control_artifacts(
    *,
    dataset_kind: DatasetKind,
    source_split: SourceSplit,
    dataset_manifest_path: Path,
    source_parent_path: Path,
    candidate_pool_path: Path,
    component_directory: Path,
    assignment_path: Path | None = None,
) -> TopKControlArtifacts:
    """Recompute R002, scan one pool once, and build canonical TopK10/9/8/7 traces."""

    if source_split not in ("train", "dev"):
        raise ValueError(
            "R003 TopK controls are restricted to train/dev; sealed/heldout is forbidden"
        )
    if dataset_kind == "niah" and assignment_path is None:
        raise ValueError("NIAH TopK controls require an assignment JSONL")
    if dataset_kind == "2wiki" and assignment_path is not None:
        raise ValueError("2Wiki TopK controls must not receive a NIAH assignment JSONL")

    components = verify_selector_component_artifacts(
        output_directory=component_directory,
        dataset_kind=dataset_kind,
        source_split=source_split,
        dataset_manifest_path=dataset_manifest_path,
        source_parent_path=source_parent_path,
        candidate_pool_path=candidate_pool_path,
        assignment_path=assignment_path,
    )
    if (
        not bool(components.manifest.get("candidate_pool_manifest_verified"))
        and Path(candidate_pool_path).is_dir()
    ):
        raise ValueError("R002 did not verify the supplied candidate pool manifest")

    bindings = _read_role_bindings(components.files[ROLE_ASSIGNMENTS_FILE])
    bundle = JsonlDatasetAdapter.load(dataset_manifest_path)
    query_by_id = {query.query_id: query for query in bundle.queries}
    gold_by_query = {
        case.query_id: tuple(case.relevant_document_ids or ()) for case in bundle.gold_cases
    }
    candidate_path = _candidate_file(candidate_pool_path)
    candidates_by_query = _read_candidate_sets(candidate_path)
    _same_keys(
        query_by_id,
        candidates_by_query,
        left_label="dataset-query",
        right_label="candidate-pool",
    )
    assignments: dict[str, NiahSelectorAssignment] = {}
    if assignment_path is not None:
        assignments = _read_niah_assignments(assignment_path)
        _same_keys(
            assignments,
            bindings,
            left_label="NIAH-assignment",
            right_label="R002-role",
        )
    else:
        _same_keys(
            query_by_id,
            bindings,
            left_label="dataset-query",
            right_label="R002-role",
        )

    dataset_ids = {binding.dataset_id for binding in bindings.values()}
    if dataset_ids != {bundle.manifest.dataset_id}:
        raise ValueError("R002 role dataset_id differs from the supplied dataset")

    selector = TopKSelector()
    observations: list[_Observation] = []
    for query_id in sorted(bindings):
        query = query_by_id[query_id]
        candidate_set = candidates_by_query[query_id]
        ranked_by_id = {candidate.evidence_id: candidate for candidate in candidate_set.candidates}
        # The only Selector invocation per query.  K=9/8/7 are literal prefixes of this S0.
        s0 = selector.select(query, candidate_set, TOP_K_BASELINE)
        s0_candidates = tuple(ranked_by_id[item.evidence_id] for item in s0.items)
        if len(s0_candidates) != TOP_K_BASELINE:
            raise ValueError(f"query {query_id} has fewer than {TOP_K_BASELINE} S0 candidates")
        if len(candidate_set.candidates) != EXPECTED_POOL_SIZE:
            raise ValueError(f"query {query_id} is not an exact Top{EXPECTED_POOL_SIZE} pool")
        pool_documents = {candidate.document_id for candidate in candidate_set.candidates}
        baseline_documents = _ordered_unique([candidate.document_id for candidate in s0_candidates])
        gold_documents = gold_by_query[query_id]
        if not gold_documents:
            raise ValueError(f"query {query_id} has no official gold documents")
        binding = bindings[query_id]
        computed_eligible = set(gold_documents).issubset(baseline_documents)
        if computed_eligible != binding.chain_eligible_topk10:
            raise ValueError(
                f"R002 chain eligibility differs from recomputed TopK10 for {query_id}"
            )
        assignment = assignments.get(query_id)

        for k in TOP_K_VALUES:
            selected_candidates = s0_candidates[:k]
            dropped_candidates = s0_candidates[k:]
            selected_evidence_ids = tuple(item.evidence_id for item in selected_candidates)
            selected_documents = _ordered_unique([item.document_id for item in selected_candidates])
            dropped_evidence_ids = tuple(item.evidence_id for item in dropped_candidates)
            dropped_documents = tuple(item.document_id for item in dropped_candidates)
            chain_loss = conditional_chain_loss(
                baseline_document_ids=baseline_documents,
                selector_document_ids=selected_documents,
                gold_document_ids=gold_documents,
            )
            required_rate = required_deletion_rate(
                dropped_document_ids=dropped_documents,
                required_document_ids=gold_documents,
            )
            observation = _Observation(
                query_id=query_id,
                component_id=binding.component_id,
                role=binding.role,
                k=k,
                selected_evidence_ids=selected_evidence_ids,
                selected_document_ids=selected_documents,
                dropped_evidence_ids=dropped_evidence_ids,
                dropped_document_ids=dropped_documents,
                required_document_ids=gold_documents,
                recall=document_recall(selected_documents, gold_documents),
                relative_recall_loss=relative_recall_loss(
                    baseline_document_ids=baseline_documents,
                    selector_document_ids=selected_documents,
                    gold_document_ids=gold_documents,
                ),
                chain_eligible_topk10=computed_eligible,
                conditional_chain_loss=chain_loss,
                required_deletion_rate=required_rate,
            )
            if assignment is not None:
                harmful_id = assignment.harmful_document_id
                harmful_in_pool = harmful_id in pool_documents
                harmful_in_topk10 = harmful_id in set(baseline_documents)
                selected_harm = float(harmful_id in set(selected_documents))
                baseline_harm = float(harmful_in_topk10)
                observation = _Observation(
                    **{
                        **observation.__dict__,
                        "harmful_document_id": harmful_id,
                        "harmful_in_pool": harmful_in_pool,
                        "harmful_in_topk10": harmful_in_topk10,
                        "pool_conditional_harmful_exposure": (
                            selected_harm if harmful_in_pool else None
                        ),
                        "pool_conditional_harmful_reduction": (
                            baseline_harm - selected_harm if harmful_in_pool else None
                        ),
                        "baseline_exposed_harmful_deletion": (
                            1.0 - selected_harm if harmful_in_topk10 else None
                        ),
                        "unconditional_harmful_exposure": selected_harm,
                        "deletion_precision": deletion_precision(
                            dropped_document_ids=dropped_documents,
                            harmful_document_ids=(harmful_id,),
                        ),
                    }
                )
            observations.append(observation)

    rows = [
        observation.as_row(dataset_id=bundle.manifest.dataset_id, dataset_kind=dataset_kind)
        for observation in observations
    ]
    rows_bytes = _jsonl_bytes(rows)
    roles = sorted({observation.role for observation in observations})
    overall = _aggregates_by_system(observations, dataset_kind=dataset_kind)
    paired_vs_topk10 = _paired_comparisons(observations, dataset_kind=dataset_kind)
    by_role = {
        role: _aggregates_by_system(
            [observation for observation in observations if observation.role == role],
            dataset_kind=dataset_kind,
        )
        for role in roles
    }
    paired_vs_topk10_by_role = {
        role: _paired_comparisons(
            [observation for observation in observations if observation.role == role],
            dataset_kind=dataset_kind,
        )
        for role in roles
    }
    report: dict[str, object] = {
        "schema_version": "1.0",
        "protocol_version": TOPK_PROTOCOL_VERSION,
        "dataset_kind": dataset_kind,
        "source_split": source_split,
        "dataset_id": bundle.manifest.dataset_id,
        "dataset_version": bundle.manifest.dataset_version,
        "declared_dataset_split": bundle.manifest.split,
        "dataset_signature": bundle.dataset_signature,
        "candidate_pool_status": components.manifest["candidate_pool_status"],
        "counts": {
            "dataset_queries": len(bundle.queries),
            "eligible_input_queries": len(bindings),
            "components": len({binding.component_id for binding in bindings.values()}),
            "trace_rows": len(rows),
        },
        "rules": {
            "pool": "same frozen Hybrid Top20 for every system",
            "anchor": "TopK10 S0",
            "systems": [f"topk{k}" for k in TOP_K_VALUES],
            "smaller_systems": "literal S0 prefix; ranks 11-20 never replace a deletion",
            "recall_unit": "document_id",
            "chain_denominator": "queries whose TopK10 contains every official gold document",
            "niah_query_universe": "frozen assignment subset",
            "twowiki_query_universe": "all source-split queries",
            "twowiki_harmful_metrics": "not defined and not emitted",
            "paired_scopes": "overall and each frozen R002 derived role separately",
            "paired_alignment": "strict identical query keys and identical metric masks",
            "paired_resampling_unit": "frozen R002 component_id",
            "paired_bootstrap_and_sign_flip_seed": PAIRED_INFERENCE_SEED,
            "paired_bootstrap_and_sign_flip_iterations": PAIRED_INFERENCE_ITERATIONS,
            "paired_p_value": "two-sided Monte Carlo sign-flip with plus-one correction",
            "paired_ci": "95% percentile paired component-cluster bootstrap",
        },
        "overall": overall,
        "by_role": by_role,
        "paired_vs_topk10": paired_vs_topk10,
        "paired_vs_topk10_by_role": paired_vs_topk10_by_role,
        "artifacts": {TOPK_ROWS_FILE: _bytes_pin(rows_bytes, records=len(rows))},
    }
    report_bytes = _json_bytes(report)

    dataset_root = Path(dataset_manifest_path).parent
    inputs: dict[str, object] = {
        "dataset_manifest": _file_pin(dataset_manifest_path),
        "dataset_documents": _file_pin(dataset_root / bundle.manifest.documents_file),
        "dataset_queries": _file_pin(dataset_root / bundle.manifest.queries_file),
        "dataset_gold_cases": _file_pin(dataset_root / bundle.manifest.gold_cases_file),
        "source_parent": _file_pin(source_parent_path),
        "candidate_pool": _file_pin(candidate_path),
        "selector_components": {
            filename: _file_pin(Path(component_directory) / filename)
            for filename in COMPONENT_OUTPUT_FILES
        },
    }
    pool_manifest = Path(candidate_pool_path) / "selector_candidate_pool_manifest_v2.json"
    if Path(candidate_pool_path).is_dir():
        inputs["selector_candidate_pool_manifest_v2"] = _file_pin(pool_manifest)
    if assignment_path is not None:
        inputs["niah_assignment"] = _file_pin(assignment_path)

    output_pins = {
        TOPK_ROWS_FILE: _bytes_pin(rows_bytes, records=len(rows)),
        TOPK_REPORT_FILE: _bytes_pin(report_bytes),
    }
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "protocol_version": TOPK_PROTOCOL_VERSION,
        "dataset_kind": dataset_kind,
        "source_split": source_split,
        "dataset_id": bundle.manifest.dataset_id,
        "dataset_version": bundle.manifest.dataset_version,
        "dataset_signature": bundle.dataset_signature,
        "candidate_pool_manifest_verified": components.manifest["candidate_pool_manifest_verified"],
        "candidate_pool_status": components.manifest["candidate_pool_status"],
        "r002_manifest_file": COMPONENT_MANIFEST_FILE,
        "inputs": inputs,
        "outputs": output_pins,
        "counts": report["counts"],
        "rules": report["rules"],
    }
    manifest_bytes = _json_bytes(manifest)
    files = {
        TOPK_ROWS_FILE: rows_bytes,
        TOPK_REPORT_FILE: report_bytes,
        TOPK_MANIFEST_FILE: manifest_bytes,
    }
    return TopKControlArtifacts(files=files, report=report, manifest=manifest)


def freeze_topk_control_artifacts(output_directory: Path, artifacts: TopKControlArtifacts) -> Path:
    """Write one complete TopK bundle; an existing non-empty directory is immutable."""

    directory = Path(output_directory)
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"TopK control output is not a directory: {directory}")
    if directory.is_dir():
        existing = sorted(path.name for path in directory.iterdir())
        if existing:
            raise ValueError(
                f"TopK control output {directory} is not empty; refusing to overwrite "
                f"write-once artifacts: {existing[:5]}"
            )
    directory.mkdir(parents=True, exist_ok=True)
    for filename in TOPK_OUTPUT_FILES:
        (directory / filename).write_bytes(artifacts.files[filename])
    return directory / TOPK_MANIFEST_FILE


def verify_topk_control_artifacts(
    *,
    output_directory: Path,
    dataset_kind: DatasetKind,
    source_split: SourceSplit,
    dataset_manifest_path: Path,
    source_parent_path: Path,
    candidate_pool_path: Path,
    component_directory: Path,
    assignment_path: Path | None = None,
) -> TopKControlArtifacts:
    """Recompute from every source and require a byte-identical frozen bundle."""

    expected = build_topk_control_artifacts(
        dataset_kind=dataset_kind,
        source_split=source_split,
        dataset_manifest_path=dataset_manifest_path,
        source_parent_path=source_parent_path,
        candidate_pool_path=candidate_pool_path,
        component_directory=component_directory,
        assignment_path=assignment_path,
    )
    directory = Path(output_directory)
    for filename in TOPK_OUTPUT_FILES:
        path = directory / filename
        try:
            actual = path.read_bytes()
        except OSError as error:
            raise ValueError(
                f"missing or unreadable TopK control artifact {path}: {error}"
            ) from error
        wanted = expected.files[filename]
        if actual != wanted:
            raise ValueError(
                f"TopK control artifact differs from recomputation: {path}; "
                f"actual_sha256={_sha256_bytes(actual)}, "
                f"expected_sha256={_sha256_bytes(wanted)}"
            )
    return expected


def repeat_seed_token(repeat_index: int) -> str:
    """Return the frozen SHA-256 seed token for repeat 0..99."""

    if (
        isinstance(repeat_index, bool)
        or not isinstance(repeat_index, int)
        or not 0 <= repeat_index < COUNT_MATCHED_REPEATS
    ):
        raise ValueError(f"repeat_index must be in [0, {COUNT_MATCHED_REPEATS - 1}]")
    payload = json.dumps(
        [COUNT_MATCHED_PROTOCOL_VERSION, COUNT_MATCHED_MASTER_SEED, repeat_index],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_s0(candidates: Sequence[EvidenceCandidate]) -> tuple[EvidenceCandidate, ...]:
    values = tuple(candidates)
    if len(values) != TOP_K_BASELINE:
        raise ValueError(f"S0 must contain exactly {TOP_K_BASELINE} candidates")
    if len({candidate.evidence_id for candidate in values}) != len(values):
        raise ValueError("S0 evidence IDs must be unique")
    ranks = {candidate.retrieval_rank for candidate in values}
    if ranks != set(range(1, TOP_K_BASELINE + 1)):
        raise ValueError(f"S0 retrieval ranks must be exactly 1..{TOP_K_BASELINE}")
    return tuple(sorted(values, key=lambda item: (item.retrieval_rank, item.evidence_id)))


def _validated_delete_count(deletion_count: int) -> int:
    if (
        isinstance(deletion_count, bool)
        or not isinstance(deletion_count, int)
        or not 0 <= deletion_count <= COUNT_MATCHED_MAX_DELETIONS
    ):
        raise ValueError(f"deletion_count must be in [0, {COUNT_MATCHED_MAX_DELETIONS}]")
    return deletion_count


def _digest(value: str, *, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def random_candidate_priority(
    *,
    repeat_index: int,
    dataset_id: str,
    dataset_signature: str,
    pool_sha256: str,
    query_id: str,
    evidence_id: str,
) -> str:
    """Stable random-control priority; Python RNG and process hash are not used."""

    values = {
        "dataset_id": dataset_id,
        "query_id": query_id,
        "evidence_id": evidence_id,
    }
    for label, value in values.items():
        _nonblank(value, label=label)
    payload = json.dumps(
        [
            COUNT_MATCHED_PROTOCOL_VERSION,
            "random-priority",
            repeat_seed_token(repeat_index),
            dataset_id,
            _digest(dataset_signature, label="dataset_signature"),
            _digest(pool_sha256, label="pool_sha256"),
            query_id,
            evidence_id,
            TOP_K_BASELINE,
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def count_matched_random_drop(
    candidates: Sequence[EvidenceCandidate],
    *,
    deletion_count: int,
    repeat_index: int,
    dataset_id: str,
    dataset_signature: str,
    pool_sha256: str,
    query_id: str,
) -> tuple[str, ...]:
    """Choose exactly the future trace's count from S0 by frozen SHA priorities."""

    count = _validated_delete_count(deletion_count)
    s0 = _validated_s0(candidates)
    ordered = sorted(
        s0,
        key=lambda candidate: (
            random_candidate_priority(
                repeat_index=repeat_index,
                dataset_id=dataset_id,
                dataset_signature=dataset_signature,
                pool_sha256=pool_sha256,
                query_id=query_id,
                evidence_id=candidate.evidence_id,
            ),
            candidate.retrieval_rank,
            candidate.evidence_id,
        ),
    )
    return tuple(candidate.evidence_id for candidate in ordered[:count])


def count_matched_bottom_rank_drop(
    candidates: Sequence[EvidenceCandidate], *, deletion_count: int
) -> tuple[str, ...]:
    """Choose exactly the future trace's count from the lowest-ranked end of S0."""

    count = _validated_delete_count(deletion_count)
    s0 = _validated_s0(candidates)
    ordered = sorted(s0, key=lambda candidate: (-candidate.retrieval_rank, candidate.evidence_id))
    return tuple(candidate.evidence_id for candidate in ordered[:count])


def generate_count_matched_drops(
    candidates: Sequence[EvidenceCandidate],
    *,
    deletion_count: int,
    repeat_index: int,
    dataset_id: str,
    dataset_signature: str,
    pool_sha256: str,
    query_id: str,
) -> CountMatchedDrops:
    """Generate both controls after, and only after, a real deletion count is supplied."""

    return CountMatchedDrops(
        deletion_count=_validated_delete_count(deletion_count),
        random_dropped_evidence_ids=count_matched_random_drop(
            candidates,
            deletion_count=deletion_count,
            repeat_index=repeat_index,
            dataset_id=dataset_id,
            dataset_signature=dataset_signature,
            pool_sha256=pool_sha256,
            query_id=query_id,
        ),
        bottom_rank_dropped_evidence_ids=count_matched_bottom_rank_drop(
            candidates, deletion_count=deletion_count
        ),
    )


def build_count_matched_protocol_artifacts() -> CountMatchedProtocolArtifacts:
    """Build the write-once 100-repeat protocol without fabricating Selector controls."""

    repeats = [
        {
            "repeat_index": repeat_index,
            "seed_sha256": repeat_seed_token(repeat_index),
            "seed_uint64": int(repeat_seed_token(repeat_index)[:16], 16),
        }
        for repeat_index in range(COUNT_MATCHED_REPEATS)
    ]
    priority_golden = {
        "repeat_index": 0,
        "dataset_id": "golden-dataset",
        "dataset_signature": "a" * 64,
        "candidate_pool_file_sha256": "b" * 64,
        "query_id": "golden-query",
        "evidence_id": "golden-evidence",
    }
    priority_golden["priority_sha256"] = random_candidate_priority(
        repeat_index=0,
        dataset_id="golden-dataset",
        dataset_signature="a" * 64,
        pool_sha256="b" * 64,
        query_id="golden-query",
        evidence_id="golden-evidence",
    )
    protocol: dict[str, object] = {
        "schema_version": "1.0",
        "protocol_version": COUNT_MATCHED_PROTOCOL_VERSION,
        "status": "GENERATOR_PROTOCOL_ONLY_NO_SELECTOR_TRACE",
        "master_seed": COUNT_MATCHED_MASTER_SEED,
        "repeat_count": COUNT_MATCHED_REPEATS,
        "repeat_index_range": [0, COUNT_MATCHED_REPEATS - 1],
        "baseline": "frozen TopK10 S0",
        "allowed_deletion_counts": list(range(COUNT_MATCHED_MAX_DELETIONS + 1)),
        "deletion_count_source": (
            "the matching query row from a real Selector trace; never sampled or imputed"
        ),
        "repeat_seed_payload": [
            COUNT_MATCHED_PROTOCOL_VERSION,
            COUNT_MATCHED_MASTER_SEED,
            "<repeat_index>",
        ],
        "repeat_seed_encoding": (
            "sha256(json.dumps(payload, ensure_ascii=True, separators=(',', ':')).encode('utf-8'))"
        ),
        "random_priority_payload": [
            COUNT_MATCHED_PROTOCOL_VERSION,
            "random-priority",
            "<repeat_seed_sha256>",
            "<dataset_id>",
            "<dataset_signature>",
            "<candidate_pool_file_sha256>",
            "<query_id>",
            "<evidence_id>",
            TOP_K_BASELINE,
        ],
        "random_rule": (
            "sort S0 by (sha256(canonical priority payload), retrieval_rank, evidence_id); "
            "drop the first deletion_count candidates"
        ),
        "bottom_rank_rule": (
            "sort S0 by (-retrieval_rank, evidence_id); drop the first deletion_count candidates"
        ),
        "nesting": (
            "within one query and repeat, the m-drop set is a prefix of one fixed priority "
            "order, so drops at m are a subset of drops at m+1"
        ),
        "replacement": "forbidden; controls only delete from S0",
        "result_timing": (
            "materialize results only after a real Selector trace fixes every query's deletion count"
        ),
        "repeats": repeats,
    }
    protocol_bytes = _json_bytes(protocol)
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "protocol_version": COUNT_MATCHED_PROTOCOL_VERSION,
        "status": protocol["status"],
        "inputs": {},
        "outputs": {COUNT_MATCHED_PROTOCOL_FILE: _bytes_pin(protocol_bytes)},
        "golden_vectors": {
            "repeat_0_seed_sha256": repeats[0]["seed_sha256"],
            "repeat_99_seed_sha256": repeats[-1]["seed_sha256"],
            "random_priority": priority_golden,
        },
    }
    manifest_bytes = _json_bytes(manifest)
    files = {
        COUNT_MATCHED_PROTOCOL_FILE: protocol_bytes,
        COUNT_MATCHED_MANIFEST_FILE: manifest_bytes,
    }
    return CountMatchedProtocolArtifacts(files, protocol, manifest)


def freeze_count_matched_protocol_artifacts(
    output_directory: Path, artifacts: CountMatchedProtocolArtifacts
) -> Path:
    """Write the protocol exactly once."""

    directory = Path(output_directory)
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"count-matched protocol output is not a directory: {directory}")
    if directory.is_dir():
        existing = sorted(path.name for path in directory.iterdir())
        if existing:
            raise ValueError(
                f"count-matched protocol output {directory} is not empty; refusing to "
                f"overwrite write-once artifacts: {existing[:5]}"
            )
    directory.mkdir(parents=True, exist_ok=True)
    for filename in COUNT_MATCHED_OUTPUT_FILES:
        (directory / filename).write_bytes(artifacts.files[filename])
    return directory / COUNT_MATCHED_MANIFEST_FILE


def verify_count_matched_protocol_artifacts(
    output_directory: Path,
) -> CountMatchedProtocolArtifacts:
    """Rebuild the input-free protocol and require byte-identical frozen files."""

    expected = build_count_matched_protocol_artifacts()
    directory = Path(output_directory)
    for filename in COUNT_MATCHED_OUTPUT_FILES:
        path = directory / filename
        try:
            actual = path.read_bytes()
        except OSError as error:
            raise ValueError(
                f"missing or unreadable count-matched protocol artifact {path}: {error}"
            ) from error
        wanted = expected.files[filename]
        if actual != wanted:
            raise ValueError(
                f"count-matched protocol artifact differs from recomputation: {path}; "
                f"actual_sha256={_sha256_bytes(actual)}, "
                f"expected_sha256={_sha256_bytes(wanted)}"
            )
    return expected
