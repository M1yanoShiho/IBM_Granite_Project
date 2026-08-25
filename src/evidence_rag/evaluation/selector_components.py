"""Reproducible leakage components and derived Selector data roles.

The source dataset split is never randomly re-split query by query.  Queries are first joined
through a disjoint-set union (DSU) using only the pre-registered relation keys, then whole
components are assigned to folds with a deterministic largest-first, query-balanced rule.

NIAH uses query, labelled source-parent and synthetic-family keys.  2Wiki uses query and the
parents of *official gold documents only*; parents seen only on retrieved distractors are never
component edges.  The same component map is written once and reused by role allocation,
component-level risk representatives and later cluster-bootstrap inference.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from evidence_rag.contracts.models import CandidateSet
from evidence_rag.evaluation.selector_risk import (
    COMPONENT_REPRESENTATIVE_SEED,
    NIAH_CHAIN_RISK,
    NIAH_RECALL_RISK,
    TWOWIKI_CHAIN_RISK,
    TWOWIKI_RECALL_RISK,
    ComponentCandidate,
    representative_digest,
    select_component_representatives,
)
from evidence_rag.infrastructure.datasets import DatasetBundle, JsonlDatasetAdapter
from evidence_rag.materializer.selector_pool import verify_selector_candidate_pool_v2

DatasetKind: TypeAlias = Literal["niah", "2wiki"]
SourceSplit: TypeAlias = Literal["train", "dev"]
DerivedRole: TypeAlias = Literal[
    "train-fit", "train-modelval", "crc-calibration", "decision-dev"
]
AllowedKey: TypeAlias = tuple[str, str]

COMPONENT_MAP_FILE = "component_map.jsonl"
ROLE_ASSIGNMENTS_FILE = "role_assignments.jsonl"
RECALL_REPRESENTATIVES_FILE = "recall_representatives.jsonl"
CHAIN_REPRESENTATIVES_FILE = "chain_representatives.jsonl"
REPORT_FILE = "selector_components_report.json"
MANIFEST_FILE = "selector_components_manifest.json"
OUTPUT_FILES = (
    COMPONENT_MAP_FILE,
    ROLE_ASSIGNMENTS_FILE,
    RECALL_REPRESENTATIVES_FILE,
    CHAIN_REPRESENTATIVES_FILE,
    REPORT_FILE,
    MANIFEST_FILE,
)
PROTOCOL_VERSION = "selector-components-v1"
TOP_K_BASELINE = 10
EXPECTED_POOL_SIZE = 20

NonEmpty = Annotated[str, Field(min_length=1)]


class NiahSelectorAssignment(BaseModel):
    """The frozen, leakage-filtered assignment schema produced by Beam M0."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: NonEmpty
    required_document_ids: tuple[NonEmpty, ...]
    harmful_document_id: NonEmpty
    source_parent_ids: tuple[NonEmpty, ...]
    synthetic_family: NonEmpty

    @model_validator(mode="after")
    def labels_and_keys_are_well_formed(self) -> NiahSelectorAssignment:
        for field_name in (
            "query_id",
            "harmful_document_id",
            "synthetic_family",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be blank")
        if not self.required_document_ids:
            raise ValueError("required_document_ids must not be empty")
        if not self.source_parent_ids:
            raise ValueError("source_parent_ids must not be empty")
        for field_name, values in (
            ("required_document_ids", self.required_document_ids),
            ("source_parent_ids", self.source_parent_ids),
        ):
            if any(not value.strip() for value in values):
                raise ValueError(f"{field_name} must not contain blank strings")
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicates")
        if self.harmful_document_id in self.required_document_ids:
            raise ValueError("harmful_document_id must not also be required")
        return self


@dataclass(frozen=True)
class SelectorComponent:
    component_id: str
    query_ids: tuple[str, ...]
    allowed_keys: tuple[AllowedKey, ...]
    smallest_key: str

    @property
    def n_queries(self) -> int:
        return len(self.query_ids)


@dataclass(frozen=True)
class SelectorComponentArtifacts:
    """Canonical bytes plus parsed report/manifest for one write-once bundle."""

    files: Mapping[str, bytes]
    report: Mapping[str, object]
    manifest: Mapping[str, object]


@dataclass(frozen=True)
class _PreparedInputs:
    bundle: DatasetBundle
    candidate_sets: Mapping[str, CandidateSet]
    candidate_path: Path
    pool_manifest_path: Path | None
    pool_manifest_verified: bool
    parent_by_document: Mapping[str, str]
    assignments: Mapping[str, NiahSelectorAssignment]
    assignment_path: Path | None
    query_ids: tuple[str, ...]
    gold_by_query: Mapping[str, tuple[str, ...]]
    keys_by_query: Mapping[str, tuple[AllowedKey, ...]]
    chain_eligible: Mapping[str, bool]


class _Dsu:
    def __init__(self, items: Collection[str]) -> None:
        self._parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self._parent[item]
        if parent != item:
            self._parent[item] = self.find(parent)
        return self._parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        # Root identity is irrelevant to membership, but choosing the lexical root makes
        # debugging deterministic too.
        low, high = sorted((left_root, right_root))
        self._parent[high] = low


def _require_nonblank(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-blank string")
    return value


def _key_string(key: AllowedKey) -> str:
    """The pre-frozen complete relation key used by the remote split audit."""

    axis, value = key
    return f"{axis}:{value}"


def component_id_for_queries(query_ids: Collection[str]) -> str:
    """Return SHA256(''.join(q + '\\n' for q in sorted members))."""

    if isinstance(query_ids, str) or not query_ids:
        raise ValueError("component query IDs must be a non-empty collection")
    members = tuple(sorted(query_ids))
    if len(members) != len(set(members)):
        raise ValueError("component query IDs must be unique")
    if any(not isinstance(query_id, str) or not query_id for query_id in members):
        raise ValueError("component query IDs must be non-empty strings")
    payload = "".join(f"{query_id}\n" for query_id in members).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_allowed_key_components(
    keys_by_query: Mapping[str, Collection[AllowedKey]],
) -> tuple[SelectorComponent, ...]:
    """Join queries that share any namespaced allowed key, including transitively."""

    if not keys_by_query:
        raise ValueError("component input must contain at least one query")
    normalized: dict[str, tuple[AllowedKey, ...]] = {}
    for query_id, keys in keys_by_query.items():
        _require_nonblank(query_id, label="query_id")
        if not keys:
            raise ValueError(f"query {query_id!r} has no allowed component keys")
        clean: set[AllowedKey] = set()
        for key in keys:
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError(f"query {query_id!r} has an invalid allowed key")
            axis = _require_nonblank(key[0], label="allowed-key axis")
            value = _require_nonblank(key[1], label="allowed-key value")
            clean.add((axis, value))
        normalized[query_id] = tuple(sorted(clean, key=_key_string))

    dsu = _Dsu(set(normalized))
    owner_by_key: dict[AllowedKey, str] = {}
    for query_id in sorted(normalized):
        for key in normalized[query_id]:
            owner = owner_by_key.setdefault(key, query_id)
            dsu.union(query_id, owner)

    members_by_root: dict[str, list[str]] = {}
    for query_id in sorted(normalized):
        members_by_root.setdefault(dsu.find(query_id), []).append(query_id)

    components: list[SelectorComponent] = []
    seen_component_ids: set[str] = set()
    for members in members_by_root.values():
        query_ids = tuple(sorted(members))
        component_id = component_id_for_queries(query_ids)
        if component_id in seen_component_ids:
            raise AssertionError("SHA-256 collision between distinct query components")
        seen_component_ids.add(component_id)
        component_keys = tuple(
            sorted(
                {key for query_id in query_ids for key in normalized[query_id]},
                key=_key_string,
            )
        )
        components.append(
            SelectorComponent(
                component_id=component_id,
                query_ids=query_ids,
                allowed_keys=component_keys,
                smallest_key=_key_string(component_keys[0]),
            )
        )
    return tuple(sorted(components, key=lambda item: item.component_id))


def assign_query_balanced_folds(
    components: Sequence[SelectorComponent], *, n_folds: int
) -> tuple[dict[str, int], tuple[int, ...]]:
    """Assign whole components largest-first to the least-loaded query fold."""

    if n_folds <= 0:
        raise ValueError("n_folds must be positive")
    if not components:
        raise ValueError("cannot assign an empty component set")
    if len({component.component_id for component in components}) != len(components):
        raise ValueError("component IDs must be unique")

    loads = [0] * n_folds
    fold_by_component: dict[str, int] = {}
    ordered = sorted(
        components,
        key=lambda item: (-item.n_queries, item.smallest_key, item.component_id),
    )
    for component in ordered:
        fold = min(range(n_folds), key=lambda index: (loads[index], index))
        fold_by_component[component.component_id] = fold
        loads[fold] += component.n_queries
    return fold_by_component, tuple(loads)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_pin(path: Path) -> dict[str, object]:
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise ValueError(f"unable to read input artifact {path}: {error}") from error
    return {"sha256": _sha256_bytes(payload), "bytes": len(payload)}


def _bytes_pin(payload: bytes, *, records: int | None = None) -> dict[str, object]:
    pin: dict[str, object] = {"sha256": _sha256_bytes(payload), "bytes": len(payload)}
    if records is not None:
        pin["records"] = records
    return pin


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


def _read_object_lines(path: Path, *, label: str) -> tuple[tuple[int, Mapping[str, object]], ...]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"unable to read {label} {path}: {error}") from error
    rows: list[tuple[int, Mapping[str, object]]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"blank {label} record at {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{label} record at {path}:{line_number} must be an object")
        rows.append((line_number, row))
    if not rows:
        raise ValueError(f"{label} {path} is empty")
    return tuple(rows)


def _read_parent_map(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, row in _read_object_lines(path, label="source-parent"):
        document_id = _require_nonblank(
            row.get("document_id"), label=f"document_id at {path}:{line_number}"
        )
        source_parent_id = _require_nonblank(
            row.get("source_parent_id"),
            label=f"source_parent_id at {path}:{line_number}",
        )
        if document_id in values:
            raise ValueError(f"duplicate document_id in source-parent index: {document_id}")
        values[document_id] = source_parent_id
    return values


def _resolve_candidate_input(path: Path) -> tuple[Path, Path | None]:
    candidate_input = Path(path)
    if candidate_input.is_dir():
        candidate_path = candidate_input / "candidate_sets.jsonl"
        pool_manifest = candidate_input / "selector_candidate_pool_manifest_v2.json"
        if not pool_manifest.is_file():
            raise ValueError(
                f"candidate pool directory has no frozen Selector-v2 manifest: {pool_manifest}"
            )
        return candidate_path, pool_manifest
    return candidate_input, None


def _read_candidate_sets(path: Path) -> dict[str, CandidateSet]:
    values: dict[str, CandidateSet] = {}
    for line_number, row in _read_object_lines(path, label="candidate-pool"):
        try:
            candidate_set = CandidateSet.model_validate(row)
        except ValidationError as error:
            raise ValueError(f"invalid candidate set at {path}:{line_number}: {error}") from error
        if candidate_set.query_id in values:
            raise ValueError(f"duplicate candidate query ID: {candidate_set.query_id}")
        ranks = {candidate.retrieval_rank for candidate in candidate_set.candidates}
        if len(candidate_set.candidates) != EXPECTED_POOL_SIZE or ranks != set(
            range(1, EXPECTED_POOL_SIZE + 1)
        ):
            raise ValueError(
                f"candidate query {candidate_set.query_id} must contain exact ranks "
                f"1..{EXPECTED_POOL_SIZE}"
            )
        if candidate_set.retriever is None or candidate_set.retriever.name != "hybrid":
            raise ValueError(
                f"candidate query {candidate_set.query_id} is not from a declared hybrid pool"
            )
        values[candidate_set.query_id] = candidate_set
    return values


def _read_assignments(path: Path) -> dict[str, NiahSelectorAssignment]:
    values: dict[str, NiahSelectorAssignment] = {}
    for line_number, row in _read_object_lines(path, label="NIAH assignment"):
        try:
            assignment = NiahSelectorAssignment.model_validate(row)
        except ValidationError as error:
            raise ValueError(f"invalid NIAH assignment at {path}:{line_number}: {error}") from error
        if assignment.query_id in values:
            raise ValueError(f"duplicate NIAH assignment query ID: {assignment.query_id}")
        values[assignment.query_id] = assignment
    return values


def _same_keys(
    left: Collection[str], right: Collection[str], *, left_label: str, right_label: str
) -> None:
    left_keys = set(left)
    right_keys = set(right)
    if left_keys != right_keys:
        raise ValueError(
            f"{left_label}/{right_label} keys differ: "
            f"missing from {right_label}={sorted(left_keys - right_keys)[:5]}, "
            f"missing from {left_label}={sorted(right_keys - left_keys)[:5]}"
        )


def _prepare_inputs(
    *,
    dataset_kind: DatasetKind,
    source_split: SourceSplit,
    dataset_manifest_path: Path,
    source_parent_path: Path,
    candidate_pool_path: Path,
    assignment_path: Path | None,
) -> _PreparedInputs:
    if dataset_kind == "niah" and assignment_path is None:
        raise ValueError("NIAH component construction requires an assignment JSONL")
    if dataset_kind == "2wiki" and assignment_path is not None:
        raise ValueError("2Wiki component construction must not receive a NIAH assignment JSONL")

    bundle = JsonlDatasetAdapter.load(dataset_manifest_path)
    if not bundle.queries:
        raise ValueError("dataset must contain at least one query")
    query_by_id = {query.query_id: query for query in bundle.queries}
    gold_by_query: dict[str, tuple[str, ...]] = {}
    for gold_case in bundle.gold_cases:
        documents = tuple(gold_case.relevant_document_ids or ())
        if not documents:
            raise ValueError(f"query {gold_case.query_id} has no official gold documents")
        gold_by_query[gold_case.query_id] = documents
    _same_keys(
        query_by_id,
        gold_by_query,
        left_label="dataset-query",
        right_label="gold",
    )

    document_ids = {document.document_id for document in bundle.documents}
    parent_by_document = _read_parent_map(source_parent_path)
    unknown_parent_documents = sorted(set(parent_by_document) - document_ids)
    if unknown_parent_documents:
        raise ValueError(
            "source-parent index references unknown dataset documents: "
            f"{unknown_parent_documents[:5]}"
        )
    official_documents = {
        document_id for documents in gold_by_query.values() for document_id in documents
    }
    missing_official_parents = sorted(official_documents - set(parent_by_document))
    if missing_official_parents:
        raise ValueError(
            "official gold documents have no source-parent mapping: "
            f"{missing_official_parents[:5]}"
        )

    candidate_path, pool_manifest_path = _resolve_candidate_input(candidate_pool_path)
    pool_manifest_verified = False
    if pool_manifest_path is not None:
        pool_manifest = verify_selector_candidate_pool_v2(
            candidate_pool_path, dataset_manifest_path
        )
        expected_role = f"{dataset_kind}-{source_split}"
        if pool_manifest.data_role != expected_role:
            raise ValueError(
                f"candidate pool data_role={pool_manifest.data_role!r}, expected {expected_role!r}"
            )
        if pool_manifest.dataset_signature != bundle.dataset_signature:
            raise ValueError("candidate pool manifest and dataset content signatures differ")
        pool_manifest_verified = True
    candidate_sets = _read_candidate_sets(candidate_path)
    _same_keys(
        query_by_id,
        candidate_sets,
        left_label="dataset-query",
        right_label="candidate-pool",
    )
    unknown_candidate_documents = sorted(
        {
            candidate.document_id
            for candidate_set in candidate_sets.values()
            for candidate in candidate_set.candidates
        }
        - document_ids
    )
    if unknown_candidate_documents:
        raise ValueError(
            "candidate pool references unknown dataset documents: "
            f"{unknown_candidate_documents[:5]}"
        )

    assignments: dict[str, NiahSelectorAssignment] = {}
    keys_by_query: dict[str, tuple[AllowedKey, ...]] = {}
    if dataset_kind == "niah":
        if assignment_path is None:  # narrowed above; retained for mypy and fail-closed clarity
            raise AssertionError("NIAH assignment path was not narrowed")
        assignments = _read_assignments(assignment_path)
        unknown_assignment_queries = sorted(set(assignments) - set(query_by_id))
        if unknown_assignment_queries:
            raise ValueError(
                "NIAH assignments reference unknown dataset queries: "
                f"{unknown_assignment_queries[:5]}"
            )
        for query_id, assignment in assignments.items():
            if set(assignment.required_document_ids) != set(gold_by_query[query_id]):
                raise ValueError(
                    f"NIAH assignment {query_id} required documents differ from official gold"
                )
            if assignment.harmful_document_id not in document_ids:
                raise ValueError(
                    f"NIAH assignment {query_id} harmful document is absent from the dataset"
                )
            labelled_documents = {
                *assignment.required_document_ids,
                assignment.harmful_document_id,
            }
            missing_parents = sorted(labelled_documents - set(parent_by_document))
            if missing_parents:
                raise ValueError(
                    f"NIAH assignment {query_id} labelled documents have no parent: "
                    f"{missing_parents[:5]}"
                )
            expected_parents = {parent_by_document[item] for item in labelled_documents}
            if set(assignment.source_parent_ids) != expected_parents:
                raise ValueError(
                    f"NIAH assignment {query_id} source_parent_ids differ from the parent index"
                )
            keys_by_query[query_id] = tuple(
                [
                    ("query", query_id),
                    *(
                        ("parent", parent)
                        for parent in sorted(expected_parents)
                    ),
                    ("family", assignment.synthetic_family),
                ]
            )
        query_ids = tuple(sorted(assignments))
    else:
        for query_id in sorted(query_by_id):
            official_parents = sorted(
                {parent_by_document[item] for item in gold_by_query[query_id]}
            )
            keys_by_query[query_id] = tuple(
                [("query", query_id)]
                + [("parent", parent) for parent in official_parents]
            )
        query_ids = tuple(sorted(query_by_id))

    chain_eligible: dict[str, bool] = {}
    for query_id in query_ids:
        ranked = sorted(
            candidate_sets[query_id].candidates,
            key=lambda candidate: candidate.retrieval_rank,
        )
        top_k_documents = {
            candidate.document_id for candidate in ranked[:TOP_K_BASELINE]
        }
        chain_eligible[query_id] = set(gold_by_query[query_id]).issubset(top_k_documents)

    return _PreparedInputs(
        bundle=bundle,
        candidate_sets=candidate_sets,
        candidate_path=candidate_path,
        pool_manifest_path=pool_manifest_path,
        pool_manifest_verified=pool_manifest_verified,
        parent_by_document=parent_by_document,
        assignments=assignments,
        assignment_path=assignment_path,
        query_ids=query_ids,
        gold_by_query=gold_by_query,
        keys_by_query=keys_by_query,
        chain_eligible=chain_eligible,
    )


def _role_for_fold(source_split: SourceSplit, fold: int) -> DerivedRole:
    if source_split == "train":
        return "train-modelval" if fold == 0 else "train-fit"
    return "crc-calibration" if fold == 0 else "decision-dev"


def _risk_names(dataset_kind: DatasetKind) -> tuple[str, str]:
    if dataset_kind == "niah":
        return NIAH_RECALL_RISK, NIAH_CHAIN_RISK
    return TWOWIKI_RECALL_RISK, TWOWIKI_CHAIN_RISK


def _representative_rows(
    *,
    dataset_id: str,
    risk_name: str,
    formal_risk_name: str,
    purpose: str,
    source_role: DerivedRole,
    crc_representative: bool,
    query_ids: Collection[str],
    component_by_query: Mapping[str, str],
) -> list[dict[str, object]]:
    candidates = tuple(
        ComponentCandidate(
            dataset_id=dataset_id,
            risk_name=risk_name,
            component_id=component_by_query[query_id],
            query_id=query_id,
        )
        for query_id in sorted(query_ids)
    )
    ownership: dict[tuple[str, str, str], str] = {}
    for candidate in candidates:
        identity = (candidate.dataset_id, candidate.risk_name, candidate.query_id)
        previous = ownership.setdefault(identity, candidate.component_id)
        if previous != candidate.component_id:
            raise ValueError(
                "one dataset/risk/query is assigned to multiple components: "
                f"{identity}, {previous!r} versus {candidate.component_id!r}"
            )
        if component_by_query[candidate.query_id] != candidate.component_id:
            raise AssertionError("representative candidate is not bound to the frozen component map")
    # These calls are deliberately shared with the CRC implementation.  Copying the payload or
    # min-hash rule here would let the artifact generator and evaluator silently disagree.
    representatives = select_component_representatives(
        candidates, seed=COMPONENT_REPRESENTATIVE_SEED
    )
    return [
        {
            "schema_version": "1.0",
            "dataset_id": candidate.dataset_id,
            "risk_name": candidate.risk_name,
            "formal_risk_name": formal_risk_name,
            "component_id": candidate.component_id,
            "query_id": candidate.query_id,
            "representative_digest": representative_digest(
                candidate, seed=COMPONENT_REPRESENTATIVE_SEED
            ),
            "representative_seed": COMPONENT_REPRESENTATIVE_SEED,
            "source_role": source_role,
            "purpose": purpose,
            "crc_representative": crc_representative,
        }
        for candidate in representatives
    ]


def _count_components(
    query_ids: Collection[str], component_by_query: Mapping[str, str]
) -> int:
    return len({component_by_query[query_id] for query_id in query_ids})


def build_selector_component_artifacts(
    *,
    dataset_kind: DatasetKind,
    source_split: SourceSplit,
    dataset_manifest_path: Path,
    source_parent_path: Path,
    candidate_pool_path: Path,
    assignment_path: Path | None = None,
) -> SelectorComponentArtifacts:
    """Validate every input and return canonical bytes without writing them."""

    prepared = _prepare_inputs(
        dataset_kind=dataset_kind,
        source_split=source_split,
        dataset_manifest_path=dataset_manifest_path,
        source_parent_path=source_parent_path,
        candidate_pool_path=candidate_pool_path,
        assignment_path=assignment_path,
    )
    components = build_allowed_key_components(prepared.keys_by_query)
    component_by_query = {
        query_id: component.component_id
        for component in components
        for query_id in component.query_ids
    }
    n_folds = 10 if source_split == "train" else 2
    fold_by_component, fold_loads = assign_query_balanced_folds(
        components, n_folds=n_folds
    )
    fold_by_query = {
        query_id: fold_by_component[component_by_query[query_id]]
        for query_id in prepared.query_ids
    }
    role_by_query = {
        query_id: _role_for_fold(source_split, fold_by_query[query_id])
        for query_id in prepared.query_ids
    }

    component_lookup = {component.component_id: component for component in components}
    component_rows: list[dict[str, object]] = []
    role_rows: list[dict[str, object]] = []
    for query_id in prepared.query_ids:
        component = component_lookup[component_by_query[query_id]]
        component_rows.append(
            {
                "schema_version": "1.0",
                "dataset_id": prepared.bundle.manifest.dataset_id,
                "query_id": query_id,
                "component_id": component.component_id,
                "component_size": component.n_queries,
                "component_root": component.smallest_key,
                "allowed_keys": [
                    {"axis": axis, "value": value, "key": _key_string((axis, value))}
                    for axis, value in prepared.keys_by_query[query_id]
                ],
            }
        )
        role_rows.append(
            {
                "schema_version": "1.0",
                "dataset_id": prepared.bundle.manifest.dataset_id,
                "query_id": query_id,
                "component_id": component.component_id,
                "fold": fold_by_query[query_id],
                "role": role_by_query[query_id],
                "chain_eligible_topk10": prepared.chain_eligible[query_id],
            }
        )

    recall_risk, chain_risk = _risk_names(dataset_kind)
    if source_split == "dev":
        representative_role: DerivedRole = "crc-calibration"
        calibration_queries = {
            query_id
            for query_id in prepared.query_ids
            if role_by_query[query_id] == representative_role
        }
        recall_rows = _representative_rows(
            dataset_id=prepared.bundle.manifest.dataset_id,
            risk_name=recall_risk,
            formal_risk_name=recall_risk,
            purpose="crc-calibration",
            source_role=representative_role,
            crc_representative=True,
            query_ids=calibration_queries,
            component_by_query=component_by_query,
        )
        chain_rows = _representative_rows(
            dataset_id=prepared.bundle.manifest.dataset_id,
            risk_name=chain_risk,
            formal_risk_name=chain_risk,
            purpose="crc-calibration",
            source_role=representative_role,
            crc_representative=True,
            query_ids={
                query_id
                for query_id in calibration_queries
                if prepared.chain_eligible[query_id]
            },
            component_by_query=component_by_query,
        )
    else:
        representative_role = "train-modelval"
        # Train/modelval carries per-query TopK10 chain eligibility and per-role component
        # counts in role_assignments/report.  It deliberately emits no risk representative:
        # CRC representatives belong exclusively to dev crc-calibration.
        recall_rows = []
        chain_rows = []

    recall_rows.sort(key=lambda row: (str(row["component_id"]), str(row["query_id"])))
    chain_rows.sort(key=lambda row: (str(row["component_id"]), str(row["query_id"])))

    component_bytes = _jsonl_bytes(component_rows)
    role_bytes = _jsonl_bytes(role_rows)
    recall_bytes = _jsonl_bytes(recall_rows)
    chain_bytes = _jsonl_bytes(chain_rows)
    first_artifacts = {
        COMPONENT_MAP_FILE: _bytes_pin(component_bytes, records=len(component_rows)),
        ROLE_ASSIGNMENTS_FILE: _bytes_pin(role_bytes, records=len(role_rows)),
        RECALL_REPRESENTATIVES_FILE: _bytes_pin(recall_bytes, records=len(recall_rows)),
        CHAIN_REPRESENTATIVES_FILE: _bytes_pin(chain_bytes, records=len(chain_rows)),
    }
    component_projection = [
        {
            "component_id": row["component_id"],
            "dataset_id": row["dataset_id"],
            "query_id": row["query_id"],
        }
        for row in sorted(
            component_rows, key=lambda item: (str(item["dataset_id"]), str(item["query_id"]))
        )
    ]
    role_projection = [
        {
            "component_id": row["component_id"],
            "dataset_id": row["dataset_id"],
            "query_id": row["query_id"],
            "role": row["role"],
        }
        for row in sorted(
            role_rows, key=lambda item: (str(item["dataset_id"]), str(item["query_id"]))
        )
    ]

    def representative_projection(
        rows: Sequence[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        return [
            {
                "component_id": row["component_id"],
                "dataset_id": row["dataset_id"],
                "query_id": row["query_id"],
                "risk_name": row["risk_name"],
                "role": row["source_role"],
            }
            for row in sorted(
                rows,
                key=lambda item: (str(item["component_id"]), str(item["query_id"])),
            )
        ]

    projections = {
        "component_rows": component_projection,
        "role_rows": role_projection,
        "recall_representative_rows": representative_projection(recall_rows),
        "chain_representative_rows": representative_projection(chain_rows),
    }
    projection_fingerprints: dict[str, object] = {
        name: _bytes_pin(_jsonl_bytes(rows), records=len(rows))
        for name, rows in projections.items()
    }
    expected_roles: tuple[DerivedRole, DerivedRole] = (
        ("train-fit", "train-modelval")
        if source_split == "train"
        else ("crc-calibration", "decision-dev")
    )
    role_projection_fingerprints = {
        role: _bytes_pin(
            _jsonl_bytes([row for row in role_projection if row["role"] == role]),
            records=sum(row["role"] == role for row in role_projection),
        )
        for role in expected_roles
    }
    projection_fingerprints["role_rows_by_role"] = role_projection_fingerprints

    role_counts: dict[str, dict[str, int]] = {}
    for role in sorted(set(role_by_query.values())):
        role_queries = {
            query_id for query_id in prepared.query_ids if role_by_query[query_id] == role
        }
        eligible = {
            query_id for query_id in role_queries if prepared.chain_eligible[query_id]
        }
        role_counts[role] = {
            "queries": len(role_queries),
            "components": _count_components(role_queries, component_by_query),
            "chain_eligible_queries": len(eligible),
            "chain_eligible_components": _count_components(eligible, component_by_query),
        }

    component_fold_sets: dict[str, set[int]] = {}
    component_role_sets: dict[str, set[str]] = {}
    for query_id in prepared.query_ids:
        component_id = component_by_query[query_id]
        component_fold_sets.setdefault(component_id, set()).add(fold_by_query[query_id])
        component_role_sets.setdefault(component_id, set()).add(role_by_query[query_id])
    crossing = {
        "components_across_folds": sum(
            len(folds) > 1 for folds in component_fold_sets.values()
        ),
        "components_across_roles": sum(
            len(roles) > 1 for roles in component_role_sets.values()
        ),
    }
    if any(crossing.values()):
        raise AssertionError(f"component allocation crossed a boundary: {crossing}")

    report: dict[str, object] = {
        "schema_version": "1.0",
        "protocol_version": PROTOCOL_VERSION,
        "dataset_kind": dataset_kind,
        "source_split": source_split,
        "dataset_id": prepared.bundle.manifest.dataset_id,
        "dataset_version": prepared.bundle.manifest.dataset_version,
        "declared_dataset_split": prepared.bundle.manifest.split,
        "dataset_signature": prepared.bundle.dataset_signature,
        "candidate_pool_manifest_verified": prepared.pool_manifest_verified,
        "candidate_pool_status": (
            "VERIFIED_SELECTOR_POOL_V2"
            if prepared.pool_manifest_verified
            else "UNVERIFIED_TEST_INPUT"
        ),
        "counts": {
            "dataset_queries": len(prepared.bundle.queries),
            "gold_cases": len(prepared.bundle.gold_cases),
            "candidate_windows": len(prepared.candidate_sets),
            "eligible_input_queries": len(prepared.query_ids),
            "niah_assignments": len(prepared.assignments),
            "components": len(components),
            "chain_eligible_queries": sum(prepared.chain_eligible.values()),
            "chain_eligible_components": _count_components(
                {
                    query_id
                    for query_id in prepared.query_ids
                    if prepared.chain_eligible[query_id]
                },
                component_by_query,
            ),
            "max_component_queries": max(component.n_queries for component in components),
        },
        "fold_query_counts": {
            str(index): count for index, count in enumerate(fold_loads)
        },
        "roles": role_counts,
        "crossing": crossing,
        "representatives": {
            "recall": {
                "records": len(recall_rows),
                "crc": source_split == "dev",
                "source_role": representative_role,
            },
            "chain": {
                "records": len(chain_rows),
                "crc": source_split == "dev",
                "source_role": representative_role,
            },
        },
        "evaluation_semantics": {
            "crc_representatives": (
                "dev crc-calibration only" if source_split == "dev" else "none"
            ),
            "decision_dev_inference": (
                "all eligible decision-dev queries; frozen components are cluster-bootstrap "
                "units, not CRC representatives"
                if source_split == "dev"
                else None
            ),
            "train_modelval_chain": (
                "diagnostic representatives only; never CRC"
                if source_split == "train"
                else None
            ),
        },
        "artifacts": first_artifacts,
        "canonical_projection_fingerprints": projection_fingerprints,
    }
    report_bytes = _json_bytes(report)

    manifest_path = Path(dataset_manifest_path)
    root = manifest_path.parent
    input_pins: dict[str, object] = {
        "dataset_manifest": _file_pin(manifest_path),
        "dataset_documents": _file_pin(root / prepared.bundle.manifest.documents_file),
        "dataset_queries": _file_pin(root / prepared.bundle.manifest.queries_file),
        "dataset_gold_cases": _file_pin(root / prepared.bundle.manifest.gold_cases_file),
        "source_parent": _file_pin(source_parent_path),
        "candidate_pool": _file_pin(prepared.candidate_path),
    }
    if prepared.pool_manifest_path is not None:
        input_pins["selector_candidate_pool_manifest_v2"] = _file_pin(
            prepared.pool_manifest_path
        )
    if prepared.assignment_path is not None:
        input_pins["niah_assignment"] = _file_pin(prepared.assignment_path)

    output_pins = dict(first_artifacts)
    output_pins[REPORT_FILE] = _bytes_pin(report_bytes)
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "protocol_version": PROTOCOL_VERSION,
        "dataset_kind": dataset_kind,
        "source_split": source_split,
        "dataset_id": prepared.bundle.manifest.dataset_id,
        "dataset_version": prepared.bundle.manifest.dataset_version,
        "declared_dataset_split": prepared.bundle.manifest.split,
        "dataset_signature": prepared.bundle.dataset_signature,
        "candidate_pool_manifest_verified": prepared.pool_manifest_verified,
        "candidate_pool_status": (
            "VERIFIED_SELECTOR_POOL_V2"
            if prepared.pool_manifest_verified
            else "UNVERIFIED_TEST_INPUT"
        ),
        "rules": {
            "allowed_key_prefixes": (
                ["query:", "parent:", "family:"]
                if dataset_kind == "niah"
                else ["query:", "parent:"]
            ),
            "twowiki_parent_scope": (
                "official-gold-only" if dataset_kind == "2wiki" else None
            ),
            "component_id": "sha256(''.join(query_id + '\\n' for sorted members))",
            "component_root": "min(all complete allowed-key strings)",
            "component_order": ["-n_queries", "component_root", "component_id"],
            "fold_assignment": "least current query count; ties use smaller fold",
            "n_folds": n_folds,
            "fold_roles": (
                {"0": "train-modelval", "1-9": "train-fit"}
                if source_split == "train"
                else {"0": "crc-calibration", "1": "decision-dev"}
            ),
            "top_k_chain_eligibility": TOP_K_BASELINE,
            "representative_seed": COMPONENT_REPRESENTATIVE_SEED,
            "representative_selector": (
                "evidence_rag.evaluation.selector_risk.select_component_representatives"
            ),
            "representative_digest": (
                "evidence_rag.evaluation.selector_risk.representative_digest"
            ),
            "representative_binding": (
                "each dataset/risk/query maps to exactly one explicit component_id"
            ),
            "train_crc_representatives": False,
        },
        "inputs": input_pins,
        "outputs": output_pins,
        "canonical_projection_fingerprints": projection_fingerprints,
        "counts": report["counts"],
        "crossing": crossing,
    }
    manifest_bytes = _json_bytes(manifest)
    files: dict[str, bytes] = {
        COMPONENT_MAP_FILE: component_bytes,
        ROLE_ASSIGNMENTS_FILE: role_bytes,
        RECALL_REPRESENTATIVES_FILE: recall_bytes,
        CHAIN_REPRESENTATIVES_FILE: chain_bytes,
        REPORT_FILE: report_bytes,
        MANIFEST_FILE: manifest_bytes,
    }
    return SelectorComponentArtifacts(files=files, report=report, manifest=manifest)


def freeze_selector_component_artifacts(
    output_directory: Path, artifacts: SelectorComponentArtifacts
) -> Path:
    """Write the complete bundle once; never replace or merge an existing bundle."""

    directory = Path(output_directory)
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"selector component output is not a directory: {directory}")
    if directory.is_dir():
        existing = sorted(path.name for path in directory.iterdir())
        if existing:
            raise ValueError(
                f"selector component output {directory} is not empty; refusing to overwrite "
                f"write-once artifacts: {existing[:5]}"
            )
    directory.mkdir(parents=True, exist_ok=True)
    for filename in OUTPUT_FILES:
        directory.joinpath(filename).write_bytes(artifacts.files[filename])
    return directory / MANIFEST_FILE


def verify_selector_component_artifacts(
    *,
    output_directory: Path,
    dataset_kind: DatasetKind,
    source_split: SourceSplit,
    dataset_manifest_path: Path,
    source_parent_path: Path,
    candidate_pool_path: Path,
    assignment_path: Path | None = None,
) -> SelectorComponentArtifacts:
    """Recompute the bundle from source inputs and require byte-identical frozen outputs."""

    expected = build_selector_component_artifacts(
        dataset_kind=dataset_kind,
        source_split=source_split,
        dataset_manifest_path=dataset_manifest_path,
        source_parent_path=source_parent_path,
        candidate_pool_path=candidate_pool_path,
        assignment_path=assignment_path,
    )
    directory = Path(output_directory)
    for filename in OUTPUT_FILES:
        path = directory / filename
        try:
            actual = path.read_bytes()
        except OSError as error:
            raise ValueError(f"missing or unreadable selector component artifact {path}: {error}") from error
        wanted = expected.files[filename]
        if actual != wanted:
            raise ValueError(
                f"selector component artifact differs from recomputation: {path}; "
                f"actual_sha256={_sha256_bytes(actual)}, "
                f"expected_sha256={_sha256_bytes(wanted)}"
            )
    return expected
