"""Strict, masked supervision artifacts for the conservative Selector.

The label builder deliberately keeps two boundaries separate:

* IDs, provenance and frozen split/component metadata are label/audit sidecars.
* The scorer-facing :class:`TextPair` contains only the question and one candidate passage.

Unjudged never means negative.  In particular, a retrieved NIAH distractor (including another
query's synthetic document) and a non-supporting 2Wiki Top20 candidate are masked unless a
separate audited source explicitly judges them.  R004 has no such irrelevant-evidence sidecar,
so it does not manufacture ``0/0`` examples.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from evidence_rag.contracts.models import CandidateSet
from evidence_rag.evaluation.selector_components import (
    OUTPUT_FILES as COMPONENT_OUTPUT_FILES,
)
from evidence_rag.evaluation.selector_components import (
    ROLE_ASSIGNMENTS_FILE,
    DatasetKind,
    NiahSelectorAssignment,
    SelectorComponentArtifacts,
    verify_selector_component_artifacts,
)
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.materializer.injector import alias_occurrences
from evidence_rag.materializer.provenance import MutationRecord
from evidence_rag.materializer.selector_pool import (
    CANDIDATE_FILE,
    SELECTOR_POOL_MANIFEST_FILE,
    verify_selector_candidate_pool_v2,
)
from evidence_rag.selector.answer_norm import canonicalize_answer

LABELS_FILE = "selector_labels.jsonl"
REPORT_FILE = "selector_labels_report.json"
MANIFEST_FILE = "selector_labels_manifest.json"
OUTPUT_FILES = (LABELS_FILE, REPORT_FILE, MANIFEST_FILE)
PROTOCOL_VERSION: Literal["selector-labels-v2"] = "selector-labels-v2"
SOURCE_SPLIT: Literal["train"] = "train"
EXPECTED_POOL_SIZE = 20

BinaryLabel: TypeAlias = Literal[0, 1]
TrainRole: TypeAlias = Literal["train-fit", "train-modelval"]
NonEmpty = Annotated[str, Field(min_length=1)]


@dataclass(frozen=True)
class TextPair:
    """The complete scorer input contract: no IDs, rank, source or provenance."""

    question: str
    candidate_text: str

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("TextPair question must not be blank")
        if not self.candidate_text.strip():
            raise ValueError("TextPair candidate_text must not be blank")

    def as_tokenizer_pair(self) -> tuple[str, str]:
        return self.question, self.candidate_text


def project_text_pair(*, question: str, candidate_text: str) -> TextPair:
    """Project already-selected text fields into the only model-facing record."""

    return TextPair(question=question, candidate_text=candidate_text)


def text_pair_sha256(pair: TextPair) -> str:
    """Bind a label row to exact model text without placing that text in the sidecar."""

    payload = _json_bytes(
        {"candidate_text": pair.candidate_text, "question": pair.question}, indent=None
    )
    return hashlib.sha256(payload).hexdigest()


class SelectorLabelRow(BaseModel):
    """One candidate's two independent binary targets and masks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    protocol_version: Literal["selector-labels-v2"] = PROTOCOL_VERSION
    dataset_id: NonEmpty
    dataset_kind: DatasetKind
    query_id: NonEmpty
    component_id: NonEmpty
    role: TrainRole
    evidence_id: NonEmpty
    document_id: NonEmpty
    text_pair_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    protect_label: BinaryLabel | None
    protect_mask: bool
    protect_source: NonEmpty
    harm_label: BinaryLabel | None
    harm_mask: bool
    harm_source: NonEmpty

    @model_validator(mode="after")
    def masks_match_labels(self) -> SelectorLabelRow:
        for head in ("protect", "harm"):
            label = getattr(self, f"{head}_label")
            mask = getattr(self, f"{head}_mask")
            if mask != (label is not None):
                raise ValueError(f"{head}_mask must be true exactly when {head}_label is set")
        return self


class _RoleBinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: NonEmpty
    query_id: NonEmpty
    component_id: NonEmpty
    fold: Annotated[int, Field(ge=0)]
    role: TrainRole
    chain_eligible_topk10: bool


@dataclass(frozen=True)
class SelectorLabelArtifacts:
    """Canonical bytes plus parsed summaries for one write-once label bundle."""

    files: Mapping[str, bytes]
    report: Mapping[str, object]
    manifest: Mapping[str, object]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_pin(path: Path) -> dict[str, object]:
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise ValueError(f"unable to read required Selector label input {path}: {error}") from error
    return {"bytes": len(payload), "sha256": _sha256_bytes(payload)}


def _bytes_pin(payload: bytes, *, records: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {"bytes": len(payload), "sha256": _sha256_bytes(payload)}
    if records is not None:
        result["records"] = records
    return result


def _json_bytes(payload: Mapping[str, object], *, indent: int | None = 2) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            sort_keys=True,
            allow_nan=False,
        )
        + ("" if indent is None else "\n")
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return (
        "\n".join(
            json.dumps(
                row,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
            for row in rows
        )
        + ("\n" if rows else "")
    ).encode("utf-8")


def _read_nonblank_jsonl(path: Path, *, label: str) -> tuple[tuple[int, bytes], ...]:
    try:
        lines = Path(path).read_bytes().splitlines()
    except OSError as error:
        raise ValueError(f"unable to read {label} {path}: {error}") from error
    if not lines:
        raise ValueError(f"{label} {path} is empty")
    values: list[tuple[int, bytes]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"blank {label} record at {path}:{line_number}")
        values.append((line_number, line))
    return tuple(values)


def _read_candidate_sets(path: Path) -> dict[str, CandidateSet]:
    values: dict[str, CandidateSet] = {}
    for line_number, line in _read_nonblank_jsonl(path, label="candidate-pool"):
        try:
            candidate_set = CandidateSet.model_validate_json(line)
        except (ValidationError, ValueError) as error:
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
        values[candidate_set.query_id] = candidate_set
    return values


def _read_role_bindings(payload: bytes, *, dataset_id: str) -> dict[str, _RoleBinding]:
    bindings: dict[str, _RoleBinding] = {}
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank R002 role record at line {line_number}")
        try:
            binding = _RoleBinding.model_validate_json(line)
        except (ValidationError, ValueError) as error:
            raise ValueError(f"invalid R002 role record at line {line_number}: {error}") from error
        if binding.dataset_id != dataset_id:
            raise ValueError(
                f"R002 role dataset_id={binding.dataset_id!r} differs from {dataset_id!r}"
            )
        if binding.query_id in bindings:
            raise ValueError(f"duplicate R002 role query ID: {binding.query_id}")
        bindings[binding.query_id] = binding
    if not bindings:
        raise ValueError("R002 role assignments are empty")
    return bindings


def _read_assignments(path: Path) -> dict[str, NiahSelectorAssignment]:
    values: dict[str, NiahSelectorAssignment] = {}
    for line_number, line in _read_nonblank_jsonl(path, label="NIAH assignment"):
        try:
            assignment = NiahSelectorAssignment.model_validate_json(line)
        except (ValidationError, ValueError) as error:
            raise ValueError(f"invalid NIAH assignment at {path}:{line_number}: {error}") from error
        if assignment.query_id in values:
            raise ValueError(f"duplicate NIAH assignment query ID: {assignment.query_id}")
        values[assignment.query_id] = assignment
    return values


def _read_provenance(path: Path) -> dict[str, MutationRecord]:
    values: dict[str, MutationRecord] = {}
    for line_number, line in _read_nonblank_jsonl(path, label="NIAH provenance"):
        try:
            record = MutationRecord.model_validate_json(line)
        except (ValidationError, ValueError) as error:
            raise ValueError(f"invalid NIAH provenance at {path}:{line_number}: {error}") from error
        if record.query_id in values:
            raise ValueError(f"duplicate NIAH provenance query ID: {record.query_id}")
        values[record.query_id] = record
    return values


def _same_keys(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    left_label: str,
    right_label: str,
) -> None:
    left_keys, right_keys = set(left), set(right)
    if left_keys != right_keys:
        raise ValueError(
            f"{left_label}/{right_label} query keys differ: "
            f"missing from {right_label}={sorted(left_keys - right_keys)[:5]}, "
            f"missing from {left_label}={sorted(right_keys - left_keys)[:5]}"
        )


def _verify_counterfactual_pair(
    *,
    assignment: NiahSelectorAssignment,
    record: MutationRecord,
    gold_document_ids: set[str],
    reference_answers: tuple[str, ...],
    text_by_document: Mapping[str, str],
) -> None:
    query_id = assignment.query_id
    if record.query_id != query_id:
        raise ValueError(f"NIAH provenance query mismatch for {query_id}")
    if assignment.harmful_document_id != record.counterfactual_document_id:
        raise ValueError(f"NIAH assignment/provenance harmful document mismatch for {query_id}")
    if set(assignment.required_document_ids) != gold_document_ids:
        raise ValueError(f"NIAH assignment/official gold mismatch for {query_id}")
    if record.needle_document_id not in gold_document_ids:
        raise ValueError(f"NIAH clean needle is not official required evidence for {query_id}")
    if record.counterfactual_document_id in gold_document_ids:
        raise ValueError(f"NIAH counterfactual is also official required evidence for {query_id}")
    expected_counterfactual_id = f"cf::{record.query_id}::{record.needle_document_id}"
    if record.counterfactual_document_id != expected_counterfactual_id:
        raise ValueError(f"NIAH counterfactual document ID is not canonical for {query_id}")
    if not reference_answers:
        raise ValueError(f"NIAH official reference answers are empty for {query_id}")
    if record.gold_alias_used not in reference_answers:
        raise ValueError(f"NIAH gold_alias_used is not an official reference answer for {query_id}")
    canonical_answers = {canonicalize_answer(answer) for answer in reference_answers}
    if canonical_answers != {record.gold_value}:
        raise ValueError(
            f"NIAH provenance gold value differs from official reference answers for {query_id}"
        )
    if canonicalize_answer(record.replacement_value) in canonical_answers:
        raise ValueError(f"NIAH replacement remains an accepted answer for {query_id}")

    try:
        clean = text_by_document[record.needle_document_id]
        counterfactual = text_by_document[record.counterfactual_document_id]
    except KeyError as error:
        raise ValueError(
            f"NIAH verified pair document is absent from the dataset for {query_id}: {error}"
        ) from error
    if hashlib.sha256(clean.encode("utf-8")).hexdigest() != record.text_hash_before:
        raise ValueError(f"NIAH clean text hash mismatch for {query_id}")
    if hashlib.sha256(counterfactual.encode("utf-8")).hexdigest() != record.text_hash_after:
        raise ValueError(f"NIAH counterfactual text hash mismatch for {query_id}")

    start, end = record.char_span
    if not (0 <= start < end <= len(clean)):
        raise ValueError(f"NIAH mutation span is outside the clean text for {query_id}")
    if clean[start:end] != record.gold_alias_used:
        raise ValueError(f"NIAH mutation span does not select gold_alias_used for {query_id}")
    if alias_occurrences(clean, record.gold_alias_used) != [(start, end)]:
        raise ValueError(f"NIAH gold alias is not unique at the recorded span for {query_id}")
    expected = clean[:start] + record.replacement_value + clean[end:]
    if counterfactual != expected:
        raise ValueError(f"NIAH counterfactual is not an exact one-span replacement for {query_id}")
    restored = (
        counterfactual[:start]
        + record.gold_alias_used
        + counterfactual[start + len(record.replacement_value) :]
    )
    if restored != clean:
        raise ValueError(
            f"NIAH counterfactual cannot be restored to its clean needle for {query_id}"
        )
    residual_aliases = [
        alias for alias in reference_answers if alias_occurrences(counterfactual, alias)
    ]
    if residual_aliases:
        raise ValueError(
            f"NIAH counterfactual retains official answer aliases for {query_id}: "
            f"{residual_aliases[:3]}"
        )


def _label_row(
    *,
    dataset_id: str,
    dataset_kind: DatasetKind,
    binding: _RoleBinding,
    evidence_id: str,
    document_id: str,
    pair: TextPair,
    protect_label: BinaryLabel | None,
    protect_source: str,
    harm_label: BinaryLabel | None,
    harm_source: str,
) -> SelectorLabelRow:
    return SelectorLabelRow(
        dataset_id=dataset_id,
        dataset_kind=dataset_kind,
        query_id=binding.query_id,
        component_id=binding.component_id,
        role=binding.role,
        evidence_id=evidence_id,
        document_id=document_id,
        text_pair_sha256=text_pair_sha256(pair),
        protect_label=protect_label,
        protect_mask=protect_label is not None,
        protect_source=protect_source,
        harm_label=harm_label,
        harm_mask=harm_label is not None,
        harm_source=harm_source,
    )


def _niah_targets(
    *,
    document_id: str,
    assignment: NiahSelectorAssignment,
    record: MutationRecord,
) -> tuple[BinaryLabel | None, str, BinaryLabel | None, str]:
    if document_id == record.counterfactual_document_id:
        return 0, "verified_own_counterfactual", 1, "verified_own_counterfactual"
    if document_id in assignment.required_document_ids:
        if document_id == record.needle_document_id:
            return 1, "official_required", 0, "verified_clean_counterpart"
        return 1, "official_required", None, "masked_no_clean_counterpart_judgment"
    return None, "masked_unjudged", None, "masked_unjudged"


def _twowiki_targets(
    *, document_id: str, gold_document_ids: set[str]
) -> tuple[BinaryLabel | None, str, BinaryLabel | None, str]:
    if document_id in gold_document_ids:
        return 1, "official_supporting", None, "masked_no_harm_labels"
    return None, "masked_unjudged", None, "masked_unjudged"


def _count_rows(rows: Sequence[SelectorLabelRow]) -> dict[str, object]:
    def label_name(value: BinaryLabel | None) -> str:
        return "mask" if value is None else str(value)

    label_pairs: dict[str, int] = {}
    roles: dict[str, int] = {}
    for row in rows:
        key = f"protect={label_name(row.protect_label)},harm={label_name(row.harm_label)}"
        label_pairs[key] = label_pairs.get(key, 0) + 1
        roles[row.role] = roles.get(row.role, 0) + 1
    return {
        "candidate_rows": len(rows),
        "queries": len({row.query_id for row in rows}),
        "components": len({row.component_id for row in rows}),
        "roles": dict(sorted(roles.items())),
        "label_pairs": dict(sorted(label_pairs.items())),
        "protect_supervised": sum(row.protect_mask for row in rows),
        "protect_masked": sum(not row.protect_mask for row in rows),
        "harm_supervised": sum(row.harm_mask for row in rows),
        "harm_masked": sum(not row.harm_mask for row in rows),
        "both_supervised": sum(row.protect_mask and row.harm_mask for row in rows),
        "both_masked": sum(not row.protect_mask and not row.harm_mask for row in rows),
    }


def _component_inputs(directory: Path) -> dict[str, object]:
    return {filename: _file_pin(Path(directory) / filename) for filename in COMPONENT_OUTPUT_FILES}


def _verify_upstream(
    *,
    dataset_kind: DatasetKind,
    dataset_manifest_path: Path,
    source_parent_path: Path,
    candidate_pool_path: Path,
    component_directory: Path,
    assignment_path: Path | None,
) -> SelectorComponentArtifacts:
    pool_path = Path(candidate_pool_path)
    if not pool_path.is_dir():
        raise ValueError("Selector labels require a frozen Selector-v2 candidate pool directory")
    pool_manifest_path = pool_path / SELECTOR_POOL_MANIFEST_FILE
    if not pool_manifest_path.is_file():
        raise ValueError(f"candidate pool is missing {SELECTOR_POOL_MANIFEST_FILE}")
    pool_manifest = verify_selector_candidate_pool_v2(pool_path, dataset_manifest_path)
    expected_role = f"{dataset_kind}-{SOURCE_SPLIT}"
    if pool_manifest.data_role != expected_role:
        raise ValueError(
            f"candidate pool data_role={pool_manifest.data_role!r}, expected {expected_role!r}"
        )
    components = verify_selector_component_artifacts(
        output_directory=component_directory,
        dataset_kind=dataset_kind,
        source_split=SOURCE_SPLIT,
        dataset_manifest_path=dataset_manifest_path,
        source_parent_path=source_parent_path,
        candidate_pool_path=candidate_pool_path,
        assignment_path=assignment_path,
    )
    if components.manifest.get("candidate_pool_manifest_verified") is not True:
        raise ValueError("R002 components are not bound to a verified Selector-v2 pool")
    if components.manifest.get("source_split") != SOURCE_SPLIT:
        raise ValueError("R002 components are not the frozen train split")
    if components.manifest.get("dataset_kind") != dataset_kind:
        raise ValueError("R002 component dataset kind differs from the label request")
    return components


def build_selector_label_artifacts(
    *,
    dataset_kind: DatasetKind,
    dataset_manifest_path: Path,
    source_parent_path: Path,
    candidate_pool_path: Path,
    component_directory: Path,
    assignment_path: Path | None = None,
    provenance_path: Path | None = None,
) -> SelectorLabelArtifacts:
    """Validate every frozen input and build canonical R004 label/audit bytes."""

    if dataset_kind == "niah" and (assignment_path is None or provenance_path is None):
        raise ValueError("NIAH labels require both assignment and provenance JSONL inputs")
    if dataset_kind == "2wiki" and (assignment_path is not None or provenance_path is not None):
        raise ValueError("2Wiki labels must not receive NIAH assignment or provenance inputs")

    components = _verify_upstream(
        dataset_kind=dataset_kind,
        dataset_manifest_path=dataset_manifest_path,
        source_parent_path=source_parent_path,
        candidate_pool_path=candidate_pool_path,
        component_directory=component_directory,
        assignment_path=assignment_path,
    )
    bundle = JsonlDatasetAdapter.load(dataset_manifest_path)
    if components.manifest.get("dataset_signature") != bundle.dataset_signature:
        raise ValueError("R002 components and dataset signatures differ")
    query_by_id = {query.query_id: query for query in bundle.queries}
    gold_by_query = {
        gold.query_id: set(gold.relevant_document_ids or ()) for gold in bundle.gold_cases
    }
    answers_by_query = {
        gold.query_id: tuple(gold.reference_answers or ()) for gold in bundle.gold_cases
    }
    _same_keys(query_by_id, gold_by_query, left_label="dataset", right_label="official gold")
    if any(not documents for documents in gold_by_query.values()):
        raise ValueError("every Selector label query must have official required evidence")
    text_by_document = {document.document_id: document.text for document in bundle.documents}

    roles_payload = components.files.get(ROLE_ASSIGNMENTS_FILE)
    if roles_payload is None:
        raise ValueError("verified R002 components omitted role_assignments.jsonl")
    bindings = _read_role_bindings(roles_payload, dataset_id=bundle.manifest.dataset_id)
    candidate_path = Path(candidate_pool_path) / CANDIDATE_FILE
    candidate_sets = _read_candidate_sets(candidate_path)

    assignments: dict[str, NiahSelectorAssignment] = {}
    provenance: dict[str, MutationRecord] = {}
    verified_pairs = 0
    provenance_records = 0
    unused_provenance_records = 0
    verified_clean_candidate_texts = 0
    verified_harmful_candidate_texts = 0
    if dataset_kind == "niah":
        if assignment_path is None or provenance_path is None:
            raise AssertionError("NIAH paths were not narrowed")
        assignments = _read_assignments(assignment_path)
        provenance = _read_provenance(provenance_path)
        provenance_records = len(provenance)
        _same_keys(bindings, assignments, left_label="R002 roles", right_label="NIAH assignment")
        missing_provenance = sorted(set(assignments) - set(provenance))
        if missing_provenance:
            raise ValueError(
                f"NIAH assignments have no provenance records: {missing_provenance[:5]}"
            )
        provenance_outside_dataset = sorted(set(provenance) - set(query_by_id))
        if provenance_outside_dataset:
            raise ValueError(
                "NIAH provenance contains query IDs outside the pinned source-train dataset: "
                f"{provenance_outside_dataset[:5]}"
            )
        for query_id in sorted(assignments):
            if query_id not in query_by_id:
                raise ValueError(f"NIAH assignment query is absent from the dataset: {query_id}")
            _verify_counterfactual_pair(
                assignment=assignments[query_id],
                record=provenance[query_id],
                gold_document_ids=gold_by_query[query_id],
                reference_answers=answers_by_query[query_id],
                text_by_document=text_by_document,
            )
            verified_pairs += 1
        unused_provenance_records = len(set(provenance) - set(assignments))
    else:
        _same_keys(bindings, query_by_id, left_label="R002 roles", right_label="dataset")

    missing_candidate_queries = sorted(set(bindings) - set(candidate_sets))
    if missing_candidate_queries:
        raise ValueError(
            f"R002 role queries are absent from the candidate pool: {missing_candidate_queries[:5]}"
        )

    label_rows: list[SelectorLabelRow] = []
    projection_rows: list[dict[str, object]] = []
    for query_id in sorted(bindings):
        binding = bindings[query_id]
        query = query_by_id[query_id]
        candidates = sorted(
            candidate_sets[query_id].candidates,
            key=lambda item: (item.retrieval_rank, item.evidence_id),
        )
        for candidate in candidates:
            pair = project_text_pair(question=query.text, candidate_text=candidate.text)
            if dataset_kind == "niah":
                record = provenance[query_id]
                if candidate.document_id == record.counterfactual_document_id:
                    if record.replacement_value not in candidate.text:
                        raise ValueError(
                            f"NIAH harmful candidate text omits the replacement for {query_id}"
                        )
                    residual = [
                        alias
                        for alias in answers_by_query[query_id]
                        if alias_occurrences(candidate.text, alias)
                    ]
                    if residual:
                        raise ValueError(
                            f"NIAH harmful candidate text retains official answers for "
                            f"{query_id}: {residual[:3]}"
                        )
                    verified_harmful_candidate_texts += 1
                if candidate.document_id == record.needle_document_id:
                    if not alias_occurrences(candidate.text, record.gold_alias_used):
                        raise ValueError(
                            f"NIAH clean candidate text omits gold_alias_used for {query_id}"
                        )
                    verified_clean_candidate_texts += 1
                protect, protect_source, harm, harm_source = _niah_targets(
                    document_id=candidate.document_id,
                    assignment=assignments[query_id],
                    record=record,
                )
            else:
                protect, protect_source, harm, harm_source = _twowiki_targets(
                    document_id=candidate.document_id,
                    gold_document_ids=gold_by_query[query_id],
                )
            row = _label_row(
                dataset_id=bundle.manifest.dataset_id,
                dataset_kind=dataset_kind,
                binding=binding,
                evidence_id=candidate.evidence_id,
                document_id=candidate.document_id,
                pair=pair,
                protect_label=protect,
                protect_source=protect_source,
                harm_label=harm,
                harm_source=harm_source,
            )
            label_rows.append(row)
            projection_rows.append(
                {
                    "dataset_id": row.dataset_id,
                    "evidence_id": row.evidence_id,
                    "query_id": row.query_id,
                    "text_pair_sha256": row.text_pair_sha256,
                }
            )

    row_payloads = [row.model_dump(mode="json") for row in label_rows]
    labels_bytes = _jsonl_bytes(row_payloads)
    projection_bytes = _jsonl_bytes(projection_rows)
    counts = _count_rows(label_rows)
    report: dict[str, object] = {
        "schema_version": "1.0",
        "protocol_version": PROTOCOL_VERSION,
        "dataset_kind": dataset_kind,
        "source_split": SOURCE_SPLIT,
        "dataset_id": bundle.manifest.dataset_id,
        "dataset_version": bundle.manifest.dataset_version,
        "dataset_signature": bundle.dataset_signature,
        "status": "LABEL_AUDIT_READY",
        "counts": counts,
        "niah_verified_clean_counterfactual_pairs": verified_pairs,
        "niah_provenance_scope": {
            "source_train_records": provenance_records,
            "used_assignment_records": verified_pairs,
            "unused_source_train_records": unused_provenance_records,
            "outside_pinned_dataset_records": 0,
        },
        "niah_candidate_text_semantics": {
            "verified_clean_candidates_with_gold_alias": verified_clean_candidate_texts,
            "verified_harmful_candidates_with_replacement_and_no_gold_alias": (
                verified_harmful_candidate_texts
            ),
            "violations": 0,
        },
        "rules": {
            "heads": "two independent binary targets with independent masks",
            "niah_official_required": "protect=1; harm is masked except verified clean needle",
            "niah_verified_clean_needle": "protect=1,harm=0",
            "niah_own_counterfactual": "protect=0,harm=1",
            "niah_other_candidates": "protect=mask,harm=mask",
            "twowiki_official_supporting": "protect=1,harm=mask",
            "twowiki_unjudged_top20": "protect=mask,harm=mask",
            "explicit_irrelevant": (
                "none: no audited irrelevant sidecar exists in the frozen R004 inputs"
            ),
            "model_input_fields": ["question", "candidate_text"],
            "forbidden_model_input_fields": [
                "query_id",
                "evidence_id",
                "document_id",
                "retrieval_rank",
                "retrieval_score",
                "source_uri",
                "metadata",
                "component_id",
                "role",
                "source_parent_id",
                "synthetic_family",
                "provenance",
                "labels",
            ],
        },
        "artifacts": {LABELS_FILE: _bytes_pin(labels_bytes, records=len(label_rows))},
        "model_text_projection": _bytes_pin(projection_bytes, records=len(projection_rows)),
    }
    report_bytes = _json_bytes(report)

    dataset_root = Path(dataset_manifest_path).parent
    inputs: dict[str, object] = {
        "dataset_manifest": _file_pin(dataset_manifest_path),
        "dataset_documents": _file_pin(dataset_root / bundle.manifest.documents_file),
        "dataset_queries": _file_pin(dataset_root / bundle.manifest.queries_file),
        "dataset_gold_cases": _file_pin(dataset_root / bundle.manifest.gold_cases_file),
        "source_parent_split_only": _file_pin(source_parent_path),
        "candidate_pool": _file_pin(candidate_path),
        "selector_candidate_pool_manifest_v2": _file_pin(
            Path(candidate_pool_path) / SELECTOR_POOL_MANIFEST_FILE
        ),
        "r002_components": _component_inputs(component_directory),
    }
    if assignment_path is not None and provenance_path is not None:
        inputs["niah_assignment_label_sidecar"] = _file_pin(assignment_path)
        inputs["niah_provenance_label_sidecar"] = _file_pin(provenance_path)

    output_pins = {
        LABELS_FILE: _bytes_pin(labels_bytes, records=len(label_rows)),
        REPORT_FILE: _bytes_pin(report_bytes),
    }
    manifest: dict[str, object] = {
        "schema_version": "2.0",
        "manifest_type": "selector_label_manifest",
        "protocol_version": PROTOCOL_VERSION,
        "dataset_kind": dataset_kind,
        "source_split": SOURCE_SPLIT,
        "dataset_id": bundle.manifest.dataset_id,
        "dataset_version": bundle.manifest.dataset_version,
        "dataset_signature": bundle.dataset_signature,
        "candidate_pool_manifest_verified": True,
        "r002_components_verified_byte_identical": True,
        "inputs": inputs,
        "outputs": output_pins,
        "counts": counts,
        "rules": report["rules"],
        "model_text_projection": report["model_text_projection"],
        "artifact_status": {
            "labels": "COMPLETE",
            "label_audit": "COMPLETE",
            "model_checkpoint": "NOT_APPLICABLE_R004_LABEL_STAGE",
            "selector_decision_trace": "NOT_APPLICABLE_R004_LABEL_STAGE",
            "sealed_or_heldout_effect": "NOT_ACCESSED",
        },
    }
    manifest_bytes = _json_bytes(manifest)
    return SelectorLabelArtifacts(
        files={
            LABELS_FILE: labels_bytes,
            REPORT_FILE: report_bytes,
            MANIFEST_FILE: manifest_bytes,
        },
        report=report,
        manifest=manifest,
    )


def freeze_selector_label_artifacts(
    output_directory: Path, artifacts: SelectorLabelArtifacts
) -> Path:
    """Write a complete label bundle once; never merge or overwrite it."""

    directory = Path(output_directory)
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"Selector label output is not a directory: {directory}")
    if directory.is_dir():
        existing = sorted(path.name for path in directory.iterdir())
        if existing:
            raise ValueError(
                f"Selector label output {directory} is not empty; refusing to overwrite "
                f"write-once artifacts: {existing[:5]}"
            )
    directory.mkdir(parents=True, exist_ok=True)
    for filename in OUTPUT_FILES:
        (directory / filename).write_bytes(artifacts.files[filename])
    return directory / MANIFEST_FILE


def verify_selector_label_artifacts(
    *,
    output_directory: Path,
    dataset_kind: DatasetKind,
    dataset_manifest_path: Path,
    source_parent_path: Path,
    candidate_pool_path: Path,
    component_directory: Path,
    assignment_path: Path | None = None,
    provenance_path: Path | None = None,
) -> SelectorLabelArtifacts:
    """Recompute from all frozen inputs and require byte-identical label artifacts."""

    expected = build_selector_label_artifacts(
        dataset_kind=dataset_kind,
        dataset_manifest_path=dataset_manifest_path,
        source_parent_path=source_parent_path,
        candidate_pool_path=candidate_pool_path,
        component_directory=component_directory,
        assignment_path=assignment_path,
        provenance_path=provenance_path,
    )
    directory = Path(output_directory)
    for filename in OUTPUT_FILES:
        path = directory / filename
        try:
            actual = path.read_bytes()
        except OSError as error:
            raise ValueError(
                f"missing or unreadable Selector label artifact {path}: {error}"
            ) from error
        wanted = expected.files[filename]
        if actual != wanted:
            raise ValueError(
                f"Selector label artifact differs from recomputation: {path}; "
                f"actual_sha256={_sha256_bytes(actual)}, "
                f"expected_sha256={_sha256_bytes(wanted)}"
            )
    return expected
