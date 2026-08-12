"""Run the pre-registered R005 dual-head sanity gate.

R005 is deliberately smaller than a real training run.  It asks whether the audited labels and
dual-head implementation can (a) memorise a fixed 16+16-query ``train-fit`` sample and (b) expose
at least one conservative, non-zero deletion corner on the untouched ``train-modelval`` role.
It never reads dev, sealed, heldout, CRC-calibration or decision-dev data, and its checkpoint and
diagnostic thresholds are explicitly forbidden as initialisation or policy choices for R006+.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import platform
import socket
import sys
import time
import tomllib
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

from pydantic import ValidationError

import evidence_rag.cli.finalize_selector_r005 as finalize_r005_module
import evidence_rag.evaluation.selector_sanity as selector_sanity_module
import evidence_rag.selector.dual_head as dual_head_module
import evidence_rag.selector.models as selector_models_module
import evidence_rag.selector.risk_controlled as risk_controlled_module
from evidence_rag.cli.finalize_selector_r004 import finalize_selector_r004
from evidence_rag.cli.finalize_selector_r005 import (
    R005BoundaryReport,
    R005CheckpointManifest,
    R005SanityManifest,
    build_r005_checkpoint_manifest,
    build_r005_sanity_manifest,
    finalize_selector_r005,
    freeze_or_verify_r005_checkpoint_manifest,
    freeze_or_verify_r005_sanity_manifest,
)
from evidence_rag.cli.run_selector_preflight import (
    _audit_model_snapshot,
    _DatasetArguments,
    _expected_model_identity,
    _git_snapshot,
    _label_artifacts,
    _load_config,
    _module_path,
    _Pair,
    _prepare_dataset,
    _PreparedDataset,
)
from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate, Query
from evidence_rag.evaluation.selector_artifacts import (
    DeviceInfo,
    DeviceKind,
    FilePin,
    GitPin,
    RuntimeInfo,
    SelectorExperimentManifestV2,
    SelectorInputPins,
    SelectorRunStatus,
    pin_file,
    verify_file_pin,
)
from evidence_rag.evaluation.selector_components import DatasetKind, NiahSelectorAssignment
from evidence_rag.evaluation.selector_controls import (
    COUNT_MATCHED_REPEATS,
    generate_count_matched_drops,
    verify_count_matched_protocol_artifacts,
)
from evidence_rag.evaluation.selector_risk import (
    conditional_chain_loss,
    document_recall,
    relative_recall_loss,
)
from evidence_rag.evaluation.selector_sanity import (
    SanityCandidateScoreRow,
    SanityPolicyQueryOutcome,
    SanityQuantileThreshold,
    SanitySourceHeadLossTrend,
    SanityTrainingPrediction,
    SanityTrainingTrace,
    compute_class_weights,
    decide_safe_corner,
    derive_train_fit_thresholds,
    evaluate_policy_metrics,
    evaluate_training_sanity,
    select_sanity_queries,
)
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.materializer.selector_labels import project_text_pair, text_pair_sha256
from evidence_rag.materializer.selector_pool import CANDIDATE_FILE
from evidence_rag.selector.dual_head import (
    fingerprint_dual_head_model,
    load_dual_head_checkpoint,
    load_dual_head_model,
    masked_dual_head_bce,
    save_dual_head_checkpoint,
)
from evidence_rag.selector.models import CandidateRiskScore, SelectorAction
from evidence_rag.selector.risk_controlled import RiskControlledSelector

PROTOCOL_VERSION = "selector-r005-sanity-v1"
RUN_ID = "R005"
SEED = 13
SAMPLE_SEED = 20260811
QUESTIONS_PER_DATASET = 16
SANITY_EPOCHS = 30
MINIMUM_ACCURACY = 0.95
MINIMUM_NIAH_PAIRS = 12
PAIR_DIRECTION_ACCURACY = 0.95
CLASSIFICATION_THRESHOLD = 0.5
DIAGNOSTIC_QUANTILES = (0.99, 0.975, 0.95, 0.90)
DIAGNOSTIC_CAP: Literal[1] = 1
MAX_HELDOUT_LOSS = 0.03
DATASET_KINDS: tuple[DatasetKind, DatasetKind] = ("niah", "2wiki")
QueryValue = TypeVar("QueryValue")

CONFIG_FILE = "config.toml"
CHECKPOINT_DIRECTORY = "checkpoint"
SANITY_DIRECTORY = "sanity"
CHECKPOINT_FILE = "checkpoint/model.safetensors"
CHECKPOINT_MANIFEST_FILE = "checkpoint/checkpoint_manifest.json"
SAMPLE_FILE = "sanity/sanity_sample.jsonl"
TRAINING_TRACE_FILE = "sanity/training_trace.jsonl"
CANDIDATE_SCORES_FILE = "sanity/candidate_scores.jsonl"
QUANTILE_POLICIES_FILE = "sanity/quantile_policies.json"
DECISION_TRACE_FILE = "sanity/decision_trace.jsonl"
SELECTION_RESULTS_FILE = "sanity/selection_results.jsonl"
SELECTED_SETS_FILE = "sanity/selected_sets.jsonl"
COUNT_MATCHED_FILE = "sanity/count_matched_controls.jsonl"
REPORT_FILE = "sanity/sanity_report.json"
LOG_FILE = "sanity/sanity_log.jsonl"
SANITY_MANIFEST_FILE = "sanity/sanity_manifest.json"
PREREQUISITE_FILES = (
    CONFIG_FILE,
    CHECKPOINT_FILE,
    CHECKPOINT_MANIFEST_FILE,
    SAMPLE_FILE,
    TRAINING_TRACE_FILE,
    CANDIDATE_SCORES_FILE,
    QUANTILE_POLICIES_FILE,
    DECISION_TRACE_FILE,
    SELECTION_RESULTS_FILE,
    SELECTED_SETS_FILE,
    COUNT_MATCHED_FILE,
    REPORT_FILE,
    LOG_FILE,
    SANITY_MANIFEST_FILE,
)


@dataclass(frozen=True)
class _SanityConfig:
    raw: Mapping[str, object]
    learning_rate: float
    batch_size: int
    accumulation: int
    weight_decay: float
    beta1: float
    beta2: float
    epsilon: float


@dataclass(frozen=True)
class _LoadedDataset:
    prepared: _PreparedDataset
    query_by_id: Mapping[str, Query]
    candidate_by_query: Mapping[str, CandidateSet]
    gold_by_query: Mapping[str, tuple[str, ...]]
    dataset_id: str
    dataset_signature: str
    pool_sha256: str
    assignment_by_query: Mapping[str, NiahSelectorAssignment]


@dataclass(frozen=True)
class _ScoreRow:
    dataset_kind: DatasetKind
    query_id: str
    evidence_id: str
    document_id: str
    retrieval_rank: int
    role: str
    score_scope: str
    text_pair_sha256: str
    protect_score: float
    harm_score: float
    safe_score: float

    def as_row(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "protocol_version": PROTOCOL_VERSION,
            **asdict(self),
        }


@dataclass(frozen=True)
class _PolicyOutput:
    policy_id: str
    quantile: float | None
    threshold: float
    selection_by_query: Mapping[tuple[DatasetKind, str], tuple[str, ...]]
    result_rows: tuple[Mapping[str, object], ...]
    trace_rows: tuple[Mapping[str, object], ...]
    selected_rows: tuple[Mapping[str, object], ...]


TrainingTermination = Literal["completed", "sample-coverage", "cuda-oom", "nonfinite"]


@dataclass(frozen=True)
class _SampleAssessment:
    class_counts: Counter[tuple[DatasetKind, str, int]]
    paired_query_ids: tuple[str, ...]
    missing_required_classes: tuple[tuple[DatasetKind, str, int], ...]
    strict_niah_pair_count: int
    required_strict_niah_pairs: int
    failed_checks: tuple[str, ...]
    status: Literal["PASS", "FAIL"]

    def as_report(self) -> Mapping[str, object]:
        return {
            "status": self.status,
            "missing_required_classes": [
                f"{kind}.{head}:{label}" for kind, head, label in self.missing_required_classes
            ],
            "strict_niah_pair_count": self.strict_niah_pair_count,
            "required_strict_niah_pairs": self.required_strict_niah_pairs,
            "failed_checks": list(self.failed_checks),
        }


@dataclass(frozen=True)
class _TrainingOutcome:
    epoch_rows: tuple[Mapping[str, object], ...]
    optimizer_steps: int
    epochs_completed: int
    termination_reason: TrainingTermination
    failure_epoch: int | None


class _NonFiniteTrainingError(RuntimeError):
    """Expected fail-closed termination for non-finite R005 training state."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or verify formal R005 dual-head sanity")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--model-snapshot", required=True, type=Path)
    parser.add_argument("--r004-root", required=True, type=Path)
    parser.add_argument("--count-matched-protocol-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed table")
    return cast(Mapping[str, object], value)


def _integer(section: Mapping[str, object], name: str) -> int:
    value = section.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"config {name} must be an integer")
    return value


def _number(section: Mapping[str, object], name: str) -> float:
    value = section.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"config {name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"config {name} must be finite")
    return numeric


def _load_sanity_config(path: Path) -> tuple[Any, _SanityConfig]:
    common = _load_config(path)
    try:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"unable to load R005 config {path}: {error}") from error
    training = _mapping(raw.get("training"), label="[training]")
    sanity = _mapping(raw.get("sanity"), label="[sanity]")
    expected: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "sample_seed": SAMPLE_SEED,
        "sample_role": "train-fit",
        "questions_per_dataset": QUESTIONS_PER_DATASET,
        "sample_order": (
            "ascending-sha256(protocol,newline,dataset-kind,newline,query-id,newline,seed)"
        ),
        "sample_selection_uses_labels_lengths_or_model_scores": False,
        "epochs": SANITY_EPOCHS,
        "checkpoint_rule": "final-epoch-only",
        "checkpoint_purpose": "R005-sanity-only-never-used-to-initialize-R006",
        "minimum_training_accuracy": MINIMUM_ACCURACY,
        "required_niah_active_classes": ["protect:0", "protect:1", "harm:0", "harm:1"],
        "required_twowiki_active_classes": ["protect:1"],
        "twowiki_not_applicable_classes": ["protect:0", "harm:0", "harm:1"],
        "minimum_niah_paired_queries": MINIMUM_NIAH_PAIRS,
        "minimum_pair_direction_accuracy": PAIR_DIRECTION_ACCURACY,
        "pair_direction_rule": (
            "protect(clean)>protect(cf),harm(cf)>harm(clean),safe(cf)>safe(clean);ties-fail"
        ),
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "heldout_role": "train-modelval",
        "heldout_scope": "all-403-queries-top20-score-top10-action",
        "expected_niah_heldout_queries": 103,
        "expected_twowiki_heldout_queries": 300,
        "diagnostic_policy": "safe-score-0-to-cap1-only",
        "diagnostic_cap": DIAGNOSTIC_CAP,
        "diagnostic_quantiles": list(DIAGNOSTIC_QUANTILES),
        "diagnostic_quantile_source": (
            "all-train-fit-topk10-safe-scores-merged-across-datasets-without-label-filtering"
        ),
        "diagnostic_quantile_method": "nearest-rank",
        "duplicate_quantile_threshold_policy": ("retain-duplicate-policies-without-replacement"),
        "diagnostic_thresholds_are_reused_by_r006_or_r007": False,
        "maximum_heldout_recall_loss": MAX_HELDOUT_LOSS,
        "maximum_heldout_conditional_chain_loss": MAX_HELDOUT_LOSS,
        "harmful_reduction_requirement": "pool-conditional-point>0",
        "deletion_precision_requirement": (
            "strictly-greater-than-100-repeat-count-matched-random-mean"
        ),
        "heldout_may_change_checkpoint_or_thresholds": False,
        "diagnostic_witness_rule": (
            "first-passing-policy-in-conservative-to-aggressive-config-order"
        ),
    }
    mismatches = sorted(name for name, value in expected.items() if sanity.get(name) != value)
    if mismatches:
        raise ValueError(f"R005 config differs from the pre-registered sanity rules: {mismatches}")
    training_expected: dict[str, object] = {
        "class_weighting": "inverse-sqrt-frequency-per-source-head-active-classes-only",
        "class_weight_normalization": "mean-active-weight-one-per-source-head",
        "weighted_loss_normalization": "active-count-per-head",
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 0.00000001,
        "scheduler": "none",
        "gradient_clipping": "none",
    }
    training_mismatches = sorted(
        name for name, value in training_expected.items() if training.get(name) != value
    )
    if training_mismatches:
        raise ValueError(
            f"R005 config differs from the frozen optimizer/weighting rules: {training_mismatches}"
        )
    return common, _SanityConfig(
        raw=raw,
        learning_rate=_number(training, "learning_rate"),
        batch_size=_integer(training, "batch_size"),
        accumulation=_integer(training, "gradient_accumulation_steps"),
        weight_decay=_number(training, "weight_decay"),
        beta1=_number(training, "adam_beta1"),
        beta2=_number(training, "adam_beta2"),
        epsilon=_number(training, "adam_epsilon"),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(values: Sequence[Mapping[str, object]]) -> bytes:
    return (
        "\n".join(
            json.dumps(
                value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True
            )
            for value in values
        )
        + ("\n" if values else "")
    ).encode("utf-8")


def _write_once(path: Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite R005 artifact: {destination}") from error


def _resolved_pin_path(pin: FilePin, *, root: Path) -> Path:
    path = Path(pin.path)
    return (path if path.is_absolute() else Path(root) / path).resolve()


def _pin_index(manifest: SelectorExperimentManifestV2) -> dict[str, FilePin]:
    pins = manifest.inputs.all_pins()
    result = {pin.label: pin for pin in pins}
    if len(result) != len(pins):
        raise ValueError("R004 input pin labels are not globally unique")
    return result


def _required_pin(index: Mapping[str, FilePin], label: str, *, root: Path) -> Path:
    try:
        pin = index[label]
    except KeyError as error:
        raise ValueError(f"R004 manifest omits required input pin {label!r}") from error
    path = verify_file_pin(pin, root=root)
    _refuse_forbidden_path(path)
    return path.resolve()


def _refuse_forbidden_path(path: Path) -> None:
    raw = str(path).lower()
    resolved = str(Path(path).resolve(strict=False)).lower()
    if "sealed" in raw or "heldout" in raw or "sealed" in resolved or "heldout" in resolved:
        raise ValueError(f"R005 refuses sealed/heldout data paths: {path}")


def _dataset_arguments_from_r004(
    manifest: SelectorExperimentManifestV2, *, r004_root: Path
) -> tuple[_DatasetArguments, ...]:
    index = _pin_index(manifest)
    root = Path(r004_root).resolve()
    niah_pool_file = _required_pin(index, "niah/candidate_pool", root=root)
    wiki_pool_file = _required_pin(index, "2wiki/candidate_pool", root=root)
    niah_component = _required_pin(index, "niah/components/component_map.jsonl", root=root)
    wiki_component = _required_pin(index, "2wiki/components/component_map.jsonl", root=root)
    niah_labels = _required_pin(index, "niah/labels/selector_labels.jsonl", root=root)
    wiki_labels = _required_pin(index, "2wiki/labels/selector_labels.jsonl", root=root)
    return (
        _DatasetArguments(
            kind="niah",
            dataset_manifest=_required_pin(index, "niah/dataset_manifest", root=root),
            source_parent=_required_pin(index, "niah/source_parent", root=root),
            candidate_pool=niah_pool_file.parent,
            components_dir=niah_component.parent,
            labels_dir=niah_labels.parent,
            assignment=_required_pin(index, "niah/assignment", root=root),
            provenance=_required_pin(index, "niah/provenance", root=root),
        ),
        _DatasetArguments(
            kind="2wiki",
            dataset_manifest=_required_pin(index, "2wiki/dataset_manifest", root=root),
            source_parent=_required_pin(index, "2wiki/source_parent", root=root),
            candidate_pool=wiki_pool_file.parent,
            components_dir=wiki_component.parent,
            labels_dir=wiki_labels.parent,
        ),
    )


def _read_candidate_sets(path: Path) -> dict[str, CandidateSet]:
    values: dict[str, CandidateSet] = {}
    for line_number, line in enumerate(Path(path).read_bytes().splitlines(), start=1):
        try:
            value = CandidateSet.model_validate_json(line)
        except (ValidationError, ValueError) as error:
            raise ValueError(f"invalid candidate pool row {path}:{line_number}: {error}") from error
        if value.query_id in values:
            raise ValueError(f"duplicate candidate query ID: {value.query_id}")
        values[value.query_id] = value
    if not values:
        raise ValueError(f"candidate pool is empty: {path}")
    return values


def _read_assignments(path: Path) -> dict[str, NiahSelectorAssignment]:
    values: dict[str, NiahSelectorAssignment] = {}
    for line_number, line in enumerate(Path(path).read_bytes().splitlines(), start=1):
        try:
            value = NiahSelectorAssignment.model_validate_json(line)
        except (ValidationError, ValueError) as error:
            raise ValueError(f"invalid NIAH assignment {path}:{line_number}: {error}") from error
        if value.query_id in values:
            raise ValueError(f"duplicate NIAH assignment query ID: {value.query_id}")
        values[value.query_id] = value
    return values


def _project_query_universe(
    values: Mapping[str, QueryValue], expected_keys: set[str], *, label: str
) -> dict[str, QueryValue]:
    """Project a pinned source superset onto the exact R004 labelled-query universe."""

    missing = sorted(expected_keys - set(values))
    if missing:
        raise ValueError(f"{label} omits R004 labelled queries: {missing[:5]}")
    return {query_id: values[query_id] for query_id in sorted(expected_keys)}


def _load_dataset(
    arguments: _DatasetArguments,
    *,
    expected_counts: Mapping[str, int],
    pool_sha256: str,
) -> _LoadedDataset:
    artifacts = _label_artifacts(arguments)
    prepared = _prepare_dataset(arguments, artifacts, expected_counts)
    bundle = JsonlDatasetAdapter.load(arguments.dataset_manifest)
    all_query_by_id = {query.query_id: query for query in bundle.queries}
    all_candidate_by_query = _read_candidate_sets(arguments.candidate_pool / CANDIDATE_FILE)
    all_gold_by_query = {
        case.query_id: tuple(case.relevant_document_ids or ()) for case in bundle.gold_cases
    }
    expected_keys = set(prepared.query_role)
    query_by_id = _project_query_universe(
        all_query_by_id, expected_keys, label=f"{arguments.kind} source queries"
    )
    candidate_by_query = _project_query_universe(
        all_candidate_by_query, expected_keys, label=f"{arguments.kind} candidate pool"
    )
    gold_by_query = _project_query_universe(
        all_gold_by_query, expected_keys, label=f"{arguments.kind} gold cases"
    )
    assignment_by_query: Mapping[str, NiahSelectorAssignment] = {}
    if arguments.assignment is not None:
        assignment_by_query = _read_assignments(arguments.assignment)
        if set(assignment_by_query) != expected_keys:
            raise ValueError("NIAH assignment/R004 label query keys differ")
    return _LoadedDataset(
        prepared=prepared,
        query_by_id=query_by_id,
        candidate_by_query=candidate_by_query,
        gold_by_query=gold_by_query,
        dataset_id=bundle.manifest.dataset_id,
        dataset_signature=bundle.dataset_signature,
        pool_sha256=pool_sha256,
        assignment_by_query=assignment_by_query,
    )


def _sample_digest(kind: DatasetKind, query_id: str) -> str:
    return hashlib.sha256(
        f"{PROTOCOL_VERSION}\n{kind}\n{query_id}\n{SAMPLE_SEED}".encode()
    ).hexdigest()


def _select_sample(
    datasets: Mapping[DatasetKind, _LoadedDataset],
) -> dict[DatasetKind, tuple[str, ...]]:
    typed = select_sanity_queries(
        {
            kind: tuple(
                query_id
                for query_id, role in datasets[kind].prepared.query_role.items()
                if role == "train-fit"
            )
            for kind in DATASET_KINDS
        }
    )
    result: dict[DatasetKind, tuple[str, ...]] = {}
    for kind in DATASET_KINDS:
        result[kind] = tuple(row.query_id for row in typed.queries if row.dataset_kind == kind)
    return result


def _active_sample_pairs(
    datasets: Mapping[DatasetKind, _LoadedDataset],
    sample: Mapping[DatasetKind, Sequence[str]],
) -> dict[DatasetKind, tuple[_Pair, ...]]:
    result: dict[DatasetKind, tuple[_Pair, ...]] = {}
    for kind in DATASET_KINDS:
        wanted = set(sample[kind])
        pairs = tuple(
            pair
            for pair in datasets[kind].prepared.pairs
            if pair.query_id in wanted and (pair.protect_mask or pair.harm_mask)
        )
        result[kind] = pairs
    return result


def _class_counts(
    pairs: Mapping[DatasetKind, Sequence[_Pair]],
) -> Counter[tuple[DatasetKind, str, int]]:
    counts: Counter[tuple[DatasetKind, str, int]] = Counter()
    for kind, values in pairs.items():
        for pair in values:
            if pair.protect_mask:
                if type(pair.protect_label) is not int or pair.protect_label not in (0, 1):
                    raise ValueError("active protect label is not binary")
                counts[(kind, "protect", pair.protect_label)] += 1
            if pair.harm_mask:
                if type(pair.harm_label) is not int or pair.harm_label not in (0, 1):
                    raise ValueError("active harm label is not binary")
                counts[(kind, "harm", pair.harm_label)] += 1
    return counts


def _assess_sample_classes(
    pairs: Mapping[DatasetKind, Sequence[_Pair]],
) -> _SampleAssessment:
    counts = _class_counts(pairs)
    required: set[tuple[DatasetKind, str, int]] = {
        ("niah", "protect", 0),
        ("niah", "protect", 1),
        ("niah", "harm", 0),
        ("niah", "harm", 1),
        ("2wiki", "protect", 1),
    }
    missing = tuple(sorted(key for key in required if counts[key] <= 0))
    forbidden: set[tuple[DatasetKind, str, int]] = {
        ("2wiki", "protect", 0),
        ("2wiki", "harm", 0),
        ("2wiki", "harm", 1),
    }
    present_forbidden = tuple(sorted(key for key in forbidden if counts[key] > 0))
    if present_forbidden:
        raise ValueError(
            "fixed R005 sample contains protocol-forbidden 2Wiki active labels: "
            f"{present_forbidden}"
        )
    pairs_by_query: dict[str, list[_Pair]] = defaultdict(list)
    for pair in pairs["niah"]:
        pairs_by_query[pair.query_id].append(pair)
    paired_query_ids = tuple(
        sorted(
            query_id
            for query_id, values in pairs_by_query.items()
            if sum(
                pair.protect_mask
                and pair.harm_mask
                and pair.protect_label == 1
                and pair.harm_label == 0
                for pair in values
            )
            == 1
            and sum(
                pair.protect_mask
                and pair.harm_mask
                and pair.protect_label == 0
                and pair.harm_label == 1
                for pair in values
            )
            == 1
        )
    )
    failed_checks: list[str] = []
    if missing:
        failed_checks.append("required-active-class-coverage")
    if len(paired_query_ids) < MINIMUM_NIAH_PAIRS:
        failed_checks.append("minimum-strict-niah-pairs")
    return _SampleAssessment(
        class_counts=counts,
        paired_query_ids=paired_query_ids,
        missing_required_classes=missing,
        strict_niah_pair_count=len(paired_query_ids),
        required_strict_niah_pairs=MINIMUM_NIAH_PAIRS,
        failed_checks=tuple(failed_checks),
        status="FAIL" if failed_checks else "PASS",
    )


def _class_weights(
    counts: Mapping[tuple[DatasetKind, str, int], int],
) -> dict[tuple[DatasetKind, str, int], float]:
    weights: dict[tuple[DatasetKind, str, int], float] = {}
    for kind in DATASET_KINDS:
        for head in ("protect", "harm"):
            observed = {
                label: counts.get((kind, head, label), 0)
                for label in (0, 1)
                if counts.get((kind, head, label), 0) > 0
            }
            if not observed:
                continue
            raw = {label: 1.0 / math.sqrt(count) for label, count in observed.items()}
            active_total = sum(observed.values())
            sqrt_frequency_total = sum(math.sqrt(count) for count in observed.values())
            scale = active_total / sqrt_frequency_total
            for label in observed:
                weights[(kind, head, label)] = raw[label] * scale
    return weights


def _core_class_weights(
    pairs: Mapping[DatasetKind, Sequence[_Pair]],
) -> tuple[dict[tuple[DatasetKind, str, int], float], tuple[Mapping[str, object], ...]]:
    rows = [
        {
            "dataset_kind": pair.dataset_kind,
            "query_id": pair.query_id,
            "evidence_id": pair.evidence_id,
            "role": "train-fit",
            "protect_label": pair.protect_label,
            "protect_mask": pair.protect_mask,
            "harm_label": pair.harm_label,
            "harm_mask": pair.harm_mask,
        }
        for kind in DATASET_KINDS
        for pair in pairs[kind]
    ]
    typed = compute_class_weights(rows)
    weights: dict[tuple[DatasetKind, str, int], float] = {
        (row.dataset_kind, row.head, row.class_label): float(row.normalized_weight) for row in typed
    }
    return weights, tuple(row.model_dump(mode="json") for row in typed)


def _pair_order(pair: _Pair, *, epoch: int) -> str:
    return hashlib.sha256(
        (
            f"{PROTOCOL_VERSION}\ntrain\n{pair.dataset_kind}\n{epoch}\n"
            f"{pair.query_id}\n{pair.evidence_id}\n{SEED}"
        ).encode()
    ).hexdigest()


def _source_batches(
    values: Sequence[_Pair], *, epoch: int, batch_size: int
) -> tuple[tuple[_Pair, ...], ...]:
    ordered = sorted(
        values,
        key=lambda pair: (_pair_order(pair, epoch=epoch), pair.query_id, pair.evidence_id),
    )
    batches = tuple(
        tuple(ordered[start : start + batch_size]) for start in range(0, len(ordered), batch_size)
    )
    if not batches:
        raise ValueError("R005 source has no active training batches")
    return batches


def _loss_tensors(
    *,
    torch: Any,
    output: Any,
    batch: Sequence[_Pair],
    weights: Mapping[tuple[DatasetKind, str, int], float],
) -> Any:
    nan = float("nan")
    protect_labels = torch.tensor(
        [nan if pair.protect_label is None else float(pair.protect_label) for pair in batch],
        device=output.protect_logits.device,
        dtype=output.protect_logits.dtype,
    )
    harm_labels = torch.tensor(
        [nan if pair.harm_label is None else float(pair.harm_label) for pair in batch],
        device=output.harm_logits.device,
        dtype=output.harm_logits.dtype,
    )
    protect_mask = torch.tensor(
        [pair.protect_mask for pair in batch], device=output.protect_logits.device
    )
    harm_mask = torch.tensor([pair.harm_mask for pair in batch], device=output.harm_logits.device)
    protect_weights = torch.tensor(
        [
            (
                weights[(pair.dataset_kind, "protect", cast(int, pair.protect_label))]
                if pair.protect_mask
                else nan
            )
            for pair in batch
        ],
        device=output.protect_logits.device,
        dtype=output.protect_logits.dtype,
    )
    harm_weights = torch.tensor(
        [
            (
                weights[(pair.dataset_kind, "harm", cast(int, pair.harm_label))]
                if pair.harm_mask
                else nan
            )
            for pair in batch
        ],
        device=output.harm_logits.device,
        dtype=output.harm_logits.dtype,
    )
    return masked_dual_head_bce(
        protect_logits=output.protect_logits,
        harm_logits=output.harm_logits,
        protect_labels=protect_labels,
        harm_labels=harm_labels,
        protect_mask=protect_mask,
        harm_mask=harm_mask,
        protect_weights=protect_weights,
        harm_weights=harm_weights,
        weight_normalization="active_count",
    )


def _predict_pairs(
    *, model: Any, torch: Any, pairs: Sequence[_Pair], batch_size: int
) -> dict[tuple[DatasetKind, str, str], tuple[float, float, float]]:
    model.eval()
    scores: dict[tuple[DatasetKind, str, str], tuple[float, float, float]] = {}
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            output = model(
                question=[pair.question for pair in batch],
                candidate_text=[pair.candidate_text for pair in batch],
            )
            protect_values = output.protect_scores.detach().cpu().tolist()
            harm_values = output.harm_scores.detach().cpu().tolist()
            for pair, protect_raw, harm_raw in zip(batch, protect_values, harm_values, strict=True):
                protect_score = float(protect_raw)
                harm_score = float(harm_raw)
                safe_score = min(harm_score, 1.0 - protect_score)
                if not all(
                    math.isfinite(value) and 0.0 <= value <= 1.0
                    for value in (protect_score, harm_score, safe_score)
                ):
                    raise _NonFiniteTrainingError(
                        "R005 model emitted a non-finite/out-of-range score"
                    )
                key = (pair.dataset_kind, pair.query_id, pair.evidence_id)
                if key in scores:
                    raise ValueError(f"duplicate scorer pair: {key}")
                scores[key] = (protect_score, harm_score, safe_score)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    return scores


def _weighted_bce_from_scores(
    *,
    pairs: Sequence[_Pair],
    scores: Mapping[tuple[DatasetKind, str, str], tuple[float, float, float]],
    weights: Mapping[tuple[DatasetKind, str, int], float],
) -> dict[str, float]:
    numerators: dict[tuple[DatasetKind, str], float] = defaultdict(float)
    denominators: dict[tuple[DatasetKind, str], float] = defaultdict(float)
    for pair in pairs:
        protect_score, harm_score, _ = scores[(pair.dataset_kind, pair.query_id, pair.evidence_id)]
        for head, label, active, probability in (
            ("protect", pair.protect_label, pair.protect_mask, protect_score),
            ("harm", pair.harm_label, pair.harm_mask, harm_score),
        ):
            if not active:
                continue
            binary = cast(int, label)
            weight = weights[(pair.dataset_kind, head, binary)]
            clipped = min(max(probability, 1e-12), 1.0 - 1e-12)
            loss = -(binary * math.log(clipped) + (1 - binary) * math.log(1.0 - clipped))
            numerators[(pair.dataset_kind, head)] += weight * loss
            denominators[(pair.dataset_kind, head)] += 1.0
    return {
        f"{kind}.{head}": numerators[(kind, head)] / denominator
        for (kind, head), denominator in sorted(denominators.items())
        if denominator > 0.0
    }


def _initial_active_losses(
    *,
    model: Any,
    torch: Any,
    pairs: Sequence[_Pair],
    weights: Mapping[tuple[DatasetKind, str, int], float],
    batch_size: int,
) -> Mapping[str, float]:
    """Compute the pre-training baseline; any failure here is a hard execution error."""

    scores = _predict_pairs(model=model, torch=torch, pairs=pairs, batch_size=batch_size)
    return _weighted_bce_from_scores(pairs=pairs, scores=scores, weights=weights)


def _training_assessment(
    *,
    pairs: Mapping[DatasetKind, Sequence[_Pair]],
    scores: Mapping[tuple[DatasetKind, str, str], tuple[float, float, float]],
    paired_query_ids: Sequence[str],
) -> dict[str, object]:
    correct: Counter[tuple[DatasetKind, str, int]] = Counter()
    total: Counter[tuple[DatasetKind, str, int]] = Counter()
    pair_by_query: dict[str, dict[str, _Pair]] = defaultdict(dict)
    for kind, values in pairs.items():
        for pair in values:
            protect_score, harm_score, _ = scores[(kind, pair.query_id, pair.evidence_id)]
            if pair.protect_mask:
                label = cast(int, pair.protect_label)
                key = (kind, "protect", label)
                total[key] += 1
                correct[key] += int((protect_score >= CLASSIFICATION_THRESHOLD) == bool(label))
            if pair.harm_mask:
                label = cast(int, pair.harm_label)
                key = (kind, "harm", label)
                total[key] += 1
                correct[key] += int((harm_score >= CLASSIFICATION_THRESHOLD) == bool(label))
            if kind == "niah" and pair.protect_label == 1 and pair.harm_label == 0:
                pair_by_query[pair.query_id]["clean"] = pair
            if kind == "niah" and pair.protect_label == 0 and pair.harm_label == 1:
                pair_by_query[pair.query_id]["counterfactual"] = pair
    class_rows: list[dict[str, object]] = []
    class_pass = True
    for key in sorted(total):
        accuracy = correct[key] / total[key]
        class_rows.append(
            {
                "dataset_kind": key[0],
                "head": key[1],
                "label": key[2],
                "correct": correct[key],
                "total": total[key],
                "accuracy": accuracy,
                "minimum": MINIMUM_ACCURACY,
                "pass": accuracy >= MINIMUM_ACCURACY,
            }
        )
        class_pass = class_pass and accuracy >= MINIMUM_ACCURACY

    pair_rows: list[dict[str, object]] = []
    pair_correct = 0
    for query_id in paired_query_ids:
        clean = pair_by_query[query_id]["clean"]
        counterfactual = pair_by_query[query_id]["counterfactual"]
        clean_scores = scores[("niah", query_id, clean.evidence_id)]
        counterfactual_scores = scores[("niah", query_id, counterfactual.evidence_id)]
        protect_direction = clean_scores[0] > counterfactual_scores[0]
        harm_direction = counterfactual_scores[1] > clean_scores[1]
        safe_direction = counterfactual_scores[2] > clean_scores[2]
        passed = protect_direction and harm_direction and safe_direction
        pair_correct += int(passed)
        pair_rows.append(
            {
                "query_id": query_id,
                "clean_evidence_id": clean.evidence_id,
                "counterfactual_evidence_id": counterfactual.evidence_id,
                "protect_direction": protect_direction,
                "harm_direction": harm_direction,
                "safe_direction": safe_direction,
                "pass": passed,
            }
        )
    pair_accuracy = pair_correct / len(pair_rows) if pair_rows else 0.0
    return {
        "class_accuracy": class_rows,
        "all_required_classes_pass": class_pass,
        "niah_pair_direction": {
            "pairs": len(pair_rows),
            "correct": pair_correct,
            "accuracy": pair_accuracy,
            "minimum_pairs": MINIMUM_NIAH_PAIRS,
            "minimum_accuracy": PAIR_DIRECTION_ACCURACY,
            "pass": (
                len(pair_rows) >= MINIMUM_NIAH_PAIRS and pair_accuracy >= PAIR_DIRECTION_ACCURACY
            ),
            "rows": pair_rows,
        },
    }


def _twowiki_harm_gradient_probe(
    *,
    model: Any,
    torch: Any,
    pairs: Sequence[_Pair],
    weights: Mapping[tuple[DatasetKind, str, int], float],
    batch_size: int,
) -> dict[str, object]:
    batch = tuple(pairs[:batch_size])
    if not batch or any(pair.harm_mask for pair in batch):
        raise ValueError("2Wiki gradient probe must contain only harm-masked active rows")
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(
        question=[pair.question for pair in batch],
        candidate_text=[pair.candidate_text for pair in batch],
    )
    loss = _loss_tensors(torch=torch, output=output, batch=batch, weights=weights)
    loss.total.backward()
    gradient_l1 = 0.0
    for parameter in model.harm_head.parameters():
        if parameter.grad is not None:
            gradient_l1 += float(parameter.grad.detach().abs().sum().cpu().item())
    model.zero_grad(set_to_none=True)
    return {
        "dataset_kind": "2wiki",
        "batch_size": len(batch),
        "harm_effective_labels": loss.harm_count,
        "harm_loss": float(loss.harm.detach().cpu().item()),
        "harm_head_gradient_l1": gradient_l1,
        "pass": loss.harm_count == 0 and gradient_l1 == 0.0,
    }


def _classify_training_failure(
    error: Exception, *, oom_error_type: type[BaseException]
) -> Literal["cuda-oom", "nonfinite"] | None:
    if isinstance(error, _NonFiniteTrainingError):
        return "nonfinite"
    if isinstance(error, oom_error_type):
        return "cuda-oom"
    return None


def _run_epoch_boundaries(
    *,
    run_epoch: Callable[[int], tuple[Mapping[str, object], int]],
    snapshot_state: Callable[[], object],
    restore_state: Callable[[object], None],
    oom_error_type: type[BaseException],
    on_expected_failure: Callable[[], None] = lambda: None,
    epochs: int = SANITY_EPOCHS,
) -> _TrainingOutcome:
    """Run epochs transactionally and retain only the last complete model boundary."""

    boundary_state = snapshot_state()
    rows: list[Mapping[str, object]] = []
    optimizer_steps = 0
    for epoch in range(1, epochs + 1):
        try:
            row, epoch_steps = run_epoch(epoch)
            next_boundary = snapshot_state()
        except Exception as error:
            reason = _classify_training_failure(error, oom_error_type=oom_error_type)
            if reason is None:
                raise
            on_expected_failure()
            restore_state(boundary_state)
            on_expected_failure()
            return _TrainingOutcome(
                epoch_rows=tuple(rows),
                optimizer_steps=optimizer_steps,
                epochs_completed=len(rows),
                termination_reason=reason,
                failure_epoch=epoch,
            )
        rows.append(row)
        optimizer_steps += epoch_steps
        boundary_state = next_boundary
    return _TrainingOutcome(
        epoch_rows=tuple(rows),
        optimizer_steps=optimizer_steps,
        epochs_completed=len(rows),
        termination_reason="completed",
        failure_epoch=None,
    )


def _finite_cpu_model_state(model: Any, torch: Any) -> Mapping[str, Any]:
    state: dict[str, Any] = {}
    for name, tensor in model.state_dict().items():
        detached = tensor.detach()
        if not bool(torch.all(torch.isfinite(detached)).item()):
            raise _NonFiniteTrainingError("R005 model parameters became non-finite")
        state[name] = detached.to(device="cpu").contiguous().clone()
    if not state:
        raise ValueError("R005 model state must not be empty")
    return state


def _train_sanity_epoch(
    *,
    model: Any,
    torch: Any,
    optimizer: Any,
    pairs: Mapping[DatasetKind, Sequence[_Pair]],
    weights: Mapping[tuple[DatasetKind, str, int], float],
    config: _SanityConfig,
    epoch: int,
) -> tuple[Mapping[str, object], int]:
    batches = {
        kind: _source_batches(values, epoch=epoch, batch_size=config.batch_size)
        for kind, values in pairs.items()
    }
    paired_steps = max(len(batches["niah"]), len(batches["2wiki"]))
    microbatches: list[tuple[DatasetKind, tuple[_Pair, ...]]] = []
    for index in range(paired_steps):
        microbatches.append(("niah", batches["niah"][index % len(batches["niah"])]))
        microbatches.append(("2wiki", batches["2wiki"][index % len(batches["2wiki"])]))
    if len(microbatches) % config.accumulation:
        raise ValueError("R005 1:1 microbatch schedule must fill complete accumulation windows")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    protect_losses: list[float] = []
    harm_losses: list[float] = []
    effective_protect = 0
    effective_harm = 0
    started = time.perf_counter()
    epoch_optimizer_steps = 0
    for microbatch_index, (_, batch) in enumerate(microbatches, start=1):
        output = model(
            question=[pair.question for pair in batch],
            candidate_text=[pair.candidate_text for pair in batch],
        )
        loss = _loss_tensors(torch=torch, output=output, batch=batch, weights=weights)
        values = (
            float(loss.total.detach().cpu().item()),
            float(loss.protect.detach().cpu().item()),
            float(loss.harm.detach().cpu().item()),
        )
        if not all(math.isfinite(value) for value in values):
            raise _NonFiniteTrainingError(f"non-finite R005 loss at epoch {epoch}")
        (loss.total / config.accumulation).backward()
        losses.append(values[0])
        protect_losses.append(values[1])
        harm_losses.append(values[2])
        effective_protect += loss.protect_count
        effective_harm += loss.harm_count
        if microbatch_index % config.accumulation == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_optimizer_steps += 1
    return (
        {
            "schema_version": "1.0",
            "protocol_version": PROTOCOL_VERSION,
            "epoch": epoch,
            "microbatches": len(microbatches),
            "optimizer_steps": epoch_optimizer_steps,
            "mean_total_loss": sum(losses) / len(losses),
            "mean_protect_loss": sum(protect_losses) / len(protect_losses),
            "mean_harm_loss": sum(harm_losses) / len(harm_losses),
            "protect_effective_labels_with_deterministic_repetition": effective_protect,
            "harm_effective_labels_with_deterministic_repetition": effective_harm,
            "wall_time_seconds": time.perf_counter() - started,
            "nonfinite": False,
        },
        epoch_optimizer_steps,
    )


def _train_sanity_model(
    *,
    model: Any,
    torch: Any,
    pairs: Mapping[DatasetKind, Sequence[_Pair]],
    weights: Mapping[tuple[DatasetKind, str, int], float],
    config: _SanityConfig,
) -> _TrainingOutcome:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )

    def cleanup() -> None:
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

    return _run_epoch_boundaries(
        run_epoch=lambda epoch: _train_sanity_epoch(
            model=model,
            torch=torch,
            optimizer=optimizer,
            pairs=pairs,
            weights=weights,
            config=config,
            epoch=epoch,
        ),
        snapshot_state=lambda: _finite_cpu_model_state(model, torch),
        restore_state=lambda state: model.load_state_dict(state, strict=True),
        oom_error_type=torch.cuda.OutOfMemoryError,
        on_expected_failure=cleanup,
    )


def _trainfit_scoring_pairs(
    datasets: Mapping[DatasetKind, _LoadedDataset],
    sample: Mapping[DatasetKind, Sequence[str]],
) -> tuple[tuple[_Pair, str], ...]:
    values: list[tuple[_Pair, str]] = []
    for kind in DATASET_KINDS:
        sample_ids = set(sample[kind])
        for pair in datasets[kind].prepared.pairs:
            scope: str | None = None
            if pair.role == "train-fit" and pair.retrieval_rank <= 10:
                scope = (
                    "train-fit-quantile-and-overfit"
                    if pair.query_id in sample_ids
                    else "train-fit-quantile"
                )
            elif (
                pair.role == "train-fit"
                and pair.query_id in sample_ids
                and pair.retrieval_rank > 10
            ):
                scope = "sanity-sample-overfit-extra"
            if scope is not None:
                values.append((pair, scope))
    keys = [(pair.dataset_kind, pair.query_id, pair.evidence_id) for pair, _ in values]
    if len(keys) != len(set(keys)):
        raise ValueError("R005 scoring scope contains duplicate candidate pairs")
    expected = {
        "niah": 920 * 10 + QUESTIONS_PER_DATASET * 10,
        "2wiki": 2700 * 10 + QUESTIONS_PER_DATASET * 10,
    }
    actual = Counter(pair.dataset_kind for pair, _ in values)
    if actual != expected:
        raise ValueError(
            f"R005 train-fit scoring cardinality differs from frozen roles: {actual} != {expected}"
        )
    return tuple(
        sorted(
            values,
            key=lambda item: (
                item[0].dataset_kind,
                item[0].query_id,
                item[0].retrieval_rank,
                item[0].evidence_id,
            ),
        )
    )


def _modelval_scoring_pairs(
    datasets: Mapping[DatasetKind, _LoadedDataset],
) -> tuple[tuple[_Pair, str], ...]:
    values = [
        (pair, "train-modelval-heldout")
        for kind in DATASET_KINDS
        for pair in datasets[kind].prepared.pairs
        if pair.role == "train-modelval"
    ]
    keys = [(pair.dataset_kind, pair.query_id, pair.evidence_id) for pair, _ in values]
    if len(keys) != len(set(keys)):
        raise ValueError("R005 modelval scoring scope contains duplicate candidate pairs")
    actual = Counter(pair.dataset_kind for pair, _ in values)
    expected = {"niah": 103 * 20, "2wiki": 300 * 20}
    if actual != expected:
        raise ValueError(
            f"R005 modelval scoring cardinality differs from frozen roles: {actual} != {expected}"
        )
    return tuple(
        sorted(
            values,
            key=lambda item: (
                item[0].dataset_kind,
                item[0].query_id,
                item[0].retrieval_rank,
                item[0].evidence_id,
            ),
        )
    )


def _build_score_rows(
    *,
    scoped_pairs: Sequence[tuple[_Pair, str]],
    scores: Mapping[tuple[DatasetKind, str, str], tuple[float, float, float]],
) -> tuple[_ScoreRow, ...]:
    rows: list[_ScoreRow] = []
    for pair, scope in scoped_pairs:
        protect_score, harm_score, safe_score = scores[
            (pair.dataset_kind, pair.query_id, pair.evidence_id)
        ]
        pair_hash = text_pair_sha256(
            project_text_pair(question=pair.question, candidate_text=pair.candidate_text)
        )
        rows.append(
            _ScoreRow(
                dataset_kind=pair.dataset_kind,
                query_id=pair.query_id,
                evidence_id=pair.evidence_id,
                document_id=pair.document_id,
                retrieval_rank=pair.retrieval_rank,
                role=pair.role,
                score_scope=scope,
                text_pair_sha256=pair_hash,
                protect_score=protect_score,
                harm_score=harm_score,
                safe_score=safe_score,
            )
        )
    return tuple(rows)


def _nearest_rank(values: Sequence[float], quantile: float) -> tuple[int, float]:
    if not values:
        raise ValueError("cannot compute a diagnostic quantile from no scores")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("diagnostic quantile must lie in (0,1]")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("diagnostic quantile input contains a non-finite value")
    position = math.ceil(quantile * len(ordered))
    return position, ordered[position - 1]


def _quantile_policies(
    score_rows: Sequence[_ScoreRow],
) -> tuple[tuple[Mapping[str, object], ...], tuple[SanityQuantileThreshold, ...]]:
    universe_rows = tuple(
        SanityCandidateScoreRow.model_validate(row.as_row())
        for row in score_rows
        if row.role == "train-fit" and row.retrieval_rank <= 10
    )
    points = derive_train_fit_thresholds(universe_rows)
    policies: list[Mapping[str, object]] = []
    for point in points:
        quantile_code = str(int(round(float(point.quantile) * 1000))).zfill(3)
        policies.append(
            {
                "policy_id": f"Q{quantile_code}_CAP1",
                "point_index": point.point_index,
                "quantile": float(point.quantile),
                "nearest_rank_position": point.nearest_rank,
                "universe_count": point.score_count,
                "safe_threshold": float(point.threshold),
                "max_delete": DIAGNOSTIC_CAP,
                "min_keep": 7,
                "policy_enabled": True,
                "purpose": "R005_DIAGNOSTIC_ONLY_NOT_R006_OR_R007_POLICY",
            }
        )
    return tuple(policies), points


def _score_table(
    rows: Sequence[_ScoreRow], *, role: str, dataset_kind: DatasetKind
) -> dict[str, dict[str, CandidateRiskScore]]:
    values: dict[str, dict[str, CandidateRiskScore]] = defaultdict(dict)
    for row in rows:
        if row.role != role or row.dataset_kind != dataset_kind:
            continue
        if row.evidence_id in values[row.query_id]:
            raise ValueError(f"duplicate score for {row.query_id}/{row.evidence_id}")
        values[row.query_id][row.evidence_id] = CandidateRiskScore(
            protect_score=row.protect_score,
            harm_score=row.harm_score,
        )
    return dict(values)


def _policy_outputs(
    *,
    datasets: Mapping[DatasetKind, _LoadedDataset],
    score_rows: Sequence[_ScoreRow],
    policies: Sequence[Mapping[str, object]],
) -> tuple[_PolicyOutput, ...]:
    score_tables = {
        kind: _score_table(score_rows, role="train-modelval", dataset_kind=kind)
        for kind in cast(tuple[DatasetKind, ...], ("niah", "2wiki"))
    }
    definitions: list[tuple[str, float | None, float, bool]] = [
        ("P0_KEEP_TOPK10", None, 0.0, False)
    ]
    for policy in policies:
        policy_id = cast(str, policy["policy_id"])
        quantile = float(cast(float, policy["quantile"]))
        threshold = float(cast(float, policy["safe_threshold"]))
        definitions.append((policy_id, quantile, threshold, True))

    outputs: list[_PolicyOutput] = []
    for policy_id, quantile_value, threshold, policy_enabled in definitions:
        selection_by_query: dict[tuple[DatasetKind, str], tuple[str, ...]] = {}
        result_rows: list[Mapping[str, object]] = []
        trace_rows: list[Mapping[str, object]] = []
        selected_rows: list[Mapping[str, object]] = []
        for kind in DATASET_KINDS:
            selector = RiskControlledSelector(
                scores_by_query=(score_tables[kind] if policy_enabled else {}),
                safe_threshold=threshold,
                max_delete=DIAGNOSTIC_CAP,
                policy_enabled=policy_enabled,
                dependency_ready=True,
            )
            dataset = datasets[kind]
            query_ids = sorted(
                query_id
                for query_id, role in dataset.prepared.query_role.items()
                if role == "train-modelval"
            )
            for query_id in query_ids:
                query = dataset.query_by_id[query_id]
                candidates = dataset.candidate_by_query[query_id]
                result, trace = selector.select_with_trace(query, candidates, max_selected=10)
                selected_ids = tuple(item.evidence_id for item in result.items)
                selection_by_query[(kind, query_id)] = selected_ids
                result_rows.append(
                    {
                        "schema_version": "1.0",
                        "protocol_version": PROTOCOL_VERSION,
                        "policy_id": policy_id,
                        "dataset_kind": kind,
                        **result.model_dump(mode="json"),
                    }
                )
                trace_rows.append(
                    {
                        "schema_version": "1.0",
                        "sanity_protocol_version": PROTOCOL_VERSION,
                        "policy_id": policy_id,
                        "dataset_kind": kind,
                        **trace.model_dump(mode="json"),
                    }
                )
                candidate_index = {
                    candidate.evidence_id: candidate for candidate in candidates.candidates
                }
                selected_rows.append(
                    {
                        "schema_version": "1.0",
                        "protocol_version": PROTOCOL_VERSION,
                        "policy_id": policy_id,
                        "dataset_kind": kind,
                        "query_id": query_id,
                        "selected_evidence_ids": list(selected_ids),
                        "selected_document_ids": list(
                            dict.fromkeys(
                                candidate_index[evidence_id].document_id
                                for evidence_id in selected_ids
                            )
                        ),
                    }
                )
        outputs.append(
            _PolicyOutput(
                policy_id=policy_id,
                quantile=quantile_value,
                threshold=threshold,
                selection_by_query=selection_by_query,
                result_rows=tuple(result_rows),
                trace_rows=tuple(trace_rows),
                selected_rows=tuple(selected_rows),
            )
        )

    expected_rows = (103 + 300) * len(outputs)
    if any(len(output.result_rows) != 103 + 300 for output in outputs):
        raise ValueError("R005 policy output does not cover all 403 modelval queries")
    if sum(len(output.result_rows) for output in outputs) != expected_rows:
        raise AssertionError("R005 total policy row accounting failed")
    for kind in DATASET_KINDS:
        query_ids = sorted(
            query_id
            for query_id, role in datasets[kind].prepared.query_role.items()
            if role == "train-modelval"
        )
        for query_id in query_ids:
            previous = set(outputs[0].selection_by_query[(kind, query_id)])
            baseline = previous.copy()
            for output in outputs[1:]:
                current = set(output.selection_by_query[(kind, query_id)])
                if not current.issubset(previous):
                    raise RuntimeError(
                        f"diagnostic selected sets are not nested for {kind}/{query_id}"
                    )
                if not current.issubset(baseline):
                    raise RuntimeError("R005 policy introduced a rank11-20 replacement")
                previous = current
    return tuple(outputs)


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _aggregate_selection(
    *,
    kind: DatasetKind,
    dataset: _LoadedDataset,
    selection_by_query: Mapping[tuple[DatasetKind, str], Sequence[str]],
) -> dict[str, object]:
    query_ids = sorted(
        query_id
        for query_id, role in dataset.prepared.query_role.items()
        if role == "train-modelval"
    )
    recall_losses: list[float] = []
    baseline_recalls: list[float] = []
    selected_recalls: list[float] = []
    chain_losses: list[float] = []
    dropped_actions = 0
    queries_with_drops = 0
    required_drops = 0
    harmful_drops = 0
    pool_harm_reductions: list[float] = []
    baseline_exposed_deletions: list[float] = []
    unconditional_harm_reductions: list[float] = []
    for query_id in query_ids:
        candidates = tuple(
            sorted(
                dataset.candidate_by_query[query_id].candidates,
                key=lambda item: (item.retrieval_rank, item.evidence_id),
            )
        )
        baseline = candidates[:10]
        baseline_ids = tuple(item.evidence_id for item in baseline)
        selected_ids = tuple(selection_by_query[(kind, query_id)])
        if not set(selected_ids).issubset(baseline_ids):
            raise ValueError(f"R005 selection is not a TopK10 subset for {kind}/{query_id}")
        candidate_by_id = {item.evidence_id: item for item in baseline}
        selected_documents = _ordered_unique(
            [candidate_by_id[evidence_id].document_id for evidence_id in selected_ids]
        )
        baseline_documents = _ordered_unique([item.document_id for item in baseline])
        gold = dataset.gold_by_query[query_id]
        if not gold:
            raise ValueError(f"R005 gold set is empty for {kind}/{query_id}")
        baseline_recalls.append(document_recall(baseline_documents, gold))
        selected_recalls.append(document_recall(selected_documents, gold))
        recall_losses.append(
            relative_recall_loss(
                baseline_document_ids=baseline_documents,
                selector_document_ids=selected_documents,
                gold_document_ids=gold,
            )
        )
        chain = conditional_chain_loss(
            baseline_document_ids=baseline_documents,
            selector_document_ids=selected_documents,
            gold_document_ids=gold,
        )
        if chain is not None:
            chain_losses.append(chain)
        dropped = [item for item in baseline if item.evidence_id not in set(selected_ids)]
        dropped_actions += len(dropped)
        queries_with_drops += int(bool(dropped))
        required_drops += sum(item.document_id in set(gold) for item in dropped)
        if kind == "niah":
            harmful_id = dataset.assignment_by_query[query_id].harmful_document_id
            pool_documents = {item.document_id for item in candidates}
            baseline_harm = float(harmful_id in set(baseline_documents))
            selected_harm = float(harmful_id in set(selected_documents))
            harmful_drops += sum(item.document_id == harmful_id for item in dropped)
            if harmful_id in pool_documents:
                pool_harm_reductions.append(baseline_harm - selected_harm)
            if baseline_harm:
                baseline_exposed_deletions.append(1.0 - selected_harm)
            unconditional_harm_reductions.append(baseline_harm - selected_harm)

    def mean(values: Sequence[float]) -> float | None:
        return sum(values) / len(values) if values else None

    result: dict[str, object] = {
        "queries": len(query_ids),
        "baseline_document_recall": mean(baseline_recalls),
        "selected_document_recall": mean(selected_recalls),
        "relative_recall_loss": mean(recall_losses),
        "conditional_chain": {
            "eligible_queries": len(chain_losses),
            "failures": int(sum(chain_losses)),
            "loss": mean(chain_losses),
        },
        "actions": {
            "queries_with_deletions": queries_with_drops,
            "dropped_candidate_actions": dropped_actions,
            "required_dropped_candidate_actions": required_drops,
            "required_deletion_rate": (
                None if dropped_actions == 0 else required_drops / dropped_actions
            ),
        },
    }
    if kind == "niah":
        result["harmful"] = {
            "pool_conditional_denominator": len(pool_harm_reductions),
            "pool_conditional_harmful_reduction": mean(pool_harm_reductions),
            "baseline_exposed_denominator": len(baseline_exposed_deletions),
            "baseline_exposed_harmful_deletion": mean(baseline_exposed_deletions),
            "unconditional_harmful_reduction": mean(unconditional_harm_reductions),
            "harmful_dropped_candidate_actions": harmful_drops,
            "deletion_precision": (
                None if dropped_actions == 0 else harmful_drops / dropped_actions
            ),
        }
    return result


def _row_index(
    rows: Sequence[Mapping[str, object]], *, label: str
) -> dict[tuple[DatasetKind, str], Mapping[str, object]]:
    result: dict[tuple[DatasetKind, str], Mapping[str, object]] = {}
    for row in rows:
        kind = row.get("dataset_kind")
        query_id = row.get("query_id")
        if kind not in {"niah", "2wiki"} or not isinstance(query_id, str):
            raise ValueError(f"{label} row has an invalid dataset/query identity")
        key = (cast(DatasetKind, kind), query_id)
        if key in result:
            raise ValueError(f"duplicate {label} row: {key}")
        result[key] = row
    return result


def _string_tuple(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{label} must be a sequence of non-empty strings")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicate IDs")
    return result


def _validate_policy_outputs(
    *,
    datasets: Mapping[DatasetKind, _LoadedDataset],
    outputs: Sequence[_PolicyOutput],
) -> Mapping[str, object]:
    expected_policy_ids = (
        "P0_KEEP_TOPK10",
        "Q990_CAP1",
        "Q975_CAP1",
        "Q950_CAP1",
        "Q900_CAP1",
    )
    if tuple(output.policy_id for output in outputs) != expected_policy_ids:
        raise ValueError("R005 policy outputs differ from the frozen P0+four-point order")
    action_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for output in outputs:
        trace_index = _row_index(output.trace_rows, label=f"{output.policy_id} trace")
        result_index = _row_index(output.result_rows, label=f"{output.policy_id} result")
        selected_index = _row_index(output.selected_rows, label=f"{output.policy_id} selected")
        expected_keys = set(output.selection_by_query)
        if set(trace_index) != expected_keys or set(result_index) != expected_keys:
            raise ValueError(f"{output.policy_id} trace/result query coverage differs")
        if set(selected_index) != expected_keys:
            raise ValueError(f"{output.policy_id} selected-set query coverage differs")
        for key in sorted(expected_keys):
            kind, query_id = key
            candidates = tuple(
                sorted(
                    datasets[kind].candidate_by_query[query_id].candidates,
                    key=lambda item: (item.retrieval_rank, item.evidence_id),
                )
            )
            baseline_ids = tuple(item.evidence_id for item in candidates[:10])
            selected_ids = tuple(output.selection_by_query[key])
            if not set(selected_ids).issubset(baseline_ids):
                raise ValueError(f"{output.policy_id} introduced a non-TopK10 item at {key}")
            trace = trace_index[key]
            trace_baseline = _string_tuple(
                trace.get("baseline_evidence_ids"), label=f"{output.policy_id} baseline"
            )
            trace_selected = _string_tuple(
                trace.get("selected_evidence_ids"), label=f"{output.policy_id} selected"
            )
            trace_dropped = _string_tuple(
                trace.get("dropped_evidence_ids"), label=f"{output.policy_id} dropped"
            )
            if trace_baseline != baseline_ids or trace_selected != selected_ids:
                raise ValueError(
                    f"{output.policy_id} trace differs from independent TopK10 at {key}"
                )
            expected_dropped = tuple(
                evidence_id for evidence_id in baseline_ids if evidence_id not in set(selected_ids)
            )
            if set(trace_dropped) != set(expected_dropped):
                raise ValueError(
                    f"{output.policy_id} DROP_HARM set differs from selection at {key}"
                )
            if output.policy_id == "P0_KEEP_TOPK10":
                if selected_ids != baseline_ids or trace_dropped:
                    raise ValueError(f"P0 is not the exact TopK10 baseline at {key}")
            elif len(trace_dropped) > DIAGNOSTIC_CAP:
                raise ValueError(f"{output.policy_id} exceeds diagnostic cap1 at {key}")

            result_items = result_index[key].get("items")
            if not isinstance(result_items, list):
                raise ValueError(f"{output.policy_id} result items are invalid at {key}")
            result_ids = tuple(
                cast(str, item.get("evidence_id"))
                for item in result_items
                if isinstance(item, Mapping)
            )
            result_ranks = tuple(
                item.get("selection_rank") for item in result_items if isinstance(item, Mapping)
            )
            if result_ids != selected_ids or result_ranks != tuple(range(1, len(result_ids) + 1)):
                raise ValueError(f"{output.policy_id} SelectionResult differs at {key}")
            recorded_selected = _string_tuple(
                selected_index[key].get("selected_evidence_ids"),
                label=f"{output.policy_id} selected-set IDs",
            )
            if recorded_selected != selected_ids:
                raise ValueError(f"{output.policy_id} selected-set artifact differs at {key}")

            decisions = trace.get("decisions")
            if not isinstance(decisions, list) or len(decisions) != len(baseline_ids):
                raise ValueError(f"{output.policy_id} decision rows are incomplete at {key}")
            drop_actions: set[str] = set()
            for decision in decisions:
                if not isinstance(decision, Mapping):
                    raise ValueError(f"{output.policy_id} decision row is invalid at {key}")
                evidence_id = decision.get("evidence_id")
                action = decision.get("action")
                reason = decision.get("reason")
                if not isinstance(evidence_id, str) or not isinstance(action, str):
                    raise ValueError(
                        f"{output.policy_id} decision identity/action invalid at {key}"
                    )
                if not isinstance(reason, str):
                    raise ValueError(f"{output.policy_id} decision reason invalid at {key}")
                action_counts[action] += 1
                reason_counts[reason] += 1
                if action == SelectorAction.DROP_HARM.value:
                    drop_actions.add(evidence_id)
                elif evidence_id not in set(selected_ids):
                    raise ValueError(f"non-DROP action removed evidence at {key}")
            if drop_actions != set(trace_dropped):
                raise ValueError(f"{output.policy_id} action/trace drop sets differ at {key}")
    return {
        "status": "PASS",
        "policies": len(outputs),
        "queries_per_policy": 403,
        "action_counts": dict(sorted(action_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "p0_exact_topk10": True,
        "abstain_is_keep": True,
        "rank11_to_20_replacement": False,
    }


def _policy_outcomes(
    *,
    output: _PolicyOutput,
    datasets: Mapping[DatasetKind, _LoadedDataset],
) -> tuple[SanityPolicyQueryOutcome, ...]:
    trace_index = _row_index(output.trace_rows, label=f"{output.policy_id} trace")
    rows: list[SanityPolicyQueryOutcome] = []
    for kind in cast(tuple[DatasetKind, ...], ("niah", "2wiki")):
        dataset = datasets[kind]
        query_ids = sorted(
            query_id
            for query_id, role in dataset.prepared.query_role.items()
            if role == "train-modelval"
        )
        for query_id in query_ids:
            candidates = tuple(
                sorted(
                    dataset.candidate_by_query[query_id].candidates,
                    key=lambda item: (item.retrieval_rank, item.evidence_id),
                )
            )
            baseline = candidates[:10]
            baseline_ids = tuple(item.evidence_id for item in baseline)
            selected_ids = tuple(output.selection_by_query[(kind, query_id)])
            trace_dropped = _string_tuple(
                trace_index[(kind, query_id)].get("dropped_evidence_ids"),
                label=f"{output.policy_id} dropped IDs",
            )
            inferred_drops = tuple(
                evidence_id for evidence_id in baseline_ids if evidence_id not in set(selected_ids)
            )
            if set(trace_dropped) != set(inferred_drops):
                raise ValueError(f"actual DROP_HARM trace differs at {kind}/{query_id}")
            candidate_index = {item.evidence_id: item for item in baseline}
            baseline_documents = _ordered_unique([item.document_id for item in baseline])
            selected_documents = _ordered_unique(
                [candidate_index[evidence_id].document_id for evidence_id in selected_ids]
            )
            dropped_documents = tuple(
                candidate_index[evidence_id].document_id for evidence_id in trace_dropped
            )
            harmful_document_id: str | None = None
            harmful_in_pool: bool | None = None
            if kind == "niah":
                harmful_document_id = dataset.assignment_by_query[query_id].harmful_document_id
                harmful_in_pool = harmful_document_id in {item.document_id for item in candidates}
            rows.append(
                SanityPolicyQueryOutcome(
                    dataset_kind=kind,
                    query_id=query_id,
                    baseline_document_ids=baseline_documents,
                    selected_document_ids=selected_documents,
                    dropped_evidence_ids=trace_dropped,
                    dropped_document_ids=dropped_documents,
                    required_document_ids=tuple(dataset.gold_by_query[query_id]),
                    harmful_document_id=harmful_document_id,
                    harmful_in_top20_pool=harmful_in_pool,
                )
            )
    return tuple(rows)


def _count_matched_controls(
    *,
    outputs: Sequence[_PolicyOutput],
    datasets: Mapping[DatasetKind, _LoadedDataset],
) -> tuple[
    tuple[Mapping[str, object], ...],
    Mapping[str, tuple[float | None, ...]],
    Mapping[str, Mapping[str, object]],
]:
    rows: list[Mapping[str, object]] = []
    precision_by_policy: dict[str, tuple[float | None, ...]] = {}
    report_by_policy: dict[str, Mapping[str, object]] = {}
    for output in outputs:
        if output.policy_id == "P0_KEEP_TOPK10":
            continue
        random_maps: list[dict[tuple[DatasetKind, str], tuple[str, ...]]] = [
            dict() for _ in range(COUNT_MATCHED_REPEATS)
        ]
        bottom_map: dict[tuple[DatasetKind, str], tuple[str, ...]] = {}
        for kind in cast(tuple[DatasetKind, ...], ("niah", "2wiki")):
            dataset = datasets[kind]
            query_ids = sorted(
                query_id
                for query_id, role in dataset.prepared.query_role.items()
                if role == "train-modelval"
            )
            for query_id in query_ids:
                baseline = tuple(
                    sorted(
                        dataset.candidate_by_query[query_id].candidates,
                        key=lambda item: (item.retrieval_rank, item.evidence_id),
                    )[:10]
                )
                baseline_ids = tuple(item.evidence_id for item in baseline)
                actual_selected = tuple(output.selection_by_query[(kind, query_id)])
                actual_dropped = tuple(
                    evidence_id
                    for evidence_id in baseline_ids
                    if evidence_id not in set(actual_selected)
                )
                deletion_count = len(actual_dropped)
                if deletion_count > DIAGNOSTIC_CAP:
                    raise ValueError("R005 count-matched trace exceeds diagnostic cap1")
                repeat_choices: list[Mapping[str, object]] = []
                bottom_ids: tuple[str, ...] | None = None
                for repeat_index in range(COUNT_MATCHED_REPEATS):
                    generated = generate_count_matched_drops(
                        baseline,
                        deletion_count=deletion_count,
                        repeat_index=repeat_index,
                        dataset_id=dataset.dataset_id,
                        dataset_signature=dataset.dataset_signature,
                        pool_sha256=dataset.pool_sha256,
                        query_id=query_id,
                    )
                    if generated.deletion_count != deletion_count:
                        raise AssertionError("count-matched generator changed deletion count")
                    if bottom_ids is None:
                        bottom_ids = generated.bottom_rank_dropped_evidence_ids
                    elif bottom_ids != generated.bottom_rank_dropped_evidence_ids:
                        raise AssertionError("bottom-rank control changed across repeats")
                    random_drop_set = set(generated.random_dropped_evidence_ids)
                    random_maps[repeat_index][(kind, query_id)] = tuple(
                        evidence_id
                        for evidence_id in baseline_ids
                        if evidence_id not in random_drop_set
                    )
                    repeat_choices.append(
                        {
                            "repeat_index": repeat_index,
                            "random_dropped_evidence_ids": list(
                                generated.random_dropped_evidence_ids
                            ),
                        }
                    )
                assert bottom_ids is not None
                bottom_drop_set = set(bottom_ids)
                bottom_map[(kind, query_id)] = tuple(
                    evidence_id
                    for evidence_id in baseline_ids
                    if evidence_id not in bottom_drop_set
                )
                rows.append(
                    {
                        "schema_version": "1.0",
                        "protocol_version": PROTOCOL_VERSION,
                        "count_matched_protocol_version": "selector-count-matched-v1",
                        "policy_id": output.policy_id,
                        "quantile": output.quantile,
                        "safe_threshold": output.threshold,
                        "dataset_kind": kind,
                        "query_id": query_id,
                        "actual_deletion_count": deletion_count,
                        "actual_dropped_evidence_ids": list(actual_dropped),
                        "bottom_rank_dropped_evidence_ids": list(bottom_ids),
                        "random_repeats": repeat_choices,
                    }
                )

        random_summaries: list[Mapping[str, object]] = []
        random_precisions: list[float | None] = []
        for repeat_index, selection in enumerate(random_maps):
            niah = _aggregate_selection(
                kind="niah", dataset=datasets["niah"], selection_by_query=selection
            )
            twowiki = _aggregate_selection(
                kind="2wiki", dataset=datasets["2wiki"], selection_by_query=selection
            )
            harmful = cast(Mapping[str, object], niah["harmful"])
            precision_raw = harmful["deletion_precision"]
            precision = None if precision_raw is None else float(cast(float, precision_raw))
            random_precisions.append(precision)
            random_summaries.append(
                {
                    "repeat_index": repeat_index,
                    "niah": niah,
                    "2wiki": twowiki,
                }
            )
        bottom_summary = {
            "niah": _aggregate_selection(
                kind="niah", dataset=datasets["niah"], selection_by_query=bottom_map
            ),
            "2wiki": _aggregate_selection(
                kind="2wiki", dataset=datasets["2wiki"], selection_by_query=bottom_map
            ),
        }
        precision_by_policy[output.policy_id] = tuple(random_precisions)
        defined = [value for value in random_precisions if value is not None]
        report_by_policy[output.policy_id] = {
            "random_repeats": random_summaries,
            "random_deletion_precision": {
                "values": random_precisions,
                "mean": None if not defined else sum(defined) / len(defined),
                "minimum": None if not defined else min(defined),
                "maximum": None if not defined else max(defined),
            },
            "bottom_rank": bottom_summary,
        }
    expected_rows = 4 * (103 + 300)
    if len(rows) != expected_rows:
        raise ValueError(f"R005 count-matched rows={len(rows)}, expected {expected_rows}")
    return tuple(rows), precision_by_policy, report_by_policy


def _module_weights_sha256(module: Any, torch: Any) -> str:
    digest = hashlib.sha256()
    state = module.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        specification = f"{name}|{tuple(tensor.shape)}|{tensor.dtype}".encode()
        raw = memoryview(tensor.view(torch.uint8).numpy()).cast("B")
        digest.update(len(specification).to_bytes(8, "big"))
        digest.update(specification)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _head_state(module: Any) -> dict[str, Any]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def _head_l2_change(module: Any, initial: Mapping[str, Any], torch: Any) -> float:
    current = module.state_dict()
    if set(current) != set(initial):
        raise ValueError("R005 head parameter names changed during training")
    squared = 0.0
    for name in sorted(current):
        difference = current[name].detach().cpu().to(dtype=torch.float64) - initial[name].to(
            dtype=torch.float64
        )
        squared += float((difference * difference).sum().item())
    result = math.sqrt(squared)
    if not math.isfinite(result):
        raise RuntimeError("R005 head parameter change is non-finite")
    return result


def _sample_all_pairs(
    datasets: Mapping[DatasetKind, _LoadedDataset],
    sample: Mapping[DatasetKind, Sequence[str]],
) -> tuple[_Pair, ...]:
    values = tuple(
        pair
        for kind in DATASET_KINDS
        for pair in datasets[kind].prepared.pairs
        if pair.query_id in set(sample[kind])
    )
    expected = QUESTIONS_PER_DATASET * 20 * 2
    if len(values) != expected:
        raise ValueError(f"R005 sample scoring pairs={len(values)}, expected {expected}")
    return tuple(
        sorted(
            values,
            key=lambda pair: (
                pair.dataset_kind,
                pair.query_id,
                pair.retrieval_rank,
                pair.evidence_id,
            ),
        )
    )


def _training_predictions(
    *,
    active_pairs: Mapping[DatasetKind, Sequence[_Pair]],
    scores: Mapping[tuple[DatasetKind, str, str], tuple[float, float, float]],
) -> tuple[SanityTrainingPrediction, ...]:
    return tuple(
        SanityTrainingPrediction(
            dataset_kind=pair.dataset_kind,
            query_id=pair.query_id,
            evidence_id=pair.evidence_id,
            protect_label=cast(Literal[0, 1] | None, pair.protect_label),
            protect_mask=pair.protect_mask,
            harm_label=cast(Literal[0, 1] | None, pair.harm_label),
            harm_mask=pair.harm_mask,
            protect_score=scores[(pair.dataset_kind, pair.query_id, pair.evidence_id)][0],
            harm_score=scores[(pair.dataset_kind, pair.query_id, pair.evidence_id)][1],
            safe_score=scores[(pair.dataset_kind, pair.query_id, pair.evidence_id)][2],
        )
        for kind in DATASET_KINDS
        for pair in active_pairs[kind]
    )


def _contract_candidate(rank: int) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=f"probe-e{rank:02d}",
        document_id=f"probe-d{rank:02d}",
        chunk_id=f"probe-c{rank:02d}",
        text=f"probe candidate {rank}",
        source_uri=f"probe://{rank}",
        retrieval_score=float(100 - rank),
        retrieval_rank=rank,
    )


def _policy_contract_probes() -> Mapping[str, object]:
    query = Query(query_id="r005-contract-probe", text="R005 policy contract probe")
    candidates20 = CandidateSet(
        query_id=query.query_id,
        candidates=tuple(_contract_candidate(rank) for rank in range(1, 21)),
    )
    baseline_scores = {
        query.query_id: {
            f"probe-e{rank:02d}": CandidateRiskScore(protect_score=0.9, harm_score=0.1)
            for rank in range(1, 21)
        }
    }

    p0_result, p0_trace = RiskControlledSelector(
        scores_by_query={}, safe_threshold=0.8, max_delete=1, policy_enabled=False
    ).select_with_trace(query, candidates20, max_selected=10)
    baseline_ids = tuple(f"probe-e{rank:02d}" for rank in range(1, 11))
    p0_ok = (
        tuple(item.evidence_id for item in p0_result.items) == baseline_ids
        and not p0_trace.dropped_evidence_ids
        and all(decision.action is SelectorAction.KEEP for decision in p0_trace.decisions)
    )

    conflict_scores = {key: dict(value) for key, value in baseline_scores.items()}
    conflict_scores[query.query_id]["probe-e10"] = CandidateRiskScore(
        protect_score=0.9, harm_score=0.95
    )
    conflict_result, conflict_trace = RiskControlledSelector(
        scores_by_query=conflict_scores, safe_threshold=0.8, max_delete=1
    ).select_with_trace(query, candidates20, max_selected=10)
    conflict_decision = next(
        decision for decision in conflict_trace.decisions if decision.evidence_id == "probe-e10"
    )
    conflict_ok = conflict_decision.action is SelectorAction.ABSTAIN_KEEP and "probe-e10" in {
        item.evidence_id for item in conflict_result.items
    }

    high_scores = {query.query_id: dict(baseline_scores[query.query_id])}
    for rank in range(8, 11):
        high_scores[query.query_id][f"probe-e{rank:02d}"] = CandidateRiskScore(
            protect_score=0.0, harm_score=1.0
        )
    cap_result, cap_trace = RiskControlledSelector(
        scores_by_query=high_scores, safe_threshold=0.8, max_delete=1
    ).select_with_trace(query, candidates20, max_selected=10)
    cap_ok = (
        len(cap_trace.dropped_evidence_ids) == 1
        and len(cap_result.items) == 9
        and any(decision.reason.value == "MAX_DELETE_REACHED" for decision in cap_trace.decisions)
    )

    candidates8 = CandidateSet(query_id=query.query_id, candidates=candidates20.candidates[:8])
    min_keep_scores = {query.query_id: dict(baseline_scores[query.query_id])}
    for rank in (7, 8):
        min_keep_scores[query.query_id][f"probe-e{rank:02d}"] = CandidateRiskScore(
            protect_score=0.0, harm_score=1.0
        )
    min_keep_result, min_keep_trace = RiskControlledSelector(
        scores_by_query=min_keep_scores, safe_threshold=0.8, max_delete=3
    ).select_with_trace(query, candidates8, max_selected=10)
    min_keep_ok = (
        len(min_keep_result.items) == 7
        and len(min_keep_trace.dropped_evidence_ids) == 1
        and any(
            decision.reason.value == "MIN_KEEP_REACHED" for decision in min_keep_trace.decisions
        )
    )

    missing_scores = {query.query_id: dict(baseline_scores[query.query_id])}
    del missing_scores[query.query_id]["probe-e04"]
    missing_result, missing_trace = RiskControlledSelector(
        scores_by_query=missing_scores, safe_threshold=0.8, max_delete=1
    ).select_with_trace(query, candidates20, max_selected=10)
    missing_ok = (
        len(missing_result.items) == 10
        and missing_trace.fallback_reason is not None
        and missing_trace.fallback_reason.value == "MISSING_SCORE"
    )

    dependency_result, dependency_trace = RiskControlledSelector(
        scores_by_query=baseline_scores,
        safe_threshold=0.8,
        max_delete=1,
        dependency_ready=False,
        dependency_reason="synthetic missing dependency",
    ).select_with_trace(query, candidates20, max_selected=10)
    dependency_ok = (
        len(dependency_result.items) == 10
        and dependency_trace.fallback_reason is not None
        and dependency_trace.fallback_reason.value == "MISSING_DEPENDENCY"
    )

    candidates6 = CandidateSet(query_id=query.query_id, candidates=candidates20.candidates[:6])
    few_result, few_trace = RiskControlledSelector(
        scores_by_query=baseline_scores, safe_threshold=0.8, max_delete=1
    ).select_with_trace(query, candidates6, max_selected=10)
    few_ok = (
        len(few_result.items) == 6
        and few_trace.fallback_reason is not None
        and few_trace.fallback_reason.value == "TOO_FEW_CANDIDATES"
    )

    fail_fast = False
    try:
        RiskControlledSelector(
            scores_by_query=baseline_scores, safe_threshold=0.8, max_delete=1
        ).select(query, candidates20, max_selected=9)
    except ValueError:
        fail_fast = True

    rank11_scores = {query.query_id: dict(baseline_scores[query.query_id])}
    rank11_scores[query.query_id]["probe-e20"] = CandidateRiskScore(
        protect_score=0.0, harm_score=1.0
    )
    rank11_result, rank11_trace = RiskControlledSelector(
        scores_by_query=rank11_scores, safe_threshold=0.8, max_delete=1
    ).select_with_trace(query, candidates20, max_selected=10)
    rank11_ok = (
        tuple(item.evidence_id for item in rank11_result.items) == baseline_ids
        and "probe-e20" not in rank11_trace.baseline_evidence_ids
    )

    checks = {
        "explicit_p0_exact_topk10": p0_ok,
        "protect_harm_conflict_abstains_and_keeps": conflict_ok,
        "cap1_blocks_extra_deletions": cap_ok,
        "min_keep_blocks_extra_deletions": min_keep_ok,
        "missing_score_whole_query_fallback": missing_ok,
        "missing_dependency_whole_query_fallback": dependency_ok,
        "fewer_than_seven_whole_query_fallback": few_ok,
        "max_selected_not_ten_fails_fast": fail_fast,
        "rank11_to_20_never_replaces": rank11_ok,
    }
    if not all(checks.values()):
        raise RuntimeError(f"R005 policy contract probes failed: {checks}")
    return {"status": "PASS", "checks": checks}


def _absolute_input_pin(pin: FilePin, *, root: Path) -> FilePin:
    path = _resolved_pin_path(pin, root=root)
    verify_file_pin(pin, root=root)
    return FilePin(
        label=pin.label,
        path=path.as_posix(),
        sha256=pin.sha256,
        bytes=pin.bytes,
        records=pin.records,
        schema_version=pin.schema_version,
    )


def _r005_inputs(
    *,
    r004_manifest: SelectorExperimentManifestV2,
    r004_root: Path,
    count_matched_protocol_dir: Path,
) -> SelectorInputPins:
    root = Path(r004_root).resolve()
    categories = {
        category: tuple(_absolute_input_pin(pin, root=root) for pin in pins)
        for category, pins in r004_manifest.inputs.categories()
    }
    other = list(categories["other"])
    for label, path in (
        (
            "r004/selector_experiment_manifest.json",
            root / "selector_experiment_manifest.json",
        ),
        ("r004/CHECKSUMS.sha256", root / "CHECKSUMS.sha256"),
        (
            "r003/count_matched_protocol.json",
            Path(count_matched_protocol_dir).resolve() / "count_matched_protocol.json",
        ),
        (
            "r003/count_matched_protocol_manifest.json",
            Path(count_matched_protocol_dir).resolve() / "count_matched_protocol_manifest.json",
        ),
    ):
        other.append(pin_file(path, label=label, recorded_path=path.resolve().as_posix()))
    return SelectorInputPins(
        source=categories["source"],
        label=categories["label"],
        data=categories["data"],
        pool=categories["pool"],
        component=categories["component"],
        provenance=categories["provenance"],
        other=tuple(other),
    )


def _sample_rows(
    sample: Mapping[DatasetKind, Sequence[str]],
) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    for kind in cast(tuple[DatasetKind, ...], ("niah", "2wiki")):
        for index, query_id in enumerate(sample[kind], start=1):
            rows.append(
                {
                    "schema_version": "1.0",
                    "protocol_version": PROTOCOL_VERSION,
                    "dataset_kind": kind,
                    "query_id": query_id,
                    "role": "train-fit",
                    "selection_index": index,
                    "sample_digest": _sample_digest(kind, query_id),
                }
            )
    return tuple(rows)


def _prepare_output_root(path: Path) -> Path:
    root = Path(path).resolve()
    if root.exists() and not root.is_dir():
        raise ValueError(f"R005 output root is not a directory: {root}")
    if root.is_dir():
        existing = sorted(item.name for item in root.iterdir())
        if existing:
            raise ValueError(f"R005 output root must be empty; refusing overwrite: {existing[:5]}")
    root.mkdir(parents=True, exist_ok=True)
    (root / CHECKPOINT_DIRECTORY).mkdir()
    (root / SANITY_DIRECTORY).mkdir()
    return root


def _load_json_mapping(path: Path, *, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read {label} at {path}: {error}") from error
    return _mapping(value, label=label)


def _run(arguments: argparse.Namespace) -> Mapping[str, object]:
    started = datetime.now(UTC)
    started_clock = time.perf_counter()
    runtime_log: list[Mapping[str, object]] = [
        {"event": "start", "run_id": RUN_ID, "time_utc": started.isoformat()}
    ]

    common_config, sanity_config = _load_sanity_config(arguments.config)
    git_raw = _git_snapshot(
        arguments.repo_root,
        required_tracked_paths=(
            Path(__file__),
            _module_path(finalize_r005_module),
            _module_path(selector_sanity_module),
            _module_path(dual_head_module),
            _module_path(selector_models_module),
            _module_path(risk_controlled_module),
            arguments.config,
        ),
    )
    git = GitPin.model_validate(git_raw)
    r004_manifest = finalize_selector_r004(arguments.r004_root, verify_only=True)
    if r004_manifest.status is not SelectorRunStatus.PASS:
        raise ValueError("R005 requires a verified R004 PASS")
    verify_count_matched_protocol_artifacts(arguments.count_matched_protocol_dir)
    runtime_log.append(
        {
            "event": "r003-r004-prerequisites-verified",
            "time_utc": datetime.now(UTC).isoformat(),
        }
    )

    snapshot_pins = _audit_model_snapshot(arguments.model_snapshot, common_config)
    base_model_identity = _expected_model_identity(common_config, snapshot_pins)
    if r004_manifest.model.identity_sha256 != base_model_identity["snapshot_identity_sha256"]:
        raise ValueError("R005 model snapshot identity differs from verified R004")
    dataset_arguments = _dataset_arguments_from_r004(r004_manifest, r004_root=arguments.r004_root)
    pin_index = _pin_index(r004_manifest)
    datasets = {
        item.kind: _load_dataset(
            item,
            expected_counts=common_config.expected_label_counts[item.kind],
            pool_sha256=pin_index[f"{item.kind}/candidate_pool"].sha256,
        )
        for item in dataset_arguments
    }
    dataset_query_ids = {kind: set(datasets[kind].prepared.query_role) for kind in DATASET_KINDS}
    overlapping_query_ids = dataset_query_ids["niah"] & dataset_query_ids["2wiki"]
    if overlapping_query_ids:
        raise ValueError(
            f"R005 refuses cross-dataset query-ID collisions: {sorted(overlapping_query_ids)[:5]}"
        )
    inputs = _r005_inputs(
        r004_manifest=r004_manifest,
        r004_root=arguments.r004_root,
        count_matched_protocol_dir=arguments.count_matched_protocol_dir,
    )
    sample = _select_sample(datasets)
    sample_rows = _sample_rows(sample)
    active_pairs = _active_sample_pairs(datasets, sample)
    sample_assessment = _assess_sample_classes(active_pairs)
    class_counts = sample_assessment.class_counts
    paired_query_ids = sample_assessment.paired_query_ids
    if sample_assessment.status == "PASS":
        weights, class_weight_rows = _core_class_weights(active_pairs)
        if weights != _class_weights(class_counts):
            raise AssertionError("R005 core and runner class-weight implementations disagree")
    else:
        weights = {}
        class_weight_rows = ()
    runtime_log.append(
        {
            "event": "fixed-sample-and-active-labels-assessed",
            "time_utc": datetime.now(UTC).isoformat(),
            "sample_queries": 32,
            "active_rows": sum(len(values) for values in active_pairs.values()),
            "status": sample_assessment.status,
            "failed_checks": list(sample_assessment.failed_checks),
        }
    )
    contract_probes = _policy_contract_probes()

    if arguments.verify_only:
        return _verify_existing_run(
            arguments=arguments,
            common_config=common_config,
            sanity_config=sanity_config,
            git=git,
            base_model_identity=base_model_identity,
            datasets=datasets,
            inputs=inputs,
            sample=sample,
            sample_rows=sample_rows,
            active_pairs=active_pairs,
            sample_assessment=sample_assessment,
            weights=weights,
            class_weight_rows=class_weight_rows,
            contract_probes=contract_probes,
        )

    if not str(arguments.device).startswith("cuda"):
        raise ValueError("formal R005 requires one CUDA device")
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for formal R005")
    device = torch.device(arguments.device)
    torch.cuda.set_device(device)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats(device)

    root = _prepare_output_root(arguments.output_dir)
    _write_once(root / CONFIG_FILE, Path(arguments.config).read_bytes())
    model = load_dual_head_model(
        str(Path(arguments.model_snapshot).resolve()),
        revision=common_config.revision,
        identity_model_id=common_config.model_id,
        local_files_only=True,
        device=str(arguments.device),
    )
    if model.protect_head is model.harm_head:
        raise RuntimeError("R005 protect and harm heads share one module")
    if any(
        left.data_ptr() == right.data_ptr()
        for left, right in zip(
            model.protect_head.parameters(), model.harm_head.parameters(), strict=True
        )
    ):
        raise RuntimeError("R005 protect and harm heads share parameter storage")
    initial_fingerprint = fingerprint_dual_head_model(model)
    initial_protect_hash = _module_weights_sha256(model.protect_head, torch)
    initial_harm_hash = _module_weights_sha256(model.harm_head, torch)
    initial_protect_state = _head_state(model.protect_head)
    initial_harm_state = _head_state(model.harm_head)
    all_active_pairs = tuple(pair for kind in DATASET_KINDS for pair in active_pairs[kind])
    initial_losses: Mapping[str, float] = {}
    if sample_assessment.status == "FAIL":
        training_outcome = _TrainingOutcome(
            epoch_rows=(),
            optimizer_steps=0,
            epochs_completed=0,
            termination_reason="sample-coverage",
            failure_epoch=None,
        )
    else:
        initial_losses = _initial_active_losses(
            model=model,
            torch=torch,
            pairs=all_active_pairs,
            weights=weights,
            batch_size=sanity_config.batch_size,
        )
        training_outcome = _train_sanity_model(
            model=model,
            torch=torch,
            pairs=active_pairs,
            weights=weights,
            config=sanity_config,
        )
    training_epoch_rows = training_outcome.epoch_rows
    optimizer_steps = training_outcome.optimizer_steps
    final_protect_hash = _module_weights_sha256(model.protect_head, torch)
    final_harm_hash = _module_weights_sha256(model.harm_head, torch)
    protect_l2_change = _head_l2_change(model.protect_head, initial_protect_state, torch)
    harm_l2_change = _head_l2_change(model.harm_head, initial_harm_state, torch)
    saved_fingerprint = save_dual_head_checkpoint(model, root / CHECKPOINT_FILE)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    reloaded_model = load_dual_head_model(
        str(Path(arguments.model_snapshot).resolve()),
        revision=common_config.revision,
        identity_model_id=common_config.model_id,
        local_files_only=True,
        device=str(arguments.device),
    )
    reloaded_fingerprint = load_dual_head_checkpoint(reloaded_model, root / CHECKPOINT_FILE)
    if reloaded_fingerprint != saved_fingerprint:
        raise RuntimeError("strict R005 checkpoint reload changed the model fingerprint")
    sample_pairs: tuple[_Pair, ...] = ()
    sample_scores: Mapping[tuple[DatasetKind, str, str], tuple[float, float, float]] = {}
    loss_trends: tuple[SanitySourceHeadLossTrend, ...] = ()
    gradient_probe: Mapping[str, object] | None = None
    predictions: tuple[SanityTrainingPrediction, ...] = ()
    if training_outcome.termination_reason == "completed":
        sample_pairs = _sample_all_pairs(datasets, sample)
        sample_scores = _predict_pairs(
            model=reloaded_model,
            torch=torch,
            pairs=sample_pairs,
            batch_size=sanity_config.batch_size,
        )
        final_active_scores = {
            key: sample_scores[key]
            for key in (
                (pair.dataset_kind, pair.query_id, pair.evidence_id) for pair in all_active_pairs
            )
        }
        final_losses = _weighted_bce_from_scores(
            pairs=all_active_pairs, scores=final_active_scores, weights=weights
        )
        required_loss_keys = ("niah.protect", "niah.harm", "2wiki.protect")
        if set(initial_losses) != set(required_loss_keys) or set(final_losses) != set(
            required_loss_keys
        ):
            raise ValueError("R005 active source/head loss keys differ from the frozen protocol")
        gradient_probe = _twowiki_harm_gradient_probe(
            model=reloaded_model,
            torch=torch,
            pairs=active_pairs["2wiki"],
            weights=weights,
            batch_size=sanity_config.batch_size,
        )
        loss_trends = tuple(
            SanitySourceHeadLossTrend(
                dataset_kind=cast(DatasetKind, name.split(".", maxsplit=1)[0]),
                head=cast(Literal["protect", "harm"], name.split(".", maxsplit=1)[1]),
                initial_loss=initial_losses[name],
                final_loss=final_losses[name],
                decreased=final_losses[name] < initial_losses[name],
            )
            for name in required_loss_keys
        )
        predictions = _training_predictions(active_pairs=active_pairs, scores=final_active_scores)
    training_trace = SanityTrainingTrace(
        termination_reason=training_outcome.termination_reason,
        epochs_completed=training_outcome.epochs_completed,
        failure_epoch=training_outcome.failure_epoch,
        final_checkpoint_epoch=training_outcome.epochs_completed,
        epoch_mean_losses=tuple(
            float(cast(float, row["mean_total_loss"])) for row in training_epoch_rows
        ),
        active_source_head_loss_trends=loss_trends,
        protect_head_parameter_l2_change=protect_l2_change,
        harm_head_parameter_l2_change=harm_l2_change,
        twowiki_harm_head_gradient_l1=(
            None
            if gradient_probe is None
            else float(cast(float, gradient_probe["harm_head_gradient_l1"]))
        ),
        independent_sigmoid_heads=True,
        oom_encountered=training_outcome.termination_reason == "cuda-oom",
        nonfinite_encountered=training_outcome.termination_reason == "nonfinite",
    )
    training_decision = evaluate_training_sanity(predictions, training_trace)
    if training_outcome.termination_reason != "completed":
        runtime_log.append(
            {
                "event": "training-aborted",
                "time_utc": datetime.now(UTC).isoformat(),
                "termination_reason": training_outcome.termination_reason,
                "epochs_completed": training_outcome.epochs_completed,
                "failure_epoch": training_outcome.failure_epoch,
            }
        )
    runtime_log.append(
        {
            "event": "training-gate-complete",
            "time_utc": datetime.now(UTC).isoformat(),
            "status": training_decision.status,
            "checkpoint_weights_sha256": saved_fingerprint.weights_sha256,
        }
    )

    score_rows: tuple[_ScoreRow, ...]
    policy_outputs: tuple[_PolicyOutput, ...] = ()
    count_rows: tuple[Mapping[str, object], ...] = ()
    policies: tuple[Mapping[str, object], ...] = ()
    policy_metrics: tuple[Any, ...] = ()
    actual_metrics: dict[str, Mapping[str, object]] = {}
    controls_report: Mapping[str, Mapping[str, object]] = {}
    safe_corner: Mapping[str, object]
    if training_decision.status == "FAIL":
        if training_outcome.termination_reason == "completed":
            failed_scoped_pairs = tuple(
                (
                    pair,
                    (
                        "train-fit-quantile-and-overfit"
                        if pair.retrieval_rank <= 10
                        else "sanity-sample-overfit-extra"
                    ),
                )
                for pair in sample_pairs
            )
            score_rows = _build_score_rows(scoped_pairs=failed_scoped_pairs, scores=sample_scores)
        else:
            score_rows = ()
        quantile_payload: Mapping[str, object] = {
            "schema_version": "1.0",
            "protocol_version": PROTOCOL_VERSION,
            "status": "NOT_EVALUATED_TRAINING_GATE_FAIL",
            "policies": [],
            "modelval_scored": False,
        }
        _write_once(root / QUANTILE_POLICIES_FILE, _json_bytes(quantile_payload))
        safe_corner = {
            "status": "NOT_EVALUATED",
            "reason": "training-gate-failed-before-threshold-and-modelval-stages",
        }
        status: Literal["PASS", "CUT", "FAIL"] = "FAIL"
    else:
        trainfit_scoped = _trainfit_scoring_pairs(datasets, sample)
        trainfit_scores = _predict_pairs(
            model=reloaded_model,
            torch=torch,
            pairs=tuple(pair for pair, _ in trainfit_scoped),
            batch_size=sanity_config.batch_size,
        )
        trainfit_rows = _build_score_rows(scoped_pairs=trainfit_scoped, scores=trainfit_scores)
        policies, threshold_points = _quantile_policies(trainfit_rows)
        quantile_payload = {
            "schema_version": "1.0",
            "protocol_version": PROTOCOL_VERSION,
            "status": "FROZEN_BEFORE_MODELVAL",
            "threshold_source_role": "train-fit",
            "threshold_source_scope": "all-3620-queries-topk10-merged-no-label-filter",
            "nearest_rank": True,
            "duplicate_thresholds_retained": True,
            "checkpoint_weights_sha256": saved_fingerprint.weights_sha256,
            "policies": list(policies),
            "modelval_may_change_checkpoint_or_thresholds": False,
        }
        _write_once(root / QUANTILE_POLICIES_FILE, _json_bytes(quantile_payload))
        runtime_log.append(
            {
                "event": "trainfit-thresholds-frozen-before-modelval",
                "time_utc": datetime.now(UTC).isoformat(),
                "quantile_file_sha256": _sha256_file(root / QUANTILE_POLICIES_FILE),
                "checkpoint_weights_sha256": saved_fingerprint.weights_sha256,
            }
        )

        modelval_scoped = _modelval_scoring_pairs(datasets)
        modelval_scores = _predict_pairs(
            model=reloaded_model,
            torch=torch,
            pairs=tuple(pair for pair, _ in modelval_scoped),
            batch_size=sanity_config.batch_size,
        )
        modelval_rows = _build_score_rows(scoped_pairs=modelval_scoped, scores=modelval_scores)
        score_rows = (*trainfit_rows, *modelval_rows)
        if len(score_rows) != 44_580:
            raise ValueError(f"R005 candidate scores={len(score_rows)}, expected 44580")
        policy_outputs = _policy_outputs(
            datasets=datasets, score_rows=score_rows, policies=policies
        )
        policy_contract = _validate_policy_outputs(datasets=datasets, outputs=policy_outputs)
        count_rows, random_precision_by_policy, controls_report = _count_matched_controls(
            outputs=policy_outputs, datasets=datasets
        )
        metrics_values: list[Any] = []
        for point, output in zip(threshold_points, policy_outputs[1:], strict=True):
            outcomes = _policy_outcomes(output=output, datasets=datasets)
            metrics_values.append(
                evaluate_policy_metrics(
                    point=point,
                    outcomes=outcomes,
                    random_repeat_precisions=random_precision_by_policy[output.policy_id],
                )
            )
            actual_metrics[output.policy_id] = {
                "niah": _aggregate_selection(
                    kind="niah",
                    dataset=datasets["niah"],
                    selection_by_query=output.selection_by_query,
                ),
                "2wiki": _aggregate_selection(
                    kind="2wiki",
                    dataset=datasets["2wiki"],
                    selection_by_query=output.selection_by_query,
                ),
            }
        policy_metrics = tuple(metrics_values)
        safe_decision = decide_safe_corner(policy_metrics)
        safe_corner = safe_decision.model_dump(mode="json")
        safe_corner = {**safe_corner, "policy_contract": policy_contract}
        status = safe_decision.status
        runtime_log.append(
            {
                "event": "modelval-safe-corner-complete",
                "time_utc": datetime.now(UTC).isoformat(),
                "status": status,
                "witness_point_index": safe_decision.witness_point_index,
            }
        )

    peak_memory = int(torch.cuda.max_memory_allocated(device))
    finished = datetime.now(UTC)
    wall_time = time.perf_counter() - started_clock
    runtime_log.append({"event": "complete", "status": status, "time_utc": finished.isoformat()})
    properties = torch.cuda.get_device_properties(device)
    device_info = DeviceInfo(
        kind=DeviceKind.CUDA,
        name=str(properties.name),
        precision="float32",
        runtime_version=str(torch.version.cuda),
    )
    runtime_info = RuntimeInfo(
        host=socket.gethostname(),
        python_version=platform.python_version(),
        command=tuple(sys.argv),
        started_at_utc=started,
        finished_at_utc=finished,
        wall_time_seconds=wall_time,
        peak_device_memory_bytes=peak_memory,
    )
    base_model = r004_manifest.model
    training_summary = {
        "schema_version": "1.0",
        "protocol_version": PROTOCOL_VERSION,
        "row_type": "training-summary",
        "training_trace": training_trace.model_dump(mode="json"),
        "training_decision": training_decision.model_dump(mode="json"),
        "optimizer_steps": optimizer_steps,
        "class_weights": list(class_weight_rows),
        "sample_assessment": sample_assessment.as_report(),
        "not_applicable_classes": [
            "2wiki.protect:0",
            "2wiki.harm:0",
            "2wiki.harm:1",
        ],
        "gradient_probe": gradient_probe,
        "initial_fingerprint": initial_fingerprint.to_dict(),
        "final_fingerprint": saved_fingerprint.to_dict(),
        "strict_reload_fingerprint": reloaded_fingerprint.to_dict(),
        "head_fingerprints": {
            "protect": {"initial": initial_protect_hash, "final": final_protect_hash},
            "harm": {"initial": initial_harm_hash, "final": final_harm_hash},
        },
    }
    training_rows: tuple[Mapping[str, object], ...] = (
        training_summary,
        *({"row_type": "epoch", **row} for row in training_epoch_rows),
    )
    report: Mapping[str, object] = {
        "schema_version": "1.0",
        "protocol_version": PROTOCOL_VERSION,
        "run_id": RUN_ID,
        "stage": "dual-head-sanity",
        "status": status,
        "interpretation": "SANITY_ONLY_NOT_A_FORMAL_SELECTOR_EFFECT_CLAIM",
        "training_gate": training_decision.model_dump(mode="json"),
        "safe_corner": safe_corner,
        "sampling": {
            "role": "train-fit",
            "queries_per_dataset": QUESTIONS_PER_DATASET,
            "selection_uses_labels_lengths_or_model_scores": False,
            "paired_niah_queries": len(paired_query_ids),
            "active_class_counts": {
                f"{kind}.{head}:{label}": count
                for (kind, head, label), count in sorted(class_counts.items())
            },
            "assessment": sample_assessment.as_report(),
        },
        "training": training_summary,
        "diagnostic": {
            "policies": list(policies),
            "metrics": [item.model_dump(mode="json") for item in policy_metrics],
            "detailed_actual_metrics": actual_metrics,
            "count_matched": controls_report,
            "candidate_score_rows": len(score_rows),
            "decision_rows": sum(len(output.trace_rows) for output in policy_outputs),
            "selection_rows": sum(len(output.result_rows) for output in policy_outputs),
        },
        "contract_probes": contract_probes,
        "model": {
            "base_snapshot_identity_sha256": base_model_identity["snapshot_identity_sha256"],
            "final_checkpoint_weights_sha256": saved_fingerprint.weights_sha256,
            "checkpoint_purpose": "R005-sanity-only-never-used-to-initialize-R006",
        },
        "device": device_info.model_dump(mode="json"),
        "runtime": runtime_info.model_dump(mode="json"),
        "boundaries": R005BoundaryReport().model_dump(mode="json"),
    }

    _write_once(root / SAMPLE_FILE, _jsonl_bytes(sample_rows))
    _write_once(root / TRAINING_TRACE_FILE, _jsonl_bytes(training_rows))
    _write_once(
        root / CANDIDATE_SCORES_FILE,
        _jsonl_bytes(tuple(row.as_row() for row in score_rows)),
    )
    _write_once(
        root / DECISION_TRACE_FILE,
        _jsonl_bytes(tuple(row for output in policy_outputs for row in output.trace_rows)),
    )
    _write_once(
        root / SELECTION_RESULTS_FILE,
        _jsonl_bytes(tuple(row for output in policy_outputs for row in output.result_rows)),
    )
    _write_once(
        root / SELECTED_SETS_FILE,
        _jsonl_bytes(tuple(row for output in policy_outputs for row in output.selected_rows)),
    )
    _write_once(root / COUNT_MATCHED_FILE, _jsonl_bytes(count_rows))
    _write_once(root / REPORT_FILE, _json_bytes(report))
    _write_once(root / LOG_FILE, _jsonl_bytes(runtime_log))

    checkpoint_manifest = build_r005_checkpoint_manifest(
        root,
        status=status,
        epochs_completed=training_outcome.epochs_completed,
        model=base_model,
        checkpoint_fingerprint=saved_fingerprint.to_dict(),
    )
    freeze_or_verify_r005_checkpoint_manifest(root, checkpoint_manifest)
    sanity_manifest = build_r005_sanity_manifest(
        root,
        status=status,
        git=git,
        model=base_model,
        inputs=inputs,
        device=device_info,
        runtime=runtime_info,
        boundaries=R005BoundaryReport(),
    )
    freeze_or_verify_r005_sanity_manifest(root, sanity_manifest)
    finalized = finalize_selector_r005(root)
    finalize_selector_r005(root, verify_only=True)
    return {
        "action": "frozen-and-verified",
        "status": finalized.status.value,
        "output_dir": root.as_posix(),
        "training_gate": training_decision.status,
        "termination_reason": training_outcome.termination_reason,
        "epochs_completed": training_outcome.epochs_completed,
        "safe_corner": cast(Mapping[str, object], report["safe_corner"])["status"],
        "candidate_scores": len(score_rows),
        "checkpoint_weights_sha256": saved_fingerprint.weights_sha256,
        "peak_memory_allocated_bytes": peak_memory,
        "wall_time_seconds": wall_time,
    }


def _read_jsonl_mappings(path: Path, *, label: str) -> tuple[Mapping[str, object], ...]:
    try:
        lines = Path(path).read_bytes().splitlines()
    except OSError as error:
        raise ValueError(f"unable to read {label} at {path}: {error}") from error
    rows: list[Mapping[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid {label} row {line_number}: {error}") from error
        rows.append(_mapping(value, label=f"{label} row {line_number}"))
    return tuple(rows)


def _score_row_from_typed(row: SanityCandidateScoreRow) -> _ScoreRow:
    return _ScoreRow(
        dataset_kind=row.dataset_kind,
        query_id=row.query_id,
        evidence_id=row.evidence_id,
        document_id=row.document_id,
        retrieval_rank=row.retrieval_rank,
        role=row.role,
        score_scope=row.score_scope,
        text_pair_sha256=row.text_pair_sha256,
        protect_score=float(row.protect_score),
        harm_score=float(row.harm_score),
        safe_score=float(row.safe_score),
    )


def _validate_stored_epoch_rows(
    *,
    training_summary: Mapping[str, object],
    epoch_rows: Sequence[Mapping[str, object]],
    trace: SanityTrainingTrace,
) -> None:
    if any(row.get("row_type") != "epoch" for row in epoch_rows):
        raise ValueError("R005 training trace contains a non-epoch row after the summary")
    if len(epoch_rows) != trace.epochs_completed:
        raise ValueError("R005 training trace row count differs from actual completed epochs")
    expected_epochs = tuple(range(1, trace.epochs_completed + 1))
    actual_epochs = tuple(row.get("epoch") for row in epoch_rows)
    if actual_epochs != expected_epochs:
        raise ValueError("R005 epoch rows must be numbered consecutively from one")
    epoch_optimizer_steps = tuple(row.get("optimizer_steps") for row in epoch_rows)
    if any(type(value) is not int or value < 0 for value in epoch_optimizer_steps):
        raise ValueError("R005 epoch optimizer-step counts must be non-negative integers")
    for row in epoch_rows:
        for field in (
            "microbatches",
            "protect_effective_labels_with_deterministic_repetition",
            "harm_effective_labels_with_deterministic_repetition",
        ):
            value = row.get(field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"R005 epoch {field} must be a positive integer")
        for field in (
            "mean_total_loss",
            "mean_protect_loss",
            "mean_harm_loss",
            "wall_time_seconds",
        ):
            value = row.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"R005 epoch {field} must be finite and non-negative")
        if row.get("nonfinite") is not False:
            raise ValueError("R005 completed epoch rows must record nonfinite=false")
    if training_summary.get("optimizer_steps") != sum(cast(tuple[int, ...], epoch_optimizer_steps)):
        raise ValueError("R005 summary optimizer steps differ from complete epoch rows")


def _validate_training_runtime_log(
    log_rows: Sequence[Mapping[str, object]],
    trace: SanityTrainingTrace,
    *,
    expected_gate_status: object | None = None,
    expected_checkpoint_sha256: object | None = None,
    expected_run_status: object | None = None,
) -> tuple[object, ...]:
    events = tuple(row.get("event") for row in log_rows)
    for event in ("start", "training-gate-complete", "complete"):
        if events.count(event) != 1:
            raise ValueError(f"R005 runtime log requires exactly one {event} event")
    start_index = events.index("start")
    gate_index = events.index("training-gate-complete")
    complete_index = events.index("complete")
    if not start_index < gate_index < complete_index:
        raise ValueError("R005 runtime log start/gate/complete events are out of order")
    gate_row = log_rows[gate_index]
    complete_row = log_rows[complete_index]
    if expected_gate_status is not None and gate_row.get("status") != expected_gate_status:
        raise ValueError("R005 training-gate log status disagrees with the training decision")
    if (
        expected_checkpoint_sha256 is not None
        and gate_row.get("checkpoint_weights_sha256") != expected_checkpoint_sha256
    ):
        raise ValueError("R005 training-gate log checkpoint hash is invalid")
    if expected_run_status is not None and complete_row.get("status") != expected_run_status:
        raise ValueError("R005 complete log status disagrees with the sealed run status")
    threshold_event = "trainfit-thresholds-frozen-before-modelval"
    modelval_event = "modelval-safe-corner-complete"
    if expected_run_status == "FAIL":
        if events.count(threshold_event) or events.count(modelval_event):
            raise ValueError("R005 FAIL runtime log cannot contain threshold/modelval events")
    elif expected_run_status in ("PASS", "CUT"):
        if events.count(threshold_event) != 1 or events.count(modelval_event) != 1:
            raise ValueError(
                "R005 PASS/CUT runtime log requires exactly one threshold and modelval event"
            )
        threshold_index = events.index(threshold_event)
        modelval_index = events.index(modelval_event)
        if not gate_index < threshold_index < modelval_index < complete_index:
            raise ValueError(
                "R005 runtime log gate/threshold/modelval/complete events are out of order"
            )
    aborted_rows = tuple(row for row in log_rows if row.get("event") == "training-aborted")
    if trace.termination_reason != "completed":
        if len(aborted_rows) != 1:
            raise ValueError("R005 early failure requires exactly one training-aborted log row")
        aborted = aborted_rows[0]
        expected_abort = {
            "termination_reason": trace.termination_reason,
            "epochs_completed": trace.epochs_completed,
            "failure_epoch": trace.failure_epoch,
        }
        if any(aborted.get(key) != value for key, value in expected_abort.items()):
            raise ValueError("R005 training-aborted log row disagrees with the training trace")
        abort_index = events.index("training-aborted")
        if not start_index < abort_index < gate_index:
            raise ValueError("R005 training-aborted log event is out of order")
        forbidden_events = {threshold_event, modelval_event}
        if forbidden_events.intersection(events):
            raise ValueError("R005 early failure log contains a threshold/modelval event")
    elif aborted_rows:
        raise ValueError("R005 completed training cannot contain a training-aborted log row")
    return events


def _require_empty_artifacts(root: Path, relatives: Sequence[str], *, context: str) -> None:
    for relative in relatives:
        if (root / relative).read_bytes() != b"":
            raise ValueError(f"R005 {context} artifact must be empty: {relative}")


def _verify_existing_run(
    *,
    arguments: argparse.Namespace,
    common_config: Any,
    sanity_config: _SanityConfig,
    git: GitPin,
    base_model_identity: Mapping[str, object],
    datasets: Mapping[DatasetKind, _LoadedDataset],
    inputs: SelectorInputPins,
    sample: Mapping[DatasetKind, Sequence[str]],
    sample_rows: Sequence[Mapping[str, object]],
    active_pairs: Mapping[DatasetKind, Sequence[_Pair]],
    sample_assessment: _SampleAssessment,
    weights: Mapping[tuple[DatasetKind, str, int], float],
    class_weight_rows: Sequence[Mapping[str, object]],
    contract_probes: Mapping[str, object],
) -> Mapping[str, object]:
    root = Path(arguments.output_dir).resolve()
    finalized = finalize_selector_r005(root, verify_only=True)
    if (root / CONFIG_FILE).read_bytes() != Path(arguments.config).read_bytes():
        raise ValueError("R005 run-local config differs from the requested tracked config")
    sanity_manifest = R005SanityManifest.model_validate_json(
        (root / SANITY_MANIFEST_FILE).read_bytes()
    )
    checkpoint_manifest = R005CheckpointManifest.model_validate_json(
        (root / CHECKPOINT_MANIFEST_FILE).read_bytes()
    )
    if sanity_manifest.git != git:
        raise ValueError("R005 Git pin differs from the current clean worktree")
    if sanity_manifest.inputs != inputs:
        raise ValueError("R005 input pins differ from current R003/R004 recomputation")
    if sanity_manifest.status != finalized.status.value:
        raise ValueError("R005 runner and generic manifest statuses differ")
    if checkpoint_manifest.status != sanity_manifest.status:
        raise ValueError("R005 checkpoint and sanity statuses differ")
    if (root / SAMPLE_FILE).read_bytes() != _jsonl_bytes(tuple(sample_rows)):
        raise ValueError("R005 sample differs from hash-based recomputation")

    training_rows = _read_jsonl_mappings(root / TRAINING_TRACE_FILE, label="training trace")
    if not training_rows:
        raise ValueError("R005 training trace requires a summary row")
    training_summary = training_rows[0]
    if training_summary.get("row_type") != "training-summary":
        raise ValueError("R005 training trace first row is not the summary")
    epoch_rows = training_rows[1:]
    stored_training_trace = SanityTrainingTrace.model_validate(
        _mapping(training_summary.get("training_trace"), label="stored training trace")
    )
    _validate_stored_epoch_rows(
        training_summary=training_summary,
        epoch_rows=epoch_rows,
        trace=stored_training_trace,
    )
    if training_summary.get("class_weights") != list(class_weight_rows):
        raise ValueError("R005 stored class weights differ from sample recomputation")
    if training_summary.get("sample_assessment") != sample_assessment.as_report():
        raise ValueError("R005 sample assessment differs from fixed-sample recomputation")
    if (sample_assessment.status == "FAIL") != (
        stored_training_trace.termination_reason == "sample-coverage"
    ):
        raise ValueError("R005 sample assessment and termination reason disagree")
    if checkpoint_manifest.epochs_completed != stored_training_trace.epochs_completed:
        raise ValueError("R005 checkpoint epoch differs from the training trace")
    stored_training_decision = _mapping(
        training_summary.get("training_decision"), label="stored training decision"
    )
    report = _load_json_mapping(root / REPORT_FILE, label="R005 sanity report")
    if report.get("training") != training_summary:
        raise ValueError("R005 report training summary differs from training_trace.jsonl")
    if report.get("contract_probes") != contract_probes:
        raise ValueError("R005 stored policy contract probes differ from recomputation")
    sampling_report = _mapping(report.get("sampling"), label="R005 sampling report")
    expected_sampling_report = {
        "role": "train-fit",
        "queries_per_dataset": QUESTIONS_PER_DATASET,
        "selection_uses_labels_lengths_or_model_scores": False,
        "paired_niah_queries": len(sample_assessment.paired_query_ids),
        "active_class_counts": {
            f"{kind}.{head}:{label}": count
            for (kind, head, label), count in sorted(sample_assessment.class_counts.items())
        },
        "assessment": sample_assessment.as_report(),
    }
    if sampling_report != expected_sampling_report:
        raise ValueError("R005 sampling report differs from fixed-sample recomputation")

    early_failure = stored_training_trace.termination_reason != "completed"
    log_rows = _read_jsonl_mappings(root / LOG_FILE, label="R005 runtime log")
    events = _validate_training_runtime_log(
        log_rows,
        stored_training_trace,
        expected_gate_status=stored_training_decision.get("status"),
        expected_checkpoint_sha256=checkpoint_manifest.checkpoint_fingerprint.weights_sha256,
        expected_run_status=sanity_manifest.status,
    )
    if early_failure:
        _require_empty_artifacts(
            root,
            (
                CANDIDATE_SCORES_FILE,
                DECISION_TRACE_FILE,
                SELECTION_RESULTS_FILE,
                SELECTED_SETS_FILE,
                COUNT_MATCHED_FILE,
            ),
            context="early failure",
        )

    if not str(arguments.device).startswith("cuda"):
        raise ValueError("formal R005 verify-only requires one CUDA device")
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for formal R005 verify-only")
    device = torch.device(arguments.device)
    torch.cuda.set_device(device)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model = load_dual_head_model(
        str(Path(arguments.model_snapshot).resolve()),
        revision=common_config.revision,
        identity_model_id=common_config.model_id,
        local_files_only=True,
        device=str(arguments.device),
    )
    initial_fingerprint = fingerprint_dual_head_model(model)
    initial_protect_hash = _module_weights_sha256(model.protect_head, torch)
    initial_harm_hash = _module_weights_sha256(model.harm_head, torch)
    initial_protect_state = _head_state(model.protect_head)
    initial_harm_state = _head_state(model.harm_head)
    all_active_pairs = tuple(pair for kind in DATASET_KINDS for pair in active_pairs[kind])
    initial_losses: Mapping[str, float] = {}
    if not early_failure:
        initial_losses = _initial_active_losses(
            model=model,
            torch=torch,
            pairs=all_active_pairs,
            weights=weights,
            batch_size=sanity_config.batch_size,
        )
    final_fingerprint = load_dual_head_checkpoint(model, root / CHECKPOINT_FILE)
    if finalized.model.identity_sha256 != base_model_identity["snapshot_identity_sha256"]:
        raise ValueError("R005 generic model identity differs from the frozen base snapshot")
    if checkpoint_manifest.model != finalized.model:
        raise ValueError("R005 checkpoint and generic manifests disagree on the base model")
    if (
        checkpoint_manifest.checkpoint_fingerprint.model_dump(mode="json")
        != final_fingerprint.to_dict()
    ):
        raise ValueError("R005 checkpoint manifest does not bind the loaded state")
    final_protect_hash = _module_weights_sha256(model.protect_head, torch)
    final_harm_hash = _module_weights_sha256(model.harm_head, torch)
    protect_l2_change = _head_l2_change(model.protect_head, initial_protect_state, torch)
    harm_l2_change = _head_l2_change(model.harm_head, initial_harm_state, torch)
    fingerprint_rows = {
        "initial_fingerprint": initial_fingerprint.to_dict(),
        "final_fingerprint": final_fingerprint.to_dict(),
        "strict_reload_fingerprint": final_fingerprint.to_dict(),
    }
    for key, expected in fingerprint_rows.items():
        if training_summary.get(key) != expected:
            raise ValueError(f"R005 {key} differs from independent checkpoint loading")
    expected_head_fingerprints = {
        "protect": {"initial": initial_protect_hash, "final": final_protect_hash},
        "harm": {"initial": initial_harm_hash, "final": final_harm_hash},
    }
    if training_summary.get("head_fingerprints") != expected_head_fingerprints:
        raise ValueError("R005 head fingerprints differ from independent recomputation")
    if (
        early_failure
        and stored_training_trace.epochs_completed == 0
        and final_fingerprint != initial_fingerprint
    ):
        raise ValueError("R005 epoch-0 failure must preserve the initialized model state")

    status = sanity_manifest.status
    expected_safe: Mapping[str, object]
    if early_failure:
        if status != "FAIL":
            raise ValueError("R005 early training termination must seal status FAIL")
        scoped_pairs: tuple[tuple[_Pair, str], ...] = ()
    elif status == "FAIL":
        sample_pairs = _sample_all_pairs(datasets, sample)
        scoped_pairs = tuple(
            (
                pair,
                (
                    "train-fit-quantile-and-overfit"
                    if pair.retrieval_rank <= 10
                    else "sanity-sample-overfit-extra"
                ),
            )
            for pair in sample_pairs
        )
    else:
        scoped_pairs = (
            *_trainfit_scoring_pairs(datasets, sample),
            *_modelval_scoring_pairs(datasets),
        )
    recomputed_scores = (
        {}
        if early_failure
        else _predict_pairs(
            model=model,
            torch=torch,
            pairs=tuple(pair for pair, _ in scoped_pairs),
            batch_size=sanity_config.batch_size,
        )
    )
    recomputed_score_rows = _build_score_rows(scoped_pairs=scoped_pairs, scores=recomputed_scores)
    expected_score_bytes = _jsonl_bytes(tuple(row.as_row() for row in recomputed_score_rows))
    actual_score_bytes = (root / CANDIDATE_SCORES_FILE).read_bytes()
    if actual_score_bytes != expected_score_bytes:
        raise ValueError(
            "R005 candidate scores differ from checkpoint recomputation: "
            f"actual={hashlib.sha256(actual_score_bytes).hexdigest()}, "
            f"expected={hashlib.sha256(expected_score_bytes).hexdigest()}"
        )
    typed_score_rows = tuple(
        SanityCandidateScoreRow.model_validate_json(line)
        for line in actual_score_bytes.splitlines()
    )
    score_rows = tuple(_score_row_from_typed(row) for row in typed_score_rows)
    loss_trends: tuple[SanitySourceHeadLossTrend, ...] = ()
    gradient_probe: Mapping[str, object] | None = None
    predictions: tuple[SanityTrainingPrediction, ...] = ()
    if not early_failure:
        final_active_scores = {
            (pair.dataset_kind, pair.query_id, pair.evidence_id): recomputed_scores[
                (pair.dataset_kind, pair.query_id, pair.evidence_id)
            ]
            for pair in all_active_pairs
        }
        final_losses = _weighted_bce_from_scores(
            pairs=all_active_pairs, scores=final_active_scores, weights=weights
        )
        gradient_probe = _twowiki_harm_gradient_probe(
            model=model,
            torch=torch,
            pairs=active_pairs["2wiki"],
            weights=weights,
            batch_size=sanity_config.batch_size,
        )
        required_loss_keys = ("niah.protect", "niah.harm", "2wiki.protect")
        loss_trends = tuple(
            SanitySourceHeadLossTrend(
                dataset_kind=cast(DatasetKind, name.split(".", maxsplit=1)[0]),
                head=cast(Literal["protect", "harm"], name.split(".", maxsplit=1)[1]),
                initial_loss=initial_losses[name],
                final_loss=final_losses[name],
                decreased=final_losses[name] < initial_losses[name],
            )
            for name in required_loss_keys
        )
        predictions = _training_predictions(active_pairs=active_pairs, scores=final_active_scores)
    recomputed_trace = SanityTrainingTrace(
        termination_reason=stored_training_trace.termination_reason,
        epochs_completed=stored_training_trace.epochs_completed,
        failure_epoch=stored_training_trace.failure_epoch,
        final_checkpoint_epoch=stored_training_trace.epochs_completed,
        epoch_mean_losses=tuple(float(cast(float, row["mean_total_loss"])) for row in epoch_rows),
        active_source_head_loss_trends=loss_trends,
        protect_head_parameter_l2_change=protect_l2_change,
        harm_head_parameter_l2_change=harm_l2_change,
        twowiki_harm_head_gradient_l1=(
            None
            if gradient_probe is None
            else float(cast(float, gradient_probe["harm_head_gradient_l1"]))
        ),
        independent_sigmoid_heads=True,
        oom_encountered=stored_training_trace.termination_reason == "cuda-oom",
        nonfinite_encountered=stored_training_trace.termination_reason == "nonfinite",
    )
    if recomputed_trace != stored_training_trace:
        raise ValueError("R005 training gate trace differs from checkpoint recomputation")
    recomputed_decision = evaluate_training_sanity(predictions, recomputed_trace)
    if recomputed_decision.model_dump(mode="json") != stored_training_decision:
        raise ValueError("R005 training decision differs from independent recomputation")
    if report.get("training_gate") != stored_training_decision:
        raise ValueError("R005 report/training-trace gate decisions differ")
    if training_summary.get("gradient_probe") != gradient_probe:
        raise ValueError("R005 stored gradient probe differs from recomputation")

    if status == "FAIL":
        expected_quantile = {
            "schema_version": "1.0",
            "protocol_version": PROTOCOL_VERSION,
            "status": "NOT_EVALUATED_TRAINING_GATE_FAIL",
            "policies": [],
            "modelval_scored": False,
        }
        if (root / QUANTILE_POLICIES_FILE).read_bytes() != _json_bytes(expected_quantile):
            raise ValueError("R005 FAIL quantile marker differs from recomputation")
        expected_safe = {
            "status": "NOT_EVALUATED",
            "reason": "training-gate-failed-before-threshold-and-modelval-stages",
        }
        if report.get("safe_corner") != expected_safe:
            raise ValueError("R005 FAIL safe-corner marker is invalid")
        if recomputed_decision.status != "FAIL":
            raise ValueError("R005 manifest says FAIL but recomputed training gate passes")
        _require_empty_artifacts(
            root,
            (
                DECISION_TRACE_FILE,
                SELECTION_RESULTS_FILE,
                SELECTED_SETS_FILE,
                COUNT_MATCHED_FILE,
            ),
            context="training-gate FAIL",
        )
        diagnostic = _mapping(report.get("diagnostic"), label="R005 diagnostic report")
        expected_diagnostic = {
            "policies": [],
            "metrics": [],
            "detailed_actual_metrics": {},
            "count_matched": {},
            "candidate_score_rows": len(score_rows),
            "decision_rows": 0,
            "selection_rows": 0,
        }
        if diagnostic != expected_diagnostic:
            raise ValueError("R005 FAIL diagnostic report differs from recomputation")
    else:
        if recomputed_decision.status != "PASS":
            raise ValueError("R005 reached modelval despite a failed recomputed training gate")
        trainfit_rows = tuple(row for row in score_rows if row.role == "train-fit")
        policies, points = _quantile_policies(trainfit_rows)
        expected_quantile = {
            "schema_version": "1.0",
            "protocol_version": PROTOCOL_VERSION,
            "status": "FROZEN_BEFORE_MODELVAL",
            "threshold_source_role": "train-fit",
            "threshold_source_scope": "all-3620-queries-topk10-merged-no-label-filter",
            "nearest_rank": True,
            "duplicate_thresholds_retained": True,
            "checkpoint_weights_sha256": final_fingerprint.weights_sha256,
            "policies": list(policies),
            "modelval_may_change_checkpoint_or_thresholds": False,
        }
        actual_quantile_bytes = (root / QUANTILE_POLICIES_FILE).read_bytes()
        if actual_quantile_bytes != _json_bytes(expected_quantile):
            raise ValueError("R005 quantile policies differ from train-fit score recomputation")
        outputs = _policy_outputs(datasets=datasets, score_rows=score_rows, policies=policies)
        policy_contract = _validate_policy_outputs(datasets=datasets, outputs=outputs)
        expected_trace = _jsonl_bytes(tuple(row for output in outputs for row in output.trace_rows))
        expected_results = _jsonl_bytes(
            tuple(row for output in outputs for row in output.result_rows)
        )
        expected_sets = _jsonl_bytes(
            tuple(row for output in outputs for row in output.selected_rows)
        )
        for path, expected_payload in (
            (root / DECISION_TRACE_FILE, expected_trace),
            (root / SELECTION_RESULTS_FILE, expected_results),
            (root / SELECTED_SETS_FILE, expected_sets),
        ):
            if path.read_bytes() != expected_payload:
                raise ValueError(f"R005 policy artifact differs from recomputation: {path}")
        count_rows, random_precision, controls_report = _count_matched_controls(
            outputs=outputs, datasets=datasets
        )
        if (root / COUNT_MATCHED_FILE).read_bytes() != _jsonl_bytes(count_rows):
            raise ValueError("R005 count-matched controls differ from recomputation")
        metrics: list[Any] = []
        actual_metrics: dict[str, Mapping[str, object]] = {}
        for point, output in zip(points, outputs[1:], strict=True):
            metrics.append(
                evaluate_policy_metrics(
                    point=point,
                    outcomes=_policy_outcomes(output=output, datasets=datasets),
                    random_repeat_precisions=random_precision[output.policy_id],
                )
            )
            actual_metrics[output.policy_id] = {
                "niah": _aggregate_selection(
                    kind="niah",
                    dataset=datasets["niah"],
                    selection_by_query=output.selection_by_query,
                ),
                "2wiki": _aggregate_selection(
                    kind="2wiki",
                    dataset=datasets["2wiki"],
                    selection_by_query=output.selection_by_query,
                ),
            }
        safe_decision = decide_safe_corner(tuple(metrics))
        expected_safe = {
            **safe_decision.model_dump(mode="json"),
            "policy_contract": policy_contract,
        }
        if report.get("safe_corner") != expected_safe:
            raise ValueError("R005 safe-corner decision differs from recomputation")
        if safe_decision.status != status:
            raise ValueError("R005 status differs from the recomputed safe-corner status")
        diagnostic = _mapping(report.get("diagnostic"), label="R005 diagnostic report")
        expected_diagnostic = {
            "policies": list(policies),
            "metrics": [item.model_dump(mode="json") for item in metrics],
            "detailed_actual_metrics": actual_metrics,
            "count_matched": controls_report,
            "candidate_score_rows": len(score_rows),
            "decision_rows": sum(len(output.trace_rows) for output in outputs),
            "selection_rows": sum(len(output.result_rows) for output in outputs),
        }
        if diagnostic != expected_diagnostic:
            raise ValueError("R005 diagnostic report differs from complete recomputation")
        try:
            threshold_index = events.index("trainfit-thresholds-frozen-before-modelval")
            modelval_index = events.index("modelval-safe-corner-complete")
        except ValueError as error:
            raise ValueError("R005 runtime log omits threshold/modelval ordering") from error
        if threshold_index >= modelval_index:
            raise ValueError("R005 runtime log does not freeze thresholds before modelval")

    model_report = _mapping(report.get("model"), label="R005 model report")
    if model_report.get("base_snapshot_identity_sha256") != base_model_identity.get(
        "snapshot_identity_sha256"
    ):
        raise ValueError("R005 report base-model identity differs from recomputation")
    if model_report.get("final_checkpoint_weights_sha256") != final_fingerprint.weights_sha256:
        raise ValueError("R005 report checkpoint identity differs from recomputation")
    return {
        "action": "independently-verified",
        "status": finalized.status.value,
        "output_dir": root.as_posix(),
        "training_gate": recomputed_decision.status,
        "termination_reason": stored_training_trace.termination_reason,
        "epochs_completed": stored_training_trace.epochs_completed,
        "candidate_scores": len(score_rows),
        "checkpoint_weights_sha256": final_fingerprint.weights_sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = _run(arguments)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
