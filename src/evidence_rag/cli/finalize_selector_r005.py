"""Freeze or independently verify the fixed-layout R005 Selector sanity run.

R005 has two deliberately separate manifest layers:

* ``checkpoint/checkpoint_manifest.json`` binds the final sanity checkpoint to the
  trained-model fingerprint and the final completed epoch;
* ``sanity/sanity_manifest.json`` binds every other prerequisite, all frozen inputs,
  the completed runtime, and the no-leakage/no-reuse boundaries.

This finalizer verifies both layers, then creates the generic Selector experiment v2
manifest and a checksum list over all fourteen prerequisites.  Neither helper nor the
top-level ``--verify-only`` path rewrites an existing byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidence_rag.evaluation.selector_artifacts import (
    SELECTOR_EXPERIMENT_MANIFEST_FILE,
    DeviceInfo,
    FilePin,
    GitPin,
    ModelPin,
    NotApplicableArtifact,
    PresentArtifact,
    RuntimeInfo,
    SelectorArtifactInventory,
    SelectorExperimentManifestV2,
    SelectorExperimentStage,
    SelectorInputPins,
    SelectorRunStatus,
    canonical_manifest_bytes,
    pin_file,
    verify_file_pin,
    verify_selector_experiment_manifest,
    verify_selector_experiment_pins,
)

CHECKSUMS_FILE = "CHECKSUMS.sha256"
CONFIG_FILE = "config.toml"
CHECKPOINT_DIRECTORY = "checkpoint"
SANITY_DIRECTORY = "sanity"
CHECKPOINT_FILE = f"{CHECKPOINT_DIRECTORY}/model.safetensors"
CHECKPOINT_MANIFEST_FILE = f"{CHECKPOINT_DIRECTORY}/checkpoint_manifest.json"
SANITY_MANIFEST_FILE = f"{SANITY_DIRECTORY}/sanity_manifest.json"

SANITY_SAMPLE_FILE = f"{SANITY_DIRECTORY}/sanity_sample.jsonl"
TRAINING_TRACE_FILE = f"{SANITY_DIRECTORY}/training_trace.jsonl"
CANDIDATE_SCORES_FILE = f"{SANITY_DIRECTORY}/candidate_scores.jsonl"
QUANTILE_POLICIES_FILE = f"{SANITY_DIRECTORY}/quantile_policies.json"
DECISION_TRACE_FILE = f"{SANITY_DIRECTORY}/decision_trace.jsonl"
SELECTION_RESULTS_FILE = f"{SANITY_DIRECTORY}/selection_results.jsonl"
SELECTED_SETS_FILE = f"{SANITY_DIRECTORY}/selected_sets.jsonl"
COUNT_MATCHED_CONTROLS_FILE = f"{SANITY_DIRECTORY}/count_matched_controls.jsonl"
SANITY_REPORT_FILE = f"{SANITY_DIRECTORY}/sanity_report.json"
SANITY_LOG_FILE = f"{SANITY_DIRECTORY}/sanity_log.jsonl"

SANITY_OUTPUT_RELATIVE_FILES = tuple(
    sorted(
        (
            CONFIG_FILE,
            CHECKPOINT_FILE,
            CHECKPOINT_MANIFEST_FILE,
            SANITY_SAMPLE_FILE,
            TRAINING_TRACE_FILE,
            CANDIDATE_SCORES_FILE,
            QUANTILE_POLICIES_FILE,
            DECISION_TRACE_FILE,
            SELECTION_RESULTS_FILE,
            SELECTED_SETS_FILE,
            COUNT_MATCHED_CONTROLS_FILE,
            SANITY_REPORT_FILE,
            SANITY_LOG_FILE,
        )
    )
)
PREREQUISITE_RELATIVE_FILES = tuple(sorted((*SANITY_OUTPUT_RELATIVE_FILES, SANITY_MANIFEST_FILE)))

CompletedStatus: TypeAlias = Literal["PASS", "CUT", "FAIL"]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonEmpty = Annotated[str, Field(min_length=1)]


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class R005CheckpointFingerprint(_FrozenStrictModel):
    """Loaded model-state identity, separate from the frozen base-model identity."""

    schema_version: Literal["selector-dual-head-fingerprint-v1"]
    model_id: NonEmpty
    revision: NonEmpty
    max_length: Literal[512]
    weights_sha256: Sha256


class R005CheckpointManifest(_FrozenStrictModel):
    """Canonical metadata for the one R005-only final-completed-epoch checkpoint."""

    schema_version: Literal["1.0"] = "1.0"
    manifest_type: Literal["selector_r005_checkpoint_manifest"] = (
        "selector_r005_checkpoint_manifest"
    )
    run_id: Literal["R005"] = "R005"
    status: CompletedStatus
    training_seed: Literal[13] = 13
    checkpoint_rule: Literal["final-epoch-only"] = "final-epoch-only"
    checkpoint_purpose: Literal["R005-sanity-only-never-used-to-initialize-R006"] = (
        "R005-sanity-only-never-used-to-initialize-R006"
    )
    epochs_planned: Literal[30] = 30
    epochs_completed: Annotated[int, Field(ge=0, le=30)]
    checkpoint_epoch: Annotated[int, Field(ge=0, le=30)]
    checkpoint_fingerprint: R005CheckpointFingerprint
    model: ModelPin
    checkpoint: FilePin

    @model_validator(mode="after")
    def checkpoint_metadata_is_coherent(self) -> Self:
        if self.checkpoint.path != CHECKPOINT_FILE:
            raise ValueError(f"R005 checkpoint pin path must be {CHECKPOINT_FILE!r}")
        if self.checkpoint.label != CHECKPOINT_FILE:
            raise ValueError(f"R005 checkpoint pin label must be {CHECKPOINT_FILE!r}")
        if self.checkpoint_epoch != self.epochs_completed:
            raise ValueError("R005 checkpoint must be from the final completed epoch")
        if (
            self.checkpoint_fingerprint.model_id != self.model.name
            or self.checkpoint_fingerprint.revision != self.model.revision
        ):
            raise ValueError("R005 checkpoint fingerprint must identify the frozen base model")
        if self.status in {"PASS", "CUT"} and self.epochs_completed != self.epochs_planned:
            raise ValueError(f"R005 {self.status} requires all 30 pre-registered epochs")
        return self


class R005BoundaryReport(_FrozenStrictModel):
    """Statements that keep a cheap sanity run from becoming an accidental main run."""

    scorer_input_fields: Literal["question+candidate_text-only"] = "question+candidate_text-only"
    training_role: Literal["train-fit"] = "train-fit"
    diagnostic_role: Literal["train-modelval"] = "train-modelval"
    diagnostic_policy: Literal["safe-score-0-to-cap1-only"] = "safe-score-0-to-cap1-only"
    max_delete: Literal[1] = 1
    sealed_or_heldout_effect_read: Literal[False] = False
    crc_calibration_performed: Literal[False] = False
    production_default_changed: Literal[False] = False
    checkpoint_may_initialize_r006: Literal[False] = False
    witness_or_thresholds_may_be_reused_by_r006_or_r007: Literal[False] = False
    modelval_may_change_checkpoint_or_thresholds: Literal[False] = False


class R005SanityManifest(_FrozenStrictModel):
    """Strict runner-to-finalizer contract for the completed R005 evidence bundle."""

    schema_version: Literal["1.0"] = "1.0"
    manifest_type: Literal["selector_r005_sanity_manifest"] = "selector_r005_sanity_manifest"
    protocol_version: Literal["selector-r005-sanity-v1"] = "selector-r005-sanity-v1"
    run_id: Literal["R005"] = "R005"
    status: CompletedStatus
    git: GitPin
    config: FilePin
    model: ModelPin
    inputs: SelectorInputPins
    device: DeviceInfo
    runtime: RuntimeInfo
    boundaries: R005BoundaryReport
    outputs: tuple[FilePin, ...] = Field(min_length=13, max_length=13)

    @model_validator(mode="after")
    def completed_run_is_fully_pinned(self) -> Self:
        if self.git.dirty:
            raise ValueError("a completed R005 run requires a clean pinned git worktree")
        if self.config.path != CONFIG_FILE or self.config.label != CONFIG_FILE:
            raise ValueError("R005 config pin must be the run-local relative config.toml")
        if self.runtime.finished_at_utc is None or self.runtime.wall_time_seconds is None:
            raise ValueError("a completed R005 run requires a completed runtime")
        empty = [
            category
            for category, pins in self.inputs.categories()
            if category != "other" and not pins
        ]
        if empty:
            raise ValueError(f"R005 requires all six core input pin categories: empty={empty}")
        paths = tuple(pin.path for pin in self.outputs)
        labels = tuple(pin.label for pin in self.outputs)
        if paths != SANITY_OUTPUT_RELATIVE_FILES:
            raise ValueError(
                "R005 sanity output pins must be in the exact canonical file order: "
                f"actual={paths}, expected={SANITY_OUTPUT_RELATIVE_FILES}"
            )
        if labels != paths:
            raise ValueError("every R005 sanity output pin label must equal its relative path")
        if any(Path(path).is_absolute() for path in paths):
            raise ValueError("R005 sanity output pins must be relative to the run root")
        return self


def _canonical_model_bytes(model: BaseModel) -> bytes:
    return (
        json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_records(path: Path) -> int | None:
    return len(Path(path).read_bytes().splitlines()) if Path(path).suffix == ".jsonl" else None


def _run_pin(run_root: Path, relative: str, *, schema_version: str | None = None) -> FilePin:
    return pin_file(
        Path(run_root) / relative,
        label=relative,
        recorded_path=relative,
        records=_line_records(Path(run_root) / relative),
        schema_version=schema_version,
    )


def _freeze_or_verify_typed_manifest(
    path: Path,
    manifest: BaseModel,
    *,
    verify_only: bool,
    label: str,
) -> None:
    expected = _canonical_model_bytes(manifest)
    destination = Path(path)
    if verify_only:
        try:
            actual = destination.read_bytes()
        except OSError as error:
            raise ValueError(f"missing or unreadable {label} {destination}: {error}") from error
        if actual != expected:
            raise ValueError(
                f"{label} differs from canonical recomputation: "
                f"actual_sha256={_sha256_bytes(actual)}, "
                f"expected_sha256={_sha256_bytes(expected)}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(expected)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite write-once {label}: {destination}") from error


def _load_typed_manifest(path: Path, model_type: type[BaseModel], *, label: str) -> BaseModel:
    source = Path(path)
    try:
        raw = source.read_bytes()
        # Use Pydantic's JSON path rather than ``json.loads`` followed by strict validation:
        # JSON arrays are the canonical wire representation of frozen Python tuples.
        manifest = model_type.model_validate_json(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"missing or invalid {label} {source}: {error}") from error
    if raw != _canonical_model_bytes(manifest):
        raise ValueError(f"{label} is not canonical JSON: {source}")
    return manifest


def build_r005_checkpoint_manifest(
    run_root: Path,
    *,
    status: CompletedStatus,
    epochs_completed: int,
    model: ModelPin,
    checkpoint_fingerprint: Mapping[str, object],
) -> R005CheckpointManifest:
    """Build checkpoint metadata from the existing write-once safetensors file."""

    root = Path(run_root).resolve()
    return R005CheckpointManifest(
        status=status,
        epochs_completed=epochs_completed,
        checkpoint_epoch=epochs_completed,
        checkpoint_fingerprint=R005CheckpointFingerprint.model_validate(checkpoint_fingerprint),
        model=model,
        checkpoint=_run_pin(root, CHECKPOINT_FILE),
    )


def freeze_or_verify_r005_checkpoint_manifest(
    run_root: Path,
    manifest: R005CheckpointManifest,
    *,
    verify_only: bool = False,
) -> R005CheckpointManifest:
    """Write once or byte-verify the runner's checkpoint manifest."""

    root = Path(run_root).resolve()
    verify_file_pin(manifest.checkpoint, root=root)
    _freeze_or_verify_typed_manifest(
        root / CHECKPOINT_MANIFEST_FILE,
        manifest,
        verify_only=verify_only,
        label="R005 checkpoint manifest",
    )
    return cast(
        R005CheckpointManifest,
        _load_typed_manifest(
            root / CHECKPOINT_MANIFEST_FILE,
            R005CheckpointManifest,
            label="R005 checkpoint manifest",
        ),
    )


def _forbidden_role_path(path: str) -> bool:
    lowered = path.lower()
    return "sealed" in lowered or "heldout" in lowered


def _verify_external_inputs(inputs: SelectorInputPins, *, run_root: Path) -> None:
    empty = [category for category, pins in inputs.categories() if category != "other" and not pins]
    if empty:
        raise ValueError(f"R005 requires all six core input pin categories: empty={empty}")
    for pin in inputs.all_pins():
        raw = Path(pin.path)
        if not raw.is_absolute():
            raise ValueError(f"R005 external input pin must be absolute: {pin.label}={pin.path}")
        if _forbidden_role_path(pin.path):
            raise ValueError(f"R005 refuses sealed/heldout input path: {pin.path}")
        resolved = raw.resolve()
        if _forbidden_role_path(resolved.as_posix()):
            raise ValueError(f"R005 refuses resolved sealed/heldout input path: {resolved}")
        verify_file_pin(pin, root=run_root)


def build_r005_sanity_manifest(
    run_root: Path,
    *,
    status: CompletedStatus,
    git: GitPin,
    model: ModelPin,
    inputs: SelectorInputPins,
    device: DeviceInfo,
    runtime: RuntimeInfo,
    boundaries: R005BoundaryReport | None = None,
) -> R005SanityManifest:
    """Build the runner manifest after all thirteen non-recursive outputs exist."""

    root = Path(run_root).resolve()
    _verify_external_inputs(inputs, run_root=root)
    return R005SanityManifest(
        status=status,
        git=git,
        config=_run_pin(root, CONFIG_FILE),
        model=model,
        inputs=inputs,
        device=device,
        runtime=runtime,
        boundaries=boundaries or R005BoundaryReport(),
        outputs=tuple(_run_pin(root, relative) for relative in SANITY_OUTPUT_RELATIVE_FILES),
    )


def freeze_or_verify_r005_sanity_manifest(
    run_root: Path,
    manifest: R005SanityManifest,
    *,
    verify_only: bool = False,
) -> R005SanityManifest:
    """Write once or independently byte-verify the runner-level sanity manifest."""

    root = Path(run_root).resolve()
    _verify_external_inputs(manifest.inputs, run_root=root)
    verify_file_pin(manifest.config, root=root)
    for pin in manifest.outputs:
        verify_file_pin(pin, root=root)
    _freeze_or_verify_typed_manifest(
        root / SANITY_MANIFEST_FILE,
        manifest,
        verify_only=verify_only,
        label="R005 sanity manifest",
    )
    return cast(
        R005SanityManifest,
        _load_typed_manifest(
            root / SANITY_MANIFEST_FILE,
            R005SanityManifest,
            label="R005 sanity manifest",
        ),
    )


@dataclass(frozen=True)
class _FinalizationContext:
    status: SelectorRunStatus
    config: FilePin
    inputs: SelectorInputPins
    git: GitPin
    model: ModelPin
    device: DeviceInfo
    runtime: RuntimeInfo
    checkpoint: tuple[FilePin, ...]
    candidate_scores: tuple[FilePin, ...]
    decision_trace: tuple[FilePin, ...]
    selected_sets: tuple[FilePin, ...]
    count_matched: tuple[FilePin, ...]
    metrics: tuple[FilePin, ...]
    run_log: tuple[FilePin, ...]


def _load_json_mapping(path: Path, *, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"missing or invalid {label} {path}: {error}") from error
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed JSON object")
    return cast(Mapping[str, object], value)


def _nested_status(report: Mapping[str, object], key: str, *, allowed: frozenset[str]) -> str:
    value = report.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"R005 sanity report {key!r} must be an object with a status")
    status = value.get("status")
    if not isinstance(status, str) or status not in allowed:
        raise ValueError(
            f"R005 sanity report {key!r} status must be one of {sorted(allowed)}, got {status!r}"
        )
    return status


def _verify_status_semantics(*, report: Mapping[str, object], sanity: R005SanityManifest) -> None:
    if report.get("status") != sanity.status:
        raise ValueError(
            "R005 sanity report status differs from the frozen sanity manifest: "
            f"report={report.get('status')!r}, manifest={sanity.status!r}"
        )
    training = _nested_status(report, "training_gate", allowed=frozenset({"PASS", "FAIL"}))
    safe_corner = _nested_status(
        report,
        "safe_corner",
        allowed=frozenset({"PASS", "CUT", "NOT_EVALUATED"}),
    )
    if sanity.status == "FAIL":
        if training != "FAIL" or safe_corner != "NOT_EVALUATED":
            raise ValueError("R005 FAIL requires training_gate FAIL and safe_corner NOT_EVALUATED")
        return
    if training != "PASS" or safe_corner != sanity.status:
        raise ValueError(
            f"R005 {sanity.status} requires training_gate PASS and safe_corner {sanity.status}"
        )


def _verify_downstream_record_semantics(
    *, status: CompletedStatus, pins: Mapping[str, FilePin]
) -> None:
    downstream = (
        DECISION_TRACE_FILE,
        SELECTION_RESULTS_FILE,
        SELECTED_SETS_FILE,
        COUNT_MATCHED_CONTROLS_FILE,
    )
    records = {path: pins[path].records for path in downstream}
    if any(count is None for count in records.values()):
        raise ValueError("R005 downstream JSONL pins must record exact line counts")
    if status == "FAIL":
        nonempty = {path: count for path, count in records.items() if count != 0}
        if nonempty:
            raise ValueError(
                "R005 training-gate FAIL must stop before modelval and leave downstream "
                f"JSONL files empty: {nonempty}"
            )
        return
    empty = [path for path, count in records.items() if count == 0]
    if empty:
        raise ValueError(f"R005 {status} requires complete non-empty modelval evidence: {empty}")


def _pin_by_path(manifest: R005SanityManifest) -> dict[str, FilePin]:
    return {pin.path: pin for pin in manifest.outputs}


def _require_equal_pins(actual: FilePin, expected: FilePin, *, label: str) -> None:
    if actual.model_dump(mode="json") != expected.model_dump(mode="json"):
        raise ValueError(f"R005 {label} pin disagrees across manifests")


def _load_verified_context(run_root: Path) -> _FinalizationContext:
    root = Path(run_root).resolve()
    checkpoint = cast(
        R005CheckpointManifest,
        _load_typed_manifest(
            root / CHECKPOINT_MANIFEST_FILE,
            R005CheckpointManifest,
            label="R005 checkpoint manifest",
        ),
    )
    sanity = cast(
        R005SanityManifest,
        _load_typed_manifest(
            root / SANITY_MANIFEST_FILE,
            R005SanityManifest,
            label="R005 sanity manifest",
        ),
    )
    _verify_external_inputs(sanity.inputs, run_root=root)
    verify_file_pin(sanity.config, root=root)
    pins = _pin_by_path(sanity)
    for pin in sanity.outputs:
        verify_file_pin(pin, root=root)
    verify_file_pin(checkpoint.checkpoint, root=root)

    if checkpoint.status != sanity.status:
        raise ValueError("R005 checkpoint and sanity manifest statuses differ")
    if checkpoint.model != sanity.model:
        raise ValueError("R005 checkpoint and sanity manifest model metadata differ")
    _require_equal_pins(
        pins[CHECKPOINT_FILE], checkpoint.checkpoint, label="checkpoint/model.safetensors"
    )
    _require_equal_pins(pins[CONFIG_FILE], sanity.config, label="run-local config.toml")

    report = _load_json_mapping(root / SANITY_REPORT_FILE, label="R005 sanity report")
    _verify_status_semantics(report=report, sanity=sanity)
    _verify_downstream_record_semantics(status=sanity.status, pins=pins)
    model_report = report.get("model")
    if not isinstance(model_report, Mapping):
        raise ValueError("R005 sanity report model metadata must be an object")
    if model_report.get("base_snapshot_identity_sha256") != sanity.model.identity_sha256:
        raise ValueError("R005 report base-model identity differs from the frozen model pin")
    if (
        model_report.get("final_checkpoint_weights_sha256")
        != checkpoint.checkpoint_fingerprint.weights_sha256
    ):
        raise ValueError("R005 report checkpoint fingerprint differs from checkpoint manifest")

    return _FinalizationContext(
        status=SelectorRunStatus(sanity.status),
        config=sanity.config,
        inputs=sanity.inputs,
        git=sanity.git,
        model=sanity.model,
        device=sanity.device,
        runtime=sanity.runtime,
        checkpoint=(pins[CHECKPOINT_FILE], pins[CHECKPOINT_MANIFEST_FILE]),
        candidate_scores=(pins[CANDIDATE_SCORES_FILE],),
        decision_trace=(pins[DECISION_TRACE_FILE],),
        selected_sets=(pins[SELECTION_RESULTS_FILE], pins[SELECTED_SETS_FILE]),
        count_matched=(pins[COUNT_MATCHED_CONTROLS_FILE],),
        metrics=(
            pins[SANITY_SAMPLE_FILE],
            pins[TRAINING_TRACE_FILE],
            pins[QUANTILE_POLICIES_FILE],
            pins[SANITY_REPORT_FILE],
            _run_pin(root, SANITY_MANIFEST_FILE, schema_version="1.0"),
        ),
        run_log=(pins[SANITY_LOG_FILE],),
    )


def _actual_relative_files(run_root: Path) -> set[str]:
    root = Path(run_root)
    values: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"R005 run root must not contain symlinks: {path}")
        if path.is_file():
            values.add(path.relative_to(root).as_posix())
    return values


def _require_exact_run_files(run_root: Path, *, finalized: bool) -> None:
    expected = set(PREREQUISITE_RELATIVE_FILES)
    if finalized:
        expected.update({CHECKSUMS_FILE, SELECTOR_EXPERIMENT_MANIFEST_FILE})
    actual = _actual_relative_files(run_root)
    if actual != expected:
        raise ValueError(
            "R005 run root has an unexpected file set: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def _checksums_bytes(run_root: Path) -> bytes:
    return (
        "\n".join(
            f"{_sha256_file(Path(run_root) / relative)}  {relative}"
            for relative in PREREQUISITE_RELATIVE_FILES
        )
        + "\n"
    ).encode()


def _generic_manifest(
    context: _FinalizationContext, *, checksum_bytes: bytes
) -> SelectorExperimentManifestV2:
    checksum_pin = FilePin(
        label=CHECKSUMS_FILE,
        path=CHECKSUMS_FILE,
        sha256=_sha256_bytes(checksum_bytes),
        bytes=len(checksum_bytes),
        records=len(checksum_bytes.splitlines()),
    )
    return SelectorExperimentManifestV2(
        run_id="R005",
        run_name="R005 dual-head sanity",
        stage=SelectorExperimentStage.DUAL_HEAD_SANITY,
        status=context.status,
        git=context.git,
        config=context.config,
        model=context.model,
        inputs=context.inputs,
        artifacts=SelectorArtifactInventory(
            labels=NotApplicableArtifact(
                reason="R005 consumes hash-pinned R004 labels and does not rebuild them"
            ),
            preflight=NotApplicableArtifact(
                reason="R005 is a training and diagnostic sanity run, not a resource preflight"
            ),
            checkpoint=PresentArtifact(
                files=context.checkpoint,
                description="Final-completed-epoch R005-only sanity checkpoint",
            ),
            candidate_scores=PresentArtifact(files=context.candidate_scores),
            crc=NotApplicableArtifact(
                reason="R005 does not perform CRC calibration or freeze a deployable policy"
            ),
            decision_trace=PresentArtifact(files=context.decision_trace),
            selected_sets=PresentArtifact(files=context.selected_sets),
            count_matched=PresentArtifact(
                files=context.count_matched,
                description="One hundred repeats matched to each diagnostic policy trace",
            ),
            metrics=PresentArtifact(files=context.metrics),
            run_log=PresentArtifact(files=context.run_log),
            checksums=PresentArtifact(files=(checksum_pin,)),
        ),
        device=context.device,
        runtime=context.runtime,
        notes=(
            "R005 is a small-sample learning and train-modelval diagnostic only.",
            "The R005 checkpoint must never initialize R006.",
            "The R005 diagnostic witness and thresholds must not be reused by R006 or R007.",
            "R005 does not establish a harmful-reduction confidence interval or CRC guarantee.",
        ),
    )


def finalize_selector_r005(
    run_root: Path, *, verify_only: bool = False
) -> SelectorExperimentManifestV2:
    """Freeze the two top-level files once, or verify the complete R005 chain."""

    root = Path(run_root).resolve()
    if not root.is_dir():
        raise ValueError(f"missing R005 run root: {root}")
    checksum_path = root / CHECKSUMS_FILE
    manifest_path = root / SELECTOR_EXPERIMENT_MANIFEST_FILE
    if not verify_only:
        existing = [path.name for path in (checksum_path, manifest_path) if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite write-once R005 outputs: {existing}")

    _require_exact_run_files(root, finalized=verify_only)
    context = _load_verified_context(root)
    expected_checksums = _checksums_bytes(root)
    expected_manifest = _generic_manifest(context, checksum_bytes=expected_checksums)

    if verify_only:
        try:
            actual_checksums = checksum_path.read_bytes()
            actual_manifest = manifest_path.read_bytes()
        except OSError as error:
            raise ValueError(f"missing R005 finalization output: {error}") from error
        if actual_checksums != expected_checksums:
            raise ValueError(
                "R005 CHECKSUMS.sha256 differs from byte-for-byte recomputation: "
                f"actual_sha256={_sha256_bytes(actual_checksums)}, "
                f"expected_sha256={_sha256_bytes(expected_checksums)}"
            )
        expected_manifest_bytes = canonical_manifest_bytes(expected_manifest)
        if actual_manifest != expected_manifest_bytes:
            raise ValueError(
                "R005 Selector manifest differs from byte-for-byte recomputation: "
                f"actual_sha256={_sha256_bytes(actual_manifest)}, "
                f"expected_sha256={_sha256_bytes(expected_manifest_bytes)}"
            )
        return verify_selector_experiment_manifest(
            manifest_path,
            expected=expected_manifest,
            pin_root=root,
            verify_pins=True,
        )

    expected_manifest_bytes = canonical_manifest_bytes(expected_manifest)
    staged_paths: list[Path] = []
    published_paths: list[Path] = []
    try:
        for destination, payload in (
            (checksum_path, expected_checksums),
            (manifest_path, expected_manifest_bytes),
        ):
            descriptor, raw_staged = tempfile.mkstemp(
                prefix=f".{root.name}-{destination.name}-",
                suffix=".tmp",
                dir=root.parent,
            )
            staged = Path(raw_staged)
            staged_paths.append(staged)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        for staged, destination in zip(staged_paths, (checksum_path, manifest_path), strict=True):
            try:
                os.link(staged, destination)
            except FileExistsError as error:
                raise FileExistsError(
                    f"refusing to overwrite write-once R005 finalization output: {destination}"
                ) from error
            published_paths.append(destination)
    except Exception:
        for destination in reversed(published_paths):
            destination.unlink(missing_ok=True)
        raise
    finally:
        for staged in staged_paths:
            staged.unlink(missing_ok=True)

    try:
        verify_selector_experiment_pins(expected_manifest, root=root)
        _require_exact_run_files(root, finalized=True)
        return verify_selector_experiment_manifest(
            manifest_path,
            expected=expected_manifest,
            pin_root=root,
            verify_pins=True,
        )
    except Exception:
        for destination in reversed(published_paths):
            destination.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize or verify the fixed-layout R005 run")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    manifest = finalize_selector_r005(arguments.run_root, verify_only=bool(arguments.verify_only))
    root = Path(arguments.run_root).resolve()
    print(
        json.dumps(
            {
                "action": "verified" if arguments.verify_only else "frozen",
                "checksums": str(root / CHECKSUMS_FILE),
                "checksums_sha256": _sha256_file(root / CHECKSUMS_FILE),
                "kind": "selector-r005-finalization",
                "manifest": str(root / SELECTOR_EXPERIMENT_MANIFEST_FILE),
                "manifest_sha256": _sha256_bytes(canonical_manifest_bytes(manifest)),
                "status": manifest.status.value,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
