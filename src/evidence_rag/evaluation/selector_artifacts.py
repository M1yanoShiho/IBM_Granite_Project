"""Strict, write-once manifests for Selector experiments R004 and later.

The production pipeline manifest intentionally remains small.  Selector research
runs need a separate contract because their evidence includes label audits,
candidate pools, scorer checkpoints, CRC decisions, and per-query traces.  This
module keeps that contract explicit and makes absence unambiguous: an artifact is
either present, not applicable to the stage, or deferred to a later real run.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    PositiveInt,
    StrictBool,
    StrictInt,
    model_validator,
)

SCHEMA_VERSION = "2.0"
MANIFEST_TYPE = "selector_experiment_manifest"
SELECTOR_EXPERIMENT_MANIFEST_FILE = "selector_experiment_manifest.json"

NonEmpty = Annotated[str, Field(min_length=1)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
GitCommit = Annotated[str, Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
NonNegativeFloat = Annotated[FiniteFloat, Field(ge=0.0)]
PositiveFloat = Annotated[FiniteFloat, Field(gt=0.0)]


class FrozenStrictModel(BaseModel):
    """Immutable Pydantic base that rejects undeclared manifest fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class SelectorExperimentStage(StrEnum):
    """Pre-registered stages from R004 through the formal evaluation runs."""

    LABEL_AUDIT_200Q_PREFLIGHT = "label-audit-and-200q-preflight"
    DUAL_HEAD_SANITY = "dual-head-sanity"
    TRAIN_SEED = "train-seed"
    METHOD_SELECTION = "method-selection"
    CRC_CALIBRATION = "crc-calibration"
    DECISION_DEV = "decision-dev"
    TRAIN_AND_CALIBRATE = "train-and-calibrate"
    THREE_SEED_DECISION_DEV = "three-seed-decision-dev"
    FORMAL_FREEZE = "formal-freeze"
    FORMAL_EVALUATION = "formal-evaluation"


class SelectorRunStatus(StrEnum):
    RUNNING = "RUNNING"
    PASS = "PASS"
    FAIL = "FAIL"
    CUT = "CUT"
    BLOCKED = "BLOCKED"


class ArtifactApplicability(StrEnum):
    PRESENT = "PRESENT"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    DEFERRED = "DEFERRED"


class DeviceKind(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    TPU = "tpu"
    OTHER = "other"


class FilePin(FrozenStrictModel):
    """Byte-level identity for one file.

    ``path`` is the path recorded in the manifest.  It may be relative to a run
    root or absolute for a frozen external input.  ``records`` is optional because
    it only has an unambiguous meaning for line-oriented artifacts.
    """

    label: NonEmpty
    path: NonEmpty
    sha256: Sha256
    bytes: NonNegativeInt
    records: NonNegativeInt | None = None
    schema_version: NonEmpty | None = None

    @model_validator(mode="after")
    def path_is_a_file_reference(self) -> Self:
        if "\x00" in self.path:
            raise ValueError("file pin path must not contain NUL")
        if self.path.endswith(("/", "\\")):
            raise ValueError("file pin path must identify a file, not a directory")
        return self


class GitPin(FrozenStrictModel):
    branch: NonEmpty
    commit: GitCommit
    dirty: StrictBool


class ModelPin(FrozenStrictModel):
    """Identity of the model implementation/snapshot used by the run.

    ``identity_sha256`` pins a frozen model or snapshot identity document.  A
    trained checkpoint, when one exists, is independently pinned in
    ``artifacts.checkpoint``.
    """

    name: NonEmpty
    implementation_version: NonEmpty
    revision: NonEmpty
    identity_sha256: Sha256


class SelectorInputPins(FrozenStrictModel):
    """Explicit input categories needed to detect accidental data drift."""

    source: tuple[FilePin, ...]
    label: tuple[FilePin, ...]
    data: tuple[FilePin, ...]
    pool: tuple[FilePin, ...]
    component: tuple[FilePin, ...]
    provenance: tuple[FilePin, ...]
    other: tuple[FilePin, ...] = ()

    @model_validator(mode="after")
    def pin_labels_are_unique_within_each_category(self) -> Self:
        for category, pins in self.categories():
            labels = tuple(pin.label for pin in pins)
            if len(labels) != len(set(labels)):
                raise ValueError(f"duplicate input pin label in {category}")
        return self

    def categories(self) -> tuple[tuple[str, tuple[FilePin, ...]], ...]:
        return (
            ("source", self.source),
            ("label", self.label),
            ("data", self.data),
            ("pool", self.pool),
            ("component", self.component),
            ("provenance", self.provenance),
            ("other", self.other),
        )

    def all_pins(self) -> tuple[FilePin, ...]:
        return tuple(pin for _, pins in self.categories() for pin in pins)


class PresentArtifact(FrozenStrictModel):
    state: Literal[ArtifactApplicability.PRESENT] = ArtifactApplicability.PRESENT
    files: tuple[FilePin, ...] = Field(min_length=1)
    description: NonEmpty | None = None

    @model_validator(mode="after")
    def files_are_unique(self) -> Self:
        paths = tuple(pin.path for pin in self.files)
        labels = tuple(pin.label for pin in self.files)
        if len(paths) != len(set(paths)):
            raise ValueError("PRESENT artifact file paths must be unique")
        if len(labels) != len(set(labels)):
            raise ValueError("PRESENT artifact file labels must be unique")
        return self


class NotApplicableArtifact(FrozenStrictModel):
    state: Literal[ArtifactApplicability.NOT_APPLICABLE] = ArtifactApplicability.NOT_APPLICABLE
    reason: NonEmpty


class DeferredArtifact(FrozenStrictModel):
    state: Literal[ArtifactApplicability.DEFERRED] = ArtifactApplicability.DEFERRED
    reason: NonEmpty
    depends_on: tuple[NonEmpty, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def dependencies_are_unique(self) -> Self:
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("DEFERRED artifact dependencies must be unique")
        return self


ArtifactRecord = Annotated[
    PresentArtifact | NotApplicableArtifact | DeferredArtifact,
    Field(discriminator="state"),
]


class SelectorArtifactInventory(FrozenStrictModel):
    """All artifact classes are mandatory, even when absent by design."""

    labels: ArtifactRecord
    preflight: ArtifactRecord
    checkpoint: ArtifactRecord
    candidate_scores: ArtifactRecord
    crc: ArtifactRecord
    decision_trace: ArtifactRecord
    selected_sets: ArtifactRecord
    count_matched: ArtifactRecord
    metrics: ArtifactRecord
    run_log: ArtifactRecord
    checksums: ArtifactRecord

    def named_records(self) -> tuple[tuple[str, ArtifactRecord], ...]:
        return (
            ("labels", self.labels),
            ("preflight", self.preflight),
            ("checkpoint", self.checkpoint),
            ("candidate_scores", self.candidate_scores),
            ("crc", self.crc),
            ("decision_trace", self.decision_trace),
            ("selected_sets", self.selected_sets),
            ("count_matched", self.count_matched),
            ("metrics", self.metrics),
            ("run_log", self.run_log),
            ("checksums", self.checksums),
        )

    @model_validator(mode="after")
    def present_paths_are_unique_across_artifact_classes(self) -> Self:
        owners: dict[str, str] = {}
        for artifact_name, record in self.named_records():
            if not isinstance(record, PresentArtifact):
                continue
            for pin in record.files:
                previous = owners.setdefault(pin.path, artifact_name)
                if previous != artifact_name:
                    raise ValueError(
                        f"artifact file {pin.path!r} is assigned to both "
                        f"{previous} and {artifact_name}"
                    )
        return self


class DeviceInfo(FrozenStrictModel):
    kind: DeviceKind
    name: NonEmpty
    count: PositiveInt = 1
    precision: NonEmpty
    runtime_version: NonEmpty | None = None


class RuntimeInfo(FrozenStrictModel):
    host: NonEmpty
    python_version: NonEmpty
    command: tuple[NonEmpty, ...] = Field(min_length=1)
    started_at_utc: AwareDatetime
    finished_at_utc: AwareDatetime | None = None
    wall_time_seconds: NonNegativeFloat | None = None
    peak_device_memory_bytes: NonNegativeInt | None = None
    throughput_queries_per_second: PositiveFloat | None = None

    @model_validator(mode="after")
    def completion_fields_are_consistent(self) -> Self:
        has_finished = self.finished_at_utc is not None
        has_wall_time = self.wall_time_seconds is not None
        if has_finished != has_wall_time:
            raise ValueError(
                "finished_at_utc and wall_time_seconds must be present or absent together"
            )
        if self.finished_at_utc is not None and self.finished_at_utc < self.started_at_utc:
            raise ValueError("finished_at_utc cannot precede started_at_utc")
        return self


_EXPECTED_STAGE_BY_RUN: dict[int, SelectorExperimentStage] = {
    4: SelectorExperimentStage.LABEL_AUDIT_200Q_PREFLIGHT,
    5: SelectorExperimentStage.DUAL_HEAD_SANITY,
    6: SelectorExperimentStage.TRAIN_SEED,
    7: SelectorExperimentStage.METHOD_SELECTION,
    8: SelectorExperimentStage.CRC_CALIBRATION,
    9: SelectorExperimentStage.DECISION_DEV,
    10: SelectorExperimentStage.TRAIN_AND_CALIBRATE,
    11: SelectorExperimentStage.TRAIN_AND_CALIBRATE,
    12: SelectorExperimentStage.THREE_SEED_DECISION_DEV,
    13: SelectorExperimentStage.FORMAL_FREEZE,
    14: SelectorExperimentStage.FORMAL_EVALUATION,
    15: SelectorExperimentStage.FORMAL_EVALUATION,
}

_REQUIRED_PRESENT_BY_STAGE: dict[SelectorExperimentStage, frozenset[str]] = {
    SelectorExperimentStage.LABEL_AUDIT_200Q_PREFLIGHT: frozenset({"labels", "preflight"}),
    SelectorExperimentStage.DUAL_HEAD_SANITY: frozenset(
        {"checkpoint", "candidate_scores", "decision_trace", "selected_sets", "metrics"}
    ),
    SelectorExperimentStage.TRAIN_SEED: frozenset({"checkpoint", "candidate_scores", "metrics"}),
    SelectorExperimentStage.METHOD_SELECTION: frozenset(
        {
            "checkpoint",
            "candidate_scores",
            "decision_trace",
            "selected_sets",
            "count_matched",
            "metrics",
        }
    ),
    SelectorExperimentStage.CRC_CALIBRATION: frozenset(
        {"checkpoint", "crc", "decision_trace", "selected_sets", "metrics"}
    ),
    SelectorExperimentStage.DECISION_DEV: frozenset(
        {"checkpoint", "crc", "decision_trace", "selected_sets", "count_matched", "metrics"}
    ),
    SelectorExperimentStage.TRAIN_AND_CALIBRATE: frozenset(
        {
            "checkpoint",
            "candidate_scores",
            "crc",
            "decision_trace",
            "selected_sets",
            "metrics",
        }
    ),
    SelectorExperimentStage.THREE_SEED_DECISION_DEV: frozenset(
        {"checkpoint", "crc", "decision_trace", "selected_sets", "count_matched", "metrics"}
    ),
    SelectorExperimentStage.FORMAL_FREEZE: frozenset({"checkpoint", "crc"}),
    SelectorExperimentStage.FORMAL_EVALUATION: frozenset(
        {"checkpoint", "crc", "decision_trace", "selected_sets", "count_matched", "metrics"}
    ),
}

_R004_ONLY_PRESENT = frozenset({"labels", "preflight", "run_log", "checksums"})
_ALWAYS_REQUIRED_PRESENT = frozenset({"run_log", "checksums"})
_CORE_INPUT_CATEGORIES = ("source", "label", "data", "pool", "component", "provenance")
_COMPLETED_RUNTIME_STATUSES = frozenset(
    {
        SelectorRunStatus.PASS,
        SelectorRunStatus.FAIL,
        SelectorRunStatus.CUT,
    }
)


class SelectorExperimentManifestV2(FrozenStrictModel):
    """Canonical manifest contract for Selector research runs R004+."""

    schema_version: Literal["2.0"] = "2.0"
    manifest_type: Literal["selector_experiment_manifest"] = "selector_experiment_manifest"
    run_id: Annotated[str, Field(pattern=r"^R[0-9]{3,}$")]
    run_name: NonEmpty
    stage: SelectorExperimentStage
    status: SelectorRunStatus
    git: GitPin
    config: FilePin
    model: ModelPin
    inputs: SelectorInputPins
    artifacts: SelectorArtifactInventory
    device: DeviceInfo
    runtime: RuntimeInfo
    notes: tuple[NonEmpty, ...] = ()

    @model_validator(mode="after")
    def run_stage_status_and_artifacts_are_consistent(self) -> Self:
        run_number = int(self.run_id[1:])
        if run_number < 4:
            raise ValueError("Selector experiment manifest v2 is only valid for R004+")

        expected_stage = _EXPECTED_STAGE_BY_RUN.get(run_number)
        if expected_stage is not None and self.stage != expected_stage:
            raise ValueError(
                f"{self.run_id} requires stage {expected_stage.value!r}, not {self.stage.value!r}"
            )

        runtime_complete = self.runtime.finished_at_utc is not None
        if self.status in _COMPLETED_RUNTIME_STATUSES and not runtime_complete:
            raise ValueError(f"runtime must be complete when run status is {self.status.value}")
        if self.status == SelectorRunStatus.RUNNING and runtime_complete:
            raise ValueError("runtime must be incomplete when run status is RUNNING")

        if self.status != SelectorRunStatus.PASS:
            return self

        if self.git.dirty:
            raise ValueError("PASS requires a clean pinned git worktree")

        empty_categories = [
            category for category in _CORE_INPUT_CATEGORIES if not getattr(self.inputs, category)
        ]
        if empty_categories:
            raise ValueError(
                "PASS requires source/label/data/pool/component/provenance input pins; "
                f"empty categories: {empty_categories}"
            )

        required = _REQUIRED_PRESENT_BY_STAGE[self.stage] | _ALWAYS_REQUIRED_PRESENT
        records = dict(self.artifacts.named_records())
        missing = sorted(
            artifact_name
            for artifact_name in required
            if not isinstance(records[artifact_name], PresentArtifact)
        )
        if missing:
            raise ValueError(
                f"PASS at stage {self.stage.value!r} requires PRESENT artifacts: {missing}"
            )

        if self.stage == SelectorExperimentStage.LABEL_AUDIT_200Q_PREFLIGHT:
            unexpected = sorted(
                artifact_name
                for artifact_name, record in records.items()
                if artifact_name not in _R004_ONLY_PRESENT and isinstance(record, PresentArtifact)
            )
            if unexpected:
                raise ValueError(
                    f"R004 must not claim future-stage artifacts as PRESENT: {unexpected}"
                )
            if self.runtime.peak_device_memory_bytes is None:
                raise ValueError("R004 PASS requires measured peak device memory")
            if self.runtime.throughput_queries_per_second is None:
                raise ValueError("R004 PASS requires measured query throughput")
        return self


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pin_file(
    path: Path,
    *,
    label: str,
    recorded_path: str | None = None,
    records: int | None = None,
    schema_version: str | None = None,
) -> FilePin:
    """Build a byte/sha pin from an existing regular file."""

    source = Path(path)
    if not source.is_file():
        raise ValueError(f"cannot pin missing or non-file path: {source}")
    return FilePin(
        label=label,
        path=recorded_path if recorded_path is not None else source.as_posix(),
        sha256=_sha256_file(source),
        bytes=source.stat().st_size,
        records=records,
        schema_version=schema_version,
    )


def canonical_manifest_bytes(manifest: SelectorExperimentManifestV2) -> bytes:
    """Return the one permitted byte representation of a v2 manifest."""

    payload = json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{payload}\n".encode()


def manifest_sha256(manifest: SelectorExperimentManifestV2) -> str:
    return hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()


def _resolve_pin_path(pin: FilePin, root: Path) -> Path:
    path = Path(pin.path)
    return path if path.is_absolute() else root / path


def verify_file_pin(pin: FilePin, *, root: Path) -> Path:
    """Require the referenced file's size, SHA-256, and optional line count."""

    path = _resolve_pin_path(pin, Path(root))
    if not path.is_file():
        raise ValueError(f"missing pinned file for {pin.label!r}: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != pin.bytes:
        raise ValueError(
            f"pinned file size mismatch for {pin.label!r}: "
            f"actual={actual_bytes}, expected={pin.bytes}, path={path}"
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != pin.sha256:
        raise ValueError(
            f"pinned file SHA-256 mismatch for {pin.label!r}: "
            f"actual={actual_sha256}, expected={pin.sha256}, path={path}"
        )
    if pin.records is not None:
        with path.open("rb") as stream:
            actual_records = sum(1 for _ in stream)
        if actual_records != pin.records:
            raise ValueError(
                f"pinned file record count mismatch for {pin.label!r}: "
                f"actual={actual_records}, expected={pin.records}, path={path}"
            )
    return path


def verify_selector_experiment_pins(
    manifest: SelectorExperimentManifestV2, *, root: Path
) -> tuple[Path, ...]:
    """Verify config, every input, and every PRESENT artifact; ignore no pin."""

    pins: list[FilePin] = [manifest.config, *manifest.inputs.all_pins()]
    for _, record in manifest.artifacts.named_records():
        if isinstance(record, PresentArtifact):
            pins.extend(record.files)
    return tuple(verify_file_pin(pin, root=root) for pin in pins)


def write_selector_experiment_manifest(path: Path, manifest: SelectorExperimentManifestV2) -> Path:
    """Create a canonical manifest exactly once using exclusive file creation."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(canonical_manifest_bytes(manifest))
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite write-once Selector manifest: {destination}"
        ) from error
    return destination


def load_selector_experiment_manifest(path: Path) -> SelectorExperimentManifestV2:
    """Read a manifest and reject valid JSON stored in a non-canonical encoding."""

    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise ValueError(f"missing or unreadable Selector manifest {source}: {error}") from error
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Selector manifest JSON {source}: {error}") from error
    manifest = SelectorExperimentManifestV2.model_validate(parsed)
    canonical = canonical_manifest_bytes(manifest)
    if raw != canonical:
        raise ValueError(f"Selector manifest is not canonical JSON: {source}")
    return manifest


def verify_selector_experiment_manifest(
    path: Path,
    *,
    expected: SelectorExperimentManifestV2 | None = None,
    pin_root: Path | None = None,
    verify_pins: bool = False,
) -> SelectorExperimentManifestV2:
    """Verify-only path: parse canonical bytes, optionally compare and check pins."""

    source = Path(path)
    actual = load_selector_experiment_manifest(source)
    if expected is not None:
        actual_bytes = canonical_manifest_bytes(actual)
        expected_bytes = canonical_manifest_bytes(expected)
        if actual_bytes != expected_bytes:
            raise ValueError(
                "Selector manifest differs from expected recomputation: "
                f"actual_sha256={hashlib.sha256(actual_bytes).hexdigest()}, "
                f"expected_sha256={hashlib.sha256(expected_bytes).hexdigest()}"
            )
    if verify_pins:
        verify_selector_experiment_pins(
            actual,
            root=Path(pin_root) if pin_root is not None else source.parent,
        )
    return actual


def freeze_or_verify_selector_experiment_manifest(
    path: Path,
    manifest: SelectorExperimentManifestV2,
    *,
    verify_only: bool,
    pin_root: Path | None = None,
    verify_pins: bool = False,
) -> SelectorExperimentManifestV2:
    """Shared runner primitive whose verify-only branch performs no writes."""

    destination = Path(path)
    if verify_only:
        return verify_selector_experiment_manifest(
            destination,
            expected=manifest,
            pin_root=pin_root,
            verify_pins=verify_pins,
        )
    if verify_pins:
        verify_selector_experiment_pins(
            manifest,
            root=Path(pin_root) if pin_root is not None else destination.parent,
        )
    write_selector_experiment_manifest(destination, manifest)
    return manifest


def utc_now() -> datetime:
    """Small injectable-friendly helper for runners constructing runtime records."""

    return datetime.now(UTC)
