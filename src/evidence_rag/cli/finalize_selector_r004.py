"""Finalize or independently verify the fixed-layout R004 Selector run.

The command is intentionally specific to R004.  It adapts the already verified
label and resource-preflight bundles into the generic Selector manifest v2
contract without rerunning either GPU work or label construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from evidence_rag.evaluation.selector_artifacts import (
    SELECTOR_EXPERIMENT_MANIFEST_FILE,
    DeferredArtifact,
    DeviceInfo,
    DeviceKind,
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
    write_selector_experiment_manifest,
)
from evidence_rag.evaluation.selector_components import OUTPUT_FILES as COMPONENT_OUTPUT_FILES
from evidence_rag.evaluation.selector_preflight import (
    MANIFEST_FILE as PREFLIGHT_MANIFEST_FILE,
)
from evidence_rag.evaluation.selector_preflight import (
    OUTPUT_FILES as PREFLIGHT_OUTPUT_FILES,
)
from evidence_rag.evaluation.selector_preflight import (
    REPORT_FILE as PREFLIGHT_REPORT_FILE,
)
from evidence_rag.evaluation.selector_preflight import (
    RUNTIME_LOG_FILE as PREFLIGHT_RUNTIME_LOG_FILE,
)
from evidence_rag.evaluation.selector_preflight import (
    SAMPLE_FILE as PREFLIGHT_SAMPLE_FILE,
)
from evidence_rag.evaluation.selector_preflight import (
    DatasetKind,
    PreflightSampleQuery,
    R004ResourcePreflightReport,
    verify_resource_preflight_artifacts,
)
from evidence_rag.materializer.selector_labels import OUTPUT_FILES as LABEL_OUTPUT_FILES
from evidence_rag.materializer.selector_labels import verify_selector_label_artifacts
from evidence_rag.materializer.selector_pool import (
    CANDIDATE_FILE,
    SELECTOR_POOL_MANIFEST_FILE,
)

CHECKSUMS_FILE = "CHECKSUMS.sha256"
CONFIG_FILE = "config.toml"
LABELS_DIRECTORY = "labels"
PREFLIGHT_DIRECTORY = "preflight"


@dataclass(frozen=True)
class _FinalizationContext:
    config: FilePin
    inputs: SelectorInputPins
    labels: tuple[FilePin, ...]
    preflight: tuple[FilePin, ...]
    run_log: FilePin
    git: GitPin
    model: ModelPin
    device: DeviceInfo
    runtime: RuntimeInfo


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finalize or verify the fixed-layout R004 Selector run"
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_mapping(path: Path, *, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read {label} at {path}: {error}") from error
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed JSON object")
    return cast(Mapping[str, object], value)


def _input_pin_mapping(manifest: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    raw = manifest.get("inputs")
    if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
        raise ValueError("R004 preflight manifest has no string-keyed input pins")
    result: dict[str, Mapping[str, object]] = {}
    for name, pin in raw.items():
        if not isinstance(pin, Mapping) or any(not isinstance(key, str) for key in pin):
            raise ValueError(f"R004 preflight input pin is not an object: {name}")
        result[str(name)] = cast(Mapping[str, object], pin)
    return result


def _source_path(run_root: Path, name: str, pins: Mapping[str, Mapping[str, object]]) -> Path:
    try:
        value = pins[name]["path"]
    except KeyError as error:
        raise ValueError(f"R004 preflight input pins omit {name!r}") from error
    if not isinstance(value, str) or not value:
        raise ValueError(f"R004 preflight input pin {name!r} has no path")
    path = Path(value)
    return (path if path.is_absolute() else Path(run_root) / path).resolve()


def _recorded_path(run_root: Path, source: Path) -> str:
    root = Path(run_root).resolve()
    resolved = Path(source).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _adapt_input_pin(
    run_root: Path,
    *,
    name: str,
    pin: Mapping[str, object],
) -> FilePin:
    source = _source_path(run_root, name, {name: pin})
    return FilePin.model_validate(
        {
            "label": name,
            "path": _recorded_path(run_root, source),
            "sha256": pin.get("sha256"),
            "bytes": pin.get("bytes"),
        }
    )


def _require_same_path(actual: Path, expected: Path, *, label: str) -> None:
    if Path(actual).resolve() != Path(expected).resolve():
        raise ValueError(
            f"R004 preflight {label} pin does not point to the fixed run layout: "
            f"actual={actual}, expected={expected}"
        )


def _require_exact_directory(directory: Path, expected_names: Sequence[str], *, label: str) -> None:
    if not Path(directory).is_dir():
        raise ValueError(f"missing R004 {label} directory: {directory}")
    actual = {path.name for path in Path(directory).iterdir()}
    expected = set(expected_names)
    if actual != expected:
        raise ValueError(
            f"R004 {label} has an unexpected file set: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    if any(not (Path(directory) / name).is_file() for name in expected):
        raise ValueError(f"R004 {label} entries must all be regular files")


def _expected_model(report: R004ResourcePreflightReport) -> dict[str, object]:
    return {
        "model_id": report.model.model_id,
        "revision": report.model.revision,
        "snapshot_identity_sha256": report.model.snapshot_identity_sha256,
        "snapshot_files": {
            name: pin.model_dump(mode="json")
            for name, pin in sorted(report.model.snapshot_files.items())
        },
    }


def _verify_preflight(
    run_root: Path,
) -> tuple[R004ResourcePreflightReport, dict[str, Mapping[str, object]]]:
    directory = Path(run_root) / PREFLIGHT_DIRECTORY
    _require_exact_directory(directory, PREFLIGHT_OUTPUT_FILES, label="preflight bundle")
    manifest = _load_json_mapping(
        directory / PREFLIGHT_MANIFEST_FILE, label="R004 preflight manifest"
    )
    input_pins = _input_pin_mapping(manifest)
    try:
        report = R004ResourcePreflightReport.model_validate_json(
            (directory / PREFLIGHT_REPORT_FILE).read_bytes()
        )
        sample = tuple(
            PreflightSampleQuery.model_validate_json(line)
            for line in (directory / PREFLIGHT_SAMPLE_FILE).read_bytes().splitlines()
        )
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid R004 preflight report/sample: {error}") from error
    verified = verify_resource_preflight_artifacts(
        directory,
        expected_sample=sample,
        expected_input_pins=input_pins,
        expected_git=report.git.model_dump(mode="json"),
        expected_model=_expected_model(report),
    )
    typed = R004ResourcePreflightReport.model_validate_json(
        json.dumps(verified.report, ensure_ascii=True, allow_nan=False, sort_keys=True)
    )
    return typed, input_pins


def _verify_labels(
    run_root: Path,
    *,
    report: R004ResourcePreflightReport,
    input_pins: Mapping[str, Mapping[str, object]],
) -> None:
    root = Path(run_root)
    dataset_kinds: tuple[DatasetKind, ...] = ("niah", "2wiki")
    for kind in dataset_kinds:
        label_directory = root / LABELS_DIRECTORY / kind
        _require_exact_directory(label_directory, LABEL_OUTPUT_FILES, label=f"{kind} label bundle")
        for filename in LABEL_OUTPUT_FILES:
            key = f"{kind}/labels/{filename}"
            _require_same_path(
                _source_path(root, key, input_pins),
                label_directory / filename,
                label=key,
            )

        component_directory = _source_path(
            root, f"{kind}/components/{COMPONENT_OUTPUT_FILES[0]}", input_pins
        ).parent
        for filename in COMPONENT_OUTPUT_FILES:
            _require_same_path(
                _source_path(root, f"{kind}/components/{filename}", input_pins),
                component_directory / filename,
                label=f"{kind}/components/{filename}",
            )
        candidate_pool = _source_path(root, f"{kind}/candidate_pool", input_pins).parent
        _require_same_path(
            _source_path(root, f"{kind}/candidate_pool", input_pins),
            candidate_pool / CANDIDATE_FILE,
            label=f"{kind}/candidate_pool",
        )
        _require_same_path(
            _source_path(root, f"{kind}/pool_manifest", input_pins),
            candidate_pool / SELECTOR_POOL_MANIFEST_FILE,
            label=f"{kind}/pool_manifest",
        )

        artifacts = verify_selector_label_artifacts(
            output_directory=label_directory,
            dataset_kind=kind,
            dataset_manifest_path=_source_path(root, f"{kind}/dataset_manifest", input_pins),
            source_parent_path=_source_path(root, f"{kind}/source_parent", input_pins),
            candidate_pool_path=candidate_pool,
            component_directory=component_directory,
            assignment_path=(
                _source_path(root, "niah/assignment", input_pins) if kind == "niah" else None
            ),
            provenance_path=(
                _source_path(root, "niah/provenance", input_pins) if kind == "niah" else None
            ),
        )
        expected_audit = json.dumps(
            report.label_audit[kind], ensure_ascii=True, allow_nan=False, sort_keys=True
        )
        actual_audit = json.dumps(
            artifacts.report, ensure_ascii=True, allow_nan=False, sort_keys=True
        )
        if actual_audit != expected_audit:
            raise ValueError(f"{kind} label audit differs from the typed preflight report")


def _classified_inputs(
    run_root: Path, input_pins: Mapping[str, Mapping[str, object]]
) -> tuple[FilePin, SelectorInputPins]:
    if set(input_pins) == {"config"}:
        raise ValueError("R004 preflight manifest has no data input pins")
    preflight_config = _adapt_input_pin(run_root, name="preflight/config", pin=input_pins["config"])
    verify_file_pin(preflight_config, root=run_root)
    config = _run_pin(run_root, CONFIG_FILE)
    if (config.bytes, config.sha256) != (
        preflight_config.bytes,
        preflight_config.sha256,
    ):
        raise ValueError(
            "run_root/config.toml differs from the Git-tracked config pinned by preflight"
        )
    buckets: dict[str, list[FilePin]] = {
        "source": [],
        "label": [],
        "data": [],
        "pool": [],
        "component": [],
        "provenance": [],
        "other": [],
    }
    for name in sorted(set(input_pins) - {"config"}):
        if name.endswith("/source_parent"):
            category = "source"
        elif "/labels/" in name:
            category = "label"
        elif name.endswith("/dataset_manifest"):
            category = "data"
        elif name.endswith(("/candidate_pool", "/pool_manifest")):
            category = "pool"
        elif "/components/" in name:
            category = "component"
        elif name in {"niah/assignment", "niah/provenance"}:
            category = "provenance"
        elif name.startswith("model_snapshot/"):
            category = "other"
        else:
            raise ValueError(f"unclassified R004 preflight input pin: {name}")
        buckets[category].append(_adapt_input_pin(run_root, name=name, pin=input_pins[name]))
    inputs = SelectorInputPins(
        source=tuple(buckets["source"]),
        label=tuple(buckets["label"]),
        data=tuple(buckets["data"]),
        pool=tuple(buckets["pool"]),
        component=tuple(buckets["component"]),
        provenance=tuple(buckets["provenance"]),
        other=tuple(buckets["other"]),
    )
    return config, inputs


def _line_records(path: Path) -> int | None:
    if Path(path).suffix != ".jsonl":
        return None
    return len(Path(path).read_bytes().splitlines())


def _run_pin(run_root: Path, relative: str, *, schema_version: str | None = None) -> FilePin:
    path = Path(run_root) / relative
    return pin_file(
        path,
        label=relative,
        recorded_path=relative,
        records=_line_records(path),
        schema_version=schema_version,
    )


def _load_verified_context(run_root: Path) -> _FinalizationContext:
    root = Path(run_root).resolve()
    report, input_pins = _verify_preflight(root)
    _verify_labels(root, report=report, input_pins=input_pins)
    config, inputs = _classified_inputs(root, input_pins)
    for pin in (config, *inputs.all_pins()):
        verify_file_pin(pin, root=root)

    labels = tuple(
        _run_pin(
            root,
            f"{LABELS_DIRECTORY}/{kind}/{filename}",
            schema_version=("2.0" if filename.endswith("_manifest.json") else "1.0"),
        )
        for kind in ("niah", "2wiki")
        for filename in LABEL_OUTPUT_FILES
    )
    preflight = tuple(
        _run_pin(
            root,
            f"{PREFLIGHT_DIRECTORY}/{filename}",
            schema_version="1.0",
        )
        for filename in PREFLIGHT_OUTPUT_FILES
        if filename != PREFLIGHT_RUNTIME_LOG_FILE
    )
    run_log = _run_pin(root, f"{PREFLIGHT_DIRECTORY}/{PREFLIGHT_RUNTIME_LOG_FILE}")
    return _FinalizationContext(
        config=config,
        inputs=inputs,
        labels=labels,
        preflight=preflight,
        run_log=run_log,
        git=GitPin.model_validate(report.git.model_dump(mode="json")),
        model=ModelPin(
            name=report.model.model_id,
            implementation_version="shared-deberta-independent-sigmoid-heads-v1",
            revision=report.model.revision,
            identity_sha256=report.model.snapshot_identity_sha256,
        ),
        device=DeviceInfo(
            kind=DeviceKind(report.device.kind),
            name=report.device.name,
            precision=report.device.precision,
            runtime_version=report.device.cuda_runtime_version,
        ),
        runtime=RuntimeInfo(
            host=report.runtime.host,
            python_version=report.runtime.python_version,
            command=report.runtime.command,
            started_at_utc=report.runtime.started_at_utc,
            finished_at_utc=report.runtime.finished_at_utc,
            wall_time_seconds=report.runtime.wall_time_seconds,
            peak_device_memory_bytes=report.device.peak_memory_allocated_bytes,
            throughput_queries_per_second=report.forward_probe.queries_per_second,
        ),
    )


def _prerequisite_relative_files() -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                CONFIG_FILE,
                *(
                    f"{LABELS_DIRECTORY}/{kind}/{filename}"
                    for kind in ("niah", "2wiki")
                    for filename in LABEL_OUTPUT_FILES
                ),
                *(f"{PREFLIGHT_DIRECTORY}/{filename}" for filename in PREFLIGHT_OUTPUT_FILES),
            }
        )
    )


def _actual_relative_files(run_root: Path) -> set[str]:
    root = Path(run_root)
    values: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"R004 run root must not contain symlinks: {path}")
        if path.is_file():
            values.add(path.relative_to(root).as_posix())
    return values


def _require_exact_run_files(run_root: Path, *, finalized: bool) -> None:
    expected = set(_prerequisite_relative_files())
    if finalized:
        expected.update({CHECKSUMS_FILE, SELECTOR_EXPERIMENT_MANIFEST_FILE})
    actual = _actual_relative_files(run_root)
    if actual != expected:
        raise ValueError(
            "R004 run root has an unexpected file set: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def _checksums_bytes(run_root: Path) -> bytes:
    lines = [
        f"{_sha256_file(Path(run_root) / relative)}  {relative}"
        for relative in _prerequisite_relative_files()
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _manifest(
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
        run_id="R004",
        run_name="R004 label audit and 200-query resource preflight",
        stage=SelectorExperimentStage.LABEL_AUDIT_200Q_PREFLIGHT,
        status=SelectorRunStatus.PASS,
        git=context.git,
        config=context.config,
        model=context.model,
        inputs=context.inputs,
        artifacts=SelectorArtifactInventory(
            labels=PresentArtifact(
                files=context.labels,
                description="Two independently verified source-train label bundles",
            ),
            preflight=PresentArtifact(
                files=context.preflight,
                description="R004 resource proof; no Selector policy or effect estimate",
            ),
            checkpoint=NotApplicableArtifact(
                reason="R004 discards its ephemeral training probe and writes no checkpoint"
            ),
            candidate_scores=NotApplicableArtifact(
                reason="untrained resource-probe outputs are not Selector candidate scores"
            ),
            crc=NotApplicableArtifact(reason="R004 has no trained scores or calibration policy"),
            decision_trace=NotApplicableArtifact(
                reason="R004 constructs no Selector policy or decision trace"
            ),
            selected_sets=NotApplicableArtifact(
                reason="R004 constructs no Selector-selected evidence sets"
            ),
            count_matched=DeferredArtifact(
                reason="count-matched controls require a real Selector decision trace",
                depends_on=("real-selector-decision-trace",),
            ),
            metrics=NotApplicableArtifact(
                reason="R004 measures label/resource feasibility, not Selector effect"
            ),
            run_log=PresentArtifact(files=(context.run_log,)),
            checksums=PresentArtifact(files=(checksum_pin,)),
        ),
        device=context.device,
        runtime=context.runtime,
        notes=(
            "R004 PASS means label and resource feasibility only.",
            "No deletion policy, checkpoint, CRC decision, or Selector effect is claimed.",
        ),
    )


def finalize_selector_r004(
    run_root: Path, *, verify_only: bool = False
) -> SelectorExperimentManifestV2:
    """Freeze the two top-level files once, or verify both without writing."""

    root = Path(run_root).resolve()
    if not root.is_dir():
        raise ValueError(f"missing R004 run root: {root}")
    checksum_path = root / CHECKSUMS_FILE
    manifest_path = root / SELECTOR_EXPERIMENT_MANIFEST_FILE
    if not verify_only:
        existing = [path.name for path in (checksum_path, manifest_path) if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite write-once R004 outputs: {existing}")

    context = _load_verified_context(root)
    _require_exact_run_files(root, finalized=verify_only)
    expected_checksums = _checksums_bytes(root)
    expected_manifest = _manifest(context, checksum_bytes=expected_checksums)

    if verify_only:
        try:
            actual_checksums = checksum_path.read_bytes()
            actual_manifest = manifest_path.read_bytes()
        except OSError as error:
            raise ValueError(f"missing R004 finalization output: {error}") from error
        if actual_checksums != expected_checksums:
            raise ValueError(
                "R004 CHECKSUMS.sha256 differs from byte-for-byte recomputation: "
                f"actual_sha256={_sha256_bytes(actual_checksums)}, "
                f"expected_sha256={_sha256_bytes(expected_checksums)}"
            )
        expected_manifest_bytes = canonical_manifest_bytes(expected_manifest)
        if actual_manifest != expected_manifest_bytes:
            raise ValueError(
                "R004 Selector manifest differs from byte-for-byte recomputation: "
                f"actual_sha256={_sha256_bytes(actual_manifest)}, "
                f"expected_sha256={_sha256_bytes(expected_manifest_bytes)}"
            )
        return verify_selector_experiment_manifest(
            manifest_path,
            expected=expected_manifest,
            pin_root=root,
            verify_pins=True,
        )

    try:
        with checksum_path.open("xb") as stream:
            stream.write(expected_checksums)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite write-once R004 checksums: {checksum_path}"
        ) from error
    verify_selector_experiment_pins(expected_manifest, root=root)
    write_selector_experiment_manifest(manifest_path, expected_manifest)
    _require_exact_run_files(root, finalized=True)
    return verify_selector_experiment_manifest(
        manifest_path,
        expected=expected_manifest,
        pin_root=root,
        verify_pins=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    manifest = finalize_selector_r004(arguments.run_root, verify_only=bool(arguments.verify_only))
    manifest_path = Path(arguments.run_root).resolve() / SELECTOR_EXPERIMENT_MANIFEST_FILE
    checksums_path = Path(arguments.run_root).resolve() / CHECKSUMS_FILE
    print(
        json.dumps(
            {
                "action": "verified" if arguments.verify_only else "frozen",
                "checksums": str(checksums_path),
                "checksums_sha256": _sha256_file(checksums_path),
                "kind": "selector-r004-finalization",
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256_bytes(canonical_manifest_bytes(manifest)),
                "status": manifest.status.value,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
