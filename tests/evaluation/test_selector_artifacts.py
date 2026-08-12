from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from evidence_rag.evaluation.selector_artifacts import (
    ArtifactApplicability,
    DeferredArtifact,
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
    freeze_or_verify_selector_experiment_manifest,
    load_selector_experiment_manifest,
    manifest_sha256,
    pin_file,
    verify_selector_experiment_manifest,
    write_selector_experiment_manifest,
)


def _write(root: Path, name: str, payload: bytes) -> FilePin:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return pin_file(path, label=name, recorded_path=name)


def _not_applicable(reason: str = "not used at this stage") -> NotApplicableArtifact:
    return NotApplicableArtifact(reason=reason)


def _deferred(reason: str = "requires a later real Selector run") -> DeferredArtifact:
    return DeferredArtifact(reason=reason, depends_on=("real-selector-trace",))


def _r004_manifest(root: Path, *, status: SelectorRunStatus = SelectorRunStatus.PASS):
    config = _write(root, "R004_CONFIG.json", b'{"batch_size":8}\n')
    source = _write(root, "inputs/source.jsonl", b'{"query_id":"q1"}\n')
    label_spec = _write(root, "inputs/label_spec.json", b'{"version":"1"}\n')
    data = _write(root, "inputs/data.jsonl", b'{"document_id":"d1"}\n')
    pool = _write(root, "inputs/pool.jsonl", b'{"query_id":"q1","candidates":[]}\n')
    component = _write(root, "inputs/components.jsonl", b'{"query_id":"q1","c":"c1"}\n')
    provenance = _write(root, "inputs/provenance.jsonl", b'{"query_id":"q1"}\n')
    labels = _write(root, "labels/label_audit.json", b'{"status":"PASS"}\n')
    preflight = _write(root, "preflight/throughput.json", b'{"qps":12.5}\n')
    run_log = _write(root, "R004_RUN_LOG.md", b"# R004\n")
    checksums = _write(root, "CHECKSUMS.sha256", b"frozen checksums\n")
    started = datetime(2026, 8, 12, 10, 0, tzinfo=UTC)
    terminal = status != SelectorRunStatus.RUNNING
    return SelectorExperimentManifestV2(
        run_id="R004",
        run_name="label-audit-and-200q-preflight",
        stage=SelectorExperimentStage.LABEL_AUDIT_200Q_PREFLIGHT,
        status=status,
        git=GitPin(branch="refactor/three-module-baseline", commit="a" * 40, dirty=False),
        config=config,
        model=ModelPin(
            name="dual-head-base",
            implementation_version="dual-head-v1",
            revision="frozen-revision",
            identity_sha256="b" * 64,
        ),
        inputs=SelectorInputPins(
            source=(source,),
            label=(label_spec,),
            data=(data,),
            pool=(pool,),
            component=(component,),
            provenance=(provenance,),
        ),
        artifacts=SelectorArtifactInventory(
            labels=PresentArtifact(files=(labels,)),
            preflight=PresentArtifact(files=(preflight,)),
            checkpoint=_deferred("training has not started"),
            candidate_scores=_not_applicable(),
            crc=_deferred("requires trained scores and calibration"),
            decision_trace=_deferred(),
            selected_sets=_deferred(),
            count_matched=_deferred(),
            metrics=_not_applicable("Selector-effect metrics are not evaluated in R004"),
            run_log=PresentArtifact(files=(run_log,)),
            checksums=PresentArtifact(files=(checksums,)),
        ),
        device=DeviceInfo(kind="cuda", name="test-gpu", precision="fp32"),
        runtime=RuntimeInfo(
            host="test-host",
            python_version="3.11",
            command=("python", "-m", "r004"),
            started_at_utc=started,
            finished_at_utc=started + timedelta(seconds=16) if terminal else None,
            wall_time_seconds=16.0 if terminal else None,
            peak_device_memory_bytes=1024,
            throughput_queries_per_second=12.5,
        ),
    )


def test_r004_pass_expresses_present_and_absent_artifacts_explicitly(tmp_path: Path) -> None:
    manifest = _r004_manifest(tmp_path)

    assert manifest.schema_version == "2.0"
    assert manifest.artifacts.labels.state == ArtifactApplicability.PRESENT
    assert manifest.artifacts.preflight.state == ArtifactApplicability.PRESENT
    assert manifest.artifacts.checkpoint.state == ArtifactApplicability.DEFERRED
    assert manifest.artifacts.metrics.state == ArtifactApplicability.NOT_APPLICABLE
    with pytest.raises(ValidationError):
        GitPin(branch="main", commit="a" * 40, dirty=False, unexpected=True)
    with pytest.raises(ValidationError):
        manifest.status = SelectorRunStatus.FAIL  # type: ignore[misc]


def test_deferred_cannot_disguise_an_empty_or_pinned_file() -> None:
    with pytest.raises(ValidationError, match="files"):
        DeferredArtifact.model_validate(
            {
                "state": "DEFERRED",
                "reason": "later",
                "depends_on": ["trace"],
                "files": [],
            }
        )
    with pytest.raises(ValidationError, match="sha256"):
        DeferredArtifact.model_validate(
            {
                "state": "DEFERRED",
                "reason": "later",
                "depends_on": ["trace"],
                "sha256": "0" * 64,
            }
        )
    with pytest.raises(ValidationError):
        PresentArtifact(files=())


def test_pass_rejects_missing_stage_required_artifact(tmp_path: Path) -> None:
    manifest = _r004_manifest(tmp_path)
    payload = manifest.model_dump(mode="python")
    payload["artifacts"]["labels"] = {
        "state": "DEFERRED",
        "reason": "not ready",
        "depends_on": ("label-builder",),
    }

    with pytest.raises(ValidationError, match="requires PRESENT artifacts.*labels"):
        SelectorExperimentManifestV2.model_validate(payload)


def test_r004_rejects_future_artifact_claim_and_empty_core_input(tmp_path: Path) -> None:
    manifest = _r004_manifest(tmp_path)
    future_pin = _write(tmp_path, "checkpoint/model.safetensors", b"weights")
    payload = manifest.model_dump(mode="python")
    payload["artifacts"]["checkpoint"] = {
        "state": "PRESENT",
        "files": (future_pin.model_dump(mode="python"),),
    }
    with pytest.raises(ValidationError, match="future-stage artifacts"):
        SelectorExperimentManifestV2.model_validate(payload)

    payload = manifest.model_dump(mode="python")
    payload["inputs"]["provenance"] = ()
    with pytest.raises(ValidationError, match="empty categories.*provenance"):
        SelectorExperimentManifestV2.model_validate(payload)


def test_run_stage_terminal_runtime_and_clean_git_are_enforced(tmp_path: Path) -> None:
    manifest = _r004_manifest(tmp_path)
    payload = manifest.model_dump(mode="python")
    payload["stage"] = SelectorExperimentStage.TRAIN_SEED
    with pytest.raises(ValidationError, match="R004 requires stage"):
        SelectorExperimentManifestV2.model_validate(payload)

    payload = manifest.model_dump(mode="python")
    payload["git"]["dirty"] = True
    with pytest.raises(ValidationError, match="clean pinned git"):
        SelectorExperimentManifestV2.model_validate(payload)

    payload = manifest.model_dump(mode="python")
    payload["runtime"]["finished_at_utc"] = None
    payload["runtime"]["wall_time_seconds"] = None
    with pytest.raises(ValidationError, match="runtime must be complete"):
        SelectorExperimentManifestV2.model_validate(payload)


def test_r004_pass_requires_measured_memory_and_throughput(tmp_path: Path) -> None:
    manifest = _r004_manifest(tmp_path)
    payload = manifest.model_dump(mode="python")
    payload["runtime"]["peak_device_memory_bytes"] = None
    with pytest.raises(ValidationError, match="peak device memory"):
        SelectorExperimentManifestV2.model_validate(payload)

    payload = manifest.model_dump(mode="python")
    payload["runtime"]["throughput_queries_per_second"] = None
    with pytest.raises(ValidationError, match="query throughput"):
        SelectorExperimentManifestV2.model_validate(payload)


def test_canonical_json_hash_write_once_and_verify_only(tmp_path: Path) -> None:
    manifest = _r004_manifest(tmp_path)
    manifest_path = tmp_path / "selector_experiment_manifest.json"
    canonical = canonical_manifest_bytes(manifest)

    assert canonical.endswith(b"\n")
    assert b'": ' not in canonical
    assert b', "' not in canonical
    assert manifest_sha256(manifest) == hashlib.sha256(canonical).hexdigest()
    assert write_selector_experiment_manifest(manifest_path, manifest) == manifest_path
    assert manifest_path.read_bytes() == canonical
    assert load_selector_experiment_manifest(manifest_path) == manifest
    assert (
        verify_selector_experiment_manifest(
            manifest_path,
            expected=manifest,
            pin_root=tmp_path,
            verify_pins=True,
        )
        == manifest
    )

    with pytest.raises(FileExistsError, match="write-once"):
        write_selector_experiment_manifest(manifest_path, manifest)
    assert (
        freeze_or_verify_selector_experiment_manifest(
            manifest_path,
            manifest,
            verify_only=True,
            pin_root=tmp_path,
            verify_pins=True,
        )
        == manifest
    )


def test_verify_rejects_noncanonical_manifest_and_tampered_pin(tmp_path: Path) -> None:
    manifest = _r004_manifest(tmp_path)
    manifest_path = tmp_path / "selector_experiment_manifest.json"
    write_selector_experiment_manifest(manifest_path, manifest)

    parsed = json.loads(manifest_path.read_bytes())
    manifest_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="not canonical"):
        verify_selector_experiment_manifest(manifest_path)

    manifest_path.write_bytes(canonical_manifest_bytes(manifest))
    (tmp_path / "labels/label_audit.json").write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="mismatch.*label_audit"):
        verify_selector_experiment_manifest(
            manifest_path,
            pin_root=tmp_path,
            verify_pins=True,
        )


def test_file_pin_types_and_record_count_are_strict(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    rows.write_bytes(b"{}\n{}\n")
    pin = pin_file(rows, label="rows", recorded_path="rows.jsonl", records=2)
    assert pin.bytes == 6
    assert pin.sha256 == hashlib.sha256(rows.read_bytes()).hexdigest()

    with pytest.raises(ValidationError):
        FilePin(label="rows", path="rows.jsonl", sha256="a" * 64, bytes="6")
