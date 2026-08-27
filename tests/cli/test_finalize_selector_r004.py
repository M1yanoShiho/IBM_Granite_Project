from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import evidence_rag.cli.finalize_selector_r004 as finalize_cli
from evidence_rag.evaluation.selector_artifacts import (
    ArtifactApplicability,
    DeviceInfo,
    GitPin,
    ModelPin,
    RuntimeInfo,
    SelectorInputPins,
    pin_file,
)


@dataclass(frozen=True)
class _RunFixture:
    root: Path
    context: finalize_cli._FinalizationContext
    external_pin_path: Path


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _external_pin(root: Path, label: str):
    path = _write(root / "external" / f"{label}.json", f"{label}\n".encode())
    return path, pin_file(path, label=label, recorded_path=str(path.resolve()))


def _preflight_pin(path: Path, *, recorded_path: str | None = None) -> dict[str, object]:
    pin = pin_file(
        path,
        label="fixture",
        recorded_path=recorded_path if recorded_path is not None else str(path.resolve()),
    )
    return {"path": pin.path, "bytes": pin.bytes, "sha256": pin.sha256}


@pytest.fixture
def run_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _RunFixture:
    root = tmp_path / "R004"
    for relative in finalize_cli._prerequisite_relative_files():
        payload = b"frozen-config\n" if relative == finalize_cli.CONFIG_FILE else b"{}\n"
        _write(root / relative, payload)

    labels = tuple(
        finalize_cli._run_pin(root, f"labels/{kind}/{filename}")
        for kind in ("niah", "2wiki")
        for filename in finalize_cli.LABEL_OUTPUT_FILES
    )
    preflight = tuple(
        finalize_cli._run_pin(root, f"preflight/{filename}")
        for filename in finalize_cli.PREFLIGHT_OUTPUT_FILES
        if filename != finalize_cli.PREFLIGHT_RUNTIME_LOG_FILE
    )
    run_log = finalize_cli._run_pin(root, f"preflight/{finalize_cli.PREFLIGHT_RUNTIME_LOG_FILE}")
    external_root = tmp_path / "frozen-inputs"
    source_path, source = _external_pin(external_root, "source")
    _, data = _external_pin(external_root, "data")
    _, pool = _external_pin(external_root, "pool")
    _, component = _external_pin(external_root, "component")
    _, provenance = _external_pin(external_root, "provenance")
    _, model_snapshot = _external_pin(external_root, "model-snapshot")
    started = datetime(2026, 8, 12, 10, 0, tzinfo=UTC)
    context = finalize_cli._FinalizationContext(
        config=finalize_cli._run_pin(root, finalize_cli.CONFIG_FILE),
        inputs=SelectorInputPins(
            source=(source,),
            label=labels,
            data=(data,),
            pool=(pool,),
            component=(component,),
            provenance=(provenance,),
            other=(model_snapshot,),
        ),
        labels=labels,
        preflight=preflight,
        run_log=run_log,
        git=GitPin(branch="main", commit="a" * 40, dirty=False),
        model=ModelPin(
            name="cross-encoder/nli-deberta-v3-base",
            implementation_version="shared-deberta-independent-sigmoid-heads-v1",
            revision="frozen-revision",
            identity_sha256="b" * 64,
        ),
        device=DeviceInfo(
            kind="cuda",
            name="test-gpu",
            precision="float32",
            runtime_version="12.8",
        ),
        runtime=RuntimeInfo(
            host="test-host",
            python_version="3.11.14",
            command=("python", "-m", "evidence_rag.cli.run_selector_preflight"),
            started_at_utc=started,
            finished_at_utc=started + timedelta(seconds=60),
            wall_time_seconds=60.0,
            peak_device_memory_bytes=2_000_000,
            throughput_queries_per_second=20.0,
        ),
    )
    monkeypatch.setattr(finalize_cli, "_load_verified_context", lambda _: context)
    return _RunFixture(root=root, context=context, external_pin_path=source_path)


def _tree_snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_freeze_and_zero_write_verify_build_the_r004_v2_bundle(
    run_fixture: _RunFixture,
) -> None:
    manifest = finalize_cli.finalize_selector_r004(run_fixture.root)

    checksum_path = run_fixture.root / finalize_cli.CHECKSUMS_FILE
    manifest_path = run_fixture.root / finalize_cli.SELECTOR_EXPERIMENT_MANIFEST_FILE
    assert checksum_path.is_file()
    assert manifest_path.is_file()
    listed = {
        line.split("  ", 1)[1] for line in checksum_path.read_text(encoding="utf-8").splitlines()
    }
    assert listed == set(finalize_cli._prerequisite_relative_files())
    assert finalize_cli.CHECKSUMS_FILE not in listed
    assert finalize_cli.SELECTOR_EXPERIMENT_MANIFEST_FILE not in listed
    assert manifest.artifacts.labels.state == ArtifactApplicability.PRESENT
    assert manifest.artifacts.preflight.state == ArtifactApplicability.PRESENT
    assert manifest.artifacts.run_log.state == ArtifactApplicability.PRESENT
    assert manifest.artifacts.checksums.state == ArtifactApplicability.PRESENT
    assert manifest.artifacts.checkpoint.state == ArtifactApplicability.NOT_APPLICABLE
    assert manifest.artifacts.crc.state == ArtifactApplicability.NOT_APPLICABLE
    assert manifest.artifacts.decision_trace.state == ArtifactApplicability.NOT_APPLICABLE
    assert manifest.artifacts.selected_sets.state == ArtifactApplicability.NOT_APPLICABLE
    assert manifest.artifacts.count_matched.state == ArtifactApplicability.DEFERRED
    assert all(not Path(pin.path).is_absolute() for pin in manifest.inputs.label)
    assert all(Path(pin.path).is_absolute() for pin in manifest.inputs.source)

    before = _tree_snapshot(run_fixture.root)
    verified = finalize_cli.finalize_selector_r004(run_fixture.root, verify_only=True)
    after = _tree_snapshot(run_fixture.root)
    assert verified == manifest
    assert after == before


def test_input_classification_binds_local_config_copy_to_preflight_config(
    tmp_path: Path,
) -> None:
    root = tmp_path / "staging"
    local_config = _write(root / finalize_cli.CONFIG_FILE, b"same-config\n")
    external = tmp_path / "external"
    upstream_config = _write(external / "config.toml", local_config.read_bytes())
    label = _write(root / "labels/niah/selector_labels.jsonl", b"{}\n")
    sources = {
        "config": _preflight_pin(upstream_config),
        "niah/source_parent": _preflight_pin(_write(external / "source_parent.jsonl", b"{}\n")),
        "niah/labels/selector_labels.jsonl": _preflight_pin(
            label, recorded_path="labels/niah/selector_labels.jsonl"
        ),
        "niah/dataset_manifest": _preflight_pin(_write(external / "manifest.json", b"{}\n")),
        "niah/candidate_pool": _preflight_pin(_write(external / "candidate_sets.jsonl", b"{}\n")),
        "niah/components/component_map.jsonl": _preflight_pin(
            _write(external / "component_map.jsonl", b"{}\n")
        ),
        "niah/provenance": _preflight_pin(_write(external / "provenance.jsonl", b"{}\n")),
        "model_snapshot/model.safetensors": _preflight_pin(
            _write(external / "model.safetensors", b"weights\n")
        ),
    }
    config, inputs = finalize_cli._classified_inputs(root, sources)
    assert config.path == finalize_cli.CONFIG_FILE
    assert config.sha256 == sources["config"]["sha256"]
    assert inputs.label[0].path == "labels/niah/selector_labels.jsonl"
    assert Path(inputs.source[0].path).is_absolute()

    local_config.write_bytes(b"drifted-config\n")
    with pytest.raises(ValueError, match="differs from the Git-tracked config"):
        finalize_cli._classified_inputs(root, sources)


def test_finalized_bundle_survives_atomic_run_root_rename(
    run_fixture: _RunFixture,
) -> None:
    manifest = finalize_cli.finalize_selector_r004(run_fixture.root)
    final_root = run_fixture.root.parent / "formal-R004"
    run_fixture.root.rename(final_root)

    assert finalize_cli.finalize_selector_r004(final_root, verify_only=True) == manifest


@pytest.mark.parametrize(
    "target",
    ("prerequisite", "checksums", "manifest", "external-pin"),
)
def test_verify_rejects_tampering(run_fixture: _RunFixture, target: str) -> None:
    finalize_cli.finalize_selector_r004(run_fixture.root)
    paths = {
        "prerequisite": run_fixture.root / "preflight/preflight_sample.jsonl",
        "checksums": run_fixture.root / finalize_cli.CHECKSUMS_FILE,
        "manifest": run_fixture.root / finalize_cli.SELECTOR_EXPERIMENT_MANIFEST_FILE,
        "external-pin": run_fixture.external_pin_path,
    }
    with paths[target].open("ab") as stream:
        stream.write(b"tampered\n")

    with pytest.raises(ValueError):
        finalize_cli.finalize_selector_r004(run_fixture.root, verify_only=True)


def test_freeze_is_write_once(run_fixture: _RunFixture) -> None:
    finalize_cli.finalize_selector_r004(run_fixture.root)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        finalize_cli.finalize_selector_r004(run_fixture.root)


@pytest.mark.parametrize("mutation", ("extra", "missing"))
def test_freeze_rejects_extra_or_missing_prerequisite_file(
    run_fixture: _RunFixture, mutation: str
) -> None:
    if mutation == "extra":
        _write(run_fixture.root / "labels/niah/unexpected.json", b"{}\n")
    else:
        (run_fixture.root / "labels/niah/selector_labels_report.json").unlink()

    with pytest.raises(ValueError, match="unexpected file set"):
        finalize_cli.finalize_selector_r004(run_fixture.root)
