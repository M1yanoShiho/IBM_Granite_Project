from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import evidence_rag.cli.finalize_selector_r005 as finalize_cli
from evidence_rag.evaluation.selector_artifacts import (
    ArtifactApplicability,
    DeviceInfo,
    GitPin,
    ModelPin,
    RuntimeInfo,
    SelectorInputPins,
    pin_file,
)


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _external_pin(root: Path, category: str):
    path = _write(root / category / "input.json", f"{category}\n".encode())
    return path, pin_file(path, label=category, recorded_path=str(path.resolve()))


def _model(identity: str = "b") -> ModelPin:
    return ModelPin(
        name="cross-encoder/nli-deberta-v3-base",
        implementation_version="shared-deberta-independent-sigmoid-heads-v1",
        revision="frozen-revision",
        identity_sha256=identity * 64,
    )


def _checkpoint_fingerprint(weights_sha256: str = "d" * 64) -> dict[str, object]:
    return {
        "schema_version": "selector-dual-head-fingerprint-v1",
        "model_id": "cross-encoder/nli-deberta-v3-base",
        "revision": "frozen-revision",
        "max_length": 512,
        "weights_sha256": weights_sha256,
    }


def _runtime() -> RuntimeInfo:
    started = datetime(2026, 8, 12, 10, 0, tzinfo=UTC)
    return RuntimeInfo(
        host="test-host",
        python_version="3.11.14",
        command=("python", "-m", "evidence_rag.cli.run_selector_sanity"),
        started_at_utc=started,
        finished_at_utc=started + timedelta(seconds=90),
        wall_time_seconds=90.0,
        peak_device_memory_bytes=4_000_000,
        throughput_queries_per_second=10.0,
    )


def _device() -> DeviceInfo:
    return DeviceInfo(
        kind="cuda",
        name="test-gpu",
        precision="float32",
        runtime_version="12.8",
    )


def _inputs(tmp_path: Path) -> tuple[SelectorInputPins, Path]:
    external = tmp_path / "frozen-inputs"
    paths_and_pins = {
        category: _external_pin(external, category)
        for category in ("source", "label", "data", "pool", "component", "provenance")
    }
    inputs = SelectorInputPins(
        source=(paths_and_pins["source"][1],),
        label=(paths_and_pins["label"][1],),
        data=(paths_and_pins["data"][1],),
        pool=(paths_and_pins["pool"][1],),
        component=(paths_and_pins["component"][1],),
        provenance=(paths_and_pins["provenance"][1],),
    )
    return inputs, paths_and_pins["source"][0]


def _report(
    status: str,
    *,
    base_model_identity: str = "b" * 64,
    checkpoint_weights_sha256: str = "d" * 64,
) -> bytes:
    if status == "FAIL":
        training = "FAIL"
        safe_corner = "NOT_EVALUATED"
    else:
        training = "PASS"
        safe_corner = status
    return (
        json.dumps(
            {
                "status": status,
                "training_gate": {"status": training},
                "safe_corner": {"status": safe_corner},
                "model": {
                    "base_snapshot_identity_sha256": base_model_identity,
                    "final_checkpoint_weights_sha256": checkpoint_weights_sha256,
                },
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _build_run(
    tmp_path: Path,
    *,
    status: str = "PASS",
    downstream_nonempty: bool | None = None,
    report_payload: bytes | None = None,
    checkpoint_model: ModelPin | None = None,
    sanity_model: ModelPin | None = None,
) -> tuple[Path, Path]:
    root = tmp_path / "R005"
    model_for_checkpoint = checkpoint_model or _model()
    checkpoint_weights_sha256 = "d" * 64
    _write(root / finalize_cli.CONFIG_FILE, b"frozen-config\n")
    _write(root / finalize_cli.CHECKPOINT_FILE, b"safetensors-fixture\n")
    downstream_nonempty = status != "FAIL" if downstream_nonempty is None else downstream_nonempty
    downstream = {
        finalize_cli.DECISION_TRACE_FILE,
        finalize_cli.SELECTION_RESULTS_FILE,
        finalize_cli.SELECTED_SETS_FILE,
        finalize_cli.COUNT_MATCHED_CONTROLS_FILE,
    }
    for relative in set(finalize_cli.SANITY_OUTPUT_RELATIVE_FILES) - {
        finalize_cli.CONFIG_FILE,
        finalize_cli.CHECKPOINT_FILE,
        finalize_cli.CHECKPOINT_MANIFEST_FILE,
        finalize_cli.SANITY_REPORT_FILE,
    }:
        payload = b"{}\n"
        if relative in downstream and not downstream_nonempty:
            payload = b""
        _write(root / relative, payload)
    _write(
        root / finalize_cli.SANITY_REPORT_FILE,
        (
            report_payload
            if report_payload is not None
            else _report(
                status,
                base_model_identity=model_for_checkpoint.identity_sha256,
                checkpoint_weights_sha256=checkpoint_weights_sha256,
            )
        ),
    )

    checkpoint = finalize_cli.build_r005_checkpoint_manifest(
        root,
        status=status,  # type: ignore[arg-type]
        epochs_completed=30 if status != "FAIL" else 7,
        model=model_for_checkpoint,
        checkpoint_fingerprint=_checkpoint_fingerprint(checkpoint_weights_sha256),
    )
    finalize_cli.freeze_or_verify_r005_checkpoint_manifest(root, checkpoint)

    inputs, external_path = _inputs(tmp_path)
    sanity = finalize_cli.build_r005_sanity_manifest(
        root,
        status=status,  # type: ignore[arg-type]
        git=GitPin(branch="main", commit="a" * 40, dirty=False),
        model=sanity_model or model_for_checkpoint,
        inputs=inputs,
        device=_device(),
        runtime=_runtime(),
    )
    finalize_cli.freeze_or_verify_r005_sanity_manifest(root, sanity)
    return root, external_path


def _tree_snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_freeze_and_zero_write_verify_build_complete_r005_bundle(tmp_path: Path) -> None:
    root, _ = _build_run(tmp_path)

    manifest = finalize_cli.finalize_selector_r005(root)

    checksum_path = root / finalize_cli.CHECKSUMS_FILE
    manifest_path = root / finalize_cli.SELECTOR_EXPERIMENT_MANIFEST_FILE
    listed = {
        line.split("  ", 1)[1] for line in checksum_path.read_text(encoding="utf-8").splitlines()
    }
    assert listed == set(finalize_cli.PREREQUISITE_RELATIVE_FILES)
    assert len(listed) == 14
    assert finalize_cli.CHECKSUMS_FILE not in listed
    assert finalize_cli.SELECTOR_EXPERIMENT_MANIFEST_FILE not in listed
    assert manifest.run_id == "R005"
    assert manifest.stage.value == "dual-head-sanity"
    assert manifest.status.value == "PASS"
    assert manifest.artifacts.labels.state == ArtifactApplicability.NOT_APPLICABLE
    assert manifest.artifacts.preflight.state == ArtifactApplicability.NOT_APPLICABLE
    assert manifest.artifacts.crc.state == ArtifactApplicability.NOT_APPLICABLE
    for name in (
        "checkpoint",
        "candidate_scores",
        "decision_trace",
        "selected_sets",
        "count_matched",
        "metrics",
        "run_log",
        "checksums",
    ):
        assert getattr(manifest.artifacts, name).state == ArtifactApplicability.PRESENT
    assert manifest.config.path == "config.toml"
    assert all(Path(pin.path).is_absolute() for pin in manifest.inputs.all_pins())

    before = _tree_snapshot(root)
    verified = finalize_cli.finalize_selector_r005(root, verify_only=True)
    after = _tree_snapshot(root)
    assert verified == manifest
    assert after == before
    assert manifest_path.read_bytes() == before[finalize_cli.SELECTOR_EXPERIMENT_MANIFEST_FILE][0]


@pytest.mark.parametrize("status", ("PASS", "CUT", "FAIL"))
def test_all_completed_statuses_preserve_the_full_artifact_inventory(
    tmp_path: Path, status: str
) -> None:
    root, _ = _build_run(tmp_path, status=status)

    manifest = finalize_cli.finalize_selector_r005(root)

    assert manifest.status.value == status
    for name in (
        "checkpoint",
        "candidate_scores",
        "decision_trace",
        "selected_sets",
        "count_matched",
        "metrics",
        "run_log",
    ):
        assert getattr(manifest.artifacts, name).state == ArtifactApplicability.PRESENT
    if status == "FAIL":
        for relative in (
            finalize_cli.DECISION_TRACE_FILE,
            finalize_cli.SELECTION_RESULTS_FILE,
            finalize_cli.SELECTED_SETS_FILE,
            finalize_cli.COUNT_MATCHED_CONTROLS_FILE,
        ):
            assert (root / relative).read_bytes() == b""


def test_training_fail_must_stop_before_modelval(tmp_path: Path) -> None:
    root, _ = _build_run(tmp_path, status="FAIL", downstream_nonempty=True)

    with pytest.raises(ValueError, match="stop before modelval"):
        finalize_cli.finalize_selector_r005(root)


@pytest.mark.parametrize("status", ("PASS", "CUT"))
def test_pass_or_cut_requires_nonempty_modelval_evidence(tmp_path: Path, status: str) -> None:
    root, _ = _build_run(tmp_path, status=status, downstream_nonempty=False)

    with pytest.raises(ValueError, match="non-empty modelval evidence"):
        finalize_cli.finalize_selector_r005(root)


@pytest.mark.parametrize(
    "status,payload,message",
    (
        (
            "FAIL",
            {
                "status": "FAIL",
                "training_gate": {"status": "FAIL"},
                "safe_corner": {"status": "CUT"},
            },
            "NOT_EVALUATED",
        ),
        (
            "CUT",
            {
                "status": "CUT",
                "training_gate": {"status": "PASS"},
                "safe_corner": {"status": "PASS"},
            },
            "requires training_gate PASS",
        ),
    ),
)
def test_report_status_semantics_are_not_advisory(
    tmp_path: Path, status: str, payload: dict[str, object], message: str
) -> None:
    root, _ = _build_run(
        tmp_path,
        status=status,
        report_payload=(json.dumps(payload, sort_keys=True) + "\n").encode(),
    )

    with pytest.raises(ValueError, match=message):
        finalize_cli.finalize_selector_r005(root)


def test_checkpoint_metadata_must_match_sanity_model(tmp_path: Path) -> None:
    root, _ = _build_run(
        tmp_path,
        checkpoint_model=_model("b"),
        sanity_model=_model("c"),
    )

    with pytest.raises(ValueError, match="model metadata differ"):
        finalize_cli.finalize_selector_r005(root)


def test_checkpoint_fingerprint_must_identify_the_base_model(tmp_path: Path) -> None:
    root = tmp_path / "R005"
    _write(root / finalize_cli.CHECKPOINT_FILE, b"weights\n")
    fingerprint = _checkpoint_fingerprint()
    fingerprint["model_id"] = "wrong/model"

    with pytest.raises(ValueError, match="must identify the frozen base model"):
        finalize_cli.build_r005_checkpoint_manifest(
            root,
            status="FAIL",
            epochs_completed=0,
            model=_model(),
            checkpoint_fingerprint=fingerprint,
        )


def test_report_checkpoint_fingerprint_must_match_checkpoint_manifest(tmp_path: Path) -> None:
    root, _ = _build_run(
        tmp_path,
        report_payload=_report(
            "PASS",
            base_model_identity="b" * 64,
            checkpoint_weights_sha256="e" * 64,
        ),
    )

    with pytest.raises(ValueError, match="checkpoint fingerprint differs"):
        finalize_cli.finalize_selector_r005(root)


def test_cut_checkpoint_requires_the_complete_training_schedule(tmp_path: Path) -> None:
    root = tmp_path / "R005"
    _write(root / finalize_cli.CHECKPOINT_FILE, b"weights\n")

    with pytest.raises(ValueError, match="CUT requires all 30"):
        finalize_cli.build_r005_checkpoint_manifest(
            root,
            status="CUT",
            epochs_completed=29,
            model=_model(),
            checkpoint_fingerprint=_checkpoint_fingerprint(),
        )


@pytest.mark.parametrize("target", ("prerequisite", "checksums", "manifest", "external"))
def test_verify_rejects_every_layer_of_tampering(tmp_path: Path, target: str) -> None:
    root, external = _build_run(tmp_path)
    finalize_cli.finalize_selector_r005(root)
    paths = {
        "prerequisite": root / finalize_cli.CANDIDATE_SCORES_FILE,
        "checksums": root / finalize_cli.CHECKSUMS_FILE,
        "manifest": root / finalize_cli.SELECTOR_EXPERIMENT_MANIFEST_FILE,
        "external": external,
    }
    with paths[target].open("ab") as stream:
        stream.write(b"tampered\n")

    with pytest.raises(ValueError):
        finalize_cli.finalize_selector_r005(root, verify_only=True)


def test_top_level_and_runner_manifests_are_write_once(tmp_path: Path) -> None:
    root, _ = _build_run(tmp_path)
    sanity = finalize_cli._load_typed_manifest(
        root / finalize_cli.SANITY_MANIFEST_FILE,
        finalize_cli.R005SanityManifest,
        label="fixture sanity manifest",
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        finalize_cli.freeze_or_verify_r005_sanity_manifest(root, sanity)  # type: ignore[arg-type]
    finalize_cli.finalize_selector_r005(root)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        finalize_cli.finalize_selector_r005(root)


def test_failed_top_level_verification_rolls_back_both_publish_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _build_run(tmp_path)

    def fail_verification(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ValueError("injected final verification failure")

    monkeypatch.setattr(finalize_cli, "verify_selector_experiment_pins", fail_verification)
    with pytest.raises(ValueError, match="injected final verification failure"):
        finalize_cli.finalize_selector_r005(root)

    assert not (root / finalize_cli.CHECKSUMS_FILE).exists()
    assert not (root / finalize_cli.SELECTOR_EXPERIMENT_MANIFEST_FILE).exists()


@pytest.mark.parametrize("mutation", ("extra", "missing", "symlink"))
def test_exact_file_set_and_no_symlinks_are_enforced(tmp_path: Path, mutation: str) -> None:
    root, _ = _build_run(tmp_path)
    if mutation == "extra":
        _write(root / "sanity/unexpected.json", b"{}\n")
    elif mutation == "missing":
        (root / finalize_cli.TRAINING_TRACE_FILE).unlink()
    else:
        (root / "sanity/link.json").symlink_to(root / finalize_cli.SANITY_REPORT_FILE)

    with pytest.raises(ValueError, match="unexpected file set|must not contain symlinks"):
        finalize_cli.finalize_selector_r005(root)


def test_raw_heldout_input_path_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "heldout" / "source.json", b"source\n")
    pin = pin_file(path, label="source", recorded_path=str(path.resolve()))
    good, _ = _inputs(tmp_path / "good")
    bad = SelectorInputPins(
        source=(pin,),
        label=good.label,
        data=good.data,
        pool=good.pool,
        component=good.component,
        provenance=good.provenance,
    )

    with pytest.raises(ValueError, match="refuses sealed/heldout"):
        finalize_cli._verify_external_inputs(bad, run_root=tmp_path)


def test_neutral_symlink_resolving_to_sealed_input_is_rejected(tmp_path: Path) -> None:
    target = _write(tmp_path / "storage" / "sealed" / "source.json", b"source\n")
    neutral = tmp_path / "neutral" / "source.json"
    neutral.parent.mkdir(parents=True)
    neutral.symlink_to(target)
    pin = pin_file(neutral, label="source", recorded_path=str(neutral.absolute()))
    good, _ = _inputs(tmp_path / "good")
    bad = SelectorInputPins(
        source=(pin,),
        label=good.label,
        data=good.data,
        pool=good.pool,
        component=good.component,
        provenance=good.provenance,
    )

    with pytest.raises(ValueError, match="resolved sealed/heldout"):
        finalize_cli._verify_external_inputs(bad, run_root=tmp_path)


def test_noncanonical_runner_manifest_is_rejected_even_when_its_pin_matches(
    tmp_path: Path,
) -> None:
    root, _ = _build_run(tmp_path)
    checkpoint_path = root / finalize_cli.CHECKPOINT_MANIFEST_FILE
    parsed = json.loads(checkpoint_path.read_bytes())
    checkpoint_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    # Rebuild the downstream sanity manifest so its byte pin honestly matches the now-pretty
    # checkpoint manifest.  The finalizer must still reject the non-canonical inner manifest.
    (root / finalize_cli.SANITY_MANIFEST_FILE).unlink()
    inputs, _ = _inputs(tmp_path)
    sanity = finalize_cli.build_r005_sanity_manifest(
        root,
        status="PASS",
        git=GitPin(branch="main", commit="a" * 40, dirty=False),
        model=_model(),
        inputs=inputs,
        device=_device(),
        runtime=_runtime(),
    )
    finalize_cli.freeze_or_verify_r005_sanity_manifest(root, sanity)

    with pytest.raises(ValueError, match="not canonical JSON"):
        finalize_cli.finalize_selector_r005(root)
