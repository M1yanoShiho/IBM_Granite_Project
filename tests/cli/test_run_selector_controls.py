import json
from pathlib import Path

import pytest

import evidence_rag.cli.run_selector_controls as controls_cli
from evidence_rag.evaluation.selector_controls import (
    COUNT_MATCHED_MANIFEST_FILE,
    COUNT_MATCHED_PROTOCOL_FILE,
    TOPK_MANIFEST_FILE,
    TOPK_REPORT_FILE,
    CountMatchedProtocolArtifacts,
    TopKControlArtifacts,
)


def _topk_artifacts() -> TopKControlArtifacts:
    return TopKControlArtifacts(
        files={},
        report={"counts": {"eligible_input_queries": 2, "trace_rows": 8}},
        manifest={"dataset_signature": "a" * 64},
    )


def _protocol_artifacts() -> CountMatchedProtocolArtifacts:
    return CountMatchedProtocolArtifacts(
        files={},
        protocol={
            "status": "GENERATOR_PROTOCOL_ONLY_NO_SELECTOR_TRACE",
            "repeat_count": 100,
        },
        manifest={},
    )


def _write_topk_outputs(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / TOPK_MANIFEST_FILE).write_text("{}\n", encoding="utf-8")
    (output / TOPK_REPORT_FILE).write_text("{}\n", encoding="utf-8")


def _write_protocol_outputs(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / COUNT_MATCHED_MANIFEST_FILE).write_text("{}\n", encoding="utf-8")
    (output / COUNT_MATCHED_PROTOCOL_FILE).write_text("{}\n", encoding="utf-8")


def _required_dirs(tmp_path: Path) -> tuple[Path, Path]:
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "selector_candidate_pool_manifest_v2.json").write_text("{}\n", encoding="utf-8")
    components = tmp_path / "components"
    components.mkdir()
    (components / "selector_components_manifest.json").write_text("{}\n", encoding="utf-8")
    return pool, components


def test_topk_cli_requires_verified_pool_and_r002_then_freezes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifacts = _topk_artifacts()
    calls: dict[str, object] = {}
    pool, components = _required_dirs(tmp_path)

    def build(**kwargs: object) -> TopKControlArtifacts:
        calls["build"] = kwargs
        return artifacts

    def freeze(output: Path, found: TopKControlArtifacts) -> Path:
        calls["freeze"] = (output, found)
        _write_topk_outputs(output)
        return output / TOPK_MANIFEST_FILE

    monkeypatch.setattr(controls_cli, "build_topk_control_artifacts", build)
    monkeypatch.setattr(controls_cli, "freeze_topk_control_artifacts", freeze)
    output = tmp_path / "output"
    assert (
        controls_cli.main(
            [
                "topk",
                "--dataset-kind",
                "niah",
                "--source-split",
                "dev",
                "--dataset-manifest",
                str(tmp_path / "manifest.json"),
                "--source-parent",
                str(tmp_path / "parents.jsonl"),
                "--candidate-pool",
                str(pool),
                "--components-dir",
                str(components),
                "--niah-assignment",
                str(tmp_path / "assignments.jsonl"),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert calls["build"] == {
        "dataset_kind": "niah",
        "source_split": "dev",
        "dataset_manifest_path": tmp_path / "manifest.json",
        "source_parent_path": tmp_path / "parents.jsonl",
        "candidate_pool_path": pool,
        "component_directory": components,
        "assignment_path": tmp_path / "assignments.jsonl",
    }
    assert calls["freeze"] == (output, artifacts)
    result = json.loads(capsys.readouterr().out)
    assert result["action"] == "frozen"
    assert result["kind"] == "topk-controls"
    assert len(result["manifest_sha256"]) == 64


def test_topk_cli_verify_only_never_builds_or_freezes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifacts = _topk_artifacts()
    pool, components = _required_dirs(tmp_path)
    output = tmp_path / "output"
    _write_topk_outputs(output)
    calls: dict[str, object] = {}

    def verify(**kwargs: object) -> TopKControlArtifacts:
        calls["verify"] = kwargs
        return artifacts

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("verify-only must not build or freeze")

    monkeypatch.setattr(controls_cli, "verify_topk_control_artifacts", verify)
    monkeypatch.setattr(controls_cli, "build_topk_control_artifacts", forbidden)
    monkeypatch.setattr(controls_cli, "freeze_topk_control_artifacts", forbidden)
    assert (
        controls_cli.main(
            [
                "topk",
                "--dataset-kind",
                "2wiki",
                "--source-split",
                "train",
                "--dataset-manifest",
                str(tmp_path / "manifest.json"),
                "--source-parent",
                str(tmp_path / "parents.jsonl"),
                "--candidate-pool",
                str(pool),
                "--components-dir",
                str(components),
                "--output-dir",
                str(output),
                "--verify-only",
            ]
        )
        == 0
    )
    assert calls["verify"] == {
        "output_directory": output,
        "dataset_kind": "2wiki",
        "source_split": "train",
        "dataset_manifest_path": tmp_path / "manifest.json",
        "source_parent_path": tmp_path / "parents.jsonl",
        "candidate_pool_path": pool,
        "component_directory": components,
        "assignment_path": None,
    }
    assert json.loads(capsys.readouterr().out)["action"] == "verified"


def test_topk_cli_rejects_bare_pool_missing_r002_and_wrong_assignment_contract(
    tmp_path: Path,
) -> None:
    common = [
        "topk",
        "--source-split",
        "dev",
        "--dataset-manifest",
        str(tmp_path / "manifest.json"),
        "--source-parent",
        str(tmp_path / "parents.jsonl"),
        "--candidate-pool",
        str(tmp_path / "candidate_sets.jsonl"),
        "--components-dir",
        str(tmp_path / "components"),
        "--output-dir",
        str(tmp_path / "output"),
    ]
    with pytest.raises(SystemExit):
        controls_cli.main([*common, "--dataset-kind", "niah"])
    with pytest.raises(SystemExit):
        controls_cli.main(
            [
                *common,
                "--dataset-kind",
                "2wiki",
                "--niah-assignment",
                str(tmp_path / "assignments.jsonl"),
            ]
        )


def test_protocol_cli_freezes_and_verifies_without_trace_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifacts = _protocol_artifacts()
    calls: list[tuple[str, Path]] = []

    def build() -> CountMatchedProtocolArtifacts:
        return artifacts

    def freeze(output: Path, found: CountMatchedProtocolArtifacts) -> Path:
        assert found is artifacts
        calls.append(("freeze", output))
        _write_protocol_outputs(output)
        return output / COUNT_MATCHED_MANIFEST_FILE

    monkeypatch.setattr(controls_cli, "build_count_matched_protocol_artifacts", build)
    monkeypatch.setattr(controls_cli, "freeze_count_matched_protocol_artifacts", freeze)
    output = tmp_path / "protocol"
    assert controls_cli.main(["freeze-protocol", "--output-dir", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["repeat_count"] == 100
    assert calls == [("freeze", output)]

    def verify(found_output: Path) -> CountMatchedProtocolArtifacts:
        calls.append(("verify", found_output))
        return artifacts

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("protocol verify-only must not build or freeze")

    monkeypatch.setattr(controls_cli, "verify_count_matched_protocol_artifacts", verify)
    monkeypatch.setattr(controls_cli, "build_count_matched_protocol_artifacts", forbidden)
    monkeypatch.setattr(controls_cli, "freeze_count_matched_protocol_artifacts", forbidden)
    assert controls_cli.main(["freeze-protocol", "--output-dir", str(output), "--verify-only"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "verified"
    assert calls[-1] == ("verify", output)
