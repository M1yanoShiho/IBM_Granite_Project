import json
from pathlib import Path

import pytest

import evidence_rag.cli.build_selector_components as component_cli
from evidence_rag.evaluation.selector_components import (
    MANIFEST_FILE,
    REPORT_FILE,
    SelectorComponentArtifacts,
)


def _artifacts() -> SelectorComponentArtifacts:
    report: dict[str, object] = {
        "counts": {"eligible_input_queries": 2, "components": 2},
        "crossing": {"components_across_folds": 0, "components_across_roles": 0},
    }
    manifest: dict[str, object] = {"dataset_signature": "a" * 64}
    return SelectorComponentArtifacts(files={}, report=report, manifest=manifest)


def _write_cli_outputs(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / MANIFEST_FILE).write_text("{}\n", encoding="utf-8")
    (output / REPORT_FILE).write_text("{}\n", encoding="utf-8")


def test_cli_builds_and_freezes_explicit_dataset_and_source_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, object] = {}
    artifacts = _artifacts()

    def build(**kwargs: object) -> SelectorComponentArtifacts:
        calls["build"] = kwargs
        return artifacts

    def freeze(output: Path, found: SelectorComponentArtifacts) -> Path:
        calls["freeze"] = (output, found)
        _write_cli_outputs(output)
        return output / MANIFEST_FILE

    monkeypatch.setattr(component_cli, "build_selector_component_artifacts", build)
    monkeypatch.setattr(component_cli, "freeze_selector_component_artifacts", freeze)
    output = tmp_path / "output"
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "selector_candidate_pool_manifest_v2.json").write_text("{}\n", encoding="utf-8")

    assert (
        component_cli.main(
            [
                "--dataset-kind",
                "2wiki",
                "--source-split",
                "dev",
                "--dataset-manifest",
                str(tmp_path / "manifest.json"),
                "--source-parent",
                str(tmp_path / "parents.jsonl"),
                "--candidate-pool",
                str(pool),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    assert calls["build"] == {
        "dataset_kind": "2wiki",
        "source_split": "dev",
        "dataset_manifest_path": tmp_path / "manifest.json",
        "source_parent_path": tmp_path / "parents.jsonl",
        "candidate_pool_path": pool,
        "assignment_path": None,
    }
    assert calls["freeze"] == (output, artifacts)
    result = json.loads(capsys.readouterr().out)
    assert result["action"] == "frozen"
    assert result["dataset_signature"] == "a" * 64
    assert len(result["manifest_sha256"]) == 64


def test_cli_verify_only_recomputes_without_building_or_freezing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifacts = _artifacts()
    calls: dict[str, object] = {}
    output = tmp_path / "output"
    _write_cli_outputs(output)
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "selector_candidate_pool_manifest_v2.json").write_text("{}\n", encoding="utf-8")

    def verify(**kwargs: object) -> SelectorComponentArtifacts:
        calls["verify"] = kwargs
        return artifacts

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("verify-only must not build or freeze")

    monkeypatch.setattr(component_cli, "verify_selector_component_artifacts", verify)
    monkeypatch.setattr(component_cli, "build_selector_component_artifacts", forbidden)
    monkeypatch.setattr(component_cli, "freeze_selector_component_artifacts", forbidden)

    assert (
        component_cli.main(
            [
                "--dataset-kind",
                "niah",
                "--source-split",
                "train",
                "--dataset-manifest",
                str(tmp_path / "manifest.json"),
                "--source-parent",
                str(tmp_path / "parents.jsonl"),
                "--candidate-pool",
                str(pool),
                "--niah-assignment",
                str(tmp_path / "assignments.jsonl"),
                "--output-dir",
                str(output),
                "--verify-only",
            ]
        )
        == 0
    )

    assert calls["verify"] == {
        "output_directory": output,
        "dataset_kind": "niah",
        "source_split": "train",
        "dataset_manifest_path": tmp_path / "manifest.json",
        "source_parent_path": tmp_path / "parents.jsonl",
        "candidate_pool_path": pool,
        "assignment_path": tmp_path / "assignments.jsonl",
    }
    assert json.loads(capsys.readouterr().out)["action"] == "verified"


def test_cli_enforces_dataset_specific_assignment_contract(tmp_path: Path) -> None:
    common = [
        "--source-split",
        "dev",
        "--dataset-manifest",
        str(tmp_path / "manifest.json"),
        "--source-parent",
        str(tmp_path / "parents.jsonl"),
        "--candidate-pool",
        str(tmp_path / "pool"),
        "--output-dir",
        str(tmp_path / "output"),
    ]
    with pytest.raises(SystemExit):
        component_cli.main(["--dataset-kind", "niah", *common])
    with pytest.raises(SystemExit):
        component_cli.main(
            [
                "--dataset-kind",
                "2wiki",
                *common,
                "--niah-assignment",
                str(tmp_path / "assignments.jsonl"),
            ]
        )


def test_cli_rejects_bare_candidate_jsonl_even_though_pure_build_allows_toy_inputs(
    tmp_path: Path,
) -> None:
    candidate_file = tmp_path / "candidate_sets.jsonl"
    candidate_file.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        component_cli.main(
            [
                "--dataset-kind",
                "2wiki",
                "--source-split",
                "dev",
                "--dataset-manifest",
                str(tmp_path / "manifest.json"),
                "--source-parent",
                str(tmp_path / "parents.jsonl"),
                "--candidate-pool",
                str(candidate_file),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
