import json
from pathlib import Path

import pytest

import evidence_rag.cli.build_selector_labels as labels_cli
from evidence_rag.materializer.selector_labels import SelectorLabelArtifacts


def _artifacts() -> SelectorLabelArtifacts:
    return SelectorLabelArtifacts(
        files={},
        report={
            "dataset_kind": "niah",
            "status": "PASS",
            "counts": {"queries": 2, "candidate_rows": 40},
        },
        manifest={},
    )


def _required_dirs(tmp_path: Path) -> tuple[Path, Path]:
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "selector_candidate_pool_manifest_v2.json").write_text("{}\n", encoding="utf-8")
    components = tmp_path / "components"
    components.mkdir()
    (components / "selector_components_manifest.json").write_text("{}\n", encoding="utf-8")
    return pool, components


def _write_outputs(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "selector_labels_report.json").write_text("{}\n", encoding="utf-8")
    (output / "selector_labels_manifest.json").write_text("{}\n", encoding="utf-8")


def test_niah_cli_freezes_with_both_privileged_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pool, components = _required_dirs(tmp_path)
    artifacts = _artifacts()
    seen: dict[str, object] = {}

    def build(**kwargs: object) -> SelectorLabelArtifacts:
        seen["build"] = kwargs
        return artifacts

    def freeze(output: Path, found: SelectorLabelArtifacts) -> Path:
        seen["freeze"] = (output, found)
        _write_outputs(output)
        return output / "selector_labels_manifest.json"

    monkeypatch.setattr(labels_cli, "build_selector_label_artifacts", build)
    monkeypatch.setattr(labels_cli, "freeze_selector_label_artifacts", freeze)
    output = tmp_path / "output"
    assignment = tmp_path / "assignment.jsonl"
    provenance = tmp_path / "provenance.jsonl"
    assert (
        labels_cli.main(
            [
                "--dataset-kind",
                "niah",
                "--dataset-manifest",
                str(tmp_path / "manifest.json"),
                "--source-parent",
                str(tmp_path / "parents.jsonl"),
                "--candidate-pool",
                str(pool),
                "--components-dir",
                str(components),
                "--niah-assignment",
                str(assignment),
                "--niah-provenance",
                str(provenance),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert seen["build"] == {
        "dataset_kind": "niah",
        "dataset_manifest_path": tmp_path / "manifest.json",
        "source_parent_path": tmp_path / "parents.jsonl",
        "candidate_pool_path": pool,
        "component_directory": components,
        "assignment_path": assignment,
        "provenance_path": provenance,
    }
    assert seen["freeze"] == (output, artifacts)
    result = json.loads(capsys.readouterr().out)
    assert result["action"] == "frozen"
    assert result["status"] == "PASS"


def test_2wiki_verify_only_never_builds_or_freezes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pool, components = _required_dirs(tmp_path)
    output = tmp_path / "output"
    _write_outputs(output)
    artifacts = _artifacts()
    seen: dict[str, object] = {}

    def verify(**kwargs: object) -> SelectorLabelArtifacts:
        seen["verify"] = kwargs
        return artifacts

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("verify-only must not build or freeze")

    monkeypatch.setattr(labels_cli, "verify_selector_label_artifacts", verify)
    monkeypatch.setattr(labels_cli, "build_selector_label_artifacts", forbidden)
    monkeypatch.setattr(labels_cli, "freeze_selector_label_artifacts", forbidden)
    assert (
        labels_cli.main(
            [
                "--dataset-kind",
                "2wiki",
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
    assert seen["verify"] == {
        "output_directory": output,
        "dataset_kind": "2wiki",
        "dataset_manifest_path": tmp_path / "manifest.json",
        "source_parent_path": tmp_path / "parents.jsonl",
        "candidate_pool_path": pool,
        "component_directory": components,
        "assignment_path": None,
        "provenance_path": None,
    }
    assert json.loads(capsys.readouterr().out)["action"] == "verified"


def test_cli_rejects_missing_pool_components_and_wrong_sidecar_contract(tmp_path: Path) -> None:
    common = [
        "--dataset-manifest",
        str(tmp_path / "manifest.json"),
        "--source-parent",
        str(tmp_path / "parents.jsonl"),
        "--candidate-pool",
        str(tmp_path / "pool"),
        "--components-dir",
        str(tmp_path / "components"),
        "--output-dir",
        str(tmp_path / "output"),
    ]
    with pytest.raises(SystemExit):
        labels_cli.main(["--dataset-kind", "niah", *common])

    pool, components = _required_dirs(tmp_path)
    valid_common = [
        "--dataset-manifest",
        str(tmp_path / "manifest.json"),
        "--source-parent",
        str(tmp_path / "parents.jsonl"),
        "--candidate-pool",
        str(pool),
        "--components-dir",
        str(components),
        "--output-dir",
        str(tmp_path / "output"),
    ]
    with pytest.raises(SystemExit):
        labels_cli.main(["--dataset-kind", "niah", *valid_common])
    with pytest.raises(SystemExit):
        labels_cli.main(
            [
                "--dataset-kind",
                "2wiki",
                *valid_common,
                "--niah-assignment",
                str(tmp_path / "a"),
                "--niah-provenance",
                str(tmp_path / "p"),
            ]
        )
