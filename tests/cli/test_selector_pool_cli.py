from pathlib import Path
from types import SimpleNamespace

import pytest

import evidence_rag.cli.pin_selector_pool as selector_pool_cli
from evidence_rag.contracts.models import RetrieverProvenance


def a_manifest() -> SimpleNamespace:
    return SimpleNamespace(
        data_role="niah-dev",
        recovery_status="new-v2-run",
        n_queries=2,
        top_n=20,
        candidate_sha256="a" * 64,
        retriever=RetrieverProvenance(
            name="hybrid",
            implementation_version="hybrid-v1",
            parameters_sha256=(
                "67333c6f3fcf6567756b382048961bc150da0ee26af4f9cecc59f58f30b5d78d"
            ),
        ),
    )


def test_cli_builds_and_freezes_with_explicit_role_and_recovery_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: dict[str, object] = {}
    manifest = a_manifest()

    def build(
        pool: Path,
        dataset: Path,
        *,
        data_role: str,
        recovery_status: str,
    ) -> SimpleNamespace:
        calls["build"] = (pool, dataset, data_role, recovery_status)
        return manifest

    def freeze(pool: Path, found: object) -> Path:
        calls["freeze"] = (pool, found)
        return pool / "selector_candidate_pool_manifest_v2.json"

    monkeypatch.setattr(selector_pool_cli, "build_selector_candidate_pool_manifest_v2", build)
    monkeypatch.setattr(selector_pool_cli, "freeze_selector_candidate_pool_manifest_v2", freeze)
    pool = tmp_path / "pool"
    dataset = tmp_path / "dataset" / "manifest.json"

    assert (
        selector_pool_cli.main(
            [
                "--pool-dir",
                str(pool),
                "--dataset-manifest",
                str(dataset),
                "--data-role",
                "niah-dev",
                "--recovery-status",
                "new-v2-run",
            ]
        )
        == 0
    )

    assert calls["build"] == (pool, dataset, "niah-dev", "new-v2-run")
    assert calls["freeze"] == (pool, manifest)
    assert '"top_n": 20' in capsys.readouterr().out


def test_cli_verify_only_rechecks_without_building_or_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = a_manifest()
    calls: dict[str, object] = {}

    def verify(pool: Path, dataset: Path) -> SimpleNamespace:
        calls["verify"] = (pool, dataset)
        return manifest

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("verify-only must not build or freeze")

    monkeypatch.setattr(selector_pool_cli, "verify_selector_candidate_pool_v2", verify)
    monkeypatch.setattr(selector_pool_cli, "build_selector_candidate_pool_manifest_v2", forbidden)
    monkeypatch.setattr(selector_pool_cli, "freeze_selector_candidate_pool_manifest_v2", forbidden)
    pool = tmp_path / "pool"
    dataset = tmp_path / "dataset" / "manifest.json"

    assert (
        selector_pool_cli.main(
            [
                "--pool-dir",
                str(pool),
                "--dataset-manifest",
                str(dataset),
                "--verify-only",
            ]
        )
        == 0
    )

    assert calls["verify"] == (pool, dataset)
    assert '"verified"' in capsys.readouterr().out


def test_cli_requires_role_and_status_when_not_verifying(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        selector_pool_cli.main(
            [
                "--pool-dir",
                str(tmp_path / "pool"),
                "--dataset-manifest",
                str(tmp_path / "manifest.json"),
            ]
        )
