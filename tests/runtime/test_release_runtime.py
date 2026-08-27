from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from evidence_rag.infrastructure.config import load_experiment_config

ROOT = Path(__file__).resolve().parents[2]
FINAL_CONFIG = ROOT / "configs/runtime/final_seed13.toml"
CPU_SMOKE_CONFIG = ROOT / "configs/runtime/cpu_smoke.toml"
MODEL_MANIFEST = ROOT / "configs/models/final_seed13.json"


def test_release_has_one_final_runtime_config_and_one_seed13_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "EVIDENCE_RAG_DATASET_MANIFEST",
        str(ROOT / "tests/fixtures/three_module_smoke_dataset/manifest.json"),
    )
    monkeypatch.setenv("EVIDENCE_RAG_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("EVIDENCE_RAG_MODEL_CACHE", str(tmp_path / "models"))
    monkeypatch.setenv(
        "EVIDENCE_RAG_SELECTOR_CHECKPOINT",
        str(tmp_path / "selector/model.safetensors"),
    )
    monkeypatch.setenv("EVIDENCE_RAG_GENERATOR_ADAPTER", str(tmp_path / "generator"))

    config = load_experiment_config(FINAL_CONFIG)
    manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))

    assert config.dataset_manifest_path == Path(
        os.environ["EVIDENCE_RAG_DATASET_MANIFEST"]
    ).resolve()
    assert config.output_directory == (tmp_path / "output").resolve()
    assert config.retriever.name == "hybrid"
    assert config.selector.name == "nli-risk-controlled"
    assert config.generator.name == "grounded-grc"
    assert manifest["runtime_config"] == "../runtime/final_seed13.toml"
    assert manifest["selector"]["checkpoint_sha256"] == (
        config.selector.parameters["checkpoint_sha256"]
    )
    assert manifest["selector"]["base_config_sha256"] == (
        config.selector.parameters["model_config_sha256"]
    )
    assert manifest["generator"]["adapter_weights_sha256"] == (
        config.generator.parameters["adapter_weights_sha256"]
    )
    dense = config.retriever.parameters["retrievers"][1]["parameters"]
    assert manifest["retriever"]["model_id"] == dense["embedder_model_id"]
    assert manifest["retriever"]["revision"] == dense["revision"]
    assert manifest["retriever"]["config_sha256"] == dense["model_config_sha256"]
    assert Path(str(dense["model_snapshot"])).name == manifest["retriever"]["revision"]
    assert manifest["selector"]["model_id"] == config.selector.parameters["model_id"]
    assert manifest["selector"]["revision"] == config.selector.parameters["revision"]
    assert Path(str(config.selector.parameters["model_snapshot"])).name == (
        manifest["selector"]["revision"]
    )
    assert Path(str(config.generator.parameters["model_snapshot"])).name == (
        manifest["generator"]["base_revision"]
    )
    assert manifest["generator"]["base_config_sha256"] == (
        config.generator.parameters["model_config_sha256"]
    )
    assert manifest["generator"]["adapter_config_sha256"] == (
        config.generator.parameters["adapter_config_sha256"]
    )
    assert Path(str(config.generator.parameters["true_snapshot"])).name == (
        manifest["generator"]["verifier_revision"]
    )
    assert manifest["generator"]["verifier_config_sha256"] == (
        config.generator.parameters["true_config_sha256"]
    )


def test_cpu_smoke_config_uses_final_module_classes_without_external_paths() -> None:
    config = load_experiment_config(CPU_SMOKE_CONFIG)

    assert config.retriever.name == "hybrid"
    assert config.selector.name == "nli-risk-controlled"
    assert config.generator.name == "grounded-grc"
    serialized = config.model_dump_json()
    assert "EVIDENCE_RAG_" not in serialized
    assert "/scratch" + "/" not in serialized
    assert "/user" + "/work/" not in serialized
