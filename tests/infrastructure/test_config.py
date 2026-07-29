import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from evidence_rag.infrastructure.config import (
    ModuleConfig,
    load_experiment_config,
    load_ingestion_config,
)


def write_config(root: Path, extra: str = "") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "experiment.toml"
    path.write_text(
        """
[dataset]
manifest = "data/manifest.json"

[output]
directory = "artifacts/run-1"

[retriever]
name = "bm25"
parameters = { k1 = 1.5 }

[selector]
name = "top-k"

[generator]
name = "extractive"

[run]
top_k = 3
max_selected = 2
seed = 7
""".lstrip()
        + extra,
        encoding="utf-8",
    )
    return path


def test_config_paths_are_relative_to_config_file_not_cwd(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "project" / "config")
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        config = load_experiment_config(config_path)
    finally:
        os.chdir(original_cwd)

    assert config.dataset_manifest_path == (config_path.parent / "data/manifest.json").resolve()
    assert config.output_directory == (config_path.parent / "artifacts/run-1").resolve()


def test_unknown_toml_fields_are_rejected(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, "\n[retriever.extra]\nenabled = true\n")

    with pytest.raises(ValidationError, match="extra"):
        load_experiment_config(config_path)


def _write_ingestion(root: Path, body: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "ingestion.toml"
    path.write_text(body.lstrip(), encoding="utf-8")
    return path


def test_ingestion_config_reads_switches_and_resolves_cache_dir(tmp_path: Path) -> None:
    config_path = _write_ingestion(
        tmp_path / "cfg",
        """
[ingestion]
pdf_mode = "pages"
caption_pdf_pictures = true
image_ocr = false
caption_prompt = "Describe the chart."
cache_dir = "../.cache/ingest"
""",
    )

    config = load_ingestion_config(config_path)

    assert config.pdf_mode == "pages"
    assert config.caption_pdf_pictures is True
    assert config.image_ocr is False
    assert config.caption_prompt == "Describe the chart."
    assert config.cache_dir == (config_path.parent / "../.cache/ingest").resolve()


def test_ingestion_config_missing_section_uses_defaults(tmp_path: Path) -> None:
    config_path = _write_ingestion(tmp_path, "# no ingestion table here\n")

    config = load_ingestion_config(config_path)

    assert config.pdf_mode == "chunks"
    assert config.caption_pdf_pictures is False
    assert config.image_ocr is True
    assert config.cache_dir is None


def test_ingestion_config_rejects_unknown_keys(tmp_path: Path) -> None:
    config_path = _write_ingestion(tmp_path, "[ingestion]\ncaption_pdf_picture = true\n")

    with pytest.raises(ValidationError, match="extra"):
        load_ingestion_config(config_path)


@pytest.mark.parametrize("parameters", ({"nested": {"bad": object()}}, {"values": ("tuple",)}))
def test_module_parameters_must_be_json_compatible(parameters: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="JSON-compatible"):
        ModuleConfig(name="module", parameters=parameters)


@pytest.mark.parametrize(
    ("field", "value"),
    (("top_k", 0), ("max_selected", -1)),
)
def test_nonpositive_run_limits_are_rejected(tmp_path: Path, field: str, value: int) -> None:
    config_path = write_config(tmp_path)
    source = config_path.read_text(encoding="utf-8")
    config_path.write_text(source.replace(f"{field} = {3 if field == 'top_k' else 2}", f"{field} = {value}"), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_experiment_config(config_path)
