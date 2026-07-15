import json
from pathlib import Path

from evidence_rag.cli.experiment import main

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_MANIFEST = ROOT / "tests/fixtures/reference_dataset/manifest.json"


def write_config(root: Path) -> Path:
    config_path = root / "experiment.toml"
    config_path.write_text(
        f"""
[dataset]
manifest = "{REFERENCE_MANIFEST}"
[output]
directory = "run"
[retriever]
name = "bm25"
[selector]
name = "top-k"
[generator]
name = "extractive"
[run]
top_k = 3
max_selected = 2
seed = 7
""".lstrip(),
        encoding="utf-8",
    )
    return config_path


def test_cli_all_completes_into_configured_directory(tmp_path: Path, capsys: object) -> None:
    config_path = write_config(tmp_path)

    exit_code = main(("--config", str(config_path), "all"))

    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert exit_code == 0
    assert output == {"command": "all", "output_directory": str(tmp_path / "run")}
    assert (tmp_path / "run/evaluation_report.json").is_file()


def test_cli_prepare_prints_concise_json_summary(tmp_path: Path, capsys: object) -> None:
    config_path = write_config(tmp_path)

    exit_code = main(("--config", str(config_path), "prepare"))

    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert exit_code == 0
    assert output["command"] == "prepare"
    assert output["output_directory"] == str(tmp_path / "run")
