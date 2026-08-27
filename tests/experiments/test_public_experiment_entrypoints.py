from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_selector_training_has_an_installed_command_and_help_contract() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["scripts"]["evidence-rag-selector-train"] == (
        "evidence_rag.cli.run_selector_lean:main"
    )
    result = _run("-m", "evidence_rag.cli.run_selector_lean", "--help")
    assert result.returncode == 0, result.stderr
    assert "fit" in result.stdout
    assert "calibrate" in result.stdout
    assert "final-evaluate" in result.stdout


def test_experiment04_public_builder_rebuilds_frozen_tables(tmp_path: Path) -> None:
    output = tmp_path / "experiment04"

    result = _run(
        "experiments/experiment04/build_tables.py",
        "--output-dir",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in output.iterdir()) == [
        "final_tables.md",
        "summary_metrics.csv",
        "table1.json",
        "table1.tex",
        "table2.json",
        "table2.tex",
    ]


def test_experiment05_public_builder_rebuilds_frozen_tables(tmp_path: Path) -> None:
    output = tmp_path / "experiment05"

    result = _run(
        "experiments/experiment05/build_tables.py",
        "--output-dir",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in output.iterdir()) == [
        "final_report.md",
        "table1.csv",
        "table2.csv",
        "tables.tex",
    ]
