from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "experiment05_goal1_scorer_validation.py"


def _run(*args: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_fixture_writer_freezes_required_locked_counts_and_separation(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"

    _run("fixtures", "--output-dir", str(fixtures))

    expected = {
        "fact_match.jsonl": 160,
        "evidence_support.jsonl": 160,
        "claim_extraction.jsonl": 60,
        "contract.jsonl": 80,
    }
    for name, count in expected.items():
        locked = _rows(fixtures / "locked" / name)
        calibration = _rows(fixtures / "calibration" / name)
        assert len(locked) == count
        assert calibration
        assert {row["fixture_id"] for row in locked}.isdisjoint(
            row["fixture_id"] for row in calibration
        )

    extraction = _rows(fixtures / "locked" / "claim_extraction.jsonl")
    assert sum(len(row["gold_claims"]) for row in extraction) >= 180

    fixture_text = "\n".join(
        path.read_text() for path in sorted(fixtures.rglob("*.jsonl"))
    ).casefold()
    assert "direct-like" not in fixture_text
    assert "gr-c-like" not in fixture_text


def test_deterministic_contract_runner_requires_all_80_exact_cases(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    report = tmp_path / "contract-report.json"
    _run("fixtures", "--output-dir", str(fixtures))

    _run(
        "contract",
        "--fixtures-dir",
        str(fixtures),
        "--report",
        str(report),
    )

    result = json.loads(report.read_text())
    assert result["locked_cases"] == 80
    assert result["exact_passed"] == 80
    assert result["exact_rate"] == 1.0
    assert result["pass"] is True
