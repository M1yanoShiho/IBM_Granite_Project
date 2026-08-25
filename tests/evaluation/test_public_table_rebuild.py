from __future__ import annotations

from pathlib import Path

from evidence_rag.evaluation.public_tables import (
    rebuild_experiment04_tables,
    rebuild_experiment05_tables,
)

ROOT = Path(__file__).resolve().parents[2]


def _assert_same_files(expected_root: Path, actual_root: Path, names: tuple[str, ...]) -> None:
    for name in names:
        assert (actual_root / name).read_bytes() == (expected_root / name).read_bytes(), name


def test_experiment04_tables_rebuild_from_final_aggregate_only(tmp_path: Path) -> None:
    results = ROOT / "results/experiment04"

    generated = rebuild_experiment04_tables(results / "final_results.json", tmp_path)

    expected = (
        "table1.json",
        "table2.json",
        "summary_metrics.csv",
        "table1.tex",
        "table2.tex",
        "final_tables.md",
    )
    assert generated == tuple(tmp_path / name for name in expected)
    _assert_same_files(results, tmp_path, expected)


def test_experiment05_tables_rebuild_from_public_aggregates_only(tmp_path: Path) -> None:
    results = ROOT / "results/experiment05"

    generated = rebuild_experiment05_tables(results, tmp_path)

    expected = ("table1.csv", "table2.csv", "tables.tex", "final_report.md")
    assert generated == tuple(tmp_path / name for name in expected)
    _assert_same_files(results, tmp_path, expected[:3])
    assert (tmp_path / "final_report.md").read_bytes() == (
        ROOT / "docs/research/experiment05.md"
    ).read_bytes()
