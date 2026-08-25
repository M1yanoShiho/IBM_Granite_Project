from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/experiment04"


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_final_audit_has_exact_frozen_counts_and_all_checks_pass() -> None:
    audit = json.loads((RESULTS / "final_audit.json").read_text(encoding="utf-8"))

    assert audit["status"] == "FINAL PASS"
    assert audit["counts"] == {
        "formal_new_answer_generations": 11_000,
        "goal3_per_query_rows": 7_700,
        "goal4_reused_full_rows": 1_100,
        "goal4_table2_per_query_rows": 4_400,
        "paired_bootstrap_cells": 21,
        "summary_metric_rows": 135,
    }
    assert set(audit["checks"].values()) == {"PASS", "NONE"}


def test_unified_summary_is_long_form_and_preserves_seed_semantics() -> None:
    rows = _csv(RESULTS / "summary_metrics.csv")

    assert len(rows) == 135
    assert all(row["missing_reason"] == "" for row in rows)
    ours_distributions = [
        row
        for row in rows
        if row["table"] == "Table 1"
        and row["system"] == "Ours"
        and row["metric"] in {"ans", "cit", "rar"}
    ]
    assert len(ours_distributions) == 9
    assert all(row["seed"] == "13|42|73" for row in ours_distributions)
    assert all(row["std"] and row["variance"] for row in ours_distributions)
    deterministic = [row for row in rows if row["table"] == "Table 1" and row["system"] != "Ours"]
    assert len(deterministic) == 60
    assert all(row["seed"] == "deterministic" and not row["std"] for row in deterministic)


def test_final_markdown_and_latex_have_no_placeholders() -> None:
    paths = (
        RESULTS / "final_tables.md",
        RESULTS / "table1.tex",
        RESULTS / "table2.tex",
        ROOT / "docs/research/experiment04.md",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "XX.XX" not in text
    assert "FINAL PASS" in text
    assert r"$\pm$" in (RESULTS / "table1.tex").read_text(encoding="utf-8")
    assert "Full reuses Goal 3 seed 13" in (RESULTS / "table2.tex").read_text(encoding="utf-8")
