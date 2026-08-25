from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "docs/full-flow/experiments/04_frozen_three_module_system_evaluation_2026-08-21"
RESULTS = EXPERIMENT / "results"


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


def test_goal4_full_rows_are_exact_goal3_seed13_metric_reuse() -> None:
    goal3 = {
        (row["dataset"], row["query_id"]): row
        for row in _csv(RESULTS / "per_query_metrics.csv")
        if row["arm_id"] == "ours_seed13"
    }
    goal4 = {
        (row["dataset"], row["query_id"]): row
        for row in _csv(RESULTS / "per_query_goal4_metrics.csv")
        if row["arm_id"] == "ours_seed13"
    }

    assert len(goal3) == len(goal4) == 1_100
    assert set(goal3) == set(goal4)
    for key in goal3:
        assert {
            field: goal3[key][field]
            for field in ("component_id", "ret", "sel", "ans", "cit", "rar", "failure_reason")
        } == {
            field: goal4[key][field]
            for field in ("component_id", "ret", "sel", "ans", "cit", "rar", "failure_reason")
        }


def test_final_markdown_and_latex_have_no_placeholders() -> None:
    paths = (
        RESULTS / "FINAL_TABLES.md",
        RESULTS / "TABLE1.tex",
        RESULTS / "TABLE2.tex",
        EXPERIMENT / "FINAL_REPORT.md",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "XX.XX" not in text
    assert "FINAL PASS" in text
    assert r"$\pm$" in (RESULTS / "TABLE1.tex").read_text(encoding="utf-8")
    assert "Full reuses Goal 3 seed 13" in (RESULTS / "TABLE2.tex").read_text(encoding="utf-8")
