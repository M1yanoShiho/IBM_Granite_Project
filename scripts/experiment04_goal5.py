#!/usr/bin/env python3
"""Compile and independently audit the frozen Experiment 04 final results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import experiment04_goal3 as g3  # noqa: E402
import experiment04_goal4 as g4  # noqa: E402

from evidence_rag.evaluation.experiment04_goal3 import (  # noqa: E402
    paired_component_cluster_bootstrap,
)

EXPECTED_COUNTS = {"hotpotqa": 400, "musique-answerable": 400, "rgb-noise": 300}
GOAL3_ARMS = (
    "dense_rag",
    "hybrid_rag",
    "granite_rerank_rag",
    "provence_rag",
    "ours_seed13",
    "ours_seed42",
    "ours_seed73",
)
OURS_ARMS = GOAL3_ARMS[4:]
GOAL4_ARMS = (
    "ours_seed13",
    "ablation_dense_retriever",
    "ablation_top10",
    "ablation_direct_generator",
)
METRICS = ("ret", "sel", "ans", "cit", "rar")
TABLE1_LABELS = {
    "dense_rag": "Dense RAG",
    "hybrid_rag": "Hybrid RAG",
    "granite_rerank_rag": "Granite Rerank RAG",
    "provence_rag": "Provence RAG",
    "ours": "Ours",
}
TABLE2_LABELS = {
    "ours_seed13": "Full",
    "ablation_dense_retriever": "w/ Dense Retriever",
    "ablation_top10": "w/ Top-10",
    "ablation_direct_generator": "w/ Direct Generator",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path.name}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _close(observed: float, expected: float) -> bool:
    return math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)


def _rows_by_arm(
    rows: Sequence[Mapping[str, str]],
) -> dict[tuple[str, str], dict[str, Mapping[str, str]]]:
    grouped: dict[tuple[str, str], dict[str, Mapping[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["dataset"]), str(row["arm_id"]))
        query_id = str(row["query_id"])
        if query_id in grouped[key]:
            raise ValueError(f"duplicate per-query row: {key}")
        grouped[key][query_id] = row
    return dict(grouped)


def _validate_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    arms: Sequence[str],
    expected_total: int,
) -> dict[tuple[str, str], dict[str, Mapping[str, str]]]:
    if len(rows) != expected_total:
        raise ValueError(f"per-query row count differs: {len(rows)} != {expected_total}")
    grouped = _rows_by_arm(rows)
    expected_cells = {
        (dataset, arm): count for dataset, count in EXPECTED_COUNTS.items() for arm in arms
    }
    observed = Counter({key: len(value) for key, value in grouped.items()})
    if observed != Counter(expected_cells):
        raise ValueError("dataset-arm coverage differs from the frozen matrix")
    for row in rows:
        for metric in METRICS:
            value = float(row[metric])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"metric outside [0,1]: {metric}")
    for dataset in EXPECTED_COUNTS:
        reference = set(grouped[(dataset, arms[0])])
        for arm in arms[1:]:
            if set(grouped[(dataset, arm)]) != reference:
                raise ValueError(f"{dataset} arms do not share one exact query set")
    return grouped


def _aggregate(rows: Mapping[str, Mapping[str, str]], metric: str) -> float:
    return sum(float(row[metric]) for row in rows.values()) / len(rows)


def _validate_table1(
    table: Mapping[str, Any],
    grouped: Mapping[tuple[str, str], Mapping[str, Mapping[str, str]]],
) -> None:
    reverse = {value: key for key, value in TABLE1_LABELS.items() if key != "ours"}
    for dataset in EXPECTED_COUNTS:
        rows = table[dataset]
        if [row["system"] for row in rows] != [
            "Dense RAG",
            "Hybrid RAG",
            "Granite Rerank RAG",
            "Provence RAG",
            "Ours",
        ]:
            raise ValueError(f"{dataset} Table 1 row order differs")
        for row in rows[:4]:
            arm = reverse[str(row["system"])]
            for metric in METRICS:
                if not _close(
                    _aggregate(grouped[(dataset, arm)], metric),
                    float(row[metric]),
                ):
                    raise ValueError(f"Table 1 aggregate differs: {dataset}/{arm}/{metric}")
        ours = rows[4]
        seed_aggregates = {
            arm: {metric: _aggregate(grouped[(dataset, arm)], metric) for metric in METRICS}
            for arm in OURS_ARMS
        }
        for metric in ("ret", "sel"):
            values = [seed_aggregates[arm][metric] for arm in OURS_ARMS]
            if max(values) != min(values) or not _close(values[0], float(ours[metric])):
                raise ValueError(f"Ours fixed upstream differs: {dataset}/{metric}")
        for metric in ("ans", "cit", "rar"):
            values = [seed_aggregates[arm][metric] for arm in OURS_ARMS]
            expected = ours[metric]
            if not _close(statistics.mean(values), float(expected["mean"])):
                raise ValueError(f"Ours mean differs: {dataset}/{metric}")
            if not _close(statistics.stdev(values), float(expected["sample_sd"])):
                raise ValueError(f"Ours sample SD differs: {dataset}/{metric}")


def _validate_table2(
    table: Mapping[str, Any],
    grouped: Mapping[tuple[str, str], Mapping[str, Mapping[str, str]]],
) -> None:
    reverse = {value: key for key, value in TABLE2_LABELS.items()}
    for dataset in EXPECTED_COUNTS:
        rows = table[dataset]
        if [row["configuration"] for row in rows] != list(TABLE2_LABELS.values()):
            raise ValueError(f"{dataset} Table 2 row order differs")
        for row in rows:
            arm = reverse[str(row["configuration"])]
            for metric in METRICS:
                if not _close(
                    _aggregate(grouped[(dataset, arm)], metric),
                    float(row[metric]),
                ):
                    raise ValueError(f"Table 2 aggregate differs: {dataset}/{arm}/{metric}")


def _validate_full_reuse(
    goal3: Mapping[tuple[str, str], Mapping[str, Mapping[str, str]]],
    goal4: Mapping[tuple[str, str], Mapping[str, Mapping[str, str]]],
) -> None:
    for dataset in EXPECTED_COUNTS:
        left = goal3[(dataset, "ours_seed13")]
        right = goal4[(dataset, "ours_seed13")]
        if set(left) != set(right):
            raise ValueError(f"{dataset} Goal 4 Full IDs differ from Goal 3 seed13")
        for query_id in left:
            for field in ("component_id", *METRICS, "failure_reason"):
                if left[query_id][field] != right[query_id][field]:
                    raise ValueError(f"{dataset} Full reuse row differs: {field}")


def _recompute_goal3_bootstrap(
    grouped: Mapping[tuple[str, str], Mapping[str, Mapping[str, str]]],
) -> dict[str, Any]:
    labels = {
        "dense_rag": "Dense RAG",
        "hybrid_rag": "Hybrid RAG",
        "granite_rerank_rag": "Granite Rerank RAG",
        "provence_rag": "Provence RAG",
    }
    output: dict[str, Any] = {
        "schema_version": "experiment04.paired_rar_bootstrap.v1",
        "resamples": 10_000,
        "seed": 13,
        "datasets": {},
    }
    for dataset in EXPECTED_COUNTS:
        query_ids = tuple(grouped[(dataset, "ours_seed13")])
        components = {
            query_id: str(grouped[(dataset, "ours_seed13")][query_id]["component_id"])
            for query_id in query_ids
        }
        ours_mean = {
            query_id: statistics.mean(
                float(grouped[(dataset, arm)][query_id]["rar"]) for arm in OURS_ARMS
            )
            for query_id in query_ids
        }
        output["datasets"][dataset] = {
            label: paired_component_cluster_bootstrap(
                candidate=ours_mean,
                baseline={
                    query_id: float(grouped[(dataset, arm)][query_id]["rar"])
                    for query_id in query_ids
                },
                component_ids=components,
                resamples=10_000,
                seed=13,
            )
            for arm, label in labels.items()
        }
    return output


def _recompute_goal4_bootstrap(
    grouped: Mapping[tuple[str, str], Mapping[str, Mapping[str, str]]],
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "schema_version": "experiment04.goal4_paired_rar_bootstrap.v1",
        "direction": "Full_minus_ablation",
        "resamples": 10_000,
        "seed": 13,
        "datasets": {},
    }
    for dataset in EXPECTED_COUNTS:
        query_ids = tuple(grouped[(dataset, "ours_seed13")])
        components = {
            query_id: str(grouped[(dataset, "ours_seed13")][query_id]["component_id"])
            for query_id in query_ids
        }
        output["datasets"][dataset] = {
            TABLE2_LABELS[arm]: paired_component_cluster_bootstrap(
                candidate={
                    query_id: float(grouped[(dataset, "ours_seed13")][query_id]["rar"])
                    for query_id in query_ids
                },
                baseline={
                    query_id: float(grouped[(dataset, arm)][query_id]["rar"])
                    for query_id in query_ids
                },
                component_ids=components,
                resamples=10_000,
                seed=13,
            )
            for arm in GOAL4_ARMS[1:]
        }
    return output


def _validate_pass_manifest_hashes(
    experiment: Path,
    results: Path,
) -> None:
    goal3 = _read_json(experiment / "artifacts/goal3_pass_manifest.json")
    goal4 = _read_json(experiment / "artifacts/goal4_pass_manifest.json")
    if goal3.get("status") != "PASS" or goal4.get("status") != "PASS":
        raise ValueError("Goal 3 and Goal 4 PASS manifests are required")
    for name, expected in goal3["result_sha256"].items():
        if _sha256(results / name) != expected:
            raise ValueError(f"Goal 3 frozen result hash differs: {name}")
    goal4_names = {
        "table2_markdown_sha256": "TABLE2.md",
        "table2_json_sha256": "table2.json",
        "paired_bootstrap_sha256": "bootstrap_goal4_ci.json",
        "per_query_metrics_sha256": "per_query_goal4_metrics.csv",
        "goal4_audit_sha256": "goal4_audit.json",
    }
    for key, name in goal4_names.items():
        if _sha256(results / name) != goal4["result_artifacts"][key]:
            raise ValueError(f"Goal 4 frozen result hash differs: {name}")


def _summary_rows(table1: Mapping[str, Any], table2: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, count in EXPECTED_COUNTS.items():
        for system in table1[dataset]:
            for metric in METRICS:
                value = system[metric]
                is_distribution = isinstance(value, dict)
                rows.append(
                    {
                        "table": "Table 1",
                        "dataset": dataset,
                        "system": system["system"],
                        "metric": metric,
                        "mean": value["mean"] if is_distribution else value,
                        "std": value["sample_sd"] if is_distribution else "",
                        "variance": (float(value["sample_sd"]) ** 2 if is_distribution else ""),
                        "seed": (
                            "13|42|73"
                            if is_distribution
                            else "fixed_upstream"
                            if system["system"] == "Ours"
                            else "deterministic"
                        ),
                        "n_queries": count,
                        "missing_reason": "",
                    }
                )
        for configuration in table2[dataset]:
            for metric in METRICS:
                label = str(configuration["configuration"])
                rows.append(
                    {
                        "table": "Table 2",
                        "dataset": dataset,
                        "system": label,
                        "metric": metric,
                        "mean": configuration[metric],
                        "std": "",
                        "variance": "",
                        "seed": "frozen_base" if label == "w/ Direct Generator" else "13",
                        "n_queries": count,
                        "missing_reason": "",
                    }
                )
    return rows


def _write_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = (
        "table",
        "dataset",
        "system",
        "metric",
        "mean",
        "std",
        "variance",
        "seed",
        "n_queries",
        "missing_reason",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def _latex_escape(value: str) -> str:
    return value.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def _table1_latex(table: Mapping[str, Any]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{End-to-end performance on the three held-out datasets (percent).}",
        r"\label{tab:experiment04-main}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Dataset & System & Ret. & Sel. & Ans. & Cit. & RAR \\",
        r"\midrule",
    ]
    for dataset, rows in table.items():
        for index, row in enumerate(rows):
            values: list[str] = []
            for metric in METRICS:
                value = row[metric]
                values.append(
                    f"{_pct(float(value['mean']))} $\\pm$ {_pct(float(value['sample_sd']))}"
                    if isinstance(value, dict)
                    else _pct(float(value))
                )
            lines.append(
                f"{_latex_escape(dataset) if index == 0 else ''} & "
                f"{_latex_escape(str(row['system']))} & " + " & ".join(values) + r" \\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def _table2_latex(table: Mapping[str, Any]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Frozen single-module ablations (percent). Full reuses Goal 3 seed 13.}",
        r"\label{tab:experiment04-ablation}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Dataset & Configuration & Ret. & Sel. & Ans. & Cit. & RAR \\",
        r"\midrule",
    ]
    for dataset, rows in table.items():
        for index, row in enumerate(rows):
            values = [_pct(float(row[metric])) for metric in METRICS]
            lines.append(
                f"{_latex_escape(dataset) if index == 0 else ''} & "
                f"{_latex_escape(str(row['configuration']))} & " + " & ".join(values) + r" \\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def _final_tables_markdown(table1_md: str, table2_md: str) -> str:
    return (
        "# Experiment 04 — Final frozen tables\n\n"
        + table1_md.removeprefix("# Experiment 04 — Table 1\n\n")
        + "\n"
        + table2_md.removeprefix("# Experiment 04 — Table 2\n\n")
    )


def _final_report() -> str:
    return (
        """# Experiment 04 — Final report

**Final decision:** `FINAL PASS`"""
        + "  \n"
        + """**Date:** 2026-08-22

## Executive conclusion

The frozen three-module evaluation is technically valid and fully traceable. Goal 3
produced 7,700 formal answers and Goal 4 produced 3,300 new ablation answers, for 11,000
formal answer generations in total. Table 2 reuses 1,100 Goal 3 Full seed13 rows and does
not count them as new generations.

The primary scientific claim that Ours improves RAR is not supported. Ours has higher
citation scores than the deterministic baselines, but lower answer performance and lower
RAR. The ablation result localizes the principal observed weakness to the frozen GR-C
seed13 Generator: replacing it with frozen Direct Granite improves RAR by 3.25 percentage
points on HotpotQA (Full-minus-Direct 95% CI [-5.25, -1.50]) and by 1.00 point on MuSiQue
([-2.00, -0.25]). The RGB direction is similar but its CI includes zero. Dense Retriever
and Top-10 substitutions do not change RAR in these held-out samples.

These conclusions describe the frozen systems and datasets only. They do not authorize
post-held-out tuning or a revised method claim.

## Execution validity

- All six formal dataset jobs across Goals 3 and 4 completed with exit code 0.
- All generation bundles and scorer manifests are PASS.
- Every frozen dataset-arm query set is complete and uses the common denominator.
- Generation was gold-free; scorer-only sidecars were read only after generation hashes
  were frozen.
- The single HotpotQA Goal 4 Dense-Retriever runtime failure remains in the denominator
  (0.25%, below the preregistered 1% invalidation guard).
- No seed was selected: Table 1 uses all three independent GR-C training seeds 13/42/73.
- Goal 4 Full is byte-equivalent at the metric-row level to frozen Goal 3 Ours seed13 and
  was not regenerated.

## Statistical interpretation

Table 1 reports mean and sample SD across the three GR-C training seeds only for Ours
Ans./Cit./RAR; fixed upstream Ret./Sel. and deterministic baselines remain point estimates.
All RAR comparisons use the paired component-cluster percentile bootstrap with 10,000
resamples and seed 13. Goal 3 differences are Ours three-seed per-query mean minus the
baseline; Goal 4 differences are Full minus the named ablation.

The valid negative findings are:

1. Ours does not exceed any Table 1 baseline on RAR in the frozen evaluation.
2. Removing the NLI Selector (Top-10) does not alter RAR on any of the three datasets.
3. Replacing Hybrid retrieval with Dense retrieval does not alter RAR on any dataset.
4. Direct Granite exceeds Full GR-C seed13 RAR on HotpotQA and MuSiQue; RGB is only a
   directional signal because its interval includes zero.

## Final artifacts

- Frozen report tables: [`FINAL_TABLES.md`](results/FINAL_TABLES.md)
- Unified long-form metrics: [`summary_metrics.csv`](results/summary_metrics.csv)
- Combined machine result: [`final_results.json`](results/final_results.json)
- LaTeX tables: [`TABLE1.tex`](results/TABLE1.tex), [`TABLE2.tex`](results/TABLE2.tex)
- Final machine audit: [`final_audit.json`](results/final_audit.json)
- Goal 3 evidence: [`GOAL3_MAIN_SYSTEM_RESULTS.md`](reports/GOAL3_MAIN_SYSTEM_RESULTS.md)
- Goal 4 evidence: [`GOAL4_MODULE_ABLATION_RESULTS.md`](reports/GOAL4_MODULE_ABLATION_RESULTS.md)

## Final boundary

Experiment 04 is complete. No new held-out answers, tuning, seed selection, or method
changes were performed in Goal 5.
"""
    )


def compile_final(experiment: Path) -> dict[str, Any]:
    results = experiment / "results"
    _validate_pass_manifest_hashes(experiment, results)
    goal3_rows = _read_csv(results / "per_query_metrics.csv")
    goal4_rows = _read_csv(results / "per_query_goal4_metrics.csv")
    grouped3 = _validate_rows(goal3_rows, arms=GOAL3_ARMS, expected_total=7_700)
    grouped4 = _validate_rows(goal4_rows, arms=GOAL4_ARMS, expected_total=4_400)
    _validate_full_reuse(grouped3, grouped4)

    table1_artifact = _read_json(results / "table1.json")
    table2_artifact = _read_json(results / "table2.json")
    table1 = table1_artifact["datasets"]
    table2 = table2_artifact["datasets"]
    _validate_table1(table1, grouped3)
    _validate_table2(table2, grouped4)

    goal3_bootstrap = _read_json(results / "bootstrap_ci.json")
    goal4_bootstrap = _read_json(results / "bootstrap_goal4_ci.json")
    if _recompute_goal3_bootstrap(grouped3) != goal3_bootstrap:
        raise ValueError("Goal 3 paired bootstrap does not reproduce exactly")
    if _recompute_goal4_bootstrap(grouped4) != goal4_bootstrap:
        raise ValueError("Goal 4 paired bootstrap does not reproduce exactly")

    expected_table1_md = g3._table_markdown(table1) + "\n"
    expected_table2_md = g4._table_markdown(table2) + "\n"
    observed_table1_md = (results / "TABLE1.md").read_text(encoding="utf-8")
    observed_table2_md = (results / "TABLE2.md").read_text(encoding="utf-8")
    if expected_table1_md != observed_table1_md or expected_table2_md != observed_table2_md:
        raise ValueError("frozen Markdown tables differ from machine JSON")

    summary_path = results / "summary_metrics.csv"
    final_tables_path = results / "FINAL_TABLES.md"
    table1_tex_path = results / "TABLE1.tex"
    table2_tex_path = results / "TABLE2.tex"
    _write_summary(summary_path, _summary_rows(table1, table2))
    final_tables_path.write_text(
        _final_tables_markdown(observed_table1_md, observed_table2_md),
        encoding="utf-8",
    )
    table1_tex_path.write_text(_table1_latex(table1), encoding="utf-8")
    table2_tex_path.write_text(_table2_latex(table2), encoding="utf-8")

    source_names = (
        "table1.json",
        "TABLE1.md",
        "per_query_metrics.csv",
        "bootstrap_ci.json",
        "goal3_audit.json",
        "table2.json",
        "TABLE2.md",
        "per_query_goal4_metrics.csv",
        "bootstrap_goal4_ci.json",
        "goal4_audit.json",
    )
    final_results_path = results / "final_results.json"
    source_hashes = {name: _sha256(results / name) for name in source_names}
    generated_hashes = {
        "summary_metrics.csv": _sha256(summary_path),
        "FINAL_TABLES.md": _sha256(final_tables_path),
        "TABLE1.tex": _sha256(table1_tex_path),
        "TABLE2.tex": _sha256(table2_tex_path),
    }
    final_results = {
        "schema_version": "experiment04.final_results.v1",
        "status": "FINAL PASS",
        "formal_new_answer_generations": 11_000,
        "table2_reused_full_rows": 1_100,
        "tables": {"table1": table1, "table2": table2},
        "paired_bootstrap": {"goal3": goal3_bootstrap, "goal4": goal4_bootstrap},
        "source_sha256": source_hashes,
        "generated_sha256": generated_hashes,
        "claim_decisions": {
            "ours_rar_superiority": "NOT_SUPPORTED",
            "dense_retriever_rar_contribution": "NO_OBSERVED_DIFFERENCE",
            "nli_selector_rar_contribution": "NO_OBSERVED_DIFFERENCE",
            "grc_seed13_vs_direct_hotpotqa": "DIRECT_SIGNIFICANTLY_HIGHER",
            "grc_seed13_vs_direct_musique": "DIRECT_SIGNIFICANTLY_HIGHER",
            "grc_seed13_vs_direct_rgb": "DIRECT_DIRECTIONAL_ONLY_CI_INCLUDES_ZERO",
        },
    }
    _write_json(final_results_path, final_results)

    final_report_path = experiment / "FINAL_REPORT.md"
    final_report_path.write_text(_final_report(), encoding="utf-8")
    final_audit_path = results / "final_audit.json"
    final_audit = {
        "schema_version": "experiment04.final_audit.v1",
        "status": "FINAL PASS",
        "checks": {
            "goal3_pass_manifest_and_hashes": "PASS",
            "goal4_pass_manifest_and_hashes": "PASS",
            "goal3_7700_rows_and_unique_keys": "PASS",
            "goal4_4400_rows_and_unique_keys": "PASS",
            "full_seed13_exact_metric_row_reuse": "PASS",
            "common_denominators_and_metric_bounds": "PASS",
            "table1_aggregates_mean_and_sample_sd": "PASS",
            "table2_aggregates": "PASS",
            "goal3_12_paired_bootstrap_cells_exact_reproduction": "PASS",
            "goal4_9_paired_bootstrap_cells_exact_reproduction": "PASS",
            "markdown_json_consistency": "PASS",
            "latex_and_unified_csv_generated": "PASS",
            "selective_deletion_or_seed_selection": "NONE",
            "new_heldout_generation_in_goal5": "NONE",
        },
        "counts": {
            "formal_new_answer_generations": 11_000,
            "goal3_per_query_rows": 7_700,
            "goal4_table2_per_query_rows": 4_400,
            "goal4_reused_full_rows": 1_100,
            "summary_metric_rows": len(_summary_rows(table1, table2)),
            "paired_bootstrap_cells": 21,
        },
        "source_sha256": source_hashes,
        "generated_sha256": {
            **generated_hashes,
            "final_results.json": _sha256(final_results_path),
            "FINAL_REPORT.md": _sha256(final_report_path),
        },
    }
    _write_json(final_audit_path, final_audit)
    return final_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        audit = compile_final(args.experiment_dir.resolve())
    except Exception as error:  # noqa: BLE001 - final audit emits only machine error identity
        print(
            json.dumps(
                {
                    "status": "FINAL FAIL",
                    "error_code": type(error).__name__.casefold(),
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": audit["status"],
                "formal_new_answers": audit["counts"]["formal_new_answer_generations"],
                "summary_metric_rows": audit["counts"]["summary_metric_rows"],
                "bootstrap_cells": audit["counts"]["paired_bootstrap_cells"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
