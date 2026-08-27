"""Rebuild public tables from frozen aggregate JSON without raw experiment outputs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

_EXP04_DATASETS = ("hotpotqa", "musique-answerable", "rgb-noise")
_EXP04_COUNTS = {"hotpotqa": 400, "musique-answerable": 400, "rgb-noise": 300}
_EXP04_METRICS = ("ret", "sel", "ans", "cit", "rar")
_EXP05_DATASETS = ("kilt-nq", "kilt-tqa", "alce-asqa")
_EXP05_METRICS = ("rfc", "vrfc", "ucr", "cp", "cr", "rr")


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read frozen aggregate {path}: {error}") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"frozen aggregate must be a string-keyed object: {path}")
    return cast(dict[str, Any], value)


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed object")
    return cast(Mapping[str, Any], value)


def _rows(table: Mapping[str, Any], dataset: str) -> Sequence[Mapping[str, Any]]:
    value = table.get(dataset)
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"table rows are invalid for {dataset}")
    return cast(Sequence[Mapping[str, Any]], value)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def _experiment04_table1_markdown(table: Mapping[str, Any]) -> str:
    lines = ["# Experiment 04 — Table 1", ""]
    for dataset in _EXP04_DATASETS:
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| System | Ret. | Sel. | Ans. | Cit. | RAR |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in _rows(table, dataset):
            if row["system"] == "Ours":
                formatted = [
                    _pct(float(row["ret"])),
                    _pct(float(row["sel"])),
                    *[
                        f"{_pct(float(_mapping(row[key], label=key)['mean']))} ± "
                        f"{_pct(float(_mapping(row[key], label=key)['sample_sd']))}"
                        for key in ("ans", "cit", "rar")
                    ],
                ]
            else:
                formatted = [_pct(float(row[key])) for key in _EXP04_METRICS]
            lines.append(f"| {row['system']} | " + " | ".join(formatted) + " |")
        lines.append("")
    return "\n".join(lines)


def _experiment04_table2_markdown(table: Mapping[str, Any]) -> str:
    lines = ["# Experiment 04 — Table 2", ""]
    for dataset in _EXP04_DATASETS:
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Configuration | Ret. | Sel. | Ans. | Cit. | RAR |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in _rows(table, dataset):
            metrics = " | ".join(_pct(float(row[key])) for key in _EXP04_METRICS)
            lines.append(f"| {row['configuration']} | {metrics} |")
        lines.append("")
    lines.append(
        "Full is the frozen Goal 3 Ours seed13 artifact; each non-Full row replaces only "
        "the named module."
    )
    return "\n".join(lines)


def _experiment04_final_markdown(table1: Mapping[str, Any], table2: Mapping[str, Any]) -> str:
    table1_md = _experiment04_table1_markdown(table1) + "\n"
    table2_md = _experiment04_table2_markdown(table2) + "\n"
    return (
        "# Experiment 04 — Final frozen tables\n\n"
        + table1_md.removeprefix("# Experiment 04 — Table 1\n\n")
        + "\n"
        + table2_md.removeprefix("# Experiment 04 — Table 2\n\n")
    )


def _experiment04_summary_rows(
    table1: Mapping[str, Any], table2: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in _EXP04_DATASETS:
        count = _EXP04_COUNTS[dataset]
        for system in _rows(table1, dataset):
            for metric in _EXP04_METRICS:
                value = system[metric]
                is_distribution = isinstance(value, Mapping)
                distribution = _mapping(value, label=metric) if is_distribution else {}
                rows.append(
                    {
                        "table": "Table 1",
                        "dataset": dataset,
                        "system": system["system"],
                        "metric": metric,
                        "mean": distribution["mean"] if is_distribution else value,
                        "std": distribution["sample_sd"] if is_distribution else "",
                        "variance": (
                            float(distribution["sample_sd"]) ** 2 if is_distribution else ""
                        ),
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
        for configuration in _rows(table2, dataset):
            for metric in _EXP04_METRICS:
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


def _experiment04_write_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def _latex_escape(value: str) -> str:
    return value.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def _experiment04_table1_latex(table: Mapping[str, Any]) -> str:
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
    for dataset in _EXP04_DATASETS:
        for index, row in enumerate(_rows(table, dataset)):
            values: list[str] = []
            for metric in _EXP04_METRICS:
                value = row[metric]
                if isinstance(value, Mapping):
                    distribution = _mapping(value, label=metric)
                    values.append(
                        f"{_pct(float(distribution['mean']))} $\\pm$ "
                        f"{_pct(float(distribution['sample_sd']))}"
                    )
                else:
                    values.append(_pct(float(value)))
            lines.append(
                f"{_latex_escape(dataset) if index == 0 else ''} & "
                f"{_latex_escape(str(row['system']))} & " + " & ".join(values) + r" \\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def _experiment04_table2_latex(table: Mapping[str, Any]) -> str:
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
    for dataset in _EXP04_DATASETS:
        for index, row in enumerate(_rows(table, dataset)):
            values = [_pct(float(row[metric])) for metric in _EXP04_METRICS]
            lines.append(
                f"{_latex_escape(dataset) if index == 0 else ''} & "
                f"{_latex_escape(str(row['configuration']))} & "
                + " & ".join(values)
                + r" \\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def rebuild_experiment04_tables(final_results_path: Path, output_dir: Path) -> tuple[Path, ...]:
    """Rebuild every public Experiment 04 table from ``final_results.json``."""

    final = _read_object(final_results_path)
    if final.get("schema_version") != "experiment04.final_results.v1":
        raise ValueError("Experiment 04 final results schema differs")
    if final.get("status") != "FINAL PASS":
        raise ValueError("Experiment 04 final results are not frozen FINAL PASS")
    tables = _mapping(final.get("tables"), label="Experiment 04 tables")
    table1 = _mapping(tables.get("table1"), label="Experiment 04 Table 1")
    table2 = _mapping(tables.get("table2"), label="Experiment 04 Table 2")
    if set(table1) != set(_EXP04_DATASETS) or set(table2) != set(_EXP04_DATASETS):
        raise ValueError("Experiment 04 table datasets differ")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = tuple(
        output_dir / name
        for name in (
            "table1.json",
            "table2.json",
            "summary_metrics.csv",
            "table1.tex",
            "table2.tex",
            "final_tables.md",
        )
    )
    _write_json(paths[0], {"datasets": table1, "schema_version": "experiment04.table1.v1"})
    _write_json(paths[1], {"datasets": table2, "schema_version": "experiment04.table2.v1"})
    _experiment04_write_summary(paths[2], _experiment04_summary_rows(table1, table2))
    paths[3].write_text(_experiment04_table1_latex(table1), encoding="utf-8")
    paths[4].write_text(_experiment04_table2_latex(table2), encoding="utf-8")
    paths[5].write_text(_experiment04_final_markdown(table1, table2), encoding="utf-8")

    expected_hashes = _mapping(final.get("generated_sha256"), label="generated SHA-256")
    expected_by_name = {
        "summary_metrics.csv": expected_hashes.get("summary_metrics.csv"),
        "table1.tex": expected_hashes.get("TABLE1.tex"),
        "table2.tex": expected_hashes.get("TABLE2.tex"),
        "final_tables.md": expected_hashes.get("FINAL_TABLES.md"),
    }
    mismatches = sorted(
        name
        for name, expected in expected_by_name.items()
        if not isinstance(expected, str) or _sha256(output_dir / name) != expected
    )
    if mismatches:
        raise ValueError(f"rebuilt Experiment 04 table hashes differ: {mismatches}")
    return paths


def _experiment05_metric_value(row: Mapping[str, Any], metric: str) -> float | None:
    metrics = _mapping(row.get("metrics"), label="Experiment 05 row metrics")
    value = metrics.get(metric)
    if isinstance(value, Mapping):
        mean = value.get("mean")
        return float(mean) if mean is not None else None
    return float(value) if value is not None else None


def _experiment05_write_csv(
    path: Path,
    table: Mapping[str, Any],
    label_key: str,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", label_key, *_EXP05_METRICS))
        writer.writeheader()
        for dataset in _EXP05_DATASETS:
            for row in _rows(table, dataset):
                writer.writerow(
                    {
                        "dataset": dataset,
                        label_key: row[label_key],
                        **{
                            metric: _experiment05_metric_value(row, metric)
                            for metric in _EXP05_METRICS
                        },
                    }
                )


def _experiment05_latex(table1: Mapping[str, Any], table2: Mapping[str, Any]) -> str:
    lines = ["% Experiment 05 tables", "\\begin{table*}[t]", "\\centering"]
    for title, table, label_key in (
        ("Main systems", table1, "system"),
        ("Module ablations", table2, "configuration"),
    ):
        lines.extend(
            [
                f"% {title}",
                "\\begin{tabular}{llrrrrrr}",
                "Dataset & Configuration & RFC & VRFC & UCR & CP & CR & RR \\\\",
                "\\hline",
            ]
        )
        for dataset in _EXP05_DATASETS:
            for row in _rows(table, dataset):
                label = str(row[label_key]).replace("_", "\\_")
                values: list[str] = []
                for metric in _EXP05_METRICS:
                    value = _experiment05_metric_value(row, metric)
                    values.append("--" if value is None else f"{value:.4f}")
                lines.append(f"{dataset} & {label} & " + " & ".join(values) + " \\\\")
        lines.extend(["\\end{tabular}", "\\medskip"])
    lines.extend(["\\end{table*}", ""])
    return "\n".join(lines)


def _experiment05_markdown_table(
    rows: Sequence[Mapping[str, Any]], label_key: str
) -> list[str]:
    lines = [
        f"| {label_key} | RFC | VRFC | UCR | CP | CR | RR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values: list[str] = []
        for metric in _EXP05_METRICS:
            value = _experiment05_metric_value(row, metric)
            values.append("NA" if value is None else f"{value:.4f}")
        lines.append(f"| {row[label_key]} | " + " | ".join(values) + " |")
    return lines


def _experiment05_report(
    table1: Mapping[str, Any],
    table2: Mapping[str, Any],
    claims: Mapping[str, Any],
) -> str:
    claim_a = _mapping(claims.get("claim_a"), label="Experiment 05 claim A")
    claim_b = _mapping(claims.get("claim_b"), label="Experiment 05 claim B")
    report = [
        "# Experiment 05 Final Report",
        "",
        "**Technical status:** `FINAL PASS`",
        f"**Claim A:** `{claim_a['label']}`",
        f"**Claim B:** `{claim_b['label']}`",
        "",
        "This report evaluates complete-corpus BM25 retrieval followed by frozen Granite "
        "candidate scoring within BM25 Top-1000. It does not claim global standalone dense retrieval.",
    ]
    for dataset in _EXP05_DATASETS:
        report.extend(
            [
                "",
                f"## Table 1 — {dataset}",
                "",
                *_experiment05_markdown_table(_rows(table1, dataset), "system"),
            ]
        )
    for dataset in _EXP05_DATASETS:
        report.extend(
            [
                "",
                f"## Table 2 — {dataset}",
                "",
                *_experiment05_markdown_table(_rows(table2, dataset), "configuration"),
            ]
        )
    report.extend(
        [
            "",
            "## Registered claim decision",
            "",
            "```json",
            json.dumps(claims, indent=2, sort_keys=True),
            "```",
            "",
            "Technical PASS means the frozen protocol completed; it does not imply that either scientific claim was supported.",
        ]
    )
    return "\n".join(report) + "\n"


def rebuild_experiment05_tables(results_dir: Path, output_dir: Path) -> tuple[Path, ...]:
    """Rebuild Experiment 05 public tables from audited aggregate JSON files."""

    table1 = _read_object(results_dir / "table1.json")
    table2 = _read_object(results_dir / "table2.json")
    claims = _read_object(results_dir / "claim_labels.json")
    if set(table1) != set(_EXP05_DATASETS) or set(table2) != set(_EXP05_DATASETS):
        raise ValueError("Experiment 05 table datasets differ")
    labels = {
        str(_mapping(claims.get(name), label=name).get("label"))
        for name in ("claim_a", "claim_b")
    }
    if labels != {"NOT SUPPORTED"}:
        raise ValueError("Experiment 05 frozen claim labels differ")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = tuple(
        output_dir / name
        for name in ("table1.csv", "table2.csv", "tables.tex", "final_report.md")
    )
    _experiment05_write_csv(paths[0], table1, "system")
    _experiment05_write_csv(paths[1], table2, "configuration")
    paths[2].write_text(_experiment05_latex(table1, table2), encoding="utf-8")
    paths[3].write_text(_experiment05_report(table1, table2, claims), encoding="utf-8")
    return paths
