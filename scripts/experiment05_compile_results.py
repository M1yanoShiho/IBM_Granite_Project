#!/usr/bin/env python3
"""Compile Experiment 05 frozen scores, bootstrap inference, and final tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evidence_rag.evaluation.experiment05_data import validate_sidecar_record  # noqa: E402
from evidence_rag.evaluation.experiment05_generation import SystemOutput  # noqa: E402
from evidence_rag.evaluation.experiment05_io import read_jsonl  # noqa: E402
from evidence_rag.evaluation.experiment05_runtime import (  # noqa: E402
    PreparedEvidence,
    PreparedQuery,
)
from evidence_rag.evaluation.experiment05_scorer import aggregate_query_scores  # noqa: E402

DATASETS = ("kilt-nq", "kilt-tqa", "alce-asqa")
MAIN_ARMS = (
    "bm25_rag",
    "hybrid_rag",
    "granite_rerank_rag",
    "provence_rag",
    "ours_seed13",
    "ours_seed42",
    "ours_seed73",
)
ABLATION_ARMS = (
    "ablation_bm25_retriever",
    "ablation_no_selector",
    "ablation_direct_generator",
)
OURS_ARMS = ("ours_seed13", "ours_seed42", "ours_seed73")
METRICS = ("rfc", "vrfc", "ucr", "cp", "cr", "rr")
MAIN_LABELS = {
    "bm25_rag": "BM25 RAG",
    "hybrid_rag": "Hybrid RAG",
    "granite_rerank_rag": "Granite Rerank RAG",
    "provence_rag": "Provence RAG",
}
ABLATION_LABELS = {
    "ours_seed13": "Full",
    "ablation_bm25_retriever": "w/ BM25 Retriever",
    "ablation_no_selector": "w/o Selector / Keep-all Top10",
    "ablation_direct_generator": "w/ Direct Generator",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _load_scores(runroot: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    data: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for dataset in DATASETS:
        data[dataset] = {}
        for arm in (*MAIN_ARMS, *ABLATION_ARMS):
            path = runroot / "goal5/scoring" / dataset / arm / "query_scores.jsonl"
            rows = read_jsonl(path)
            if (
                len(rows) != 400
                or len({str(row["query_id"]) for row in rows}) != 400
                or any(row.get("dataset") != dataset or row.get("arm_id") != arm for row in rows)
                or any(row.get("scorer_error") for row in rows)
            ):
                raise ValueError(f"invalid scorer bundle: {dataset}/{arm}")
            data[dataset][arm] = rows
        reference_ids = [str(row["query_id"]) for row in data[dataset][MAIN_ARMS[0]]]
        if any(
            [str(row["query_id"]) for row in data[dataset][arm]] != reference_ids
            for arm in (*MAIN_ARMS[1:], *ABLATION_ARMS)
        ):
            raise ValueError(f"{dataset} scorer arms do not share ordered query IDs")
    return data


def _metric(rows: Sequence[Mapping[str, Any]], metric: str, indexes: Sequence[int]) -> float | None:
    if metric != "ucr":
        return sum(float(rows[index]["metrics"][metric]) for index in indexes) / len(indexes)
    claims = sum(int(rows[index]["counts"]["claims"]) for index in indexes)
    if not claims:
        return None
    unsupported = sum(int(rows[index]["counts"]["unsupported_claims"]) for index in indexes)
    return unsupported / claims


def _ci(values: Sequence[float], lower: float, upper: float) -> list[float]:
    return [
        float(np.quantile(values, lower, method="linear")),
        float(np.quantile(values, upper, method="linear")),
    ]


def _bootstrap_dataset(
    arms: Mapping[str, list[dict[str, Any]]], *, resamples: int = 10_000
) -> dict[str, Any]:
    rng = random.Random(13)
    effects: dict[str, list[float]] = {metric: [] for metric in METRICS}
    invalid_ucr = 0
    ordered_arms = ("hybrid_rag", *OURS_ARMS)
    macro_arrays = {
        metric: np.asarray(
            [
                [float(row["metrics"][metric]) for row in arms[arm]]
                for arm in ordered_arms
            ],
            dtype=np.float64,
        )
        for metric in METRICS
        if metric != "ucr"
    }
    claim_arrays = np.asarray(
        [
            [int(row["counts"]["claims"]) for row in arms[arm]]
            for arm in ordered_arms
        ],
        dtype=np.int64,
    )
    unsupported_arrays = np.asarray(
        [
            [int(row["counts"]["unsupported_claims"]) for row in arms[arm]]
            for arm in ordered_arms
        ],
        dtype=np.int64,
    )
    for _ in range(resamples):
        indexes = np.fromiter(
            (rng.randrange(400) for _index in range(400)), dtype=np.int64, count=400
        )
        for metric in METRICS:
            if metric == "ucr":
                claims = claim_arrays[:, indexes].sum(axis=1)
                if np.any(claims == 0):
                    invalid_ucr += 1
                    continue
                values = unsupported_arrays[:, indexes].sum(axis=1) / claims
            else:
                values = macro_arrays[metric][:, indexes].mean(axis=1)
            effects[metric].append(float(np.mean(values[1:] - values[0])))

    output: dict[str, Any] = {
        "resamples": resamples,
        "seed": 13,
        "invalid_ucr_replicates": invalid_ucr,
        "invalid_ucr_rate": invalid_ucr / resamples,
        "metrics": {},
    }
    indexes = list(range(400))
    for metric in METRICS:
        baseline = _metric(arms["hybrid_rag"], metric, indexes)
        ours_values = [_metric(arms[arm], metric, indexes) for arm in OURS_ARMS]
        point = (
            None
            if baseline is None or any(value is None for value in ours_values)
            else statistics.mean(float(value) - baseline for value in ours_values if value is not None)
        )
        valid = effects[metric]
        estimable = point is not None and len(valid) >= int(0.99 * resamples)
        output["metrics"][metric] = {
            "effect_ours_minus_hybrid": point,
            "seed_effects": {
                arm: None if value is None or baseline is None else value - baseline
                for arm, value in zip(OURS_ARMS, ours_values, strict=True)
            },
            "valid_replicates": len(valid),
            "status": "ESTIMABLE" if estimable else "NOT_ESTIMABLE",
            "ci_98_33_two_sided": _ci(valid, 0.00835, 0.99165) if estimable else None,
            "ci_95_two_sided": _ci(valid, 0.025, 0.975) if estimable else None,
            "ci_95_one_sided_lower": float(np.quantile(valid, 0.05)) if estimable else None,
            "ci_95_one_sided_upper": float(np.quantile(valid, 0.95)) if estimable else None,
        }

    common = [
        index
        for index in range(400)
        if float(arms["hybrid_rag"][index]["metrics"]["rr"]) == 1.0
        and all(float(arms[arm][index]["metrics"]["rr"]) == 1.0 for arm in OURS_ARMS)
    ]
    if common:
        baseline = _metric(arms["hybrid_rag"], "ucr", common)
        ours = [_metric(arms[arm], "ucr", common) for arm in OURS_ARMS]
        common_effect = (
            None
            if baseline is None or any(value is None for value in ours)
            else statistics.mean(float(value) - baseline for value in ours if value is not None)
        )
    else:
        common_effect = None
    output["common_answer_ucr_sensitivity"] = {
        "n_queries": len(common),
        "effect_ours_minus_hybrid": common_effect,
    }
    return output


def _claim_labels(bootstrap: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    def metric(dataset: str, name: str) -> Mapping[str, Any]:
        return bootstrap[dataset]["metrics"][name]

    a_superior = {
        dataset: metric(dataset, "ucr")["status"] == "ESTIMABLE"
        and metric(dataset, "ucr")["ci_98_33_two_sided"][1] < 0.0
        for dataset in DATASETS
    }
    a_ucr_noninferior = {
        dataset: a_superior[dataset]
        or (
            metric(dataset, "ucr")["status"] == "ESTIMABLE"
            and metric(dataset, "ucr")["ci_95_two_sided"][1] <= 0.05
        )
        for dataset in DATASETS
    }
    a_harm = all(
        metric(dataset, name)["status"] == "ESTIMABLE"
        and metric(dataset, name)["ci_95_one_sided_lower"] >= -0.05
        for dataset in DATASETS
        for name in ("rfc", "rr")
    )

    b_superior = {
        dataset: metric(dataset, "vrfc")["status"] == "ESTIMABLE"
        and metric(dataset, "vrfc")["ci_98_33_two_sided"][0] > 0.0
        for dataset in DATASETS
    }
    b_vrfc_noninferior = {
        dataset: b_superior[dataset]
        or (
            metric(dataset, "vrfc")["status"] == "ESTIMABLE"
            and metric(dataset, "vrfc")["ci_95_two_sided"][0] >= -0.05
        )
        for dataset in DATASETS
    }
    b_harm = all(
        metric(dataset, name)["status"] == "ESTIMABLE"
        and metric(dataset, name)["ci_95_one_sided_lower"] >= -0.05
        for dataset in DATASETS
        for name in ("cp", "cr")
    )

    def label(superior: Mapping[str, bool], noninferior: Mapping[str, bool], harm: bool) -> str:
        count = sum(superior.values())
        if not harm or not all(noninferior.values()) or count == 0:
            return "NOT SUPPORTED"
        return "SUPPORTED" if count >= 2 else "MIXED"

    return {
        "claim_a": {
            "label": label(a_superior, a_ucr_noninferior, a_harm),
            "superiority_by_dataset": a_superior,
            "ucr_noninferiority_by_dataset": a_ucr_noninferior,
            "rfc_rr_harm_gates": a_harm,
        },
        "claim_b": {
            "label": label(b_superior, b_vrfc_noninferior, b_harm),
            "superiority_by_dataset": b_superior,
            "vrfc_noninferiority_by_dataset": b_vrfc_noninferior,
            "cp_cr_harm_gates": b_harm,
        },
    }


def _tables(data: Mapping[str, Mapping[str, list[dict[str, Any]]]]) -> tuple[dict[str, Any], dict[str, Any]]:
    table1: dict[str, Any] = {}
    table2: dict[str, Any] = {}
    for dataset in DATASETS:
        aggregates = {
            arm: aggregate_query_scores(rows) for arm, rows in data[dataset].items()
        }
        table1_rows: list[dict[str, Any]] = []
        for arm, label in MAIN_LABELS.items():
            table1_rows.append({"system": label, **aggregates[arm]})
        ours_metrics: dict[str, Any] = {}
        for metric in METRICS:
            values = [aggregates[arm]["metrics"][metric] for arm in OURS_ARMS]
            ours_metrics[metric] = {
                "mean": None if any(value is None for value in values) else statistics.mean(values),
                "sample_sd": None
                if any(value is None for value in values)
                else statistics.stdev(values),
                "by_seed": dict(zip(OURS_ARMS, values, strict=True)),
            }
        table1_rows.append(
            {
                "system": "Ours",
                "metrics": ours_metrics,
                "n_total": 400,
                "n_claims_by_seed": {
                    arm: aggregates[arm]["n_claims"] for arm in OURS_ARMS
                },
                "n_runtime_error_by_seed": {
                    arm: aggregates[arm]["n_runtime_error"] for arm in OURS_ARMS
                },
            }
        )
        table1[dataset] = table1_rows
        table2[dataset] = [
            {"configuration": label, **aggregates[arm]}
            for arm, label in ABLATION_LABELS.items()
        ]
    return table1, table2


def _metric_value(row: Mapping[str, Any], metric: str) -> float | None:
    value = row["metrics"][metric]
    return value.get("mean") if isinstance(value, Mapping) else value


def _write_csv(path: Path, table: Mapping[str, Sequence[Mapping[str, Any]]], label_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", label_key, *METRICS))
        writer.writeheader()
        for dataset, rows in table.items():
            for row in rows:
                writer.writerow(
                    {
                        "dataset": dataset,
                        label_key: row[label_key],
                        **{metric: _metric_value(row, metric) for metric in METRICS},
                    }
                )


def _markdown_table(rows: Sequence[Mapping[str, Any]], label_key: str) -> list[str]:
    lines = [
        f"| {label_key} | RFC | VRFC | UCR | CP | CR | RR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = []
        for metric in METRICS:
            value = _metric_value(row, metric)
            values.append("NA" if value is None else f"{value:.4f}")
        lines.append(f"| {row[label_key]} | " + " | ".join(values) + " |")
    return lines


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"[\w]+", value.casefold()))


def _overlaps(evidence: PreparedEvidence, provenance: Mapping[str, Any]) -> bool:
    if evidence.source_kind != provenance.get("source_kind") or evidence.source_id != str(
        provenance.get("source_id")
    ):
        return False
    if evidence.start_unit is None or provenance.get("start_unit") is None:
        return True
    return int(evidence.start_unit) <= int(provenance["end_unit"]) and int(
        provenance["start_unit"]
    ) <= int(evidence.end_unit)


def _supports(
    evidence: PreparedEvidence,
    fact: Mapping[str, Any],
    provenance: Sequence[Mapping[str, Any]],
) -> bool:
    if any(
        item.get("fact_id") == fact["fact_id"] and _overlaps(evidence, item)
        for item in provenance
    ):
        return True
    text = f" {_normalized(evidence.text)} "
    return any(
        (alias := _normalized(str(raw))) and f" {alias} " in text
        for raw in fact["aliases"]
    )


def _formal_diagnostics(runroot: Path, goal1_root: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset in DATASETS:
        prepared = [
            PreparedQuery.model_validate(row)
            for row in read_jsonl(runroot / "prepared/formal" / f"{dataset}.jsonl")
        ]
        sidecars = read_jsonl(
            goal1_root / "bundles/scorer_only/formal" / f"{dataset}.jsonl"
        )
        for row in sidecars:
            validate_sidecar_record(row, dataset=dataset)
        full_outputs = [
            SystemOutput.model_validate(row)
            for row in read_jsonl(
                runroot
                / "goal3/outputs"
                / dataset
                / "generations/ours_seed13.jsonl"
            )
        ]
        hybrid_outputs = [
            SystemOutput.model_validate(row)
            for row in read_jsonl(
                runroot
                / "goal3/outputs"
                / dataset
                / "generations/hybrid_rag.jsonl"
            )
        ]
        if not (
            len(prepared) == len(sidecars) == len(full_outputs) == len(hybrid_outputs) == 400
        ):
            raise ValueError(f"formal diagnostic inputs differ: {dataset}")
        retrieved_facts = 0
        total_facts = 0
        retrieved_units = 0
        retained_units = 0
        input_tokens = 0
        selected_tokens = 0
        presented_tokens = 0
        for prep, sidecar, full_row, hybrid_row in zip(
            prepared, sidecars, full_outputs, hybrid_outputs, strict=True
        ):
            retrieved = prep.arms["hybrid_rag"].selected
            selected = {
                item.evidence_id: item for item in prep.arms["ours_seed13"].selected
            }
            for fact in sidecar["reference_fact_groups"]:
                matches = [
                    item
                    for item in retrieved
                    if _supports(item, fact, sidecar["gold_provenance"])
                ]
                total_facts += 1
                retrieved_facts += bool(matches)
                retrieved_units += len(matches)
                retained_units += sum(
                    item.evidence_id in selected
                    and _supports(
                        selected[item.evidence_id], fact, sidecar["gold_provenance"]
                    )
                    for item in matches
                )
            input_tokens += sum(
                item.token_count for item in hybrid_row.selected_evidence_records
            )
            selected_tokens += sum(
                item.token_count for item in full_row.selected_evidence_records
            )
            presented_tokens += sum(
                item.token_count for item in full_row.presented_evidence_records
            )
        output[dataset] = {
            "system": "Ours seed13",
            "support_matching": "gold_provenance_or_normalized_alias_containment",
            "er_at_10": retrieved_facts / total_facts,
            "selr": (retrieved_units - retained_units) / retrieved_units
            if retrieved_units
            else None,
            "selection_reduction_rate": 1.0 - selected_tokens / input_tokens,
            "presented_reduction_rate": 1.0 - presented_tokens / input_tokens,
            "reference_facts": total_facts,
            "retrieved_support_units": retrieved_units,
        }
    return output


def _latex_tables(
    table1: Mapping[str, Sequence[Mapping[str, Any]]],
    table2: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
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
        for dataset, rows in table.items():
            for row in rows:
                label = str(row[label_key]).replace("_", "\\_")
                values = [
                    "--"
                    if _metric_value(row, metric) is None
                    else f"{_metric_value(row, metric):.4f}"
                    for metric in METRICS
                ]
                lines.append(
                    f"{dataset} & {label} & " + " & ".join(values) + " \\\\"
                )
        lines.extend(["\\end{tabular}", "\\medskip"])
    lines.extend(["\\end{table*}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runroot", type=Path, required=True)
    parser.add_argument("--goal1-root", type=Path, required=True)
    parser.add_argument("--goal2-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    goal4_freeze = json.loads(
        (args.runroot / "goal4/freeze_manifest.json").read_text(encoding="utf-8")
    )
    if goal4_freeze.get("status") != "PASS" or goal4_freeze.get("output_count") != 3600:
        raise ValueError("Goal 4 generation bundle is not frozen PASS")
    data = _load_scores(args.runroot)
    table1, table2 = _tables(data)
    bootstrap = {
        dataset: _bootstrap_dataset(data[dataset]) for dataset in DATASETS
    }
    claims = _claim_labels(bootstrap)
    goal2 = json.loads(args.goal2_audit.read_text(encoding="utf-8"))
    appendix = {
        "formal": _formal_diagnostics(args.runroot, args.goal1_root),
        "development": {
            dataset: goal2["datasets"][dataset]["diagnostics"]
            for dataset in DATASETS
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "table1.json": table1,
        "table2.json": table2,
        "bootstrap.json": bootstrap,
        "claim_labels.json": claims,
        "appendix_diagnostics.json": appendix,
    }
    for name, value in artifacts.items():
        (args.output_dir / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    _write_csv(args.output_dir / "table1.csv", table1, "system")
    _write_csv(args.output_dir / "table2.csv", table2, "configuration")
    (args.output_dir / "tables.tex").write_text(
        _latex_tables(table1, table2), encoding="utf-8"
    )

    report = [
        "# Experiment 05 Final Report",
        "",
        "**Technical status:** `FINAL PASS`",
        f"**Claim A:** `{claims['claim_a']['label']}`",
        f"**Claim B:** `{claims['claim_b']['label']}`",
        "",
        "This report evaluates complete-corpus BM25 retrieval followed by frozen Granite "
        "candidate scoring within BM25 Top-1000. It does not claim global standalone dense retrieval.",
    ]
    for dataset in DATASETS:
        report.extend(["", f"## Table 1 — {dataset}", "", *_markdown_table(table1[dataset], "system")])
    for dataset in DATASETS:
        report.extend(["", f"## Table 2 — {dataset}", "", *_markdown_table(table2[dataset], "configuration")])
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
    (args.output_dir / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    audit = {
        "schema_version": "experiment05.final_audit.v1",
        "status": "FINAL PASS",
        "formal_generation_outputs": 12_000,
        "formal_query_scores": 12_000,
        "scorer_errors": 0,
        "claim_labels": claims,
        "artifacts": {
            path.name: _sha256(path)
            for path in sorted(args.output_dir.iterdir())
            if path.is_file()
        },
    }
    (args.output_dir / "final_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "FINAL PASS", **claims}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
