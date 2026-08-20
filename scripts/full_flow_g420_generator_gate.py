"""Combine G400/G410 into the G420 Generator gate.

G420 is a read-only gate over already completed qualification artifacts.  It
does not generate answers, train models, create utility labels, or read held-out
data.  The main job is to combine G400 NIAH and G410 2Wiki evidence, compute the
frozen component-cluster bootstrap summaries, and decide whether G430 may freeze
the new GR-C Generator as the teacher GQ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_REPORT = "full-flow-g420-generator-gate-report-v1"
SCHEMA_MANIFEST = "full-flow-g420-generator-gate-manifest-v1"

SEED_CONFIGS = ("GRC13", "GRC42", "GRC73")
BOOTSTRAP_SEED = 13
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000

G400_PRIMARY_CONTEXT = "K_topk"
G400_STRESS_CONTEXTS = (
    "S_legacy_selected",
    "O_support_only",
    "OB_support_benign",
    "OH_support_harmful",
    "OP_support_last",
)
G400_UNSUPPORTED_CONTEXT = "train_unsupported_safety"
G410_ALL_CONTEXT = "all_2wiki"
G410_CONTEXTS = ("topk", "support_only", "support_first", "support_middle", "support_last")

CORE_METRICS = (
    "correct_and_cited",
    "answer_match",
    "coverage",
    "minicheck_citation_precision",
    "minicheck_citation_recall",
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _arms(row: Mapping[str, Any]) -> Mapping[str, Mapping[str, float]]:
    value = row.get("configs") or row.get("arms")
    if not isinstance(value, Mapping):
        raise ValueError(f"{row.get('task_id')} lacks configs/arms")
    return value  # type: ignore[return-value]


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        raise ValueError("empty mean")
    return sum(vals) / len(vals)


def _candidate_delta(arms: Mapping[str, Mapping[str, float]], metric: str) -> float:
    if "G0" not in arms:
        raise ValueError("scored row lacks G0")
    missing = [name for name in SEED_CONFIGS if name not in arms]
    if missing:
        raise ValueError(f"scored row lacks candidate configs: {missing}")
    g0 = float(arms["G0"].get(metric, 0.0))
    candidate = _mean(float(arms[name].get(metric, 0.0)) for name in SEED_CONFIGS)
    return candidate - g0


def query_level_deltas(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    context: str | None = None,
    dataset: str | None = None,
    answerable: bool | None = None,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if context is not None and row.get("context") != context:
            continue
        if dataset is not None and row.get("dataset") != dataset:
            continue
        if answerable is not None and bool(row.get("answerable", True)) != answerable:
            continue
        component_id = str(row.get("component_id") or row.get("query_id") or row.get("task_id"))
        query_id = str(row.get("query_id") or row.get("task_id"))
        grouped[(component_id, query_id)].append(_candidate_delta(_arms(row), metric))
    return [
        {"component_id": component_id, "query_id": query_id, "delta": _mean(values)}
        for (component_id, query_id), values in sorted(grouped.items())
    ]


def component_cluster_bootstrap(
    query_values: Sequence[Mapping[str, object]],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    if not query_values:
        raise ValueError("cannot bootstrap empty query values")
    by_component: dict[str, list[float]] = defaultdict(list)
    for item in query_values:
        by_component[str(item["component_id"])].append(float(item["delta"]))
    component_ids = sorted(by_component)
    point = _mean(float(item["delta"]) for item in query_values)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(resamples):
        values: list[float] = []
        for _component in component_ids:
            sampled = rng.choice(component_ids)
            values.extend(by_component[sampled])
        draws.append(_mean(values))
    draws.sort()
    low_index = int(0.025 * resamples)
    high_index = max(0, int(0.975 * resamples) - 1)
    return {
        "point": point,
        "ci95_low": draws[low_index],
        "ci95_high": draws[high_index],
        "queries": len(query_values),
        "components": len(component_ids),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
    }


def _stats_for_context(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    context: str | None,
    metrics: Sequence[str],
    resamples: int,
) -> dict[str, object]:
    output: dict[str, object] = {}
    for metric in metrics:
        values = query_level_deltas(rows, metric=metric, context=context, dataset=dataset)
        output[metric] = component_cluster_bootstrap(values, resamples=resamples)
    return output


def _gate_from_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    gate = report.get("gate")
    if not isinstance(gate, Mapping):
        raise ValueError(f"{report.get('stage')} report lacks gate")
    return gate


def _check_report_statuses(
    g400_report: Mapping[str, Any],
    g410_report: Mapping[str, Any],
) -> dict[str, object]:
    g400_gate = _gate_from_report(g400_report)
    g410_gate = _gate_from_report(g410_report)
    return {
        "g400_status": g400_report.get("status"),
        "g410_status": g410_report.get("status"),
        "g400_technical_pass": bool(g400_gate.get("technical_pass")),
        "g410_technical_pass": bool(g410_gate.get("technical_pass")),
        "g400_responsibility_pass": bool(g400_gate.get("responsibility_pass")),
        "g410_responsibility_pass": bool(g410_gate.get("responsibility_pass")),
        "g400_tripwires": list(g400_gate.get("tripwires") or []),
        "g410_tripwires": list(g410_gate.get("tripwires") or []),
        "g400_positive_signal": bool(g400_gate.get("positive_signal")),
        "g410_positive_signal": bool(g410_gate.get("positive_signal")),
        "g410_formal_task_subset": bool(g410_report.get("formal_task_subset")),
    }


def _strong_claim_checks(stats: Mapping[str, Any]) -> dict[str, bool]:
    g400_primary = stats["g400"]["K_topk"]  # type: ignore[index]
    g410_all = stats["g410"]["all_2wiki"]  # type: ignore[index]
    return {
        "niah_primary_correct_and_cited_ci_low_gt_0": g400_primary[
            "correct_and_cited"
        ]["ci95_low"]
        > 0.0,
        "niah_answer_ci_low_ge_minus_2pp": g400_primary["answer_match"]["ci95_low"]
        >= -0.02,
        "niah_coverage_ci_low_ge_minus_2pp": g400_primary["coverage"]["ci95_low"]
        >= -0.02,
        "niah_citation_precision_ci_low_ge_minus_3pp": g400_primary[
            "minicheck_citation_precision"
        ]["ci95_low"]
        >= -0.03,
        "niah_citation_recall_ci_low_ge_minus_3pp": g400_primary[
            "minicheck_citation_recall"
        ]["ci95_low"]
        >= -0.03,
        "twowiki_correct_and_cited_ci_low_gt_0": g410_all["correct_and_cited"][
            "ci95_low"
        ]
        > 0.0,
    }


def _build_markdown(report: Mapping[str, Any]) -> str:
    def pct(value: float) -> str:
        return f"{value * 100:.2f}pp"

    lines = [
        "# G420 Generator Gate",
        "",
        f"**Status:** `{report['status']}`",
        f"**Teacher recommendation:** `{report['teacher_recommendation']}`",
        f"**Strong claim:** `{report['strong_claim_status']}`",
        "",
        "G420 combines the completed G400 NIAH qualification and G410 2Wiki cross-data qualification. It does not run generation, train Selector, create utility labels, or read held-out data.",
        "",
        "## Primary Bootstrap Summary",
        "",
        "| Family | Metric | Point | 95% CI | Queries | Components |",
        "|---|---|---:|---:|---:|---:|",
    ]
    primary = {
        "G400 K_topk": report["statistics"]["g400"]["K_topk"],
        "G410 all_2wiki": report["statistics"]["g410"]["all_2wiki"],
    }
    for family, values in primary.items():
        for metric in CORE_METRICS:
            item = values[metric]
            lines.append(
                "| "
                + family
                + " | "
                + metric
                + " | "
                + pct(float(item["point"]))
                + " | ["
                + pct(float(item["ci95_low"]))
                + ", "
                + pct(float(item["ci95_high"]))
                + "] | "
                + str(item["queries"])
                + " | "
                + str(item["components"])
                + " |"
            )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "```json",
            json.dumps(report["gate"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Boundary",
            "",
            "ASQA/QAMPARI revealed citation guard is recorded as boundary-confirmed but not rerun in this stage. No new independent claim is made from those data here.",
        ]
    )
    return "\n".join(lines) + "\n"


def evaluate(
    *,
    g400_score_report_path: Path,
    g400_scored_rows_path: Path,
    g410_score_report_path: Path,
    g410_scored_rows_path: Path,
    output_json: Path,
    output_report: Path,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    g400_report = _json(g400_score_report_path)
    g410_report = _json(g410_score_report_path)
    g400_rows = _jsonl(g400_scored_rows_path)
    g410_rows = _jsonl(g410_scored_rows_path)

    status_checks = _check_report_statuses(g400_report, g410_report)
    stats: dict[str, dict[str, object]] = {"g400": {}, "g410": {}}
    stats["g400"][G400_PRIMARY_CONTEXT] = _stats_for_context(
        g400_rows,
        dataset="niah",
        context=G400_PRIMARY_CONTEXT,
        metrics=CORE_METRICS,
        resamples=resamples,
    )
    for context in G400_STRESS_CONTEXTS:
        stats["g400"][context] = _stats_for_context(
            g400_rows,
            dataset="niah",
            context=context,
            metrics=CORE_METRICS,
            resamples=resamples,
        )
    stats["g400"][G400_UNSUPPORTED_CONTEXT] = _stats_for_context(
        g400_rows,
        dataset="niah",
        context=G400_UNSUPPORTED_CONTEXT,
        metrics=("unsupported_ungrounded_assertion",),
        resamples=resamples,
    )
    stats["g410"][G410_ALL_CONTEXT] = _stats_for_context(
        g410_rows,
        dataset="2wiki",
        context=None,
        metrics=CORE_METRICS,
        resamples=resamples,
    )
    for context in G410_CONTEXTS:
        stats["g410"][context] = _stats_for_context(
            g410_rows,
            dataset="2wiki",
            context=context,
            metrics=CORE_METRICS,
            resamples=resamples,
        )

    strong_checks = _strong_claim_checks(stats)
    technical_pass = bool(status_checks["g400_technical_pass"]) and bool(
        status_checks["g410_technical_pass"]
    )
    responsibility_pass = bool(status_checks["g400_responsibility_pass"]) and bool(
        status_checks["g410_responsibility_pass"]
    )
    no_tripwires = not status_checks["g400_tripwires"] and not status_checks["g410_tripwires"]
    formal_complete = not bool(status_checks["g410_formal_task_subset"])
    strong_claim_established = all(strong_checks.values())
    pass_new_generator = technical_pass and responsibility_pass and no_tripwires and formal_complete

    if pass_new_generator:
        status = "G420_NEW_GENERATOR_QUALIFIED_G430_READY"
        teacher = "FREEZE_NEW_GRC_IN_G430"
    elif technical_pass and no_tripwires and (
        status_checks["g400_positive_signal"] or status_checks["g410_positive_signal"]
    ):
        status = "G420_CONTROLLED_CONTINUATION_G0_FALLBACK_READY"
        teacher = "USE_G0_FALLBACK_IN_G430"
    else:
        status = "G420_GENERATOR_GATE_FAIL"
        teacher = "STOP_BEFORE_G430"

    gate = {
        "technical_pass": technical_pass,
        "responsibility_pass": responsibility_pass,
        "no_tripwires": no_tripwires,
        "formal_complete": formal_complete,
        "g400_status": status_checks["g400_status"],
        "g410_status": status_checks["g410_status"],
        "strong_claim_checks": strong_checks,
        "asqa_qampari_revealed_guard": "BOUNDARY_CONFIRMED_NOT_RERUN_NO_NEW_CLAIM",
        "heldout_read": False,
        "utility_labels_started": False,
    }
    report: dict[str, object] = {
        "schema_version": SCHEMA_REPORT,
        "stage": "G420",
        "status": status,
        "teacher_recommendation": teacher,
        "strong_claim_status": "ESTABLISHED" if strong_claim_established else "NOT_ESTABLISHED",
        "git_commit": _git_commit(),
        "inputs": {
            "g400_score_report": {
                "path": str(g400_score_report_path),
                "sha256": _sha256(g400_score_report_path),
            },
            "g400_scored_rows": {
                "path": str(g400_scored_rows_path),
                "sha256": _sha256(g400_scored_rows_path),
            },
            "g410_score_report": {
                "path": str(g410_score_report_path),
                "sha256": _sha256(g410_score_report_path),
            },
            "g410_scored_rows": {
                "path": str(g410_scored_rows_path),
                "sha256": _sha256(g410_scored_rows_path),
            },
        },
        "source_status_checks": status_checks,
        "statistics": stats,
        "gate": gate,
        "boundaries": {
            "no_new_generation": True,
            "no_training": True,
            "no_utility_labels": True,
            "sealed_or_heldout_read": False,
            "gq_freeze_not_performed": True,
            "next_stage": "G430 teacher Generator freeze",
        },
    }
    _write_json(output_json, report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(_build_markdown(report), encoding="utf-8")
    return report


def build_manifest(
    *,
    report: Mapping[str, object],
    output_manifest: Path,
    output_json: Path,
    output_report: Path,
) -> dict[str, object]:
    manifest = {
        "schema_version": SCHEMA_MANIFEST,
        "stage": "G420",
        "status": report["status"],
        "teacher_recommendation": report["teacher_recommendation"],
        "strong_claim_status": report["strong_claim_status"],
        "git_commit": report["git_commit"],
        "outputs": {
            "score_report": {"path": str(output_json), "sha256": _sha256(output_json)},
            "markdown_report": {"path": str(output_report), "sha256": _sha256(output_report)},
        },
        "boundaries": report["boundaries"],
        "next_stage": "G430 teacher Generator freeze",
    }
    _write_json(output_manifest, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g400-score-report", type=Path, required=True)
    parser.add_argument("--g400-scored-rows", type=Path, required=True)
    parser.add_argument("--g410-score-report", type=Path, required=True)
    parser.add_argument("--g410-scored-rows", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(
        g400_score_report_path=args.g400_score_report,
        g400_scored_rows_path=args.g400_scored_rows,
        g410_score_report_path=args.g410_score_report,
        g410_scored_rows_path=args.g410_scored_rows,
        output_json=args.output_json,
        output_report=args.output_report,
        resamples=args.resamples,
    )
    manifest = build_manifest(
        report=report,
        output_manifest=args.output_manifest,
        output_json=args.output_json,
        output_report=args.output_report,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
