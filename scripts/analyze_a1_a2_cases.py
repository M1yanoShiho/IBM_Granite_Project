"""Exploratory paired case analysis for the frozen G-A12 human annotations.

This script never edits the frozen annotation files.  It derives case-level
rates, paired deltas, bootstrap confidence intervals, and a multi-label failure
taxonomy from ``annotation_unblinded.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


A1_METRICS = ("atomic", "self_contained", "verifiable", "unresolved_pronoun")
A2_METRICS = ("atomic", "self_contained", "span_aligned", "rewrite_faithful")
COVERAGE_SCORE = {"none": 0.0, "partial": 0.5, "complete": 1.0}
PAIRS = {
    "a1_new_minus_old": ("old", "new", "a1_sentence", A1_METRICS),
    "a2_new_minus_old_on_old_a1": (
        "old_old",
        "old_new",
        "a2_claim",
        A2_METRICS,
    ),
    "a2_new_minus_old_on_new_a1": (
        "new_old",
        "new_new",
        "a2_claim",
        A2_METRICS,
    ),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_rate(rows: Iterable[dict[str, str]], metric: str) -> tuple[float | None, int, int]:
    values = [row[metric] for row in rows if row[metric] in {"0", "1"}]
    if not values:
        return None, 0, 0
    positives = sum(value == "1" for value in values)
    return positives / len(values), positives, len(values)


def bootstrap_ci(values: list[float], *, seed: int, iterations: int) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    samples = sorted(
        mean(rng.choice(values) for _ in values) for _ in range(iterations)
    )
    return samples[int(iterations * 0.025)], samples[int(iterations * 0.975)]


def direction(delta: float, *, lower_is_better: bool = False) -> str:
    adjusted = -delta if lower_is_better else delta
    if adjusted > 1e-12:
        return "improved"
    if adjusted < -1e-12:
        return "regressed"
    return "unchanged"


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["query_id"], row["arm"], row["unit_type"])].append(row)
    return grouped


def paired_metric_rows(
    rows: list[dict[str, str]], *, seed: int, iterations: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped = group_rows(rows)
    case_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    query_lookup = {row["query_id"]: row["question"] for row in rows}

    for comparison, (baseline, treatment, unit_type, metrics) in PAIRS.items():
        query_ids = sorted(
            query_id
            for query_id, arm, kind in grouped
            if arm == baseline
            and kind == unit_type
            and (query_id, treatment, unit_type) in grouped
        )
        for metric in metrics:
            metric_cases: list[dict[str, object]] = []
            for query_id in query_ids:
                before, before_pos, before_n = metric_rate(
                    grouped[(query_id, baseline, unit_type)], metric
                )
                after, after_pos, after_n = metric_rate(
                    grouped[(query_id, treatment, unit_type)], metric
                )
                if before is None or after is None:
                    continue
                delta = after - before
                record: dict[str, object] = {
                    "comparison": comparison,
                    "metric": metric,
                    "query_id": query_id,
                    "question": query_lookup[query_id],
                    "baseline_arm": baseline,
                    "treatment_arm": treatment,
                    "baseline_rate": before,
                    "treatment_rate": after,
                    "delta": delta,
                    "direction": direction(
                        delta, lower_is_better=metric == "unresolved_pronoun"
                    ),
                    "baseline_positive": before_pos,
                    "baseline_denominator": before_n,
                    "treatment_positive": after_pos,
                    "treatment_denominator": after_n,
                }
                case_rows.append(record)
                metric_cases.append(record)
            deltas = [float(item["delta"]) for item in metric_cases]
            lower, upper = bootstrap_ci(
                deltas,
                seed=seed + sum(ord(char) for char in comparison + metric),
                iterations=iterations,
            )
            counts = Counter(str(item["direction"]) for item in metric_cases)
            summaries.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "paired_cases": len(metric_cases),
                    "mean_case_delta": mean(deltas) if deltas else None,
                    "bootstrap_ci_low": lower,
                    "bootstrap_ci_high": upper,
                    "improved_cases": counts["improved"],
                    "unchanged_cases": counts["unchanged"],
                    "regressed_cases": counts["regressed"],
                    "bootstrap_iterations": iterations,
                    "bootstrap_seed": seed,
                    "analysis_status": "exploratory_post_hoc",
                }
            )

    # Coverage is already one row per query and arm.  The ordinal score is only
    # an exploratory summary; category transitions are retained in the raw table.
    for comparison, baseline, treatment in (
        ("a2_coverage_new_minus_old_on_old_a1", "old_old", "old_new"),
        ("a2_coverage_new_minus_old_on_new_a1", "new_old", "new_new"),
    ):
        metric_cases = []
        query_ids = sorted(
            query_id
            for query_id, arm, kind in grouped
            if arm == baseline
            and kind == "a2_case"
            and (query_id, treatment, "a2_case") in grouped
        )
        for query_id in query_ids:
            before_row = grouped[(query_id, baseline, "a2_case")][0]
            after_row = grouped[(query_id, treatment, "a2_case")][0]
            before_label = before_row["fact_coverage"]
            after_label = after_row["fact_coverage"]
            if before_label not in COVERAGE_SCORE or after_label not in COVERAGE_SCORE:
                continue
            delta = COVERAGE_SCORE[after_label] - COVERAGE_SCORE[before_label]
            record = {
                "comparison": comparison,
                "metric": "fact_coverage_ordinal",
                "query_id": query_id,
                "question": query_lookup[query_id],
                "baseline_arm": baseline,
                "treatment_arm": treatment,
                "baseline_rate": COVERAGE_SCORE[before_label],
                "treatment_rate": COVERAGE_SCORE[after_label],
                "delta": delta,
                "direction": direction(delta),
                "baseline_positive": before_label,
                "baseline_denominator": 1,
                "treatment_positive": after_label,
                "treatment_denominator": 1,
            }
            case_rows.append(record)
            metric_cases.append(record)
        deltas = [float(item["delta"]) for item in metric_cases]
        lower, upper = bootstrap_ci(
            deltas,
            seed=seed + sum(ord(char) for char in comparison),
            iterations=iterations,
        )
        counts = Counter(str(item["direction"]) for item in metric_cases)
        summaries.append(
            {
                "comparison": comparison,
                "metric": "fact_coverage_ordinal",
                "paired_cases": len(metric_cases),
                "mean_case_delta": mean(deltas) if deltas else None,
                "bootstrap_ci_low": lower,
                "bootstrap_ci_high": upper,
                "improved_cases": counts["improved"],
                "unchanged_cases": counts["unchanged"],
                "regressed_cases": counts["regressed"],
                "bootstrap_iterations": iterations,
                "bootstrap_seed": seed,
                "analysis_status": "exploratory_post_hoc",
            }
        )
    return case_rows, summaries


def failure_taxonomy(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_group: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_group[(row["unit_type"], row["arm"])].append(row)

    output: list[dict[str, object]] = []
    for (unit_type, arm), group in sorted(by_group.items()):
        counts: Counter[str] = Counter()
        rows_with_error = 0
        for row in group:
            errors = [token.strip() for token in row["error_type"].split(";") if token.strip()]
            if errors:
                rows_with_error += 1
                counts.update(set(errors))
        for error_type, count in sorted(counts.items()):
            output.append(
                {
                    "unit_type": unit_type,
                    "arm": arm,
                    "error_type": error_type,
                    "error_rows": count,
                    "all_rows_in_arm_unit": len(group),
                    "error_row_rate": count / len(group),
                    "rows_with_any_error": rows_with_error,
                    "any_error_rate": rows_with_error / len(group),
                    "taxonomy_note": "multi-label; one row may contribute to several types",
                }
            )
    return output


def duplicate_visible_input_audit(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Audit repeated A2 items that were independently labelled in both arms.

    Exact equality of question, claim text, and displayed source span means the
    human judgement task was visibly identical.  Label disagreement is therefore
    annotation instability, not a splitter effect.
    """
    grouped = group_rows(rows)
    details: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for comparison, baseline, treatment in (
        ("a2_new_minus_old_on_old_a1", "old_old", "old_new"),
        ("a2_new_minus_old_on_new_a1", "new_old", "new_new"),
    ):
        query_ids = sorted(
            query_id
            for query_id, arm, kind in grouped
            if arm == baseline
            and kind == "a2_claim"
            and (query_id, treatment, "a2_claim") in grouped
        )
        for query_id in query_ids:
            before_by_signature: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
            after_by_signature: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
            for row in grouped[(query_id, baseline, "a2_claim")]:
                before_by_signature[(row["text"], row["source_span_text"])].append(row)
            for row in grouped[(query_id, treatment, "a2_claim")]:
                after_by_signature[(row["text"], row["source_span_text"])].append(row)
            for signature in sorted(before_by_signature.keys() & after_by_signature.keys()):
                pairs = zip(before_by_signature[signature], after_by_signature[signature])
                for before, after in pairs:
                    for metric in A2_METRICS:
                        if before[metric] not in {"0", "1"} or after[metric] not in {"0", "1"}:
                            continue
                        details.append(
                            {
                                "comparison": comparison,
                                "query_id": query_id,
                                "question": before["question"],
                                "metric": metric,
                                "claim_text": signature[0],
                                "source_span_text": signature[1],
                                "baseline_label": int(before[metric]),
                                "treatment_label": int(after[metric]),
                                "agreement": before[metric] == after[metric],
                                "baseline_errors": before["error_type"],
                                "treatment_errors": after["error_type"],
                            }
                        )
        for metric in A2_METRICS:
            subset = [
                row
                for row in details
                if row["comparison"] == comparison and row["metric"] == metric
            ]
            agreement_count = sum(bool(row["agreement"]) for row in subset)
            summaries.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "identical_visible_items": len(subset),
                    "agreements": agreement_count,
                    "disagreements": len(subset) - agreement_count,
                    "agreement_rate": agreement_count / len(subset) if subset else None,
                    "interpretation": "intra_annotator_duplicate_input_consistency",
                }
            )
    return details, summaries


def representative_cases(
    rows: list[dict[str, str]], case_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    grouped = group_rows(rows)
    candidates: list[tuple[float, str, str, str, str]] = []
    for item in case_rows:
        delta = float(item["delta"])
        if abs(delta) < 1e-12:
            continue
        candidates.append(
            (
                abs(delta),
                str(item["comparison"]),
                str(item["metric"]),
                str(item["query_id"]),
                str(item["direction"]),
            )
        )
    candidates.sort(reverse=True)

    chosen: list[tuple[str, str, str, str]] = []
    seen_slots: set[tuple[str, str]] = set()
    for _, comparison, metric, query_id, result in candidates:
        slot = (comparison, result)
        if slot in seen_slots:
            continue
        chosen.append((comparison, metric, query_id, result))
        seen_slots.add(slot)
        if len(chosen) >= 8:
            break

    pair_lookup = {
        key: (baseline, treatment, unit_type)
        for key, (baseline, treatment, unit_type, _) in PAIRS.items()
    }
    pair_lookup.update(
        {
            "a2_coverage_new_minus_old_on_old_a1": ("old_old", "old_new", "a2_case"),
            "a2_coverage_new_minus_old_on_new_a1": ("new_old", "new_new", "a2_case"),
        }
    )

    output: list[dict[str, object]] = []
    for comparison, metric, query_id, result in chosen:
        baseline, treatment, unit_type = pair_lookup[comparison]
        before = grouped[(query_id, baseline, unit_type)]
        after = grouped[(query_id, treatment, unit_type)]
        output.append(
            {
                "comparison": comparison,
                "metric": metric,
                "direction": result,
                "query_id": query_id,
                "question": before[0]["question"],
                "baseline_arm": baseline,
                "baseline_text": "\n".join(row["text"] for row in before),
                "baseline_source_span": "\n".join(
                    row["source_span_text"] for row in before if row["source_span_text"]
                ),
                "baseline_errors": ";".join(
                    sorted(
                        {
                            token
                            for row in before
                            for token in row["error_type"].split(";")
                            if token
                        }
                    )
                ),
                "baseline_coverage": before[0]["fact_coverage"],
                "treatment_arm": treatment,
                "treatment_text": "\n".join(row["text"] for row in after),
                "treatment_source_span": "\n".join(
                    row["source_span_text"] for row in after if row["source_span_text"]
                ),
                "treatment_errors": ";".join(
                    sorted(
                        {
                            token
                            for row in after
                            for token in row["error_type"].split(";")
                            if token
                        }
                    )
                ),
                "treatment_coverage": after[0]["fact_coverage"],
                "interpretation_status": "requires_manual_narrative_review",
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    args = parser.parse_args()

    rows = read_rows(args.input)
    if len(rows) != 639:
        raise ValueError(f"expected 639 frozen rows, found {len(rows)}")
    case_rows, summaries = paired_metric_rows(
        rows, seed=args.seed, iterations=args.bootstrap_iterations
    )
    taxonomy = failure_taxonomy(rows)
    duplicate_details, duplicate_summary = duplicate_visible_input_audit(rows)
    examples = representative_cases(rows, case_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "case_level_pairs.csv", case_rows)
    write_rows(args.output_dir / "paired_summary.csv", summaries)
    write_rows(args.output_dir / "failure_taxonomy.csv", taxonomy)
    write_rows(args.output_dir / "duplicate_input_audit.csv", duplicate_details)
    write_rows(args.output_dir / "duplicate_input_summary.csv", duplicate_summary)
    write_rows(args.output_dir / "representative_cases.csv", examples)
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "source": str(args.input),
                "source_rows": len(rows),
                "case_pair_rows": len(case_rows),
                "summary_rows": len(summaries),
                "taxonomy_rows": len(taxonomy),
                "duplicate_input_audit_rows": len(duplicate_details),
                "representative_case_rows": len(examples),
                "bootstrap_seed": args.seed,
                "bootstrap_iterations": args.bootstrap_iterations,
                "coverage_ordinal_mapping": COVERAGE_SCORE,
                "status": "exploratory_post_hoc",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "analysis_tables.json").write_text(
        json.dumps(
            {
                "paired_summary": summaries,
                "case_level_pairs": case_rows,
                "failure_taxonomy": taxonomy,
                "duplicate_input_audit": duplicate_details,
                "duplicate_input_summary": duplicate_summary,
                "representative_cases": examples,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
