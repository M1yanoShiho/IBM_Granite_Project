"""Triage the G210 TRUE failures without changing any gate.

The output is diagnostic only.  It reads the already archived G210 structural
and TRUE rows, summarizes which cases/pairs failed, and writes stable samples
for human review.  It does not train, rescore, filter, or relax thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_SUMMARY = "full-flow-g210-failure-triage-summary-v1"
SCHEMA_SAMPLE = "full-flow-g210-failure-triage-sample-v1"
RELATION_PATTERN = re.compile(r"^(.+?)'s (.+?) is (.+?)[.!?]?$")


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
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relation(hypothesis: str) -> str:
    match = RELATION_PATTERN.match(hypothesis.strip())
    if match is None:
        return "<unparsed>"
    return " ".join(match.group(2).split()).lower()


def _score_bin(score: float) -> str:
    if score < 0.05:
        return "[0.00,0.05)"
    if score < 0.25:
        return "[0.05,0.25)"
    if score < 0.50:
        return "[0.25,0.50)"
    if score < 0.75:
        return "[0.50,0.75)"
    return "[0.75,1.00]"


def _load_by_key(path: Path, key: str) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in _jsonl(path):
        value = str(row.get(key, ""))
        if not value or value in output:
            raise ValueError(f"invalid or duplicate {key}: {value!r}")
        output[value] = row
    return output


def triage(
    *,
    structural_rows_path: Path,
    true_worklist_path: Path,
    true_audit_rows_path: Path,
    final_manifest_path: Path,
    output_dir: Path,
    sample_size: int,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("G210 failure triage output directory must be absent or empty")
    if sample_size <= 0:
        raise ValueError("sample size must be positive")

    structural = _load_by_key(structural_rows_path, "case_id")
    worklist = _load_by_key(true_worklist_path, "audit_id")
    true_rows = _load_by_key(true_audit_rows_path, "audit_id")
    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    if final_manifest.get("status") != "FAIL":
        raise ValueError("failure triage expects a FAIL G210 final manifest")

    pair_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    score_bins: Counter[str] = Counter()
    failed_audit_ids_by_case: dict[str, list[str]] = defaultdict(list)
    case_pair_totals: Counter[str] = Counter()
    samples: list[dict[str, object]] = []

    for audit_id in sorted(true_rows):
        true_row = true_rows[audit_id]
        work_row = worklist.get(audit_id)
        if work_row is None:
            raise ValueError(f"TRUE audit row lacks worklist pair: {audit_id}")
        case_id = str(true_row.get("case_id", ""))
        case_pair_totals[case_id] += 1
        dataset = str(true_row.get("dataset", ""))
        role = str(true_row.get("role", ""))
        target_kind = str(true_row.get("target_kind", ""))
        score = float(true_row.get("true_entailment_score", 0.0))
        entailed = bool(true_row.get("entailed"))
        key = f"{dataset}:{role}:{target_kind}:{'entailed' if entailed else 'not_entailed'}"
        pair_counts[key] += 1
        score_bins[f"{dataset}:{role}:{target_kind}:{_score_bin(score)}"] += 1
        if not entailed:
            failed_audit_ids_by_case[case_id].append(audit_id)
            relation_counts[
                f"{dataset}:{role}:{target_kind}:{_relation(str(work_row.get('hypothesis', '')))}"
            ] += 1
            samples.append(
                {
                    "schema_version": SCHEMA_SAMPLE,
                    "audit_id": audit_id,
                    "case_id": case_id,
                    "dataset": dataset,
                    "role": role,
                    "target_kind": target_kind,
                    "support_evidence_id": str(true_row.get("support_evidence_id", "")),
                    "score": score,
                    "threshold": float(true_row.get("threshold", 0.5)),
                    "relation": _relation(str(work_row.get("hypothesis", ""))),
                    "hypothesis": str(work_row.get("hypothesis", "")),
                    "premise": str(work_row.get("premise", "")),
                }
            )

    case_counts: Counter[str] = Counter()
    for case_id, structural_row in structural.items():
        audit_ids = structural_row.get("true_audit_ids")
        if not isinstance(audit_ids, list) or not audit_ids:
            continue
        failed = failed_audit_ids_by_case.get(case_id, [])
        dataset = str(structural_row.get("dataset", ""))
        role = str(structural_row.get("role", ""))
        target_kind = str(structural_row.get("target_kind", ""))
        status = "case_entailed" if not failed else "case_not_entailed"
        case_counts[f"{dataset}:{role}:{target_kind}:{status}"] += 1

    samples.sort(
        key=lambda row: (
            float(row["score"]),
            str(row["dataset"]),
            str(row["role"]),
            str(row["case_id"]),
            str(row["audit_id"]),
        )
    )
    sample_rows = samples[:sample_size]
    sample_path = output_dir / "failure_samples.jsonl"
    _write_jsonl(sample_path, sample_rows)
    summary: dict[str, object] = {
        "schema_version": SCHEMA_SUMMARY,
        "status": "COMPLETE_DIAGNOSTIC_ONLY",
        "training_started": False,
        "gate_changed": False,
        "threshold_changed": False,
        "final_manifest_status": final_manifest.get("status"),
        "final_failed_gates": {
            key: value
            for key, value in dict(final_manifest.get("gates", {})).items()
            if value is False
        },
        "final_counts": final_manifest.get("counts", {}),
        "pair_counts": dict(pair_counts),
        "case_counts": dict(case_counts),
        "failed_pair_relation_counts_top25": dict(relation_counts.most_common(25)),
        "score_bins": dict(score_bins),
        "sample_size": len(sample_rows),
        "input_sha256": {
            "structural_rows": _sha256(structural_rows_path),
            "true_worklist": _sha256(true_worklist_path),
            "true_audit_rows": _sha256(true_audit_rows_path),
            "final_manifest": _sha256(final_manifest_path),
        },
        "failure_samples_sha256": _sha256(sample_path),
    }
    _write_json(output_dir / "triage_summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-rows", required=True, type=Path)
    parser.add_argument("--true-worklist", required=True, type=Path)
    parser.add_argument("--true-audit-rows", required=True, type=Path)
    parser.add_argument("--final-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=80)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = triage(
        structural_rows_path=args.structural_rows,
        true_worklist_path=args.true_worklist,
        true_audit_rows_path=args.true_audit_rows,
        final_manifest_path=args.final_manifest,
        output_dir=args.output_dir.resolve(),
        sample_size=args.sample_size,
    )
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
