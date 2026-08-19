"""Finalize G110 after two independent double-review files exist."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROUTE = Path("docs/full-flow/experiments/03_generator_grounding_repair_2026-08-18")
SCHEMA_PREFIX = "full-flow-g110"
LABELS = {
    "UNSUPPORTED_DRAFT_CLAIM",
    "DRAFT_CITATION_MISSING_OR_WRONG",
    "SPLITTER_BOUNDARY_OR_REWRITE",
    "TRUE_ROUTING_OR_ATTACHMENT",
    "EVALUATOR_DISAGREEMENT",
    "UNCERTAIN",
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(value))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _git(repo: Path, args: Sequence[str]) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_pin(repo: Path, relative: Path) -> dict[str, Any]:
    path = repo / relative
    if not path.is_file():
        raise FileNotFoundError(f"missing G110 source file: {path}")
    return {
        "path": str(path.relative_to(repo)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _review_map(path: Path, reviewer: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in _jsonl(path):
        audit_id = str(row.get("audit_row_id", ""))
        label = str(row.get("label", ""))
        if not audit_id or label not in LABELS:
            raise ValueError(f"invalid review row in {path}: {row}")
        if audit_id in output:
            raise ValueError(f"duplicate review row in {path}: {audit_id}")
        output[audit_id] = {
            "reviewer": reviewer,
            "label": label,
            "confidence": str(row.get("confidence", "")),
            "reason": str(row.get("reason", "")),
        }
    return output


def build_outputs(
    repo: Path,
    *,
    primary_path: Path,
    reviewer_a_path: Path,
    reviewer_b_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    audit_sample = _jsonl(repo / ROUTE / "artifacts/G110/G110_AUDIT_SAMPLE.jsonl")
    double_sample = _jsonl(repo / ROUTE / "artifacts/G110/G110_DOUBLE_REVIEW_SAMPLE.jsonl")
    audit_ids = [str(row["audit_row_id"]) for row in audit_sample]
    double_ids = [str(row["audit_row_id"]) for row in double_sample]
    primary = _review_map(primary_path, "PRIMARY")
    reviewer_a = _review_map(reviewer_a_path, "A")
    reviewer_b = _review_map(reviewer_b_path, "B")
    if set(primary) != set(audit_ids):
        raise ValueError("primary review file does not exactly cover the audit sample")
    if set(reviewer_a) != set(double_ids) or set(reviewer_b) != set(double_ids):
        raise ValueError("reviewer files do not exactly cover the double-review sample")
    double_id_set = set(double_ids)
    adjudicated: list[dict[str, Any]] = []
    for row in audit_sample:
        audit_id = str(row["audit_row_id"])
        primary_row = primary[audit_id]
        a = reviewer_a.get(audit_id)
        b = reviewer_b.get(audit_id)
        double_reviewed = audit_id in double_id_set
        double_agreement = bool(double_reviewed and a and b and a["label"] == b["label"])
        final_label = primary_row["label"]
        final_policy = "primary review"
        if double_agreement and a is not None:
            final_label = a["label"]
            final_policy = "double-review agreement"
        elif double_reviewed:
            final_policy = "primary adjudication after double-review disagreement"
        adjudicated.append(
            {
                "schema_version": f"{SCHEMA_PREFIX}-adjudicated-review-row-v1",
                "audit_row_id": audit_id,
                "task_id": row["task_id"],
                "config": row["config"],
                "claim_id": row["claim_id"],
                "automated_label": row["failure_stage"],
                "primary_label": primary_row["label"],
                "primary_confidence": primary_row["confidence"],
                "reviewer_a_label": a["label"] if a else None,
                "reviewer_b_label": b["label"] if b else None,
                "reviewer_a_confidence": a["confidence"] if a else None,
                "reviewer_b_confidence": b["confidence"] if b else None,
                "double_reviewed": double_reviewed,
                "double_review_agreement": double_agreement if double_reviewed else None,
                "final_label": final_label,
                "final_label_policy": final_policy,
                "primary_reason": primary_row["reason"],
                "reviewer_a_reason": a["reason"] if a else None,
                "reviewer_b_reason": b["reason"] if b else None,
            }
        )
    double_rows = [row for row in adjudicated if row["double_reviewed"]]
    agreement_count = sum(bool(row["double_review_agreement"]) for row in double_rows)
    final_counts = Counter(row["final_label"] for row in adjudicated)
    automated_counts = Counter(row["automated_label"] for row in double_rows)
    agreement_rate = agreement_count / len(double_rows)
    routing = final_counts.get("TRUE_ROUTING_OR_ATTACHMENT", 0)
    evaluator = final_counts.get("EVALUATOR_DISAGREEMENT", 0)
    splitter = final_counts.get("SPLITTER_BOUNDARY_OR_REWRITE", 0)
    draft = final_counts.get("DRAFT_CITATION_MISSING_OR_WRONG", 0) + final_counts.get(
        "UNSUPPORTED_DRAFT_CLAIM", 0
    )
    route_decision = "DEFAULT_DRAFT_REPAIR_NO_SHARED_DOWNSTREAM_REPAIR"
    if splitter > max(routing, evaluator, draft):
        route_decision = "G120_SPLITTER_REPAIR_CANDIDATE_REQUIRES_SEPARATE_IMPLEMENTATION"
    elif routing > max(splitter, evaluator, draft):
        route_decision = "G130_ROUTING_ATTACHMENT_REPAIR_CANDIDATE_REQUIRES_SEPARATE_IMPLEMENTATION"
    elif evaluator > max(splitter, routing, draft):
        route_decision = "EVALUATOR_DISAGREEMENT_DOMINANT_NO_SHARED_RUNTIME_REPAIR"
    summary = {
        "schema_version": f"{SCHEMA_PREFIX}-final-summary-v1",
        "status": "PASS",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {
            "head": _git(repo, ["rev-parse", "HEAD"]),
            "branch": _git(repo, ["branch", "--show-current"]),
        },
        "primary_review_rows": len(adjudicated),
        "double_review_rows": len(double_rows),
        "agreement_count": agreement_count,
        "agreement_rate": agreement_rate,
        "automated_label_counts_on_double_review": dict(sorted(automated_counts.items())),
        "final_label_counts": dict(sorted(final_counts.items())),
        "route_decision_counts": {
            "draft_or_unsupported": draft,
            "evaluator_disagreement": evaluator,
            "routing_or_attachment": routing,
            "splitter": splitter,
        },
        "route_decision": route_decision,
        "conditional_repair_activated": route_decision.startswith("G120")
        or route_decision.startswith("G130"),
        "rationale": "Primary review covers all 120 rows; double-review disagreements are adjudicated by the primary label and reported.",
        "source_files": {
            "sample_summary": _source_pin(repo, ROUTE / "artifacts/G110/G110_SAMPLE_SUMMARY.json"),
            "audit_sample": _source_pin(repo, ROUTE / "artifacts/G110/G110_AUDIT_SAMPLE.jsonl"),
            "double_review_sample": _source_pin(repo, ROUTE / "artifacts/G110/G110_DOUBLE_REVIEW_SAMPLE.jsonl"),
            "primary": {
                "path": str(primary_path.relative_to(repo)),
                "bytes": primary_path.stat().st_size,
                "sha256": _sha256(primary_path),
            },
            "reviewer_a": {
                "path": str(reviewer_a_path.relative_to(repo)),
                "bytes": reviewer_a_path.stat().st_size,
                "sha256": _sha256(reviewer_a_path),
            },
            "reviewer_b": {
                "path": str(reviewer_b_path.relative_to(repo)),
                "bytes": reviewer_b_path.stat().st_size,
                "sha256": _sha256(reviewer_b_path),
            },
            "finalize_script": _source_pin(repo, Path("scripts/full_flow_g110_finalize.py")),
        },
        "runtime_boundary": {
            "gold_loaded_at_runtime": False,
            "reference_answers_loaded_at_runtime": False,
            "heldout_loaded": False,
            "training_started": False,
        },
    }
    return summary, adjudicated, render_markdown(summary)


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# G110 Independent Audit",
        "",
        "**日期：** 2026-08-18",
        f"**状态：** `{summary['status']}`",
        "",
        "## Double Review",
        "",
        f"- Primary rows: `{summary['primary_review_rows']}`",
        f"- Double-reviewed rows: `{summary['double_review_rows']}`",
        f"- Agreement: `{summary['agreement_count']}/{summary['double_review_rows']}` ({summary['agreement_rate'] * 100:.2f}%)",
        f"- Route decision: `{summary['route_decision']}`",
        f"- Conditional repair activated: `{summary['conditional_repair_activated']}`",
        "",
        "## Final Label Counts",
        "",
        "| Label | Rows |",
        "|---|---:|",
    ]
    for label, count in summary["final_label_counts"].items():
        lines.append(f"| {label} | {count} |")
    lines.extend(
        [
            "",
            "G110 only chooses the next repair path. Training remains blocked by the later data/materialization/training gates.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--primary-review", required=True, type=Path)
    parser.add_argument("--reviewer-a", required=True, type=Path)
    parser.add_argument("--reviewer-b", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    out_dir = args.out_dir or (repo / ROUTE / "artifacts/G110")
    summary, rows, markdown = build_outputs(
        repo,
        primary_path=args.primary_review.resolve(),
        reviewer_a_path=args.reviewer_a.resolve(),
        reviewer_b_path=args.reviewer_b.resolve(),
    )
    _write_json(out_dir / "G110_FINAL_SUMMARY.json", summary)
    _write_jsonl(out_dir / "G110_ADJUDICATED_DOUBLE_REVIEW.jsonl", rows)
    _write_text(repo / ROUTE / "G110_AUDIT_REPORT.md", markdown)
    print(f"written {out_dir / 'G110_FINAL_SUMMARY.json'}")
    print(f"written {out_dir / 'G110_ADJUDICATED_DOUBLE_REVIEW.jsonl'}")
    print(f"written {repo / ROUTE / 'G110_AUDIT_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
