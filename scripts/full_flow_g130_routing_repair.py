"""G130 deterministic routing/attachment repair report.

This stage is allowed only because G110 activated the routing/attachment path.
It changes runtime routing logic and trace/export observability; it does not
train, tune TRUE, read gold/reference answers at runtime, or touch held-out
datasets.
"""

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

from evidence_rag.generator.trace import ClaimTrace

ROUTE = Path("docs/full-flow/experiments/03_generator_grounding_repair_2026-08-18")
SCHEMA_PREFIX = "full-flow-g130"
REQUIRED_TRACE_FIELDS = {
    "attachment_verified",
    "declared_verified",
    "rescued_by_scan",
    "review_flagged",
    "routing_hypothesis",
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
        raise FileNotFoundError(f"missing G130 source file: {path}")
    return {
        "path": str(path.relative_to(repo)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _review_labels(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for row in rows:
        audit_id = str(row.get("audit_row_id", ""))
        final_label = str(row.get("final_label", ""))
        if not audit_id or not final_label:
            raise ValueError("G110 adjudicated row is missing audit_row_id/final_label")
        if audit_id in labels:
            raise ValueError(f"duplicate G110 adjudicated row: {audit_id}")
        labels[audit_id] = final_label
    return labels


def true_routing_breakdown(
    sample_rows: Sequence[Mapping[str, Any]],
    adjudicated_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    labels = _review_labels(adjudicated_rows)
    counts: Counter[str] = Counter()
    for row in sample_rows:
        audit_id = str(row.get("audit_row_id", ""))
        if labels.get(audit_id) != "TRUE_ROUTING_OR_ATTACHMENT":
            continue
        routing_outcome = str(row.get("routing_outcome", ""))
        gated_outcome = str(row.get("gated_outcome", ""))
        has_citation = row.get("citation") is not None
        if routing_outcome == "verified" and has_citation:
            if gated_outcome == "verified":
                counts["already_verified_attachment"] += 1
            else:
                counts["verified_attachment_with_observe_gate_warning"] += 1
        elif routing_outcome == "unverified" and not has_citation:
            counts["true_unverified_no_attachment"] += 1
        else:
            counts["other_runtime_shape"] += 1
    counts["total_true_routing_or_attachment"] = sum(
        value for key, value in counts.items() if key != "total_true_routing_or_attachment"
    )
    return dict(sorted(counts.items()))


def build_outputs(repo: Path) -> tuple[dict[str, Any], str]:
    g110_summary = _json(repo / ROUTE / "artifacts/G110/G110_FINAL_SUMMARY.json")
    if g110_summary.get("status") != "PASS":
        raise ValueError("G130 requires G110 PASS")
    if not str(g110_summary.get("route_decision", "")).startswith("G130"):
        raise ValueError("G130 requires a G110 routing/attachment route decision")

    sample_rows = _jsonl(repo / ROUTE / "artifacts/G110/G110_AUDIT_SAMPLE.jsonl")
    adjudicated_rows = _jsonl(
        repo / ROUTE / "artifacts/G110/G110_ADJUDICATED_DOUBLE_REVIEW.jsonl"
    )
    missing_fields = sorted(REQUIRED_TRACE_FIELDS - set(ClaimTrace.model_fields))
    if missing_fields:
        raise ValueError(f"G130 trace contract fields are missing: {missing_fields}")

    summary = {
        "schema_version": f"{SCHEMA_PREFIX}-routing-repair-summary-v1",
        "status": "PASS",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {
            "head": _git(repo, ["rev-parse", "HEAD"]),
            "branch": _git(repo, ["branch", "--show-current"]),
        },
        "repair_scope": {
            "runtime_uses_final_sentence_as_routing_hypothesis": True,
            "declared_index_handling_changed": False,
            "true_model_or_threshold_changed": False,
            "entity_gate_policy_changed": False,
            "final_attachment_explicitly_traced": True,
            "shared_by_g0_grf_grc": True,
        },
        "trace_contract_fields": sorted(REQUIRED_TRACE_FIELDS),
        "g110_revealed_breakdown": true_routing_breakdown(
            sample_rows, adjudicated_rows
        ),
        "source_files": {
            "g110_summary": _source_pin(
                repo, ROUTE / "artifacts/G110/G110_FINAL_SUMMARY.json"
            ),
            "g110_audit_sample": _source_pin(
                repo, ROUTE / "artifacts/G110/G110_AUDIT_SAMPLE.jsonl"
            ),
            "g110_adjudicated": _source_pin(
                repo, ROUTE / "artifacts/G110/G110_ADJUDICATED_DOUBLE_REVIEW.jsonl"
            ),
            "verify_annotate": _source_pin(
                repo, Path("src/evidence_rag/generator/verify_annotate.py")
            ),
            "trace_model": _source_pin(
                repo, Path("src/evidence_rag/generator/trace.py")
            ),
            "g230_runner": _source_pin(repo, Path("scripts/full_flow_g230.py")),
            "g130_script": _source_pin(
                repo, Path("scripts/full_flow_g130_routing_repair.py")
            ),
        },
        "runtime_boundary": {
            "gold_loaded_at_runtime": False,
            "reference_answers_loaded_at_runtime": False,
            "heldout_loaded": False,
            "training_started": False,
            "utility_labels_started": False,
        },
        "next_stage": "G200_READY_AFTER_G130_ARCHIVE",
    }
    return summary, render_markdown(summary)


def render_markdown(summary: Mapping[str, Any]) -> str:
    breakdown = summary["g110_revealed_breakdown"]
    lines = [
        "# G130 Routing/Attachment Repair",
        "",
        "**日期：** 2026-08-18",
        f"**状态：** `{summary['status']}`",
        "",
        "## Runtime Repair",
        "",
        "- TRUE hypothesis now uses the final output sentence that reaches the user.",
        "- Trace and G230 routing export now expose declared verification, scan rescue, review flag, routing hypothesis, and final attachment verification.",
        "- Observe-only entity gate warnings remain non-destructive and are not treated as failed attachment.",
        "- TRUE model, TRUE threshold, Retriever, Selector, gold/reference inputs, held-out data, and training remain unchanged.",
        "",
        "## G110 Revealed Diagnostic",
        "",
        "| Category | Rows |",
        "|---|---:|",
    ]
    for key, value in breakdown.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "G130 is a deterministic shared runtime repair. It does not rerun G230 metrics and does not claim Generator qualification.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    out_dir = args.out_dir or (repo / ROUTE / "artifacts/G130")
    summary, markdown = build_outputs(repo)
    _write_json(out_dir / "G130_ROUTING_REPAIR_SUMMARY.json", summary)
    _write_text(repo / ROUTE / "G130_ROUTING_REPAIR_REPORT.md", markdown)
    print(f"written {out_dir / 'G130_ROUTING_REPAIR_SUMMARY.json'}")
    print(f"written {repo / ROUTE / 'G130_ROUTING_REPAIR_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
