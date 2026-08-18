"""Prepare the fixed G110 citation attribution audit samples.

This stage prepares deterministic samples from G100 rows and enriches them with
the question plus cited evidence text.  It does not adjudicate the double-review
subset; two independent reviewers must audit that subset before G110 can pass.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROUTE = Path("docs/full-flow/experiments/03_generator_grounding_repair_2026-08-18")
PREVIOUS_ROUTE = Path("docs/full-flow/experiments/02_generator_selector_alignment_2026-08-15")
SCHEMA_PREFIX = "full-flow-g110"
FAILURE_ORDER = (
    "UNSUPPORTED_DRAFT_CLAIM",
    "DRAFT_CITATION_MISSING_OR_WRONG",
    "SPLITTER_BOUNDARY_OR_REWRITE",
    "TRUE_ROUTING_OR_ATTACHMENT",
    "EVALUATOR_DISAGREEMENT",
)
SEEDS = (13, 42, 73)
FAMILIES = ("GC", "GM")
FULL_CONTEXT = "K_topk"
AUDIT_TARGET = 120
DOUBLE_REVIEW_TARGET = 24
MIN_PER_STRATUM = 20


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


def _jsonl_gz(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, mode="rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
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


def row_key(row: Mapping[str, Any]) -> str:
    return "|".join(
        [
            str(row["failure_stage"]),
            str(row["config"]),
            str(row["task_id"]),
            str(row["claim_id"]),
        ]
    )


def _stable_score(row: Mapping[str, Any], namespace: str) -> str:
    return hashlib.sha256(f"13:{namespace}:{row_key(row)}".encode()).hexdigest()


def stratified_sample(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: int,
    namespace: str,
    min_per_stratum: int,
) -> list[Mapping[str, Any]]:
    by_stage: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stage[str(row["failure_stage"])].append(row)
    selected: list[Mapping[str, Any]] = []
    large: list[tuple[str, list[Mapping[str, Any]]]] = []
    for stage in FAILURE_ORDER:
        group = sorted(by_stage.get(stage, []), key=lambda item: _stable_score(item, namespace))
        if not group:
            continue
        take = min(len(group), min_per_stratum)
        selected.extend(group[:take])
        if len(group) > take:
            large.append((stage, group[take:]))
    remaining = max(target - len(selected), 0)
    large_remaining = sum(len(group) for _stage, group in large)
    allocations: dict[str, int] = {}
    for stage, group in large:
        if large_remaining == 0:
            allocations[stage] = 0
        else:
            allocations[stage] = min(len(group), int(remaining * len(group) / large_remaining))
    while sum(allocations.values()) < remaining and large:
        for stage, group in large:
            if sum(allocations.values()) >= remaining:
                break
            if allocations[stage] < len(group):
                allocations[stage] += 1
    for stage, group in large:
        selected.extend(group[: allocations[stage]])
    return sorted(selected[:target], key=row_key)


def _load_generation_index(repo: Path) -> dict[str, dict[str, Mapping[str, Any]]]:
    index: dict[str, dict[str, Mapping[str, Any]]] = {}
    for seed in SEEDS:
        path = repo / PREVIOUS_ROUTE / f"artifacts/G230/seed{seed}/generations.jsonl.gz"
        for row in _jsonl_gz(path):
            if row.get("context") != FULL_CONTEXT:
                continue
            task_id = str(row["task_id"])
            evidence_by_id = {str(item["evidence_id"]): item for item in row["evidence"]}
            for family in FAMILIES:
                config = f"{family}{seed}"
                run = row["arms"][family]
                index.setdefault(config, {})[task_id] = {
                    "question": row["question"],
                    "evidence_by_id": evidence_by_id,
                    "run": run,
                }
    return index


def _truncate(text: Any, limit: int = 900) -> Any:
    if not isinstance(text, str):
        return text
    return text if len(text) <= limit else text[:limit] + "..."


def enrich_rows(repo: Path, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = _load_generation_index(repo)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        source = index[str(row["config"])][str(row["task_id"])]
        evidence = source["evidence_by_id"]
        citation = row.get("citation")
        gated_citation = row.get("gated_citation")
        enriched.append(
            {
                **dict(row),
                "audit_row_id": hashlib.sha256(row_key(row).encode()).hexdigest()[:16],
                "question": source["question"],
                "citation_text": _truncate(evidence.get(str(citation), {}).get("text")),
                "gated_citation_text": _truncate(evidence.get(str(gated_citation), {}).get("text")),
                "review_instructions": {
                    "allowed_labels": list(FAILURE_ORDER) + ["UNCERTAIN"],
                    "decision_basis": "Use draft, split/routing fields, question, claim, and cited evidence text. Do not infer held-out results.",
                },
            }
        )
    return enriched


def build_outputs(repo: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], str]:
    g100_summary = _json(repo / ROUTE / "artifacts/G100/G100_SUMMARY.json")
    if g100_summary.get("status") != "PASS":
        raise ValueError("G110 requires G100 PASS")
    rows = _jsonl(repo / ROUTE / "artifacts/G100/G100_ATTRIBUTION_ROWS.jsonl")
    if len(rows) != g100_summary["scope"]["claim_level_rows"]:
        raise ValueError("G100 row count differs from summary")
    audit_sample = enrich_rows(
        repo,
        stratified_sample(
            rows,
            target=AUDIT_TARGET,
            namespace="g110-audit-sample",
            min_per_stratum=MIN_PER_STRATUM,
        ),
    )
    double_review = enrich_rows(
        repo,
        stratified_sample(
            audit_sample,
            target=DOUBLE_REVIEW_TARGET,
            namespace="g110-double-review",
            min_per_stratum=5,
        ),
    )
    audit_counts = Counter(row["failure_stage"] for row in audit_sample)
    double_counts = Counter(row["failure_stage"] for row in double_review)
    summary = {
        "schema_version": f"{SCHEMA_PREFIX}-sample-summary-v1",
        "status": "PREPARED / AWAITING DOUBLE REVIEW",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {
            "head": _git(repo, ["rev-parse", "HEAD"]),
            "branch": _git(repo, ["branch", "--show-current"]),
        },
        "source_rows": len(rows),
        "audit_sample_rows": len(audit_sample),
        "double_review_rows": len(double_review),
        "audit_sample_stage_counts": {stage: audit_counts.get(stage, 0) for stage in FAILURE_ORDER},
        "double_review_stage_counts": {stage: double_counts.get(stage, 0) for stage in FAILURE_ORDER},
        "sampling": {
            "seed": 13,
            "target": AUDIT_TARGET,
            "min_per_available_stratum": MIN_PER_STRATUM,
            "double_review_target": DOUBLE_REVIEW_TARGET,
            "double_review_fraction": DOUBLE_REVIEW_TARGET / AUDIT_TARGET,
        },
        "source_files": {
            "g100_summary": _source_pin(repo, ROUTE / "artifacts/G100/G100_SUMMARY.json"),
            "g100_rows": _source_pin(repo, ROUTE / "artifacts/G100/G100_ATTRIBUTION_ROWS.jsonl"),
            "g110_prepare_script": _source_pin(repo, Path("scripts/full_flow_g110_audit_prepare.py")),
        },
        "runtime_boundary": {
            "gold_loaded_at_runtime": False,
            "reference_answers_loaded_at_runtime": False,
            "heldout_loaded": False,
            "training_started": False,
        },
        "next_required": "two independent reviews for G110_DOUBLE_REVIEW_SAMPLE.jsonl, adjudication, then route decision",
    }
    return summary, audit_sample, double_review, render_markdown(summary)


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# G110 Audit Sample Preparation",
        "",
        "**日期：** 2026-08-18",
        f"**状态：** `{summary['status']}`",
        "",
        "## Samples",
        "",
        f"- Source G100 rows: `{summary['source_rows']}`",
        f"- Fixed audit sample: `{summary['audit_sample_rows']}`",
        f"- Double-review subset: `{summary['double_review_rows']}`",
        "",
        "## Double-Review Stage Counts",
        "",
        "| Stage | Rows |",
        "|---|---:|",
    ]
    for stage, count in summary["double_review_stage_counts"].items():
        lines.append(f"| {stage} | {count} |")
    lines.extend(
        [
            "",
            "G110 is not complete until two independent review files and adjudication are saved.",
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
    out_dir = args.out_dir or (repo / ROUTE / "artifacts/G110")
    summary, audit_sample, double_review, markdown = build_outputs(repo)
    _write_json(out_dir / "G110_SAMPLE_SUMMARY.json", summary)
    _write_jsonl(out_dir / "G110_AUDIT_SAMPLE.jsonl", audit_sample)
    _write_jsonl(out_dir / "G110_DOUBLE_REVIEW_SAMPLE.jsonl", double_review)
    _write_text(repo / ROUTE / "G110_AUDIT_SAMPLE_REPORT.md", markdown)
    print(f"written {out_dir / 'G110_SAMPLE_SUMMARY.json'}")
    print(f"written {out_dir / 'G110_AUDIT_SAMPLE.jsonl'}")
    print(f"written {out_dir / 'G110_DOUBLE_REVIEW_SAMPLE.jsonl'}")
    print(f"written {repo / ROUTE / 'G110_AUDIT_SAMPLE_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
