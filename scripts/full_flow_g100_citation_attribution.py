"""G100 automated citation regression attribution from frozen G230 traces.

The output is a claim-level attribution candidate table for G110 audit.  It
uses only G230 archived development rows.  MiniCheck task-level judgments are
post-generation signals; gold/reference answers are not loaded here.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROUTE = Path("docs/full-flow/experiments/03_generator_grounding_repair_2026-08-18")
PREVIOUS_ROUTE = Path("docs/full-flow/experiments/02_generator_selector_alignment_2026-08-15")
SCHEMA_PREFIX = "full-flow-g100"
SEEDS = (13, 42, 73)
CONFIGS = tuple(f"{family}{seed}" for seed in SEEDS for family in ("GC", "GM"))
FULL_CONTEXT = "K_topk"
FAILURE_ORDER = (
    "UNSUPPORTED_DRAFT_CLAIM",
    "DRAFT_CITATION_MISSING_OR_WRONG",
    "SPLITTER_BOUNDARY_OR_REWRITE",
    "TRUE_ROUTING_OR_ATTACHMENT",
    "EVALUATOR_DISAGREEMENT",
    "NO_REGRESSION",
)


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
        raise FileNotFoundError(f"missing G100 source file: {path}")
    return {
        "path": str(path.relative_to(repo)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _supported(config: Mapping[str, Any]) -> bool:
    return config.get("citation_precision") == 1.0 and config.get("citation_recall") == 1.0


def _answered(config: Mapping[str, Any]) -> bool:
    return config.get("final_empty") == 0.0 and config.get("runtime_error") == 0.0


def _load_candidate_runs(repo: Path) -> dict[str, dict[str, Mapping[str, Any]]]:
    runs: dict[str, dict[str, Mapping[str, Any]]] = {config: {} for config in CONFIGS}
    for seed in SEEDS:
        path = repo / PREVIOUS_ROUTE / f"artifacts/G230/seed{seed}/generations.jsonl.gz"
        for row in _jsonl_gz(path):
            if row.get("context") != FULL_CONTEXT:
                continue
            task_id = str(row["task_id"])
            for family in ("GC", "GM"):
                runs[f"{family}{seed}"][task_id] = row["arms"][family]
    return runs


def _load_case_maps(repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    answer_rows = {
        str(row["task_id"]): row
        for row in _jsonl(repo / PREVIOUS_ROUTE / "artifacts/G230/score/scored_cases.jsonl")
        if row.get("context") == FULL_CONTEXT
    }
    citation_rows = {
        str(row["task_id"]): row
        for row in _jsonl(repo / PREVIOUS_ROUTE / "artifacts/G230/citation/citation_cases.jsonl")
        if row.get("context") == FULL_CONTEXT
    }
    if set(answer_rows) != set(citation_rows):
        raise ValueError("G100 answer and citation full-task IDs differ")
    return answer_rows, citation_rows


def classify_claim(
    *,
    run_row: Mapping[str, Any],
    claim: Mapping[str, Any],
    candidate_supported: bool,
) -> tuple[str, str]:
    trace = run_row.get("trace")
    draft = trace.get("draft", {}) if isinstance(trace, Mapping) else {}
    splitter = draft.get("splitter", {}) if isinstance(draft, Mapping) else {}
    declared_indices = claim.get("declared_indices")
    citation = claim.get("citation")
    final_disposition = str(claim.get("final_disposition", ""))
    routing_outcome = str(claim.get("routing_outcome", ""))
    gated_outcome = str(claim.get("gated_outcome", ""))

    if final_disposition == "unsupported" or routing_outcome == "unsupported":
        return (
            "UNSUPPORTED_DRAFT_CLAIM",
            "TRUE/routing found the claim unsupported; G110 must verify whether the draft claim itself is unsupported.",
        )
    if not isinstance(declared_indices, Sequence) or len(declared_indices) == 0:
        return (
            "DRAFT_CITATION_MISSING_OR_WRONG",
            "No declared citation index was available for this final claim.",
        )
    if claim.get("declared_verified") is False:
        return (
            "DRAFT_CITATION_MISSING_OR_WRONG",
            "The draft-declared citation did not verify before scan/routing rescue.",
        )
    if splitter.get("status") != "structured" or claim.get("degraded_splitter_fallback") is True:
        return (
            "SPLITTER_BOUNDARY_OR_REWRITE",
            "The splitter was degraded or not structured for this draft.",
        )
    if claim.get("faithful_to_answer") is False:
        return (
            "SPLITTER_BOUNDARY_OR_REWRITE",
            "The split claim was marked unfaithful to the draft answer.",
        )
    if not citation or gated_outcome != "verified" or final_disposition != "verified":
        return (
            "TRUE_ROUTING_OR_ATTACHMENT",
            "The final routed claim was not a verified citation attachment.",
        )
    if not candidate_supported:
        return (
            "EVALUATOR_DISAGREEMENT",
            "TRUE/routing produced a verified attachment but MiniCheck did not mark the task fully supported.",
        )
    return ("NO_REGRESSION", "MiniCheck marked the task fully supported.")


def _claim_rows_for_regression(
    *,
    task_id: str,
    query_id: str,
    component_id: str,
    config: str,
    run_row: Mapping[str, Any],
    baseline_citation: Mapping[str, Any],
    candidate_citation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidate_supported = _supported(candidate_citation)
    claims = run_row.get("trace", {}).get("claims", [])
    if not isinstance(claims, Sequence) or not claims:
        claims = [{"claim_id": "NO_TRACE_CLAIM", "final_disposition": "missing_trace"}]
    rows: list[dict[str, Any]] = []
    for claim in claims:
        if not isinstance(claim, Mapping):
            continue
        stage, reason = classify_claim(
            run_row=run_row,
            claim=claim,
            candidate_supported=candidate_supported,
        )
        rows.append(
            {
                "schema_version": f"{SCHEMA_PREFIX}-claim-attribution-row-v1",
                "task_id": task_id,
                "query_id": query_id,
                "component_id": component_id,
                "config": config,
                "claim_id": str(claim.get("claim_id", "")),
                "failure_stage": stage,
                "attribution_reason": reason,
                "baseline_citation_precision": baseline_citation.get("citation_precision"),
                "baseline_citation_recall": baseline_citation.get("citation_recall"),
                "candidate_citation_precision": candidate_citation.get("citation_precision"),
                "candidate_citation_recall": candidate_citation.get("citation_recall"),
                "raw_draft_text": run_row.get("trace", {}).get("draft", {}).get("normalized_draft_text"),
                "claim_text": claim.get("claim_text"),
                "final_sentence": claim.get("final_sentence"),
                "declared_indices": claim.get("declared_indices"),
                "declared_verified": claim.get("declared_verified"),
                "rescued_by_scan": claim.get("rescued_by_scan"),
                "routing_outcome": claim.get("routing_outcome"),
                "gated_outcome": claim.get("gated_outcome"),
                "citation": claim.get("citation"),
                "gated_citation": claim.get("gated_citation"),
                "g110_manual_review_required": True,
            }
        )
    return rows


def build_outputs(repo: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    g010 = _json(repo / ROUTE / "artifacts/G010/G010_POWER_SCOPE.json")
    if g010.get("status") != "PASS":
        raise ValueError("G100 requires G010 PASS")
    answer_rows, citation_rows = _load_case_maps(repo)
    runs = _load_candidate_runs(repo)
    common_answered: list[str] = []
    for task_id, answer_row in answer_rows.items():
        configs = answer_row["configs"]
        if _answered(configs["G0"]) and all(_answered(configs[name]) for name in CONFIGS):
            common_answered.append(task_id)
    common_answered.sort()

    rows: list[dict[str, Any]] = []
    task_regressions: Counter[str] = Counter()
    for task_id in common_answered:
        citation_row = citation_rows[task_id]
        baseline_citation = citation_row["configs"]["G0"]
        if not _supported(baseline_citation):
            continue
        for config in CONFIGS:
            candidate_citation = citation_row["configs"][config]
            if _supported(candidate_citation):
                continue
            task_regressions[config] += 1
            rows.extend(
                _claim_rows_for_regression(
                    task_id=task_id,
                    query_id=str(citation_row["query_id"]),
                    component_id=str(citation_row["component_id"]),
                    config=config,
                    run_row=runs[config][task_id],
                    baseline_citation=baseline_citation,
                    candidate_citation=candidate_citation,
                )
            )

    stage_counts = Counter(row["failure_stage"] for row in rows)
    config_counts = Counter(row["config"] for row in rows)
    review_policy = {
        "regression_rows": len(rows),
        "sample_all_if_rows_at_most": 120,
        "fixed_manual_review_sample": "G110 samples all rows if <=120, otherwise deterministic stratified 120 with at least 20 per non-empty key stratum",
        "independent_blind_double_review_minimum": "20%",
    }
    summary = {
        "schema_version": f"{SCHEMA_PREFIX}-summary-v1",
        "status": "PASS",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {
            "head": _git(repo, ["rev-parse", "HEAD"]),
            "branch": _git(repo, ["branch", "--show-current"]),
        },
        "scope": {
            "context": FULL_CONTEXT,
            "common_answered_tasks": len(common_answered),
            "baseline_supported_filter": "G0 citation_precision == citation_recall == 1",
            "candidate_regression_filter": "candidate citation_precision/recall is not both 1",
            "claim_level_rows": len(rows),
        },
        "failure_order": list(FAILURE_ORDER),
        "failure_stage_counts": {stage: stage_counts.get(stage, 0) for stage in FAILURE_ORDER},
        "config_claim_row_counts": dict(sorted(config_counts.items())),
        "config_task_regression_counts": dict(sorted(task_regressions.items())),
        "review_policy": review_policy,
        "source_files": {
            "g010_power_scope": _source_pin(repo, ROUTE / "artifacts/G010/G010_POWER_SCOPE.json"),
            "g230_scored_cases": _source_pin(
                repo, PREVIOUS_ROUTE / "artifacts/G230/score/scored_cases.jsonl"
            ),
            "g230_citation_cases": _source_pin(
                repo, PREVIOUS_ROUTE / "artifacts/G230/citation/citation_cases.jsonl"
            ),
            "seed13_generations": _source_pin(
                repo, PREVIOUS_ROUTE / "artifacts/G230/seed13/generations.jsonl.gz"
            ),
            "seed42_generations": _source_pin(
                repo, PREVIOUS_ROUTE / "artifacts/G230/seed42/generations.jsonl.gz"
            ),
            "seed73_generations": _source_pin(
                repo, PREVIOUS_ROUTE / "artifacts/G230/seed73/generations.jsonl.gz"
            ),
            "g100_script": _source_pin(repo, Path("scripts/full_flow_g100_citation_attribution.py")),
        },
        "runtime_boundary": {
            "gold_loaded_at_runtime": False,
            "reference_answers_loaded_at_runtime": False,
            "heldout_loaded": False,
            "training_started": False,
        },
        "interpretation": {
            "automated_rows_are_not_final_route_decision": True,
            "g110_required_before_repair": True,
        },
    }
    return summary, rows, render_markdown(summary)


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# G100 Citation Attribution",
        "",
        "**日期：** 2026-08-18",
        f"**状态：** `{summary['status']}`",
        "",
        "## Scope",
        "",
        f"- Common answered full TopK tasks: `{summary['scope']['common_answered_tasks']}`",
        f"- Claim-level regression rows: `{summary['scope']['claim_level_rows']}`",
        "- Rows are automated attribution candidates for G110 audit, not final repair decisions.",
        "",
        "## Failure Stage Counts",
        "",
        "| Stage | Rows |",
        "|---|---:|",
    ]
    for stage in FAILURE_ORDER:
        lines.append(f"| {stage} | {summary['failure_stage_counts'].get(stage, 0)} |")
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "G110 must audit these rows before activating splitter or routing repair. No training has been started.",
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
    out_dir = args.out_dir or (repo / ROUTE / "artifacts/G100")
    summary, rows, markdown = build_outputs(repo)
    _write_json(out_dir / "G100_SUMMARY.json", summary)
    _write_jsonl(out_dir / "G100_ATTRIBUTION_ROWS.jsonl", rows)
    _write_text(repo / ROUTE / "G100_ATTRIBUTION_REPORT.md", markdown)
    print(f"written {out_dir / 'G100_SUMMARY.json'}")
    print(f"written {out_dir / 'G100_ATTRIBUTION_ROWS.jsonl'}")
    print(f"written {repo / ROUTE / 'G100_ATTRIBUTION_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
