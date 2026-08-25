"""Finalize the G212R fixed sample review packet.

G212M consumes the fixed 100-row G212R sample and records per-row adjudication
decisions before any G300 training.  It does not train, generate utility labels,
or read sealed, held-out, or official dev data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_MANUAL_SAMPLE_ROW = "full-flow-g212-manual-sample-row-v1"
SCHEMA_PREPARE_MANIFEST = "full-flow-g212-prepare-manifest-v1"
SCHEMA_LENGTH_AUDIT = "full-flow-g212-length-audit-v1"
SCHEMA_REVIEW_ROW = "full-flow-g212m-sample-review-row-v1"
SCHEMA_REVIEW_SUMMARY = "full-flow-g212m-sample-review-summary-v1"
SCHEMA_FREEZE_MANIFEST = "full-flow-g212m-freeze-readiness-manifest-v1"
SCHEMA_ORDERED_IDS = "full-flow-g212m-ordered-review-ids-v1"

PASS = "PASS"
FAIL = "FAIL"
UNCERTAIN = "UNCERTAIN"
PENDING = "PENDING"
VALID_DECISIONS = frozenset({PASS, FAIL, UNCERTAIN})
UNKNOWN_ANSWER = "I don't know."


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


def _require_empty(path: Path, label: str) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"{label} output directory must be absent or empty")


def _parse_index_reason(items: Sequence[str]) -> dict[int, str]:
    parsed: dict[int, str] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"decision override must be INDEX:REASON, got {item!r}")
        index_text, reason = item.split(":", 1)
        index = int(index_text)
        reason = reason.strip()
        if index <= 0 or not reason:
            raise ValueError(f"invalid decision override: {item!r}")
        if index in parsed:
            raise ValueError(f"duplicate decision override for sample_index={index}")
        parsed[index] = reason
    return parsed


def _structural_findings(row: Mapping[str, Any]) -> list[str]:
    findings: list[str] = []
    if row.get("schema_version") != SCHEMA_MANUAL_SAMPLE_ROW:
        findings.append("unexpected_sample_schema")
    if row.get("review_decision") != PENDING:
        findings.append("input_decision_not_pending")
    evidence = row.get("review_evidence")
    citations = row.get("target_citation_indices")
    cited_ids = row.get("target_cited_evidence_ids")
    if not isinstance(evidence, list):
        findings.append("review_evidence_not_list")
        evidence = []
    if not isinstance(citations, list):
        findings.append("target_citation_indices_not_list")
        citations = []
    if not isinstance(cited_ids, list):
        findings.append("target_cited_evidence_ids_not_list")
        cited_ids = []
    evidence_ids = [
        str(item.get("evidence_id", ""))
        for item in evidence
        if isinstance(item, Mapping)
    ]
    expected_cited_ids: list[str] = []
    for citation in citations:
        if not isinstance(citation, int) or citation < 1 or citation > len(evidence_ids):
            findings.append("citation_index_out_of_range")
            continue
        expected_cited_ids.append(evidence_ids[citation - 1])
    if expected_cited_ids != [str(item) for item in cited_ids]:
        findings.append("citation_to_evidence_remap_mismatch")
    answerable = bool(row.get("answerable"))
    target = str(row.get("target", ""))
    support_ids = {str(item) for item in row.get("support_evidence_ids", [])}
    if answerable:
        if not citations:
            findings.append("answerable_target_has_no_citation")
        if set(str(item) for item in cited_ids) != support_ids:
            findings.append("answerable_cited_support_ids_mismatch")
        if not target.strip() or target.strip() == UNKNOWN_ANSWER:
            findings.append("answerable_target_is_blank_or_idk")
    else:
        if target.strip() != UNKNOWN_ANSWER:
            findings.append("unsupported_target_not_idk")
        if support_ids:
            findings.append("unsupported_row_retains_support_ids")
    return sorted(set(findings))


def _default_pass_reason(row: Mapping[str, Any]) -> str:
    if bool(row.get("answerable")):
        return "Target is supported by cited evidence, citations remap to support evidence, and the answer chain is preserved for this sampled row."
    return "Unsupported target is I don't know and the visible evidence packet does not expose the removed support for this sampled row."


def adjudicate(
    *,
    manual_sample_path: Path,
    prepare_manifest_path: Path,
    length_audit_path: Path,
    output_dir: Path,
    fail: Mapping[int, str] | None = None,
    uncertain: Mapping[int, str] | None = None,
    reviewer_label: str = "project_operator",
) -> dict[str, object]:
    _require_empty(output_dir, "G212M")
    started = time.time()
    fail = dict(fail or {})
    uncertain = dict(uncertain or {})
    overlap = set(fail) & set(uncertain)
    if overlap:
        raise ValueError(f"sample indices cannot be both fail and uncertain: {sorted(overlap)}")
    prepare_manifest = _json(prepare_manifest_path)
    if prepare_manifest.get("schema_version") != SCHEMA_PREPARE_MANIFEST:
        raise ValueError("G212M expects a G212 prepare manifest")
    if prepare_manifest.get("length_audit_pass") is not True:
        raise ValueError("G212M requires length_audit_pass=true")
    if prepare_manifest.get("training_started") is not False:
        raise ValueError("G212M input must record training_started=false")
    if prepare_manifest.get("utility_labels_started") is not False:
        raise ValueError("G212M input must record utility_labels_started=false")
    if prepare_manifest.get("sealed_or_heldout_read") is not False or prepare_manifest.get("dev_read") is not False:
        raise ValueError("G212M input must not have read dev/sealed/held-out data")
    manifest_sample_sha = str(
        ((prepare_manifest.get("artifacts") or {}).get("manual_sample") or {}).get("sha256", "")
    )
    if manifest_sample_sha != _sha256(manual_sample_path):
        raise ValueError("manual sample hash differs from G212 prepare manifest")
    length_audit = _json(length_audit_path)
    if length_audit.get("schema_version") != SCHEMA_LENGTH_AUDIT:
        raise ValueError("G212M expects a G212 length audit")
    if length_audit.get("status") != PASS or int(length_audit.get("over_max_length", -1)) != 0:
        raise ValueError("G212M requires a passing length audit")

    sample_rows = _jsonl(manual_sample_path)
    if len(sample_rows) != int(prepare_manifest.get("sample_rows", -1)):
        raise ValueError("manual sample row count differs from prepare manifest")
    seen_indices: set[int] = set()
    review_rows: list[dict[str, object]] = []
    forced_structural_failures = 0
    for row in sample_rows:
        sample_index = int(row.get("sample_index", -1))
        if sample_index <= 0 or sample_index in seen_indices:
            raise ValueError(f"invalid or duplicate sample_index={sample_index}")
        seen_indices.add(sample_index)
        findings = _structural_findings(row)
        if findings:
            decision = FAIL
            reason = "Structural review checks failed: " + ", ".join(findings)
            forced_structural_failures += 1
        elif sample_index in fail:
            decision = FAIL
            reason = fail[sample_index]
        elif sample_index in uncertain:
            decision = UNCERTAIN
            reason = uncertain[sample_index]
        else:
            decision = PASS
            reason = _default_pass_reason(row)
        review_rows.append(
            {
                "schema_version": SCHEMA_REVIEW_ROW,
                "audit_id": str(row["audit_id"]),
                "sample_index": sample_index,
                "sample_seed": str(row["sample_seed"]),
                "sample_stratum": str(row["sample_stratum"]),
                "case_id": str(row["case_id"]),
                "dataset": str(row["dataset"]),
                "role": str(row["role"]),
                "target_kind": str(row["target_kind"]),
                "answerable": bool(row["answerable"]),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "target": str(row["target"]),
                "review_decision": decision,
                "reviewer": reviewer_label,
                "review_reason": reason,
                "structural_findings": findings,
            }
        )
    missing = (set(fail) | set(uncertain)) - seen_indices
    if missing:
        raise ValueError(f"decision overrides reference absent sample indices: {sorted(missing)}")

    counts = Counter(str(row["review_decision"]) for row in review_rows)
    by_stratum: dict[str, dict[str, int]] = {}
    for row in review_rows:
        stratum = str(row["sample_stratum"])
        by_stratum.setdefault(stratum, {PASS: 0, FAIL: 0, UNCERTAIN: 0})
        by_stratum[stratum][str(row["review_decision"])] += 1
    fail_rows = [
        {
            "sample_index": row["sample_index"],
            "audit_id": row["audit_id"],
            "case_id": row["case_id"],
            "sample_stratum": row["sample_stratum"],
            "review_decision": row["review_decision"],
            "review_reason": row["review_reason"],
        }
        for row in review_rows
        if row["review_decision"] != PASS
    ]
    status = "PASS" if counts[FAIL] == 0 and counts[UNCERTAIN] == 0 else "FAIL_SAMPLE_REVIEW"
    freeze_ready = status == "PASS"

    review_rows_path = output_dir / "sample_review_rows.jsonl"
    review_summary_path = output_dir / "sample_review_summary.json"
    ordered_ids_path = output_dir / "ordered_review_ids.json"
    freeze_manifest_path = output_dir / "freeze_readiness_manifest.json"
    _write_jsonl(review_rows_path, review_rows)
    summary: dict[str, object] = {
        "schema_version": SCHEMA_REVIEW_SUMMARY,
        "status": status,
        "sample_rows": len(review_rows),
        "pass": counts[PASS],
        "fail": counts[FAIL],
        "uncertain": counts[UNCERTAIN],
        "forced_structural_failures": forced_structural_failures,
        "by_stratum": dict(sorted(by_stratum.items())),
        "fail_or_uncertain_rows": fail_rows,
        "training_started": False,
        "utility_labels_started": False,
        "sealed_or_heldout_read": False,
        "dev_read": False,
    }
    _write_json(review_summary_path, summary)
    ordered_ids = {
        "schema_version": SCHEMA_ORDERED_IDS,
        "sample_seed": str(prepare_manifest.get("sample_seed", "")),
        "ordered_audit_ids": [str(row["audit_id"]) for row in review_rows],
        "ordered_case_ids": [str(row["case_id"]) for row in review_rows],
    }
    _write_json(ordered_ids_path, ordered_ids)
    freeze_manifest: dict[str, object] = {
        "schema_version": SCHEMA_FREEZE_MANIFEST,
        "status": "FREEZE_READY" if freeze_ready else "NOT_FREEZE_READY_SAMPLE_REVIEW_FAILED",
        "freeze_ready": freeze_ready,
        "g300_unlocked": freeze_ready,
        "training_started": False,
        "utility_labels_started": False,
        "sealed_or_heldout_read": False,
        "dev_read": False,
        "sample_review_status": status,
        "length_audit_status": "PASS",
        "input_sha256": {
            "prepare_manifest": _sha256(prepare_manifest_path),
            "length_audit": _sha256(length_audit_path),
            "manual_sample": _sha256(manual_sample_path),
        },
        "artifacts": {
            "sample_review_rows": {"path": str(review_rows_path), "sha256": _sha256(review_rows_path)},
            "sample_review_summary": {"path": str(review_summary_path), "sha256": _sha256(review_summary_path)},
            "ordered_review_ids": {"path": str(ordered_ids_path), "sha256": _sha256(ordered_ids_path)},
        },
        "elapsed_seconds": time.time() - started,
    }
    _write_json(freeze_manifest_path, freeze_manifest)
    return freeze_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-sample", required=True, type=Path)
    parser.add_argument("--prepare-manifest", required=True, type=Path)
    parser.add_argument("--length-audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fail", action="append", default=[], help="Sample failure as INDEX:REASON")
    parser.add_argument("--uncertain", action="append", default=[], help="Uncertain sample as INDEX:REASON")
    parser.add_argument("--reviewer-label", default="project_operator")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = adjudicate(
        manual_sample_path=args.manual_sample,
        prepare_manifest_path=args.prepare_manifest,
        length_audit_path=args.length_audit,
        output_dir=args.output_dir.resolve(),
        fail=_parse_index_reason(args.fail),
        uncertain=_parse_index_reason(args.uncertain),
        reviewer_label=args.reviewer_label,
    )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
