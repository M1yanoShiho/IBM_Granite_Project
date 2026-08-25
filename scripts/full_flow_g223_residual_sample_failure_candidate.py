"""Apply the G223 residual sample-failure candidate.

G223 consumes the G221 targeted filtered bundle, the G212M5 sample adjudication
rows, and the G222 continuation amendment.  It does not rewrite targets.  It
quarantines the residual failed sample cases and unsupported train counterparts
for failed 2Wiki train answerable groups.

This stage can only produce a limited controlled-continuation candidate.  It is
not a clean freeze, not a held-out result, and not a strong statistical claim.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from full_flow_g216_sample_review_repair import (
    SCHEMA_CASE,
    SCHEMA_G212M_REVIEW_ROW,
    SCHEMA_G212M_REVIEW_SUMMARY,
    TRAIN_ROLE,
    VALIDATION_ROLE,
    _case_counts,
    _example_updates,
    _failed_review_rows,
    _json,
    _jsonl,
    _removal_keys,
    _require_empty,
    _sha256,
    _split_leakage,
    _write_json,
    _write_jsonl,
)

SCHEMA_INPUT_G221_MANIFEST = "full-flow-g221-targeted-sample-failure-repair-manifest-v1"
SCHEMA_G222_AMENDMENT = "full-flow-g222-residual-continuation-amendment-v1"
SCHEMA_G223_MANIFEST = "full-flow-g223-residual-sample-failure-candidate-manifest-v1"
SCHEMA_ORDERED_IDS = "full-flow-g223-ordered-ids-v1"
DEFAULT_TWOWIKI_MODELVAL_FLOOR = 95


def _load_bundle(
    *,
    manifest_path: Path,
    train_cases_path: Path,
    validation_cases_path: Path,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    manifest = _json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_INPUT_G221_MANIFEST:
        raise ValueError("G223 expects the G221 targeted sample-failure manifest")
    if manifest.get("status") != "PRE_MANUAL_PASS":
        raise ValueError("G223 requires a PRE_MANUAL_PASS input bundle")
    if manifest.get("training_started") is not False:
        raise ValueError("G223 input must record training_started=false")
    if manifest.get("sealed_or_heldout_read") is not False or manifest.get("dev_read") is not False:
        raise ValueError("G223 input must not have read dev/sealed/held-out data")
    if str(manifest.get("train_cases_sha256", "")) != _sha256(train_cases_path):
        raise ValueError("train cases hash differs from G221 manifest")
    if str(manifest.get("validation_cases_sha256", "")) != _sha256(validation_cases_path):
        raise ValueError("validation cases hash differs from G221 manifest")
    train_rows = _jsonl(train_cases_path)
    validation_rows = _jsonl(validation_cases_path)
    seen: set[str] = set()
    for row in [*train_rows, *validation_rows]:
        case_id = str(row.get("case_id", ""))
        if not case_id or case_id in seen or row.get("schema_version") != SCHEMA_CASE:
            raise ValueError(f"invalid or duplicate case: {case_id!r}")
        seen.add(case_id)
    return manifest, train_rows, validation_rows


def _load_amendment(path: Path) -> Mapping[str, Any]:
    amendment = _json(path)
    if amendment.get("schema_version") != SCHEMA_G222_AMENDMENT:
        raise ValueError("G223 expects the G222 residual continuation amendment")
    decision = amendment.get("decision")
    if not isinstance(decision, Mapping):
        raise ValueError("G222 amendment lacks decision object")
    if decision.get("controlled_continuation_allowed") is not True:
        raise ValueError("G222 amendment must allow controlled continuation")
    if decision.get("direct_g300_training_allowed") is not False:
        raise ValueError("G222 amendment must not directly allow G300")
    return amendment


def candidate(
    *,
    manifest_path: Path,
    train_cases_path: Path,
    validation_cases_path: Path,
    review_rows_path: Path,
    review_summary_path: Path,
    amendment_path: Path,
    output_dir: Path,
    min_niah_train_groups: int = 400,
    min_niah_modelval_groups: int = 100,
    min_twowiki_train_groups: int = 400,
    min_twowiki_modelval_groups: int = DEFAULT_TWOWIKI_MODELVAL_FLOOR,
    unsupported_ratio_min: float = 0.10,
    unsupported_ratio_max: float = 0.15,
) -> dict[str, object]:
    _require_empty(output_dir, "G223 residual sample-failure candidate")
    input_manifest, train_rows, validation_rows = _load_bundle(
        manifest_path=manifest_path,
        train_cases_path=train_cases_path,
        validation_cases_path=validation_cases_path,
    )
    amendment = _load_amendment(amendment_path)
    failed_review_rows = _failed_review_rows(
        review_rows_path=review_rows_path,
        review_summary_path=review_summary_path,
    )
    remove_train, remove_validation = _removal_keys(failed_review_rows, train_rows, validation_rows)
    repaired_train = [row for row in train_rows if str(row["case_id"]) not in remove_train]
    repaired_validation = [
        row for row in validation_rows if str(row["case_id"]) not in remove_validation
    ]
    removed_cases = [
        row for row in [*train_rows, *validation_rows]
        if str(row["case_id"]) in remove_train or str(row["case_id"]) in remove_validation
    ]
    if len(removed_cases) != len(remove_train) + len(remove_validation):
        raise ValueError("G223 removal keys did not match the expected number of rows")

    answerable_train = [row for row in repaired_train if row.get("answerable")]
    unsupported_train = [row for row in repaired_train if not row.get("answerable")]
    answerable_updates = _example_updates(answerable_train)
    unsupported_updates = _example_updates(unsupported_train)
    unsupported_ratio = unsupported_updates / (answerable_updates + unsupported_updates)
    leakage = _split_leakage(repaired_train, repaired_validation)
    counts = _case_counts([*repaired_train, *repaired_validation])

    niah_train = counts.get("niah", {}).get(f"{TRAIN_ROLE}_answerable_groups", 0)
    niah_validation = counts.get("niah", {}).get(f"{VALIDATION_ROLE}_answerable_groups", 0)
    twiki_train = counts.get("2wiki", {}).get(f"{TRAIN_ROLE}_answerable_groups", 0)
    twiki_validation = counts.get("2wiki", {}).get(f"{VALIDATION_ROLE}_answerable_groups", 0)
    gates = {
        "niah_train_groups": niah_train >= min_niah_train_groups,
        "niah_new_modelval_groups": niah_validation >= min_niah_modelval_groups,
        "twowiki_train_groups": twiki_train >= min_twowiki_train_groups,
        "twowiki_modelval_groups": twiki_validation >= min_twowiki_modelval_groups,
        "unsupported_update_ratio": unsupported_ratio_min <= unsupported_ratio <= unsupported_ratio_max,
        "split_group_overlap_zero": leakage["group_overlap"] == 0,
        "split_component_overlap_zero": leakage["component_overlap"] == 0,
    }
    continuation_ready = all(gates.values())

    train_path = output_dir / "train_cases.jsonl"
    validation_path = output_dir / "validation_cases.jsonl"
    ordered_path = output_dir / "ordered_ids.json"
    _write_jsonl(train_path, repaired_train)
    _write_jsonl(validation_path, repaired_validation)
    _write_json(
        ordered_path,
        {
            "schema_version": SCHEMA_ORDERED_IDS,
            "train_case_ids": [str(row["case_id"]) for row in repaired_train],
            "validation_case_ids": [str(row["case_id"]) for row in repaired_validation],
        },
    )
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_G223_MANIFEST,
        "status": "CONTROLLED_CONTINUATION_READY" if continuation_ready else "FAIL",
        "repair_stage": "G223",
        "repair_reason": "residual_quarantine_g212m5_failed_sample_cases",
        "clean_freeze_ready": False,
        "controlled_continuation_ready": continuation_ready,
        "g300_unlocked": continuation_ready,
        "g300_entry_mode": "controlled_continuation_limited_internal_screen"
        if continuation_ready
        else "blocked",
        "length_rerun_required": False,
        "length_rerun_requirement_reason": "deletion_only_no_target_rewrite",
        "target_rewrite_started": False,
        "training_started": False,
        "utility_labels_started": False,
        "sealed_or_heldout_read": False,
        "dev_read": False,
        "source_manifest_schema_version": str(input_manifest.get("schema_version", "")),
        "g222_status": str(amendment.get("status", "")),
        "method_amendment": {
            "clean_freeze_ready": False,
            "controlled_continuation_ready": continuation_ready,
            "twowiki_modelval_floor": min_twowiki_modelval_groups,
            "previous_floor": 96,
            "original_floor": 100,
            "reason": "G222 permits controlled continuation after G212M5 97/100, provided residual failed rows are quarantined and the reduced 2Wiki model-val screen is reported as a limitation.",
            "not_a_true_threshold_change": True,
            "not_a_heldout_result": True,
            "not_a_strong_statistical_claim": True,
        },
        "excluded_sample_review_rows": [
            {
                "sample_index": int(row.get("sample_index", -1)),
                "case_id": str(row.get("case_id", "")),
                "sample_stratum": str(row.get("sample_stratum", "")),
                "review_decision": str(row.get("review_decision", "")),
                "review_reason": str(row.get("review_reason", "")),
            }
            for row in failed_review_rows
        ],
        "excluded_cases": [
            {
                "case_id": str(row.get("case_id", "")),
                "dataset": str(row.get("dataset", "")),
                "role": str(row.get("role", "")),
                "target_kind": str(row.get("target_kind", "")),
                "answerable": bool(row.get("answerable")),
                "group_id": str(row.get("group_id", "")),
                "variant_count": len(row.get("variants", {}))
                if isinstance(row.get("variants"), Mapping)
                else 0,
            }
            for row in removed_cases
        ],
        "excluded": {
            "sample_review_rows": len(failed_review_rows),
            "train_cases": len(remove_train),
            "validation_cases": len(remove_validation),
            "total_cases": len(removed_cases),
        },
        "counts": counts,
        "answerable_train_updates": answerable_updates,
        "unsupported_updates": unsupported_updates,
        "unsupported_update_ratio": unsupported_ratio,
        "split_leakage": leakage,
        "gates": gates,
        "gate_parameters": {
            "min_niah_train_groups": min_niah_train_groups,
            "min_niah_modelval_groups": min_niah_modelval_groups,
            "min_twowiki_train_groups": min_twowiki_train_groups,
            "min_twowiki_modelval_groups": min_twowiki_modelval_groups,
            "unsupported_ratio_min": unsupported_ratio_min,
            "unsupported_ratio_max": unsupported_ratio_max,
        },
        "train_cases": len(repaired_train),
        "validation_cases": len(repaired_validation),
        "train_cases_sha256": _sha256(train_path),
        "validation_cases_sha256": _sha256(validation_path),
        "ordered_ids_sha256": _sha256(ordered_path),
        "input_sha256": {
            "source_manifest": _sha256(manifest_path),
            "source_train_cases": _sha256(train_cases_path),
            "source_validation_cases": _sha256(validation_cases_path),
            "g212m5_review_rows": _sha256(review_rows_path),
            "g212m5_review_summary": _sha256(review_summary_path),
            "g222_amendment": _sha256(amendment_path),
        },
        "risks_and_limits": {
            "modelval_screen": "2Wiki model-val screen is smaller than the earlier internal floor and must be reported as a limitation.",
            "final_claims": "Final claims still require SystemF freeze and separately authorized held-out evaluation.",
            "clean_freeze": "This is not clean 100/100 freeze readiness.",
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--train-cases", required=True, type=Path)
    parser.add_argument("--validation-cases", required=True, type=Path)
    parser.add_argument("--review-rows", required=True, type=Path)
    parser.add_argument("--review-summary", required=True, type=Path)
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = candidate(
        manifest_path=args.manifest,
        train_cases_path=args.train_cases,
        validation_cases_path=args.validation_cases,
        review_rows_path=args.review_rows,
        review_summary_path=args.review_summary,
        amendment_path=args.amendment,
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
