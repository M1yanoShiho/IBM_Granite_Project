"""Apply the G216 controlled sample-review repair.

G216 consumes the G214 pre-manual bundle and the G212M sample-review rows.  It
removes cases that failed sample review, including unsupported counterparts for
failed 2Wiki train answerable groups, then rewrites a revised pre-manual bundle.

This stage does not train, generate utility labels, change TRUE thresholds, or
read sealed, held-out, or official dev data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_CASE = "full-flow-g200-case-v2"
SCHEMA_G214_MANIFEST = "full-flow-g214-length-repair-manifest-v1"
SCHEMA_G216_MANIFEST = "full-flow-g216-sample-review-repair-manifest-v1"
SCHEMA_G212M_REVIEW_ROW = "full-flow-g212m-sample-review-row-v1"
SCHEMA_G212M_REVIEW_SUMMARY = "full-flow-g212m-sample-review-summary-v1"
TRAIN_ROLE = "train-fit"
VALIDATION_ROLE = "train-modelval"
PASS = "PASS"


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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_empty(path: Path, label: str) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"{label} output directory must be absent or empty")


def _load_bundle(
    *,
    manifest_path: Path,
    train_cases_path: Path,
    validation_cases_path: Path,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    manifest = _json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_G214_MANIFEST:
        raise ValueError("G216 expects the G214 repair manifest")
    if manifest.get("status") != "PRE_MANUAL_PASS":
        raise ValueError("G216 requires a PRE_MANUAL_PASS input bundle")
    if manifest.get("training_started") is not False:
        raise ValueError("G216 input must record training_started=false")
    if manifest.get("sealed_or_heldout_read") is not False or manifest.get("dev_read") is not False:
        raise ValueError("G216 input must not have read dev/sealed/held-out data")
    if str(manifest.get("train_cases_sha256", "")) != _sha256(train_cases_path):
        raise ValueError("train cases hash differs from G214 manifest")
    if str(manifest.get("validation_cases_sha256", "")) != _sha256(validation_cases_path):
        raise ValueError("validation cases hash differs from G214 manifest")
    train_rows = _jsonl(train_cases_path)
    validation_rows = _jsonl(validation_cases_path)
    seen: set[str] = set()
    for row in [*train_rows, *validation_rows]:
        case_id = str(row.get("case_id", ""))
        if not case_id or case_id in seen or row.get("schema_version") != SCHEMA_CASE:
            raise ValueError(f"invalid or duplicate case: {case_id!r}")
        seen.add(case_id)
    return manifest, train_rows, validation_rows


def _failed_review_rows(
    *,
    review_rows_path: Path,
    review_summary_path: Path,
) -> list[Mapping[str, Any]]:
    summary = _json(review_summary_path)
    if summary.get("schema_version") != SCHEMA_G212M_REVIEW_SUMMARY:
        raise ValueError("G216 expects a G212M review summary")
    if summary.get("status") != "FAIL_SAMPLE_REVIEW":
        raise ValueError("G216 expects a failing G212M sample review")
    rows = _jsonl(review_rows_path)
    failed = []
    for row in rows:
        if row.get("schema_version") != SCHEMA_G212M_REVIEW_ROW:
            raise ValueError(f"unexpected G212M review row schema: {row.get('sample_index')}")
        if str(row.get("review_decision", "")) != PASS:
            failed.append(row)
    if len(failed) != int(summary.get("fail", -1)) + int(summary.get("uncertain", -1)):
        raise ValueError("G212M failed/uncertain count differs from review rows")
    if not failed:
        raise ValueError("G216 requires at least one failed or uncertain review row")
    return failed


def _case_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    seen: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        dataset = str(row.get("dataset", ""))
        role = str(row.get("role", ""))
        kind = "answerable" if row.get("answerable") else "unsupported"
        seen[(dataset, role, kind)].add(str(row.get("group_id", "")))
    for (dataset, role, kind), groups in seen.items():
        output.setdefault(dataset, {})[f"{role}_{kind}_groups"] = len(groups)
    return output


def _example_updates(rows: Sequence[Mapping[str, Any]]) -> int:
    total = 0
    for row in rows:
        variants = row.get("variants")
        if not isinstance(variants, Mapping):
            raise ValueError(f"case lacks variants: {row.get('case_id')}")
        total += len(variants)
    return total


def _split_leakage(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    train_groups = {(str(row.get("dataset")), str(row.get("group_id"))) for row in train_rows}
    validation_groups = {
        (str(row.get("dataset")), str(row.get("group_id"))) for row in validation_rows
    }
    train_components = {
        (str(row.get("dataset")), str(row.get("component_id"))) for row in train_rows
    }
    validation_components = {
        (str(row.get("dataset")), str(row.get("component_id"))) for row in validation_rows
    }
    return {
        "group_overlap": len(train_groups & validation_groups),
        "component_overlap": len(train_components & validation_components),
    }


def _removal_keys(
    failed_rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> tuple[set[str], set[str]]:
    train_by_case = {str(row["case_id"]): row for row in train_rows}
    validation_by_case = {str(row["case_id"]): row for row in validation_rows}
    remove_train_case_ids: set[str] = set()
    remove_validation_case_ids: set[str] = set()
    for row in failed_rows:
        case_id = str(row.get("case_id", ""))
        if case_id in train_by_case:
            train_case = train_by_case[case_id]
            remove_train_case_ids.add(case_id)
            if str(train_case.get("dataset", "")) == "2wiki" and bool(train_case.get("answerable")):
                counterpart = f"unsupported::2wiki::{train_case.get('group_id')}"
                if counterpart in train_by_case:
                    remove_train_case_ids.add(counterpart)
        elif case_id in validation_by_case:
            remove_validation_case_ids.add(case_id)
        else:
            raise ValueError(f"G212M failed case is absent from G214 bundle: {case_id}")
    return remove_train_case_ids, remove_validation_case_ids


def repair(
    *,
    manifest_path: Path,
    train_cases_path: Path,
    validation_cases_path: Path,
    review_rows_path: Path,
    review_summary_path: Path,
    output_dir: Path,
    min_niah_train_groups: int = 400,
    min_niah_modelval_groups: int = 100,
    min_twowiki_train_groups: int = 400,
    min_twowiki_modelval_groups: int = 100,
    unsupported_ratio_min: float = 0.10,
    unsupported_ratio_max: float = 0.15,
) -> dict[str, object]:
    _require_empty(output_dir, "G216 sample-review repair")
    input_manifest, train_rows, validation_rows = _load_bundle(
        manifest_path=manifest_path,
        train_cases_path=train_cases_path,
        validation_cases_path=validation_cases_path,
    )
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
        raise ValueError("G216 removal keys did not match the expected number of rows")

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

    train_path = output_dir / "train_cases.jsonl"
    validation_path = output_dir / "validation_cases.jsonl"
    ordered_path = output_dir / "ordered_ids.json"
    _write_jsonl(train_path, repaired_train)
    _write_jsonl(validation_path, repaired_validation)
    _write_json(
        ordered_path,
        {
            "schema_version": "full-flow-g216-ordered-ids-v1",
            "train_case_ids": [str(row["case_id"]) for row in repaired_train],
            "validation_case_ids": [str(row["case_id"]) for row in repaired_validation],
        },
    )
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_G216_MANIFEST,
        "status": "PRE_MANUAL_PASS" if all(gates.values()) else "FAIL",
        "repair_stage": "G216",
        "repair_reason": "exclude_g212m_failed_sample_review_cases",
        "manual_audit_required_before_training": True,
        "sample_review_required_before_training": True,
        "training_started": False,
        "utility_labels_started": False,
        "sealed_or_heldout_read": False,
        "dev_read": False,
        "source_manifest_schema_version": str(input_manifest.get("schema_version", "")),
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
                "variant_count": len(row.get("variants", {})) if isinstance(row.get("variants"), Mapping) else 0,
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
        "train_cases": len(repaired_train),
        "validation_cases": len(repaired_validation),
        "train_cases_sha256": _sha256(train_path),
        "validation_cases_sha256": _sha256(validation_path),
        "ordered_ids_sha256": _sha256(ordered_path),
        "input_sha256": {
            "source_manifest": _sha256(manifest_path),
            "source_train_cases": _sha256(train_cases_path),
            "source_validation_cases": _sha256(validation_cases_path),
            "g212m_review_rows": _sha256(review_rows_path),
            "g212m_review_summary": _sha256(review_summary_path),
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
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = repair(
        manifest_path=args.manifest,
        train_cases_path=args.train_cases,
        validation_cases_path=args.validation_cases,
        review_rows_path=args.review_rows,
        review_summary_path=args.review_summary,
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
