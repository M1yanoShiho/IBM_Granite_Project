"""Apply the G214 controlled length repair.

The only allowed repair is to remove train groups identified by the G212 length
audit as over the frozen max length.  The script removes the whole train group,
including its unsupported counterpart, then rewrites a revised pre-manual bundle
for G212R.  It does not train a model, change max_length, or read dev/held-out
data.
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
SCHEMA_G210_PRE_MANUAL = "full-flow-g210-v2-final-manifest-v1"
SCHEMA_G214_MANIFEST = "full-flow-g214-length-repair-manifest-v1"
TRAIN_ROLE = "train-fit"
VALIDATION_ROLE = "train-modelval"


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
    if manifest.get("schema_version") != SCHEMA_G210_PRE_MANUAL:
        raise ValueError("G214 expects the G210R2 pre-manual manifest")
    if manifest.get("status") != "PRE_MANUAL_PASS":
        raise ValueError("G214 requires a PRE_MANUAL_PASS input bundle")
    if manifest.get("training_started") is not False:
        raise ValueError("G214 input must record training_started=false")
    if manifest.get("sealed_or_heldout_read") is not False or manifest.get("dev_read") is not False:
        raise ValueError("G214 input must not have read dev/sealed/held-out data")
    if str(manifest.get("train_cases_sha256", "")) != _sha256(train_cases_path):
        raise ValueError("train cases hash differs from manifest")
    if str(manifest.get("validation_cases_sha256", "")) != _sha256(validation_cases_path):
        raise ValueError("validation cases hash differs from manifest")
    train_rows = _jsonl(train_cases_path)
    validation_rows = _jsonl(validation_cases_path)
    seen: set[str] = set()
    for row in [*train_rows, *validation_rows]:
        case_id = str(row.get("case_id", ""))
        if not case_id or case_id in seen or row.get("schema_version") != SCHEMA_CASE:
            raise ValueError(f"invalid or duplicate case: {case_id!r}")
        seen.add(case_id)
    return manifest, train_rows, validation_rows


def overlength_train_groups(length_rows_path: Path) -> set[str]:
    rows = _jsonl(length_rows_path)
    groups: set[str] = set()
    validation_over: set[str] = set()
    for row in rows:
        if not bool(row.get("over_max_length")):
            continue
        case_id = str(row.get("case_id", ""))
        split = str(row.get("split", ""))
        group_id = case_id.split("::")[-1]
        if split == "train":
            groups.add(group_id)
        else:
            validation_over.add(case_id)
    if validation_over:
        raise ValueError(
            "G214 cannot silently remove validation overlength cases: "
            + ", ".join(sorted(validation_over))
        )
    if not groups:
        raise ValueError("G214 found no train overlength groups in length rows")
    return groups


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


def repair(
    *,
    manifest_path: Path,
    train_cases_path: Path,
    validation_cases_path: Path,
    length_rows_path: Path,
    output_dir: Path,
    min_niah_train_groups: int = 400,
    min_niah_modelval_groups: int = 100,
    min_twowiki_train_groups: int = 400,
    min_twowiki_modelval_groups: int = 100,
    unsupported_ratio_min: float = 0.10,
    unsupported_ratio_max: float = 0.15,
) -> dict[str, object]:
    _require_empty(output_dir, "G214 length repair")
    input_manifest, train_rows, validation_rows = _load_bundle(
        manifest_path=manifest_path,
        train_cases_path=train_cases_path,
        validation_cases_path=validation_cases_path,
    )
    groups = overlength_train_groups(length_rows_path)
    removed = [row for row in train_rows if str(row.get("group_id", "")) in groups]
    if not removed:
        raise ValueError("G214 overlength groups did not match train rows")
    removed_by_group: dict[str, Counter[str]] = {}
    for row in removed:
        group_id = str(row.get("group_id", ""))
        kind = "answerable" if row.get("answerable") else "unsupported"
        removed_by_group.setdefault(group_id, Counter())[kind] += 1
    for group_id, counts in removed_by_group.items():
        if counts.get("answerable", 0) != 1 or counts.get("unsupported", 0) != 1:
            raise ValueError(
                f"G214 requires one answerable and one unsupported train case for {group_id}: "
                f"{dict(counts)}"
            )
    repaired_train = [row for row in train_rows if str(row.get("group_id", "")) not in groups]
    repaired_validation = list(validation_rows)
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
            "train_case_ids": [str(row["case_id"]) for row in repaired_train],
            "validation_case_ids": [str(row["case_id"]) for row in repaired_validation],
        },
    )
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_G214_MANIFEST,
        "status": "PRE_MANUAL_PASS" if all(gates.values()) else "FAIL",
        "repair_stage": "G214",
        "repair_reason": "exclude_train_groups_over_frozen_max_length",
        "max_length_unchanged": True,
        "manual_audit_required_before_training": True,
        "training_started": False,
        "sealed_or_heldout_read": False,
        "dev_read": False,
        "source_manifest_schema_version": str(input_manifest.get("schema_version", "")),
        "excluded_length_groups": sorted(groups),
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
            for row in removed
        ],
        "excluded": {
            "length_over_max_groups": len(groups),
            "length_over_max_cases": len(removed),
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
            "length_rows": _sha256(length_rows_path),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--train-cases", required=True, type=Path)
    parser.add_argument("--validation-cases", required=True, type=Path)
    parser.add_argument("--length-rows", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = repair(
        manifest_path=args.manifest,
        train_cases_path=args.train_cases,
        validation_cases_path=args.validation_cases,
        length_rows_path=args.length_rows,
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
