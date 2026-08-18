from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_flow_g214_length_repair as g214  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def _variant(target: str) -> dict[str, object]:
    return {
        "prompt": "Evidence:\n[1] (e1) Paris is the answer.\nQuestion: q?\nAnswer:",
        "target": target,
        "evidence_ids": ["e1"],
        "support_evidence_ids": ["e1"],
    }


def _case(
    case_id: str,
    *,
    dataset: str,
    role: str,
    group_id: str,
    answerable: bool,
    variants: int = 1,
) -> dict[str, object]:
    variant_map = {
        f"v{index}": _variant("Answer is Paris [1]." if answerable else "I don't know.")
        for index in range(variants)
    }
    if not answerable:
        for value in variant_map.values():
            value["support_evidence_ids"] = []
    row: dict[str, object] = {
        "schema_version": g214.SCHEMA_CASE,
        "case_id": case_id,
        "dataset": dataset,
        "source": "fixture",
        "group_id": group_id,
        "query_id": group_id,
        "role": role,
        "component_id": f"component::{group_id}",
        "answerable": answerable,
        "target_kind": "twowiki_evidence_chain" if answerable else "unsupported_support_removed",
        "question": "What is the answer?",
        "answer": "Paris" if answerable else "I don't know.",
        "semantic_target": "Answer is Paris." if answerable else "I don't know.",
        "variants": variant_map,
    }
    if not answerable:
        row["removed_support_evidence_ids"] = ["e1"]
    return row


def _bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    train_rows = [
        _case(
            "2wiki::keep",
            dataset="2wiki",
            role="train-fit",
            group_id="keep",
            answerable=True,
            variants=5,
        ),
        _case(
            "unsupported::2wiki::keep",
            dataset="2wiki",
            role="train-fit",
            group_id="keep",
            answerable=False,
        ),
        _case(
            "2wiki::drop",
            dataset="2wiki",
            role="train-fit",
            group_id="drop",
            answerable=True,
            variants=5,
        ),
        _case(
            "unsupported::2wiki::drop",
            dataset="2wiki",
            role="train-fit",
            group_id="drop",
            answerable=False,
        ),
        _case(
            "niah::keep",
            dataset="niah",
            role="train-fit",
            group_id="niah-keep",
            answerable=True,
            variants=8,
        ),
    ]
    validation_rows = [
        _case(
            "2wiki::val",
            dataset="2wiki",
            role="train-modelval",
            group_id="val",
            answerable=True,
            variants=5,
        ),
        _case(
            "niah::val",
            dataset="niah",
            role="train-modelval",
            group_id="niah-val",
            answerable=True,
            variants=7,
        ),
    ]
    train = _write_jsonl(tmp_path / "train_cases.jsonl", train_rows)
    validation = _write_jsonl(tmp_path / "validation_cases.jsonl", validation_rows)
    manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": g214.SCHEMA_G210_PRE_MANUAL,
            "status": "PRE_MANUAL_PASS",
            "training_started": False,
            "sealed_or_heldout_read": False,
            "dev_read": False,
            "train_cases_sha256": _sha(train),
            "validation_cases_sha256": _sha(validation),
        },
    )
    return manifest, train, validation


def _length_rows(tmp_path: Path, *, validation_over: bool = False) -> Path:
    rows = [
        {
            "case_id": "2wiki::drop",
            "split": "train",
            "variant": "v0",
            "length": 2500,
            "over_max_length": True,
        }
    ]
    if validation_over:
        rows.append(
            {
                "case_id": "2wiki::val",
                "split": "validation",
                "variant": "v0",
                "length": 2500,
                "over_max_length": True,
            }
        )
    return _write_jsonl(tmp_path / "length_rows.jsonl", rows)


def test_repair_removes_overlength_train_group_and_counterpart(tmp_path: Path) -> None:
    manifest, train, validation = _bundle(tmp_path)

    report = g214.repair(
        manifest_path=manifest,
        train_cases_path=train,
        validation_cases_path=validation,
        length_rows_path=_length_rows(tmp_path),
        output_dir=tmp_path / "g214",
        min_niah_train_groups=1,
        min_niah_modelval_groups=1,
        min_twowiki_train_groups=1,
        min_twowiki_modelval_groups=1,
        unsupported_ratio_min=0.0,
        unsupported_ratio_max=1.0,
    )

    assert report["status"] == "PRE_MANUAL_PASS"
    assert report["excluded"] == {"length_over_max_cases": 2, "length_over_max_groups": 1}
    assert report["excluded_length_groups"] == ["drop"]
    repaired = [
        json.loads(line)
        for line in (tmp_path / "g214/train_cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["case_id"] for row in repaired} == {
        "2wiki::keep",
        "unsupported::2wiki::keep",
        "niah::keep",
    }
    assert (tmp_path / "g214/ordered_ids.json").is_file()


def test_repair_rejects_validation_overlength(tmp_path: Path) -> None:
    manifest, train, validation = _bundle(tmp_path)

    with pytest.raises(ValueError, match="validation overlength"):
        g214.repair(
            manifest_path=manifest,
            train_cases_path=train,
            validation_cases_path=validation,
            length_rows_path=_length_rows(tmp_path, validation_over=True),
            output_dir=tmp_path / "g214",
        )
