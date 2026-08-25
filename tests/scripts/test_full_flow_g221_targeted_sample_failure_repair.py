from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_flow_g221_targeted_sample_failure_repair as g221  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _variant(target: str, *, answerable: bool = True) -> dict[str, object]:
    return {
        "evidence_ids": ["e1"],
        "support_evidence_ids": ["e1"] if answerable else [],
        "prompt": "Evidence:\n[1] (e1) Evidence.\nQuestion: q?\nAnswer:",
        "target": target,
    }


def _case(
    case_id: str,
    *,
    dataset: str,
    role: str,
    group_id: str,
    answerable: bool,
) -> dict[str, object]:
    target = "Answer is Paris [1]." if answerable else "I don't know."
    return {
        "schema_version": g221.SCHEMA_CASE,
        "case_id": case_id,
        "dataset": dataset,
        "source": "fixture",
        "group_id": group_id,
        "query_id": group_id,
        "role": role,
        "component_id": f"component::{group_id}",
        "answerable": answerable,
        "target_kind": "twowiki_evidence_chain"
        if dataset == "2wiki" and answerable
        else "niah_qa2d_single_claim",
        "question": "What is the answer?",
        "answer": "Paris" if answerable else "I don't know.",
        "semantic_target": "Answer is Paris." if answerable else "I don't know.",
        "variants": {
            "support_only" if answerable else "support_removed": _variant(
                target,
                answerable=answerable,
            )
        },
    }


def _bundle(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    train_rows = [
        _case("2wiki::drop", dataset="2wiki", role=g221.TRAIN_ROLE, group_id="drop", answerable=True),
        _case(
            "unsupported::2wiki::drop",
            dataset="2wiki",
            role=g221.TRAIN_ROLE,
            group_id="drop",
            answerable=False,
        ),
        _case("2wiki::keep", dataset="2wiki", role=g221.TRAIN_ROLE, group_id="keep", answerable=True),
        _case(
            "unsupported::2wiki::keep",
            dataset="2wiki",
            role=g221.TRAIN_ROLE,
            group_id="keep",
            answerable=False,
        ),
        _case("niah::drop-train", dataset="niah", role=g221.TRAIN_ROLE, group_id="niah-drop", answerable=True),
        _case("niah::keep-train", dataset="niah", role=g221.TRAIN_ROLE, group_id="niah-keep", answerable=True),
    ]
    validation_rows = [
        _case("2wiki::drop-val", dataset="2wiki", role=g221.VALIDATION_ROLE, group_id="tw-drop", answerable=True),
        _case("2wiki::keep-val", dataset="2wiki", role=g221.VALIDATION_ROLE, group_id="tw-keep", answerable=True),
        _case("niah::drop-val", dataset="niah", role=g221.VALIDATION_ROLE, group_id="niah-drop-val", answerable=True),
        _case("niah::keep-val", dataset="niah", role=g221.VALIDATION_ROLE, group_id="niah-keep-val", answerable=True),
    ]
    train = _write_jsonl(tmp_path / "train_cases.jsonl", train_rows)
    validation = _write_jsonl(tmp_path / "validation_cases.jsonl", validation_rows)
    manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": g221.SCHEMA_INPUT_G220_MANIFEST,
            "status": "PRE_MANUAL_PASS",
            "training_started": False,
            "sealed_or_heldout_read": False,
            "dev_read": False,
            "train_cases": len(train_rows),
            "validation_cases": len(validation_rows),
            "train_cases_sha256": _sha(train),
            "validation_cases_sha256": _sha(validation),
        },
    )
    review_rows = _write_jsonl(
        tmp_path / "review_rows.jsonl",
        [
            {
                "schema_version": g221.SCHEMA_G212M_REVIEW_ROW,
                "sample_index": 1,
                "case_id": "2wiki::drop",
                "sample_stratum": "2wiki:train-fit:twowiki_evidence_chain:answerable",
                "review_decision": "FAIL",
                "review_reason": "bad train row",
            },
            {
                "schema_version": g221.SCHEMA_G212M_REVIEW_ROW,
                "sample_index": 2,
                "case_id": "2wiki::drop-val",
                "sample_stratum": "2wiki:train-modelval:twowiki_evidence_chain:answerable",
                "review_decision": "FAIL",
                "review_reason": "bad validation row",
            },
            {
                "schema_version": g221.SCHEMA_G212M_REVIEW_ROW,
                "sample_index": 3,
                "case_id": "niah::drop-train",
                "sample_stratum": "niah:train-fit:niah_qa2d_single_claim:answerable",
                "review_decision": "FAIL",
                "review_reason": "bad niah train row",
            },
            {
                "schema_version": g221.SCHEMA_G212M_REVIEW_ROW,
                "sample_index": 4,
                "case_id": "niah::drop-val",
                "sample_stratum": "niah:train-modelval:niah_qa2d_single_claim:answerable",
                "review_decision": "FAIL",
                "review_reason": "bad niah validation row",
            },
        ],
    )
    review_summary = _write_json(
        tmp_path / "review_summary.json",
        {
            "schema_version": g221.SCHEMA_G212M_REVIEW_SUMMARY,
            "status": "FAIL_SAMPLE_REVIEW",
            "fail": 4,
            "uncertain": 0,
        },
    )
    return manifest, train, validation, review_rows, review_summary


def test_g221_filters_failed_rows_and_records_reduced_screen(tmp_path: Path) -> None:
    manifest, train, validation, review_rows, review_summary = _bundle(tmp_path)

    report = g221.repair(
        manifest_path=manifest,
        train_cases_path=train,
        validation_cases_path=validation,
        review_rows_path=review_rows,
        review_summary_path=review_summary,
        output_dir=tmp_path / "g221",
        min_niah_train_groups=1,
        min_niah_modelval_groups=1,
        min_twowiki_train_groups=1,
        min_twowiki_modelval_groups=1,
        unsupported_ratio_min=0.0,
        unsupported_ratio_max=1.0,
    )

    assert report["status"] == "PRE_MANUAL_PASS"
    assert report["requires_structural_true_rerun"] is False
    assert report["target_rewrite_started"] is False
    assert report["method_amendment"]["previous_floor"] == 99
    assert report["method_amendment"]["twowiki_modelval_floor"] == 1
    assert report["excluded"]["sample_review_rows"] == 4
    assert report["excluded"]["train_cases"] == 3
    assert report["excluded"]["validation_cases"] == 2
    train_ids = {
        json.loads(line)["case_id"]
        for line in (tmp_path / "g221/train_cases.jsonl").read_text(encoding="utf-8").splitlines()
    }
    validation_ids = {
        json.loads(line)["case_id"]
        for line in (tmp_path / "g221/validation_cases.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert "2wiki::drop" not in train_ids
    assert "unsupported::2wiki::drop" not in train_ids
    assert "niah::drop-train" not in train_ids
    assert "2wiki::drop-val" not in validation_ids
    assert "niah::drop-val" not in validation_ids
