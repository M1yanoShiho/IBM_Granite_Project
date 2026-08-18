from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_flow_g223_residual_sample_failure_candidate as g223  # noqa: E402


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
        "schema_version": g223.SCHEMA_CASE,
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
        else "unsupported_support_removed",
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


def _bundle(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    train_rows = [
        _case("2wiki::drop", dataset="2wiki", role=g223.TRAIN_ROLE, group_id="drop", answerable=True),
        _case(
            "unsupported::2wiki::drop",
            dataset="2wiki",
            role=g223.TRAIN_ROLE,
            group_id="drop",
            answerable=False,
        ),
        _case("2wiki::keep", dataset="2wiki", role=g223.TRAIN_ROLE, group_id="keep", answerable=True),
        _case(
            "unsupported::2wiki::keep",
            dataset="2wiki",
            role=g223.TRAIN_ROLE,
            group_id="keep",
            answerable=False,
        ),
    ]
    validation_rows = [
        _case("2wiki::drop-val", dataset="2wiki", role=g223.VALIDATION_ROLE, group_id="tw-drop", answerable=True),
        _case("2wiki::keep-val", dataset="2wiki", role=g223.VALIDATION_ROLE, group_id="tw-keep", answerable=True),
    ]
    train = _write_jsonl(tmp_path / "train_cases.jsonl", train_rows)
    validation = _write_jsonl(tmp_path / "validation_cases.jsonl", validation_rows)
    manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": g223.SCHEMA_INPUT_G221_MANIFEST,
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
                "schema_version": g223.SCHEMA_G212M_REVIEW_ROW,
                "sample_index": 1,
                "case_id": "2wiki::drop",
                "sample_stratum": "2wiki:train-fit:twowiki_evidence_chain:answerable",
                "review_decision": "FAIL",
                "review_reason": "bad train row",
            },
            {
                "schema_version": g223.SCHEMA_G212M_REVIEW_ROW,
                "sample_index": 2,
                "case_id": "2wiki::drop-val",
                "sample_stratum": "2wiki:train-modelval:twowiki_evidence_chain:answerable",
                "review_decision": "FAIL",
                "review_reason": "bad validation row",
            },
        ],
    )
    review_summary = _write_json(
        tmp_path / "review_summary.json",
        {
            "schema_version": g223.SCHEMA_G212M_REVIEW_SUMMARY,
            "status": "FAIL_SAMPLE_REVIEW",
            "fail": 2,
            "uncertain": 0,
        },
    )
    amendment = _write_json(
        tmp_path / "g222.json",
        {
            "schema_version": g223.SCHEMA_G222_AMENDMENT,
            "status": "CONTROLLED_CONTINUATION_ALLOWED_NOT_TRAINING_READY",
            "decision": {
                "controlled_continuation_allowed": True,
                "direct_g300_training_allowed": False,
            },
        },
    )
    return manifest, train, validation, review_rows, review_summary, amendment


def test_g223_quarantines_residual_failures_and_unlocks_limited_continuation(tmp_path: Path) -> None:
    manifest, train, validation, review_rows, review_summary, amendment = _bundle(tmp_path)

    report = g223.candidate(
        manifest_path=manifest,
        train_cases_path=train,
        validation_cases_path=validation,
        review_rows_path=review_rows,
        review_summary_path=review_summary,
        amendment_path=amendment,
        output_dir=tmp_path / "g223",
        min_niah_train_groups=0,
        min_niah_modelval_groups=0,
        min_twowiki_train_groups=1,
        min_twowiki_modelval_groups=1,
        unsupported_ratio_min=0.0,
        unsupported_ratio_max=1.0,
    )

    assert report["status"] == "CONTROLLED_CONTINUATION_READY"
    assert report["clean_freeze_ready"] is False
    assert report["controlled_continuation_ready"] is True
    assert report["g300_unlocked"] is True
    assert report["length_rerun_required"] is False
    assert report["excluded"]["sample_review_rows"] == 2
    assert report["excluded"]["train_cases"] == 2
    assert report["excluded"]["validation_cases"] == 1
    train_ids = {
        json.loads(line)["case_id"]
        for line in (tmp_path / "g223/train_cases.jsonl").read_text(encoding="utf-8").splitlines()
    }
    validation_ids = {
        json.loads(line)["case_id"]
        for line in (tmp_path / "g223/validation_cases.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert "2wiki::drop" not in train_ids
    assert "unsupported::2wiki::drop" not in train_ids
    assert "2wiki::drop-val" not in validation_ids
    assert "2wiki::keep" in train_ids
    assert "2wiki::keep-val" in validation_ids


def test_g223_rejects_amendment_without_controlled_continuation(tmp_path: Path) -> None:
    manifest, train, validation, review_rows, review_summary, amendment = _bundle(tmp_path)
    amendment_data = json.loads(amendment.read_text(encoding="utf-8"))
    amendment_data["decision"]["controlled_continuation_allowed"] = False
    _write_json(amendment, amendment_data)

    try:
        g223.candidate(
            manifest_path=manifest,
            train_cases_path=train,
            validation_cases_path=validation,
            review_rows_path=review_rows,
            review_summary_path=review_summary,
            amendment_path=amendment,
            output_dir=tmp_path / "g223",
        )
    except ValueError as exc:
        assert "must allow controlled continuation" in str(exc)
    else:
        raise AssertionError("G223 accepted an amendment that did not allow continuation")
