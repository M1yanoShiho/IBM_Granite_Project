from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_flow_g212_manual_length_audit as g212  # noqa: E402


class TinyTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
        assert tokenize is False
        suffix = "\nAssistant:" if add_generation_prompt else ""
        return "\n".join(str(message["content"]) for message in messages) + suffix

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": list(range(len(text.split())))}


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


def _variant(evidence_ids: list[str], support_ids: list[str], target: str) -> dict[str, object]:
    lines = [
        f"[{index}] ({evidence_id}) Evidence text for {evidence_id}."
        for index, evidence_id in enumerate(evidence_ids, start=1)
    ]
    return {
        "evidence_ids": evidence_ids,
        "support_evidence_ids": support_ids,
        "prompt": "Evidence:\n" + "\n".join(lines) + "\nQuestion: q?\nAnswer:",
        "target": target,
    }


def _case(
    case_id: str,
    *,
    dataset: str,
    role: str,
    target_kind: str,
    answerable: bool = True,
    long_prompt: bool = False,
) -> dict[str, object]:
    variant = _variant(["e1", "e2"], ["e1"], "Answer is Paris [1].")
    if long_prompt:
        variant["prompt"] = str(variant["prompt"]) + " " + "word " * 200
    row: dict[str, object] = {
        "schema_version": g212.SCHEMA_CASE,
        "case_id": case_id,
        "dataset": dataset,
        "source": "fixture",
        "group_id": case_id,
        "query_id": case_id,
        "role": role,
        "component_id": f"component::{case_id}",
        "answerable": answerable,
        "target_kind": target_kind,
        "question": "What is the answer?",
        "answer": "Paris" if answerable else g212.UNKNOWN_ANSWER,
        "semantic_target": "Answer is Paris." if answerable else g212.UNKNOWN_ANSWER,
        "variants": {"support_only": variant},
    }
    if not answerable:
        row["removed_support_evidence_ids"] = ["e1"]
        row["variants"] = {
            "support_removed": _variant(["d1"], [], g212.UNKNOWN_ANSWER),
        }
    return row


def _bundle(tmp_path: Path, *, long_prompt: bool = False) -> tuple[Path, Path, Path, Path]:
    train_rows = [
        _case(
            "niah-train",
            dataset="niah",
            role="train-fit",
            target_kind="niah_qa2d_single_claim",
            long_prompt=long_prompt,
        ),
        _case(
            "twiki-train",
            dataset="2wiki",
            role="train-fit",
            target_kind="twowiki_evidence_chain",
        ),
        _case(
            "twiki-unsupported",
            dataset="2wiki",
            role="train-fit",
            target_kind="unsupported_support_removed",
            answerable=False,
        ),
    ]
    validation_rows = [
        _case(
            "niah-val",
            dataset="niah",
            role="train-modelval",
            target_kind="niah_qa2d_single_claim",
        ),
        _case(
            "twiki-val",
            dataset="2wiki",
            role="train-modelval",
            target_kind="twowiki_evidence_chain",
        ),
    ]
    train = _write_jsonl(tmp_path / "train_cases.jsonl", train_rows)
    validation = _write_jsonl(tmp_path / "validation_cases.jsonl", validation_rows)
    manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": g212.SCHEMA_PRE_MANUAL,
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
    model_snapshot = tmp_path / "model"
    _write_json(model_snapshot / "config.json", {"model_type": "fixture"})
    return manifest, train, validation, model_snapshot


def test_prepare_writes_length_pass_and_pending_manual_review(tmp_path: Path) -> None:
    manifest, train, validation, model_snapshot = _bundle(tmp_path)

    report = g212.prepare(
        manifest_path=manifest,
        train_cases_path=train,
        validation_cases_path=validation,
        model_snapshot=model_snapshot,
        output_dir=tmp_path / "g212",
        max_length=128,
        sample_per_stratum=1,
        tokenizer=TinyTokenizer(),
    )

    assert report["status"] == "LENGTH_PASS_MANUAL_REVIEW_PENDING"
    assert report["length_audit_pass"] is True
    assert report["manual_review_complete"] is False
    assert report["sample_rows"] == 5
    sample_rows = [
        json.loads(line)
        for line in (tmp_path / "g212/manual_sample.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["review_decision"] for row in sample_rows} == {"PENDING"}
    assert (tmp_path / "g212/length_audit.json").is_file()


def test_prepare_marks_length_failure_without_training(tmp_path: Path) -> None:
    manifest, train, validation, model_snapshot = _bundle(tmp_path, long_prompt=True)

    report = g212.prepare(
        manifest_path=manifest,
        train_cases_path=train,
        validation_cases_path=validation,
        model_snapshot=model_snapshot,
        output_dir=tmp_path / "g212",
        max_length=32,
        sample_per_stratum=1,
        tokenizer=TinyTokenizer(),
    )

    assert report["status"] == "FAIL_LENGTH"
    assert report["length_audit_pass"] is False
    assert report["training_started"] is False
    length_audit = json.loads((tmp_path / "g212/length_audit.json").read_text(encoding="utf-8"))
    assert length_audit["over_max_length"] > 0


def test_prepare_accepts_g216_repair_manifest(tmp_path: Path) -> None:
    manifest, train, validation, model_snapshot = _bundle(tmp_path)
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_data["schema_version"] = g212.SCHEMA_G216_REPAIR
    _write_json(manifest, manifest_data)

    report = g212.prepare(
        manifest_path=manifest,
        train_cases_path=train,
        validation_cases_path=validation,
        model_snapshot=model_snapshot,
        output_dir=tmp_path / "g212",
        max_length=128,
        sample_per_stratum=1,
        tokenizer=TinyTokenizer(),
    )

    assert report["status"] == "LENGTH_PASS_MANUAL_REVIEW_PENDING"
    assert report["length_audit_pass"] is True


def test_prepare_accepts_g220_conservative_filter_manifest(tmp_path: Path) -> None:
    manifest, train, validation, model_snapshot = _bundle(tmp_path)
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_data["schema_version"] = g212.SCHEMA_G220_REPAIR
    _write_json(manifest, manifest_data)

    report = g212.prepare(
        manifest_path=manifest,
        train_cases_path=train,
        validation_cases_path=validation,
        model_snapshot=model_snapshot,
        output_dir=tmp_path / "g212",
        max_length=128,
        sample_per_stratum=1,
        tokenizer=TinyTokenizer(),
    )

    assert report["status"] == "LENGTH_PASS_MANUAL_REVIEW_PENDING"
    assert report["length_audit_pass"] is True
