from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_flow_g212m_sample_adjudication as g212m  # noqa: E402


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


def _sample_row(index: int, *, answerable: bool = True) -> dict[str, object]:
    if answerable:
        target = "Paris is the answer [1]."
        citations = [1]
        support_ids = ["e1"]
        cited_ids = ["e1"]
        answer = "Paris"
    else:
        target = g212m.UNKNOWN_ANSWER
        citations = []
        support_ids = []
        cited_ids = []
        answer = g212m.UNKNOWN_ANSWER
    return {
        "schema_version": g212m.SCHEMA_MANUAL_SAMPLE_ROW,
        "audit_id": f"audit-{index}",
        "sample_index": index,
        "sample_seed": "fixture-seed",
        "sample_stratum": "niah:train-fit:fixture:answerable" if answerable else "2wiki:train-fit:fixture:unsupported",
        "case_id": f"case-{index}",
        "dataset": "niah" if answerable else "2wiki",
        "role": "train-fit",
        "target_kind": "fixture",
        "answerable": answerable,
        "question": "What is the answer?",
        "answer": answer,
        "target": target,
        "target_citation_indices": citations,
        "target_cited_evidence_ids": cited_ids,
        "support_evidence_ids": support_ids,
        "review_evidence": [{"citation_index": 1, "evidence_id": "e1", "text": "Paris is the answer."}],
        "review_decision": g212m.PENDING,
    }


def _bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    sample = _write_jsonl(
        tmp_path / "manual_sample.jsonl",
        [_sample_row(1), _sample_row(2, answerable=False)],
    )
    prepare = _write_json(
        tmp_path / "prepare_manifest.json",
        {
            "schema_version": g212m.SCHEMA_PREPARE_MANIFEST,
            "status": "LENGTH_PASS_MANUAL_REVIEW_PENDING",
            "training_started": False,
            "utility_labels_started": False,
            "sealed_or_heldout_read": False,
            "dev_read": False,
            "length_audit_pass": True,
            "sample_rows": 2,
            "sample_seed": "fixture-seed",
            "artifacts": {"manual_sample": {"path": str(sample), "sha256": _sha(sample)}},
        },
    )
    length = _write_json(
        tmp_path / "length_audit.json",
        {
            "schema_version": g212m.SCHEMA_LENGTH_AUDIT,
            "status": g212m.PASS,
            "over_max_length": 0,
        },
    )
    return sample, prepare, length


def test_adjudicate_writes_freeze_ready_when_all_pass(tmp_path: Path) -> None:
    sample, prepare, length = _bundle(tmp_path)

    report = g212m.adjudicate(
        manual_sample_path=sample,
        prepare_manifest_path=prepare,
        length_audit_path=length,
        output_dir=tmp_path / "g212m",
    )

    assert report["status"] == "FREEZE_READY"
    assert report["freeze_ready"] is True
    summary = json.loads((tmp_path / "g212m/sample_review_summary.json").read_text(encoding="utf-8"))
    assert summary["pass"] == 2
    assert summary["fail"] == 0


def test_adjudicate_blocks_freeze_when_override_fails(tmp_path: Path) -> None:
    sample, prepare, length = _bundle(tmp_path)

    report = g212m.adjudicate(
        manual_sample_path=sample,
        prepare_manifest_path=prepare,
        length_audit_path=length,
        output_dir=tmp_path / "g212m",
        fail={1: "answer chain is not preserved"},
    )

    assert report["status"] == "NOT_FREEZE_READY_SAMPLE_REVIEW_FAILED"
    assert report["g300_unlocked"] is False
    summary = json.loads((tmp_path / "g212m/sample_review_summary.json").read_text(encoding="utf-8"))
    assert summary["pass"] == 1
    assert summary["fail"] == 1
    assert summary["fail_or_uncertain_rows"][0]["sample_index"] == 1


def test_adjudicate_forces_structural_failure(tmp_path: Path) -> None:
    sample, prepare, length = _bundle(tmp_path)
    rows = [json.loads(line) for line in sample.read_text(encoding="utf-8").splitlines()]
    rows[0]["target_cited_evidence_ids"] = ["wrong"]
    _write_jsonl(sample, rows)
    prepare_data = dict(json.loads(prepare.read_text(encoding="utf-8")))
    prepare_data["artifacts"]["manual_sample"]["sha256"] = _sha(sample)
    _write_json(prepare, prepare_data)

    report = g212m.adjudicate(
        manual_sample_path=sample,
        prepare_manifest_path=prepare,
        length_audit_path=length,
        output_dir=tmp_path / "g212m",
    )

    assert report["status"] == "NOT_FREEZE_READY_SAMPLE_REVIEW_FAILED"
    summary = json.loads((tmp_path / "g212m/sample_review_summary.json").read_text(encoding="utf-8"))
    assert summary["forced_structural_failures"] == 1
