from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_flow_g210_failure_triage as triage  # noqa: E402


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_failure_triage_summarizes_pair_case_and_relation_failures(tmp_path: Path) -> None:
    structural = _write_jsonl(
        tmp_path / "structural.jsonl",
        [
            {
                "case_id": "case-1",
                "dataset": "2wiki",
                "role": "train-modelval",
                "target_kind": "twowiki_evidence_chain",
                "true_audit_ids": ["a1", "a2"],
            },
            {
                "case_id": "case-2",
                "dataset": "niah",
                "role": "train-modelval",
                "target_kind": "niah_qa2d_single_claim",
                "true_audit_ids": ["a3"],
            },
        ],
    )
    worklist = _write_jsonl(
        tmp_path / "worklist.jsonl",
        [
            {"audit_id": "a1", "hypothesis": "Alice's country is France.", "premise": "Alice lived in Paris."},
            {"audit_id": "a2", "hypothesis": "Bob's country is Canada.", "premise": "Bob was Canadian."},
            {"audit_id": "a3", "hypothesis": "It happened in 2009.", "premise": "It happened in 2009."},
        ],
    )
    true_rows = _write_jsonl(
        tmp_path / "true.jsonl",
        [
            {
                "audit_id": "a1",
                "case_id": "case-1",
                "dataset": "2wiki",
                "role": "train-modelval",
                "target_kind": "twowiki_evidence_chain",
                "support_evidence_id": "e1",
                "true_entailment_score": 0.1,
                "threshold": 0.5,
                "entailed": False,
            },
            {
                "audit_id": "a2",
                "case_id": "case-1",
                "dataset": "2wiki",
                "role": "train-modelval",
                "target_kind": "twowiki_evidence_chain",
                "support_evidence_id": "e2",
                "true_entailment_score": 0.9,
                "threshold": 0.5,
                "entailed": True,
            },
            {
                "audit_id": "a3",
                "case_id": "case-2",
                "dataset": "niah",
                "role": "train-modelval",
                "target_kind": "niah_qa2d_single_claim",
                "support_evidence_id": "e3",
                "true_entailment_score": 0.8,
                "threshold": 0.5,
                "entailed": True,
            },
        ],
    )
    final_manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "status": "FAIL",
            "gates": {"twowiki_modelval_groups": False},
            "counts": {"2wiki": {"train-modelval_answerable_groups": 76}},
        },
    )

    summary = triage.triage(
        structural_rows_path=structural,
        true_worklist_path=worklist,
        true_audit_rows_path=true_rows,
        final_manifest_path=final_manifest,
        output_dir=tmp_path / "out",
        sample_size=5,
    )

    assert summary["status"] == "COMPLETE_DIAGNOSTIC_ONLY"
    assert summary["gate_changed"] is False
    assert summary["final_failed_gates"] == {"twowiki_modelval_groups": False}
    assert summary["pair_counts"]["2wiki:train-modelval:twowiki_evidence_chain:not_entailed"] == 1
    assert summary["case_counts"]["2wiki:train-modelval:twowiki_evidence_chain:case_not_entailed"] == 1
    assert "2wiki:train-modelval:twowiki_evidence_chain:country" in summary["failed_pair_relation_counts_top25"]
    assert _sha(tmp_path / "out/failure_samples.jsonl") == summary["failure_samples_sha256"]
