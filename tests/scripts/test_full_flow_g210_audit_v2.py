from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_flow_g210_audit_v2 as g210  # noqa: E402
from full_flow_g200_v2 import SCHEMA_CASE, UNKNOWN_ANSWER, _sha256_text  # noqa: E402


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


def _variant(
    evidence_ids: list[str],
    support_ids: list[str],
    target: str,
    *,
    texts: dict[str, str],
) -> dict[str, object]:
    lines = [f"[{index}] ({evidence_id}) {texts[evidence_id]}" for index, evidence_id in enumerate(evidence_ids, 1)]
    return {
        "evidence_ids": evidence_ids,
        "support_evidence_ids": support_ids,
        "prompt": "Evidence:\n" + "\n".join(lines) + "\nQuestion: q?",
        "target": target,
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    old = {
        "schema_version": SCHEMA_CASE,
        "case_id": "niah-old::q-old",
        "dataset": "niah",
        "source": "G200-v1-reviewed-train-fit-reuse",
        "group_id": "q-old",
        "query_id": "q-old",
        "role": "train-fit",
        "component_id": "c-old",
        "answerable": True,
        "target_kind": "niah_qa2d_single_claim",
        "question": "When?",
        "answer": "2001",
        "semantic_target": "It happened in 2001.",
        "semantic_target_sha256": _sha256_text("It happened in 2001."),
        "variants": {
            "support_only": _variant(
                ["old-e1"],
                ["old-e1"],
                "It happened in 2001 [1].",
                texts={"old-e1": "It happened in 2001."},
            )
        },
    }
    twiki = {
        "schema_version": SCHEMA_CASE,
        "case_id": "2wiki::q-train",
        "dataset": "2wiki",
        "source": "official-train-evidences-supporting-facts-context",
        "group_id": "q-train",
        "query_id": "q-train",
        "role": "train-fit",
        "component_id": "c-tw-train",
        "answerable": True,
        "target_kind": "twowiki_evidence_chain",
        "question": "Same country?",
        "answer": "no",
        "semantic_target": "A's country is France. B's country is Canada.",
        "semantic_target_sha256": _sha256_text("A's country is France. B's country is Canada."),
        "official_evidences": [["A", "country", "France"], ["B", "country", "Canada"]],
        "variants": {
            "support_only": _variant(
                ["tw-e1", "tw-e2"],
                ["tw-e1", "tw-e2"],
                "A's country is France [1]. B's country is Canada [2].",
                texts={"tw-e1": "A is in France.", "tw-e2": "B is in Canada."},
            )
        },
    }
    unsupported = {
        "schema_version": SCHEMA_CASE,
        "case_id": "unsupported::2wiki::q-train",
        "dataset": "2wiki",
        "source": "support-removed-from-train-split",
        "group_id": "q-train",
        "query_id": "q-train",
        "role": "train-fit",
        "component_id": "c-tw-train",
        "answerable": False,
        "target_kind": "unsupported_support_removed",
        "question": "Same country?",
        "answer": UNKNOWN_ANSWER,
        "semantic_target": UNKNOWN_ANSWER,
        "semantic_target_sha256": _sha256_text(UNKNOWN_ANSWER),
        "removed_support_evidence_ids": ["tw-e1", "tw-e2"],
        "variants": {
            "support_removed": _variant(
                ["tw-d1"],
                [],
                UNKNOWN_ANSWER,
                texts={"tw-d1": "Distractor."},
            )
        },
    }
    niah_val = {
        "schema_version": SCHEMA_CASE,
        "case_id": "niah-new-modelval::q-val",
        "dataset": "niah",
        "source": "G200-v2-new-parent-disjoint-modelval",
        "group_id": "q-val",
        "query_id": "q-val",
        "role": "train-modelval",
        "component_id": "c-val",
        "answerable": True,
        "target_kind": "niah_qa2d_single_claim",
        "question": "When?",
        "answer": "2009",
        "semantic_target": "It happened in 2009.",
        "semantic_target_sha256": _sha256_text("It happened in 2009."),
        "variants": {
            "support_only": _variant(
                ["val-e1"],
                ["val-e1"],
                "It happened in 2009 [1].",
                texts={"val-e1": "It happened in 2009."},
            )
        },
    }
    train = _write_jsonl(tmp_path / "train_cases.jsonl", [old, twiki, unsupported])
    validation = _write_jsonl(tmp_path / "validation_cases.jsonl", [niah_val])
    manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "status": "PRE_AUDIT",
            "train_cases_sha256": _sha(train),
            "validation_cases_sha256": _sha(validation),
        },
    )
    return manifest, train, validation


def test_structural_audit_builds_true_worklist_and_accepts_yes_no_chain(tmp_path: Path) -> None:
    manifest, train, validation = _fixture(tmp_path)

    summary = g210.structural_audit(
        data_manifest_path=manifest,
        train_cases_path=train,
        validation_cases_path=validation,
        output_dir=tmp_path / "structural",
    )

    assert summary["status"] == "PASS"
    assert summary["true_worklist_rows"] == 3
    rows = [
        json.loads(line)
        for line in (tmp_path / "structural/structural_audit_rows.jsonl").read_text().splitlines()
    ]
    by_case = {row["case_id"]: row for row in rows}
    assert by_case["niah-old::q-old"]["true_reuse"] == "G200-v1-target-audit"
    assert by_case["2wiki::q-train"]["answer_alias_mode"] == "twowiki_yes_no_evidence_chain"
    assert by_case["unsupported::2wiki::q-train"]["structural_pass"] is True


def test_true_audit_and_finalize_write_pre_manual_manifest(tmp_path: Path) -> None:
    manifest, train, validation = _fixture(tmp_path)
    g210.structural_audit(
        data_manifest_path=manifest,
        train_cases_path=train,
        validation_cases_path=validation,
        output_dir=tmp_path / "structural",
    )
    true_manifest = g210.true_audit(
        true_worklist_path=tmp_path / "structural/true_worklist.jsonl",
        true_snapshot=tmp_path / "true",
        output_dir=tmp_path / "true-audit",
        scorer=lambda _premise, _hypothesis: 0.9,
    )
    assert true_manifest["entailed"] == 3

    final = g210.finalize(
        train_cases_path=train,
        validation_cases_path=validation,
        structural_rows_path=tmp_path / "structural/structural_audit_rows.jsonl",
        true_audit_rows_path=tmp_path / "true-audit/true_audit_rows.jsonl",
        output_dir=tmp_path / "final",
        min_niah_train_groups=1,
        min_niah_modelval_groups=1,
        min_twowiki_train_groups=1,
        min_twowiki_modelval_groups=0,
        unsupported_ratio_min=0.0,
        unsupported_ratio_max=1.0,
    )

    assert final["status"] == "PRE_MANUAL_PASS"
    assert final["manual_audit_required_before_training"] is True
    assert final["excluded"] == {}
    assert final["counts"]["2wiki"]["train-fit_unsupported_groups"] == 1
