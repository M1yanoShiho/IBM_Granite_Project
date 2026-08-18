from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_flow_g218_target_repair as g218  # noqa: E402


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


def _variant(evidence_ids: list[str], support_ids: list[str], target: str) -> dict[str, object]:
    return {
        "evidence_ids": evidence_ids,
        "support_evidence_ids": support_ids,
        "prompt": "Evidence:\n"
        + "\n".join(
            f"[{index}] ({evidence_id}) Evidence text for {evidence_id}."
            for index, evidence_id in enumerate(evidence_ids, start=1)
        )
        + "\nQuestion: q?\nAnswer:",
        "target": target,
    }


def _twowiki_case(case_id: str, *, role: str = g218.TRAIN_ROLE) -> dict[str, object]:
    return {
        "schema_version": g218.SCHEMA_CASE,
        "case_id": case_id,
        "dataset": "2wiki",
        "source": "fixture",
        "group_id": case_id.removeprefix("2wiki::"),
        "query_id": case_id.removeprefix("2wiki::"),
        "role": role,
        "component_id": f"component::{case_id}",
        "answerable": True,
        "target_kind": "twowiki_evidence_chain",
        "question": "Who is the father-in-law?",
        "answer": "Eliel Saarinen",
        "semantic_target": "She was the first wife of Eero Saarinen. He was the son of Eliel Saarinen.",
        "semantic_sentences": [
            "She was the first wife of Eero Saarinen.",
            "He was the son of Eliel Saarinen.",
        ],
        "official_supporting_facts": [
            {
                "title": "Lilian Swann Saarinen",
                "sentence": "She was the first wife of Eero Saarinen.",
                "sentence_index": 1,
                "official_evidence": ["Lilian Swann Saarinen", "spouse", "Eero Saarinen"],
            },
            {
                "title": "Eero Saarinen",
                "sentence": "He was the son of Eliel Saarinen.",
                "sentence_index": 2,
                "official_evidence": ["Eero Saarinen", "father", "Eliel Saarinen"],
            },
        ],
        "variants": {
            "support_only": _variant(
                ["e1", "e2"],
                ["e1", "e2"],
                "She was the first wife of Eero Saarinen [1]. He was the son of Eliel Saarinen [2].",
            ),
            "topk": _variant(
                ["d1", "e2", "e1"],
                ["e1", "e2"],
                "She was the first wife of Eero Saarinen [3]. He was the son of Eliel Saarinen [2].",
            ),
        },
    }


def _niah_case(case_id: str, *, question: str, target: str) -> dict[str, object]:
    return {
        "schema_version": g218.SCHEMA_CASE,
        "case_id": case_id,
        "dataset": "niah",
        "source": "G200-v2-new-parent-disjoint-modelval",
        "group_id": case_id,
        "query_id": case_id,
        "role": g218.VALIDATION_ROLE,
        "component_id": f"component::{case_id}",
        "answerable": True,
        "target_kind": "niah_qa2d_single_claim",
        "question": question,
        "answer": "Ethan Suplee",
        "semantic_target": target,
        "variants": {"support_only": _variant(["n1"], ["n1"], f"{target} [1].")},
    }


def _unsupported_counterpart(case_id: str) -> dict[str, object]:
    group_id = case_id.removeprefix("2wiki::")
    return {
        "schema_version": g218.SCHEMA_CASE,
        "case_id": f"unsupported::2wiki::{group_id}",
        "dataset": "2wiki",
        "source": "fixture",
        "group_id": group_id,
        "query_id": group_id,
        "role": g218.TRAIN_ROLE,
        "component_id": f"component::{group_id}",
        "answerable": False,
        "target_kind": "unsupported_support_removed",
        "question": "q?",
        "answer": g218.UNKNOWN_ANSWER,
        "semantic_target": g218.UNKNOWN_ANSWER,
        "removed_support_evidence_ids": ["e1"],
        "variants": {"support_removed": _variant(["d1"], [], g218.UNKNOWN_ANSWER)},
    }


def _bundle(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    train_rows = [
        _twowiki_case("2wiki::pronoun-train"),
        _unsupported_counterpart("2wiki::pronoun-train"),
    ]
    validation_rows = [
        _twowiki_case("2wiki::pronoun-val", role=g218.VALIDATION_ROLE),
        _niah_case(
            "niah-new-modelval::1015",
            question="who played randy on my name is earl",
            target="Ethan Suplee played randy on my name.",
        ),
        _niah_case(
            "niah-new-modelval::keep",
            question="who played randy on my name is earl",
            target="Ethan Suplee played randy on my name is earl.",
        ),
    ]
    train = _write_jsonl(tmp_path / "train_cases.jsonl", train_rows)
    validation = _write_jsonl(tmp_path / "validation_cases.jsonl", validation_rows)
    manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": g218.SCHEMA_G216_MANIFEST,
            "status": "PRE_MANUAL_PASS",
            "training_started": False,
            "sealed_or_heldout_read": False,
            "dev_read": False,
            "train_cases_sha256": _sha(train),
            "validation_cases_sha256": _sha(validation),
        },
    )
    review_rows = _write_jsonl(
        tmp_path / "review_rows.jsonl",
        [{"case_id": "niah-new-modelval::1015", "review_decision": "FAIL"}],
    )
    return manifest, train, validation, review_rows


def test_repair_anchors_twowiki_sentences_and_filters_detectable_niah(tmp_path: Path) -> None:
    manifest, train, validation, review_rows = _bundle(tmp_path)

    report = g218.repair(
        manifest_path=manifest,
        train_cases_path=train,
        validation_cases_path=validation,
        review_rows_paths=[review_rows],
        output_dir=tmp_path / "g218",
        min_niah_train_groups=0,
        min_niah_modelval_groups=1,
        min_twowiki_train_groups=1,
        min_twowiki_modelval_groups=1,
        unsupported_ratio_min=0.0,
        unsupported_ratio_max=1.0,
    )

    assert report["status"] == "PRE_AUDIT"
    assert report["requires_structural_true_rerun"] is True
    assert report["removed_case_count"] == 1
    assert report["repair_counts"]["niah_filter_reason::acted_on_title_truncation"] == 1
    train_rows = [
        json.loads(line)
        for line in (tmp_path / "g218/train_cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    validation_rows = [
        json.loads(line)
        for line in (tmp_path / "g218/validation_cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train_case = next(row for row in train_rows if row["case_id"] == "2wiki::pronoun-train")
    val_case = next(row for row in validation_rows if row["case_id"] == "2wiki::pronoun-val")
    assert train_case["semantic_sentences"] == [
        "Lilian Swann Saarinen was the first wife of Eero Saarinen.",
        "Eero Saarinen was the son of Eliel Saarinen.",
    ]
    assert "Lilian Swann Saarinen was the first wife of Eero Saarinen [1]." in train_case["variants"]["support_only"]["target"]
    assert "Eero Saarinen was the son of Eliel Saarinen [2]." in train_case["variants"]["support_only"]["target"]
    assert "Lilian Swann Saarinen was the first wife of Eero Saarinen [3]." in train_case["variants"]["topk"]["target"]
    assert val_case["target_construction"] == g218.TWOWIKI_TARGET_CONSTRUCTION
    assert {row["case_id"] for row in validation_rows} == {
        "2wiki::pronoun-val",
        "niah-new-modelval::keep",
    }


def test_anchor_handles_short_title_and_after_release() -> None:
    short = g218._anchor_twowiki_sentence(
        "Kennedy was born in Brookline, Massachusetts.",
        {
            "title": "Robert F. Kennedy",
            "official_evidence": ["Robert F. Kennedy", "place of birth", "Brookline"],
        },
    )
    after = g218._anchor_twowiki_sentence(
        "After release it quickly became a Yugoslav hit.",
        {
            "title": "The Blue 9",
            "official_evidence": ["The Blue 9", "country of origin", "Yugoslav"],
        },
    )

    assert short == ("Robert F. Kennedy was born in Brookline, Massachusetts.", "replace_leading_suffix_anchor")
    assert after == ("After release The Blue 9 quickly became a Yugoslav hit.", "replace_after_release_it")
