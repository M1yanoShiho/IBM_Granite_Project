from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_flow_g219_target_repair as g219  # noqa: E402


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


def _twowiki_case(case_id: str, *, role: str = g219.TRAIN_ROLE) -> dict[str, object]:
    return {
        "schema_version": g219.SCHEMA_CASE,
        "case_id": case_id,
        "dataset": "2wiki",
        "source": "fixture",
        "group_id": case_id.removeprefix("2wiki::"),
        "query_id": case_id.removeprefix("2wiki::"),
        "role": role,
        "component_id": f"component::{case_id}",
        "answerable": True,
        "target_kind": "twowiki_evidence_chain",
        "question": "Where was the composer born?",
        "answer": "Madras",
        "semantic_target": "The film's score was composed by A. R. Rahman. He is nicknamed Mozart of Madras.",
        "semantic_sentences": [
            "The film's score was composed by A. R. Rahman.",
            "He is nicknamed Mozart of Madras.",
        ],
        "official_supporting_facts": [
            {
                "title": "Raavan",
                "sentence": "The film's score was composed by A. R. Rahman.",
                "sentence_index": 5,
                "official_evidence": ["Raavan", "composer", "A. R. Rahman"],
            },
            {
                "title": "A. R. Rahman",
                "sentence": "He is nicknamed Mozart of Madras.",
                "sentence_index": 2,
                "official_evidence": ["A. R. Rahman", "place of birth", "Madras"],
            },
        ],
        "variants": {
            "support_only": _variant(
                ["e1", "e2"],
                ["e1", "e2"],
                "The film's score was composed by A. R. Rahman [1]. He is nicknamed Mozart of Madras [2].",
            )
        },
    }


def _niah_case(case_id: str, *, question: str, target: str, role: str = g219.TRAIN_ROLE) -> dict[str, object]:
    return {
        "schema_version": g219.SCHEMA_CASE,
        "case_id": case_id,
        "dataset": "niah",
        "source": "fixture",
        "group_id": case_id,
        "query_id": case_id,
        "role": role,
        "component_id": f"component::{case_id}",
        "answerable": True,
        "target_kind": "niah_qa2d_single_claim",
        "question": question,
        "answer": "Tony Curran",
        "semantic_target": target,
        "variants": {"support_only": _variant(["n1"], ["n1"], f"{target} [1].")},
    }


def _unsupported_case(case_id: str) -> dict[str, object]:
    return {
        "schema_version": g219.SCHEMA_CASE,
        "case_id": f"unsupported::{case_id}",
        "dataset": "2wiki",
        "source": "fixture",
        "group_id": case_id.removeprefix("2wiki::"),
        "query_id": case_id.removeprefix("2wiki::"),
        "role": g219.TRAIN_ROLE,
        "component_id": f"component::{case_id}",
        "answerable": False,
        "target_kind": "unsupported_support_removed",
        "question": "q?",
        "answer": "I don't know.",
        "semantic_target": "I don't know.",
        "variants": {"support_removed": _variant(["d1"], [], "I don't know.")},
    }


def _bundle(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    train_rows = [
        _twowiki_case("2wiki::train"),
        _unsupported_case("2wiki::train"),
        _niah_case(
            "niah-old::10517",
            question="who plays vincent van gogh in doctor who",
            target="Tony Curran plays vincent van gogh in doctor.",
        ),
    ]
    validation_rows = [
        _twowiki_case("2wiki::val", role=g219.VALIDATION_ROLE),
        _niah_case(
            "niah-new-modelval::keep",
            question="who plays vincent van gogh in doctor who",
            target="Tony Curran plays vincent van gogh in doctor who.",
            role=g219.VALIDATION_ROLE,
        ),
    ]
    train = _write_jsonl(tmp_path / "train_cases.jsonl", train_rows)
    validation = _write_jsonl(tmp_path / "validation_cases.jsonl", validation_rows)
    manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": g219.SCHEMA_INPUT_FINAL_MANIFEST,
            "status": "PRE_MANUAL_PASS",
            "training_started": False,
            "sealed_or_heldout_read": False,
            "dev_read": False,
            "train_cases_sha256": _sha(train),
            "validation_cases_sha256": _sha(validation),
        },
    )
    review_rows = _write_jsonl(
        tmp_path / "sample_review_rows.jsonl",
        [
            {
                "case_id": "niah-old::10517",
                "dataset": "niah",
                "review_decision": "FAIL",
                "review_reason": "QA2D target truncates Doctor Who to doctor.",
            }
        ],
    )
    return manifest, train, validation, review_rows


def test_g219_rewrites_relation_targets_and_filters_malformed_niah(tmp_path: Path) -> None:
    manifest, train, validation, review_rows = _bundle(tmp_path)

    report = g219.repair(
        manifest_path=manifest,
        train_cases_path=train,
        validation_cases_path=validation,
        review_rows_paths=[review_rows],
        output_dir=tmp_path / "g219",
        min_niah_train_groups=0,
        min_niah_modelval_groups=1,
        min_twowiki_train_groups=1,
        min_twowiki_modelval_groups=1,
        unsupported_ratio_min=0.0,
        unsupported_ratio_max=1.0,
    )

    assert report["status"] == "PRE_AUDIT"
    assert report["removed_case_count"] == 1
    assert report["repair_counts"]["niah_filter_reason::acted_on_title_truncation"] == 1
    train_rows = [
        json.loads(line)
        for line in (tmp_path / "g219/train_cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train_case = next(row for row in train_rows if row["case_id"] == "2wiki::train")
    assert train_case["semantic_sentences"] == [
        "Raavan's score was composed by A. R. Rahman.",
        "A. R. Rahman was born in Madras.",
    ]
    assert "A. R. Rahman was born in Madras [2]." in train_case["variants"]["support_only"]["target"]
    assert {row["case_id"] for row in train_rows} == {"2wiki::train", "unsupported::2wiki::train"}


def test_g219_detects_sampled_niah_malformation_patterns() -> None:
    failures = {"case": ["failed in sample review"]}
    rows = [
        _niah_case(
            "case",
            question="what season of american horror story has cuba gooding jr",
            target="Cuba has gooding the sixth season of american horror story.",
        ),
        _niah_case(
            "define",
            question="raised line markers on the roadway and shoulder are used to define",
            target="Raised line markers on the roadway and shoulder are used to define for increased visibility.",
        ),
        _niah_case(
            "duet",
            question="meatloaf duet it 's all coming back to me now",
            target="Meatloaf duet Marion Raven is all coming back to me now.",
        ),
    ]

    reasons = [g219._niah_filter_reasons(row, failures) for row in rows]

    assert "has_entity_phrase_lost" in reasons[0]
    assert "g212m3_failed_niah_qa2d_malformation" in reasons[0]
    assert reasons[1] == ["qa2d_define_for_malformed"]
    assert reasons[2] == ["qa2d_duet_malformed"]
