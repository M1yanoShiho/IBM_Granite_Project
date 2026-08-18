from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import full_flow_g310_seed13_screen as g310  # noqa: E402


def _prompt() -> str:
    return (
        "Answer the question using only the evidence below.\n\n"
        "Evidence:\n"
        "[1] (ev-a) Alpha was released in 2010.\n"
        "[2] (ev-b) Beta was released in 2009.\n\n"
        "Question: Which came first, Alpha or Beta?\n"
        "Answer:"
    )


def _answerable_case() -> dict[str, object]:
    return {
        "schema_version": "full-flow-g200-case-v2",
        "case_id": "case-a",
        "query_id": "q-a",
        "component_id": "component-a",
        "dataset": "2wiki",
        "role": "train-modelval",
        "target_kind": "twowiki_evidence_chain",
        "answerable": True,
        "answer": "Beta",
        "official_answer": "Beta",
        "question": "Which came first, Alpha or Beta?",
        "variants": {
            "topk": {
                "prompt": _prompt(),
                "target": "Beta was released in 2009 [2].",
                "support_evidence_ids": ["ev-b"],
            }
        },
    }


def _unsupported_case() -> dict[str, object]:
    return {
        **_answerable_case(),
        "case_id": "unsupported::case-a",
        "query_id": "q-u",
        "role": "train-fit",
        "target_kind": "unsupported_support_removed",
        "answerable": False,
        "answer": "",
        "official_answer": "",
        "variants": {
            "support_removed": {
                "prompt": _prompt(),
                "target": "I don't know.",
                "support_evidence_ids": [],
            }
        },
    }


def _generation(answer: str, cited: list[str]) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "query_id": "task",
        "answer": answer,
        "cited_evidence_ids": cited,
    }


def _arm(answer: str, cited: list[str]) -> dict[str, object]:
    routing = []
    if answer:
        routing = [
            {
                "sentence": "Beta was released in 2009.",
                "citation": cited[0] if cited else None,
                "outcome": "verified" if cited else "unverified",
            }
        ]
    return {
        "generation": _generation(answer, cited),
        "trace": {"draft": {"raw_draft_text": answer}},
        "routing": routing,
        "error": None,
        "seconds": 0.1,
    }


def test_parse_prompt_evidence_builds_ordered_candidates() -> None:
    evidence = g310.parse_prompt_evidence(
        prompt=_prompt(),
        case_id="case-a",
        variant_name="topk",
    )

    assert [item.evidence_id for item in evidence] == ["ev-a", "ev-b"]
    assert [item.retrieval_rank for item in evidence] == [1, 2]
    assert evidence[1].text == "Beta was released in 2009."


def test_build_tasks_keeps_validation_and_unsupported_safety_separate() -> None:
    tasks = g310.build_tasks(
        train_rows=[_unsupported_case()],
        validation_rows=[_answerable_case()],
    )

    assert [task.scope for task in tasks] == [
        g310.ANSWERABLE_SCOPE,
        g310.UNSUPPORTED_SCOPE,
    ]
    assert tasks[0].reference_answers == ("Beta",)
    assert tasks[0].support_evidence_ids == ("ev-b",)
    assert tasks[1].expected_unknown is True


def test_score_selects_surviving_recipe_with_tiebreak(tmp_path: Path) -> None:
    answerable = {
        "schema_version": g310.SCHEMA_GENERATION_ROW,
        "task_id": "validation-answerable::case-a::topk",
        "scope": g310.ANSWERABLE_SCOPE,
        "dataset": "2wiki",
        "case_id": "case-a",
        "variant_name": "topk",
        "answerable": True,
        "reference_answers": ["Beta"],
        "support_evidence_ids": ["ev-b"],
        "evidence": [
            {"evidence_id": "ev-a", "text": "Alpha was released in 2010."},
            {"evidence_id": "ev-b", "text": "Beta was released in 2009."},
        ],
        "arms": {
            "G0": _arm("Beta [2].", ["ev-b"]),
            "GR-F": _arm("Beta [2].", ["ev-b"]),
            "GR-C": _arm("Beta [2].", ["ev-b"]),
        },
    }
    unsupported = {
        **answerable,
        "task_id": "train-unsupported-safety::unsupported::case-a::support_removed",
        "scope": g310.UNSUPPORTED_SCOPE,
        "case_id": "unsupported::case-a",
        "answerable": False,
        "reference_answers": [],
        "support_evidence_ids": [],
        "arms": {
            "G0": _arm("", []),
            "GR-F": _arm("", []),
            "GR-C": _arm("", []),
        },
    }
    generations = tmp_path / "generations.jsonl"
    generations.write_text(
        json.dumps(answerable) + "\n" + json.dumps(unsupported) + "\n",
        encoding="utf-8",
    )

    report = g310.score(
        generations_path=generations,
        output_json=tmp_path / "report.json",
        output_rows=tmp_path / "rows.jsonl",
        output_report=tmp_path / "report.md",
        entails=lambda premise, hypothesis: "Beta was released in 2009." in premise
        and "Beta was released in 2009." in hypothesis,
    )

    assert report["status"] == "RECIPE_SELECTED"
    assert report["decision"]["recipe"] == "GR-F"
    assert report["aggregate"]["validation_answerable"]["GR-F"]["correct_and_cited"] == 1.0
    assert (
        report["aggregate"]["validation_answerable"]["GR-F"][
            "minicheck_citation_precision"
        ]
        == 1.0
    )
