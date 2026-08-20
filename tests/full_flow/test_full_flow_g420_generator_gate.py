from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import full_flow_g420_generator_gate as g420  # noqa: E402


def _metric(value: float) -> dict[str, float]:
    return {
        "runtime_error": 0.0,
        "missing_trace": 0.0,
        "draft_invalid_citation": 0.0,
        "answer_match": value,
        "coverage": value,
        "minicheck_citation_precision": value,
        "minicheck_citation_recall": value,
        "correct_and_cited": value,
        "unsupported_ungrounded_assertion": value,
    }


def _row(
    *,
    schema_version: str,
    dataset: str,
    context: str,
    component: str,
    query: str,
    g0: float,
    candidate: float,
) -> dict[str, object]:
    arms = {
        "G0": _metric(g0),
        "GRC13": _metric(candidate),
        "GRC42": _metric(candidate),
        "GRC73": _metric(candidate),
    }
    key = "configs" if schema_version == "full-flow-g400-scored-row-v1" else "arms"
    return {
        "schema_version": schema_version,
        "task_id": f"{query}:{context}",
        "dataset": dataset,
        "context": context,
        "component_id": component,
        "query_id": query,
        key: arms,
    }


def _report(status: str) -> dict[str, object]:
    return {
        "stage": status.split("_", 1)[0],
        "status": status,
        "formal_task_subset": False,
        "gate": {
            "technical_pass": True,
            "responsibility_pass": True,
            "tripwires": [],
            "positive_signal": True,
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_query_level_deltas_average_context_variants_before_bootstrap() -> None:
    rows = [
        _row(
            schema_version="full-flow-g410-scored-row-v1",
            dataset="2wiki",
            context="topk",
            component="c1",
            query="q1",
            g0=0.0,
            candidate=1.0,
        ),
        _row(
            schema_version="full-flow-g410-scored-row-v1",
            dataset="2wiki",
            context="support_only",
            component="c1",
            query="q1",
            g0=0.0,
            candidate=0.0,
        ),
    ]

    values = g420.query_level_deltas(rows, metric="correct_and_cited", dataset="2wiki")

    assert len(values) == 1
    assert values[0]["delta"] == 0.5


def test_evaluate_marks_new_generator_ready_with_positive_g400_g410(tmp_path: Path) -> None:
    g400_report = tmp_path / "g400_score.json"
    g410_report = tmp_path / "g410_score.json"
    g400_rows = tmp_path / "g400_rows.jsonl"
    g410_rows = tmp_path / "g410_rows.jsonl"
    _write_json(g400_report, _report("G400_NIAH_RESPONSIBILITY_PASS"))
    _write_json(g410_report, _report("G410_CROSS_DATA_RESPONSIBILITY_PASS"))
    g400_contexts = [
        "K_topk",
        "S_legacy_selected",
        "O_support_only",
        "OB_support_benign",
        "OH_support_harmful",
        "OP_support_last",
        "train_unsupported_safety",
    ]
    g410_contexts = ["topk", "support_only", "support_first", "support_middle", "support_last"]
    _write_jsonl(
        g400_rows,
        [
            _row(
                schema_version="full-flow-g400-scored-row-v1",
                dataset="niah",
                context=context,
                component=f"c{idx}",
                query=f"q{idx}",
                g0=0.0,
                candidate=1.0,
            )
            for idx, context in enumerate(g400_contexts)
        ],
    )
    _write_jsonl(
        g410_rows,
        [
            _row(
                schema_version="full-flow-g410-scored-row-v1",
                dataset="2wiki",
                context=context,
                component=f"tc{idx}",
                query=f"tq{idx}",
                g0=0.0,
                candidate=1.0,
            )
            for idx, context in enumerate(g410_contexts)
        ],
    )

    report = g420.evaluate(
        g400_score_report_path=g400_report,
        g400_scored_rows_path=g400_rows,
        g410_score_report_path=g410_report,
        g410_scored_rows_path=g410_rows,
        output_json=tmp_path / "out.json",
        output_report=tmp_path / "out.md",
        resamples=100,
    )

    assert report["status"] == "G420_NEW_GENERATOR_QUALIFIED_G430_READY"
    assert report["teacher_recommendation"] == "FREEZE_NEW_GRC_IN_G430"
    assert report["boundaries"]["sealed_or_heldout_read"] is False
