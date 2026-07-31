import json
from pathlib import Path

import pytest

from evidence_rag.evaluation.cluster_rescore import (
    METRICS,
    DumpRow,
    per_query_metric,
    read_dump,
    rescore,
)
from evidence_rag.selector.answer_equivalence import lenient_equivalent


def _row(
    answers: list[str], *, alias_flags: list[bool], query_id: str = "q1"
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "needle_document_id": "needle",
        "counterfactual_document_id": "cf::needle",
        "gold_value": "Kennedy",
        "gold_aliases": ["Kennedy"],
        "window": [
            {
                "evidence_id": f"e{index}",
                "document_id": (
                    "needle" if index == 0 else ("cf::needle" if index == 1 else f"d{index}")
                ),
                "retrieval_rank": index + 1,
                "answer": answer,
                "contains_gold_alias": flag,
            }
            for index, (answer, flag) in enumerate(zip(answers, alias_flags, strict=True))
        ],
    }


def test_read_dump_parses_rows(tmp_path: Path) -> None:
    path = tmp_path / "dump.jsonl"
    path.write_text(
        json.dumps(_row(["Kennedy", "Nixon"], alias_flags=[True, False])) + "\n", encoding="utf-8"
    )
    rows = read_dump(path)
    assert len(rows) == 1
    assert isinstance(rows[0], DumpRow)
    assert rows[0].answers == ("Kennedy", "Nixon")


def test_read_dump_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "dump.jsonl"
    path.write_text(
        json.dumps(_row(["Kennedy", "Nixon"], alias_flags=[True, False])) + "\n\n",
        encoding="utf-8",
    )
    assert len(read_dump(path)) == 1


def test_rebuilt_window_reproduces_the_recorded_alias_flags() -> None:
    """The dump stores a boolean, not the passage, so `window()` must synthesise text that
    `contains_alias` scores identically — otherwise fixed-denominator rescoring silently drifts."""
    from evidence_rag.evaluation.cluster_eval import contains_alias

    row = DumpRow.from_json(_row(["Kennedy", "Nixon", "paul"], alias_flags=[True, False, True]))
    rebuilt = row.window()
    assert [contains_alias(c.text, row.gold_aliases) for c in rebuilt] == [True, False, True]


def test_rescore_exact_matches_conflict_detection() -> None:
    rows = (DumpRow.from_json(_row(["Kennedy", "Nixon"], alias_flags=[True, False])),)
    report = rescore(rows, equivalence=None)
    assert report.missed_conflict.rate == 0.0
    assert report.needle_gold_recovery.rate == 1.0


def test_rescore_lenient_merges_containment() -> None:
    rows = (
        DumpRow.from_json(
            _row(["Apostle Paul", "Nixon", "paul"], alias_flags=[True, False, True])
        ),
    )
    lenient = rescore(rows, equivalence=lenient_equivalent)
    exact = rescore(rows, equivalence=None)
    assert lenient.fixed_false_conflict.rate == 0.0
    assert exact.fixed_false_conflict.rate == 1.0


def test_per_query_metric_returns_a_paired_testable_mapping() -> None:
    rows = (
        DumpRow.from_json(_row(["Kennedy", "Nixon"], alias_flags=[True, False], query_id="q1")),
        DumpRow.from_json(_row(["Nixon", "Nixon"], alias_flags=[True, False], query_id="q2")),
    )
    mapping = per_query_metric(rows, metric="needle_gold_recovery", equivalence=None)
    assert mapping == {"q1": 1.0, "q2": 0.0}


def test_per_query_metric_reports_unscored_queries_as_none() -> None:
    """compare_paired drops a query unless BOTH arms scored it, so unscored must stay None."""
    rows = (DumpRow.from_json(_row(["Kennedy", "Nixon"], alias_flags=[True, False])),)
    mapping = per_query_metric(rows, metric="fixed_false_conflict", equivalence=None)
    assert mapping == {"q1": None}


def test_per_query_metric_rejects_an_unknown_metric() -> None:
    rows = (DumpRow.from_json(_row(["Kennedy", "Nixon"], alias_flags=[True, False])),)
    with pytest.raises(ValueError, match="unknown metric"):
        per_query_metric(rows, metric="accuracy", equivalence=None)


def test_metrics_covers_every_scored_field() -> None:
    assert set(METRICS) == {
        "missed_conflict",
        "false_conflict",
        "fixed_false_conflict",
        "needle_gold_recovery",
    }
