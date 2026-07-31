import json
from pathlib import Path

import pytest

from evidence_rag.evaluation.cluster_rescore import DumpRow
from evidence_rag.evaluation.support_strata import (
    STRATA,
    analyse,
    stratum_label,
    summarize_strata,
)


def _row(
    answers: list[str],
    *,
    document_ids: list[str] | None = None,
    query_id: str = "q1",
    gold: str = "Kennedy",
) -> DumpRow:
    ids = document_ids or (["needle", "cf::needle"] + [f"d{i}" for i in range(len(answers) - 2)])
    payload = {
        "query_id": query_id,
        "needle_document_id": "needle",
        "counterfactual_document_id": "cf::needle",
        "gold_value": gold,
        "gold_aliases": [gold],
        "window": [
            {
                "evidence_id": f"e{index}",
                "document_id": ids[index],
                "retrieval_rank": index + 1,
                "answer": answer,
                "contains_gold_alias": answer == gold,
            }
            for index, answer in enumerate(answers)
        ],
    }
    return DumpRow.from_json(payload)


def test_stratum_label_buckets() -> None:
    assert stratum_label(0) == "0"
    assert stratum_label(1) == "1"
    assert stratum_label(2) == "2"
    assert stratum_label(3) == ">=3"
    assert stratum_label(9) == ">=3"
    assert STRATA == ("0", "1", "2", ">=3")


def test_gold_support_counts_the_gold_cluster() -> None:
    rows = (_row(["Kennedy", "Nixon", "Kennedy", "Kennedy"]),)
    result = analyse(rows, equivalence=None, parent_by_document=None)
    assert result[0].gold_support == 3


def test_gold_support_is_zero_when_no_cluster_carries_the_gold_answer() -> None:
    rows = (_row(["Nixon", "Nixon", "Truman"]),)
    result = analyse(rows, equivalence=None, parent_by_document=None)
    assert result[0].gold_support == 0


def test_parent_unit_collapses_same_article_support() -> None:
    """Three passages of one article are one source, so the gate loses the votes that let it
    reach the margin — this is the whole point of the stratification."""
    rows = (
        _row(
            ["Kennedy", "Nixon", "Kennedy", "Kennedy"],
            document_ids=["needle", "cf::needle", "d1", "d2"],
        ),
    )
    parents = {"needle": "page a", "d1": "page a", "d2": "page a", "cf::needle": "page a"}
    document = analyse(rows, equivalence=None, parent_by_document=None)
    parent = analyse(rows, equivalence=None, parent_by_document=parents)
    assert document[0].gold_support == 3
    assert parent[0].gold_support == 1


def test_poison_is_droppable_only_when_gold_reaches_three() -> None:
    """The cf cluster is always a single source, so with margin 2 and cap 1 the gate can act on
    the poison only when the gold cluster has at least three independent sources."""
    three = analyse((_row(["Kennedy", "Nixon", "Kennedy", "Kennedy"]),),
                    equivalence=None, parent_by_document=None)
    two = analyse((_row(["Kennedy", "Nixon", "Kennedy"]),),
                  equivalence=None, parent_by_document=None)
    assert three[0].gold_support == 3
    assert three[0].poison_dropped is True
    assert two[0].gold_support == 2
    assert two[0].poison_dropped is False


def test_true_needle_leaves_the_gate_structurally_silent() -> None:
    """gold_support == 1 is the actual needle case. The gate can never reach the margin, so it
    cannot act on the poison at all — no tuning changes this."""
    rows = (_row(["Kennedy", "Nixon"]),)
    result = analyse(rows, equivalence=None, parent_by_document=None)
    assert result[0].gold_support == 1
    assert result[0].poison_dropped is False


def test_needle_droppable_is_reported_separately() -> None:
    """An isolated needle out-voted by a distractor answer is the recall-loss path."""
    rows = (_row(["Kennedy", "Nixon", "Nixon", "Nixon"]),)
    result = analyse(rows, equivalence=None, parent_by_document=None)
    assert result[0].gold_support == 1
    assert result[0].needle_dropped is True
    assert result[0].poison_dropped is False


def test_scores_none_when_the_document_is_absent_from_the_window() -> None:
    rows = (_row(["Nixon", "Truman"], document_ids=["d1", "d2"]),)
    result = analyse(rows, equivalence=None, parent_by_document=None)
    assert result[0].poison_dropped is None
    assert result[0].needle_dropped is None


def test_summarize_reports_counts_and_rates_per_stratum() -> None:
    rows = (
        _row(["Kennedy", "Nixon", "Kennedy", "Kennedy"], query_id="q1"),
        _row(["Kennedy", "Nixon"], query_id="q2"),
        _row(["Kennedy", "Nixon"], query_id="q3"),
    )
    summary = summarize_strata(analyse(rows, equivalence=None, parent_by_document=None))
    assert summary["n_cases"] == 3
    assert summary["gold_support_histogram"]["1"] == 2
    assert summary["gold_support_histogram"][">=3"] == 1
    assert summary["true_needle_rate"] == pytest.approx(2 / 3)
    assert summary["gate_can_fire_rate"] == pytest.approx(1 / 3)
    assert summary["strata"]["1"]["poison_dropped_rate"] == 0.0
    assert summary["strata"][">=3"]["poison_dropped_rate"] == 1.0


def test_summarize_handles_an_empty_stratum() -> None:
    rows = (_row(["Kennedy", "Nixon"]),)
    summary = summarize_strata(analyse(rows, equivalence=None, parent_by_document=None))
    assert summary["strata"]["2"]["n"] == 0
    assert summary["strata"]["2"]["poison_dropped_rate"] is None


def test_cli_round_trip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from evidence_rag.evaluation.support_strata_cli import main

    dump = tmp_path / "dump.jsonl"
    payloads = [
        {
            "query_id": "q1",
            "needle_document_id": "needle",
            "counterfactual_document_id": "cf::needle",
            "gold_value": "Kennedy",
            "gold_aliases": ["Kennedy"],
            "window": [
                {"evidence_id": "e0", "document_id": "needle", "retrieval_rank": 1,
                 "answer": "Kennedy", "contains_gold_alias": True},
                {"evidence_id": "e1", "document_id": "cf::needle", "retrieval_rank": 2,
                 "answer": "Nixon", "contains_gold_alias": False},
            ],
        }
    ]
    dump.write_text(
        "".join(json.dumps(p, sort_keys=True) + "\n" for p in payloads), encoding="utf-8"
    )
    output = tmp_path / "strata.json"
    assert main(["--dump", str(dump), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    for unit in ("document", "parent"):
        for scoring in ("exact", "lenient"):
            assert payload[unit][scoring]["n_cases"] == 1
    assert payload["document"]["exact"]["true_needle_rate"] == 1.0
