# tests/test_niah_types.py
"""Tests for src/niah/types.py — the NIAH task dataclasses."""
from __future__ import annotations

from src.niah.types import Distractor, NiahExample, NiahTask


def test_distractor_carries_source_and_parent() -> None:
    d = Distractor(doc_id="q7__cf0", text="...", source="counterfactual", parent_needle_id="d42")
    assert d.source == "counterfactual"
    assert d.parent_needle_id == "d42"


def test_niahexample_holds_needles_and_distractors() -> None:
    d = Distractor(doc_id="q7__cf0", text="wrong", source="counterfactual", parent_needle_id="d42")
    ex = NiahExample(query_id="q7", query="who?", needle_ids=["d42"], distractors=[d])
    assert ex.needle_ids == ["d42"]
    assert ex.distractors[0].doc_id == "q7__cf0"


def test_niahtask_qrels_mark_only_needles_relevant() -> None:
    task = NiahTask(
        corpus={"d42": "gold", "q7__cf0": "wrong"},
        queries={"q7": "who?"},
        qrels={"q7": {"d42": 1}},
        examples=[],
    )
    assert "q7__cf0" not in task.qrels["q7"]
    assert task.qrels["q7"]["d42"] == 1
