# tests/test_niah_hardness_gate.py
"""Tests for src/niah/hardness_gate.py — the non-saturation gate."""
from __future__ import annotations

from src.niah.hardness_gate import gate_report, is_saturated, mean_recall, recall_at_k


def test_recall_at_k_counts_found_gold_in_top_k() -> None:
    ranked = ["d3", "d1", "d9"]          # retriever output, best-first
    assert recall_at_k(ranked, {"d1"}, k=3) == 1.0
    assert recall_at_k(ranked, {"d1"}, k=1) == 0.0     # d1 is at rank 2
    assert recall_at_k(ranked, {"d1", "d9"}, k=3) == 1.0


def test_recall_at_k_no_gold_is_zero() -> None:
    assert recall_at_k(["d1", "d2"], set(), k=3) == 0.0


def test_mean_recall_averages_per_query() -> None:
    per_q = {"q1": 1.0, "q2": 0.0}
    assert mean_recall(per_q) == 0.5


def test_is_saturated_true_above_threshold() -> None:
    assert is_saturated(0.97, threshold=0.95) is True
    assert is_saturated(0.80, threshold=0.95) is False


def test_gate_report_flags_pass_or_fail() -> None:
    rep = gate_report({"q1": 0.6, "q2": 0.4}, threshold=0.95)
    assert rep["mean_recall"] == 0.5
    assert rep["saturated"] is False
    assert rep["passes_gate"] is True   # NOT saturated -> the task is hard enough
