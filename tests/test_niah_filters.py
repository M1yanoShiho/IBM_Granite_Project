# tests/test_niah_filters.py
"""Tests for src/niah/filters.py — Filter 1 (answerability/positive-anchor) + Filter 2 (hardness)."""
from __future__ import annotations

from src.niah.filters import answers_query, is_hard, keep_distractor, passes_margin


def test_passes_margin_keeps_clearly_below_positive() -> None:
    assert passes_margin(cand_score=0.40, positive_score=0.90, margin=0.05) is True


def test_passes_margin_rejects_near_positive() -> None:
    assert passes_margin(cand_score=0.88, positive_score=0.90, margin=0.05) is False


class _FakeJudge:
    def __init__(self, verdict: str) -> None:
        self.verdict = verdict

    def generate(self, prompt: str) -> str:
        return self.verdict


def test_answers_query_true_when_judge_says_yes() -> None:
    assert answers_query("Paris is the capital.", "capital of France?", _FakeJudge("YES")) is True


def test_answers_query_false_when_judge_says_no() -> None:
    assert answers_query("Rome is the capital.", "capital of France?", _FakeJudge("NO")) is False


def test_is_hard_requires_top_rank_in_both_runs() -> None:
    dense_rank = {"cf0": 2, "cf1": 50}
    sparse_rank = {"cf0": 3, "cf1": 1}
    assert is_hard("cf0", dense_rank, sparse_rank, rank_threshold=10) is True
    assert is_hard("cf1", dense_rank, sparse_rank, rank_threshold=10) is False


def test_is_hard_false_when_absent_from_a_run() -> None:
    assert is_hard("cf9", {"cf9": 1}, {}, rank_threshold=10) is False


def test_keep_distractor_applies_both_filters() -> None:
    assert keep_distractor(
        cand_score=0.4, positive_score=0.9, margin=0.05,
        cand_text="Rome is the capital.", query="capital of France?", judge=_FakeJudge("NO"),
        cand_id="cf0", dense_rank={"cf0": 1}, sparse_rank={"cf0": 2}, rank_threshold=10,
    ) is True


def test_keep_distractor_drops_answer_leaking_candidate() -> None:
    assert keep_distractor(
        cand_score=0.4, positive_score=0.9, margin=0.05,
        cand_text="Paris is the capital.", query="capital of France?", judge=_FakeJudge("YES"),
        cand_id="cf0", dense_rank={"cf0": 1}, sparse_rank={"cf0": 2}, rank_threshold=10,
    ) is False
