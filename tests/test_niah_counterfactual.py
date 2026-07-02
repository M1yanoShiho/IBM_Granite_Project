# tests/test_niah_counterfactual.py
"""Tests for src/niah/counterfactual.py — Source A (entity substitution)."""
from __future__ import annotations

import pytest

from src.niah.counterfactual import make_counterfactual, propose_wrong_entity, swap_entity


def test_swap_entity_replaces_all_occurrences() -> None:
    text = "Linda Davis won in 1994. Linda Davis was the artist."
    assert swap_entity(text, "Linda Davis", "Mary Jones") == (
        "Mary Jones won in 1994. Mary Jones was the artist."
    )


def test_swap_entity_is_case_sensitive_exact() -> None:
    assert swap_entity("Paris and paris", "Paris", "Rome") == "Rome and paris"


def test_swap_entity_no_match_returns_unchanged() -> None:
    assert swap_entity("no entity here", "Xyz", "Abc") == "no entity here"


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_propose_wrong_entity_parses_llm_reply() -> None:
    llm = _FakeLLM("  Mary Jones\n")
    assert propose_wrong_entity("Linda Davis", llm) == "Mary Jones"
    assert "Linda Davis" in llm.prompts[0]


def test_propose_wrong_entity_rejects_echo() -> None:
    llm = _FakeLLM("Linda Davis")
    with pytest.raises(ValueError, match="same entity"):
        propose_wrong_entity("Linda Davis", llm)


def test_make_counterfactual_swaps_answer_in_passage() -> None:
    llm = _FakeLLM("Mary Jones")
    out = make_counterfactual("Linda Davis won the 1994 award.", "Linda Davis", llm)
    assert out == "Mary Jones won the 1994 award."


def test_make_counterfactual_rejects_no_op_swap() -> None:
    # answer absent from the passage -> swap is a no-op -> would duplicate the needle
    llm = _FakeLLM("Mary Jones")
    with pytest.raises(ValueError, match="no-op"):
        make_counterfactual("A passage without the answer.", "Linda Davis", llm)
