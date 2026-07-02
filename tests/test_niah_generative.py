# tests/test_niah_generative.py
"""Tests for src/niah/generative.py — Source B (LLM plausible non-answer)."""
from __future__ import annotations

import pytest

from src.niah.generative import make_generative_distractor


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_make_generative_distractor_returns_passage() -> None:
    llm = _FakeLLM("The 1994 ceremony was held in Nashville and drew a large crowd.")
    out = make_generative_distractor("who won the 1994 award?", "Linda Davis won it.", llm)
    assert "Nashville" in out
    assert "who won the 1994 award?" in llm.prompts[0]
    assert "not answer" in llm.prompts[0].lower()


def test_make_generative_distractor_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        make_generative_distractor("q?", "needle", _FakeLLM("   "))
