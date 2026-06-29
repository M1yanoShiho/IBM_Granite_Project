"""Tests for the shared model-free text utilities (``src.text_utils``)."""

from __future__ import annotations

from src.text_utils import jaccard, split_sentences, tokenize


class TestTokenize:
    def test_lowercases_and_drops_stop_words(self) -> None:
        assert tokenize("The cat is on a Mat") == ["cat", "mat"]

    def test_keeps_accented_letters(self) -> None:
        # [a-z0-9]+ would have produced "beyonc"; \w keeps the accent.
        assert tokenize("Beyoncé") == ["beyoncé"]

    def test_splits_on_punctuation_and_digits_kept(self) -> None:
        assert tokenize("COVID-19 vaccine!") == ["covid", "19", "vaccine"]


class TestJaccard:
    def test_identical(self) -> None:
        assert jaccard(["a", "b"], ["a", "b"]) == 1.0

    def test_disjoint(self) -> None:
        assert jaccard(["a"], ["b"]) == 0.0

    def test_partial(self) -> None:
        assert jaccard(["a", "b"], ["b", "c"]) == 1 / 3

    def test_both_empty_is_one(self) -> None:
        assert jaccard([], []) == 1.0

    def test_one_empty_is_zero(self) -> None:
        assert jaccard(["a"], []) == 0.0


class TestSplitSentences:
    def test_splits_and_keeps_delimiter(self) -> None:
        assert split_sentences("First one. Second two!") == [
            "First one.",
            "Second two!",
        ]

    def test_blank_returns_empty(self) -> None:
        assert split_sentences("   ") == []
