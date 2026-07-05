"""Tests for the pure corroboration scoring (no LLM)."""
from src.retrieval.corroboration import (
    corroboration_scores,
    is_valid_answer,
    normalize_answer,
)


def test_normalize_strips_article_case_and_punctuation():
    assert normalize_answer("  The Origin. ") == "origin"
    assert normalize_answer("Paris") == "paris"
    assert normalize_answer("A Charles Darwin") == "charles darwin"


def test_is_valid_answer_rejects_degenerate_extractions():
    assert not is_valid_answer("the")      # stopword
    assert not is_valid_answer("NONE")     # sentinel non-answer
    assert not is_valid_answer("")         # empty
    assert not is_valid_answer("ab")       # too short, non-numeric
    assert is_valid_answer("12")           # short but numeric -> kept
    assert is_valid_answer("Charles Darwin")


def test_lone_answers_tie_at_zero():
    # needle X and counterfactual Y each appear once, rest NONE -> both 0 (the tie case)
    assert corroboration_scores(["Darwin", "Lamarck", "NONE"]) == [0.0, 0.0, 0.0]


def test_corroborated_answer_beats_lone_counterfactual():
    # X answered by two passages, Y by one -> each X gets 1 vote, Y gets 0
    assert corroboration_scores(["Darwin", "Darwin", "Lamarck"]) == [1.0, 1.0, 0.0]


def test_parametric_vote_breaks_a_tie_toward_its_answer():
    assert corroboration_scores(["Darwin", "Lamarck"], parametric="Darwin") == [1.0, 0.0]


def test_degenerate_extraction_does_not_corroborate():
    # a failed "the" extraction must not count or match; the two real X's still corroborate
    assert corroboration_scores(["the", "Darwin", "Darwin"]) == [0.0, 1.0, 1.0]
