"""Tests for RAG answer-quality metrics (``eval/rag_metrics``).

All tests use deterministic, heuristic scoring — no model download, no GPU,
no LLM-judge dependency.  The tests verify behaviour on clear-cut cases
(perfect match, no match, partial overlap, empty inputs) so the heuristics
are locked in and can be upgraded later with confidence.
"""

from __future__ import annotations

import pytest

from eval.rag_metrics import (
    evaluate_rag,
    score_answer_correctness,
    score_context_precision,
    score_faithfulness,
)


# ---------------------------------------------------------------------------
# score_answer_correctness
# ---------------------------------------------------------------------------
class TestAnswerCorrectness:
    """Tests for ``score_answer_correctness``."""

    def test_exact_match_identical(self) -> None:
        assert score_answer_correctness("Paris", "Paris", fuzzy=False) == 1.0

    def test_exact_match_case_insensitive(self) -> None:
        assert score_answer_correctness("paris", "Paris", fuzzy=False) == 1.0

    def test_exact_match_no_match(self) -> None:
        assert score_answer_correctness("Paris", "London", fuzzy=False) == 0.0

    def test_exact_match_whitespace_insensitive(self) -> None:
        assert score_answer_correctness("  Paris  ", "Paris", fuzzy=False) == 1.0

    def test_fuzzy_substring_containment(self) -> None:
        # reference is fully contained in prediction
        assert (
            score_answer_correctness(
                "The capital of France is Paris.", "Paris", fuzzy=True
            )
            == 1.0
        )

    def test_fuzzy_partial_overlap(self) -> None:
        score = score_answer_correctness(
            "The capital of France is Paris",
            "The capital of Italy is Rome",
            fuzzy=True,
        )
        # Partial token overlap (shared "the capital of is")
        assert 0.0 < score < 1.0

    def test_fuzzy_no_overlap(self) -> None:
        # "Paris" and "elephant" share essentially no content — any fuzzy
        # score from character-level similarity should be negligible.
        score = score_answer_correctness("Paris", "elephant", fuzzy=True)
        assert score < 0.2

    def test_both_empty(self) -> None:
        assert score_answer_correctness("", "", fuzzy=False) == 1.0
        assert score_answer_correctness("", "", fuzzy=True) == 1.0

    def test_one_empty(self) -> None:
        assert score_answer_correctness("Paris", "", fuzzy=False) == 0.0
        assert score_answer_correctness("", "Paris", fuzzy=True) == 0.0


# ---------------------------------------------------------------------------
# score_context_precision
# ---------------------------------------------------------------------------
class TestContextPrecision:
    """Tests for ``score_context_precision``."""

    def test_all_chunks_relevant(self) -> None:
        chunks = [
            "Paris is the capital of France.",
            "France is a country in Europe.",
        ]
        question = "What is the capital of France?"
        ground_truth = "Paris"
        assert score_context_precision(question, chunks, ground_truth) == 1.0

    def test_no_chunks_relevant(self) -> None:
        chunks = [
            "Elephants are large mammals.",
            "Bananas grow on trees.",
        ]
        question = "What is the capital of France?"
        ground_truth = "Paris"
        score = score_context_precision(question, chunks, ground_truth)
        # "bananas" vs query — pure stop-word chunks could give a false positive
        # so we allow a tiny score at most; the point is it's well below 1.0
        assert score < 0.5

    def test_partial_relevance(self) -> None:
        chunks = [
            "Paris is the capital of France.",
            "Elephants are large mammals.",
        ]
        question = "What is the capital of France?"
        ground_truth = "Paris"
        assert score_context_precision(question, chunks, ground_truth) == 0.5

    def test_empty_chunks(self) -> None:
        assert (
            score_context_precision("What is the capital?", [], "Paris") == 0.0
        )

    def test_relevance_via_ground_truth_token(self) -> None:
        # A chunk that shares a token with the *ground truth* but not the
        # question should still be counted as relevant.
        chunks = ["The Louvre is in Paris."]
        question = "Where is the Louvre?"
        ground_truth = "Paris"
        assert score_context_precision(question, chunks, ground_truth) == 1.0


# ---------------------------------------------------------------------------
# score_faithfulness
# ---------------------------------------------------------------------------
class TestFaithfulness:
    """Tests for ``score_faithfulness``."""

    def test_fully_grounded(self) -> None:
        context = "Paris is the capital of France. It is a major European city."
        answer = "Paris is the capital of France."
        assert score_faithfulness(answer, context) == pytest.approx(1.0)

    def test_fully_hallucinated(self) -> None:
        context = "Paris is the capital of France."
        answer = "Elephants eat bananas in the jungle."
        score = score_faithfulness(answer, context)
        # None of the answer's content words exist in context
        assert score == 0.0

    def test_partially_grounded(self) -> None:
        context = "Paris is the capital of France. France is a country in Europe."
        answer = "Paris is the capital but elephants fly there."
        score = score_faithfulness(answer, context)
        # "Paris" and "capital" are grounded; "elephants", "fly" are not
        assert 0.2 < score < 0.8

    def test_empty_answer(self) -> None:
        assert score_faithfulness("", "Paris is the capital of France.") == 1.0

    def test_empty_context(self) -> None:
        assert score_faithfulness("Paris is the capital.", "") == 0.0

    def test_both_empty(self) -> None:
        assert score_faithfulness("", "") == 1.0


# ---------------------------------------------------------------------------
# evaluate_rag
# ---------------------------------------------------------------------------
class TestEvaluateRag:
    """Tests for the ``evaluate_rag`` aggregation function."""

    def test_aggregates_mean_over_two_queries(self) -> None:
        predictions = {
            "q1": "Paris is the capital of France.",
            "q2": "London is the capital of England.",
        }
        references = {
            "q1": "Paris",
            "q2": "London",
        }
        contexts = {
            "q1": [
                "Paris is the capital of France.",
                "France is in Europe.",
            ],
            "q2": [
                "London is a major city in England.",
                "Elephants are large.",
            ],
        }

        result = evaluate_rag(predictions, references, contexts)

        assert set(result) == {"answer_correctness", "context_precision", "faithfulness"}
        for v in result.values():
            assert 0.0 <= v <= 1.0
        # correctness should be high (answers contain references)
        assert result["answer_correctness"] > 0.5

    def test_no_overlapping_ids_returns_zero(self) -> None:
        result = evaluate_rag(
            {"q1": "a"}, {"q2": "b"}, {"q3": ["c"]}
        )
        assert result == {
            "answer_correctness": 0.0,
            "context_precision": 0.0,
            "faithfulness": 0.0,
        }

    def test_empty_inputs(self) -> None:
        result = evaluate_rag({}, {}, {})
        assert result == {
            "answer_correctness": 0.0,
            "context_precision": 0.0,
            "faithfulness": 0.0,
        }

    def test_perfect_scores_when_all_answers_are_from_context(self) -> None:
        predictions = {
            "q1": "Paris is the capital.",
        }
        references = {
            "q1": "Paris is the capital.",
        }
        contexts = {
            "q1": ["Paris is the capital of France."],
        }
        result = evaluate_rag(predictions, references, contexts)
        assert result["answer_correctness"] == pytest.approx(1.0)
        assert result["context_precision"] == pytest.approx(1.0)
        assert result["faithfulness"] == pytest.approx(1.0)
