"""Tests for RAG answer-quality metrics (``eval/rag_metrics``).

The metrics follow the established open-domain QA / IR conventions so the numbers
are comparable to published work and not a bespoke heuristic:

- **answer correctness** = SQuAD-style normalised Exact Match + token-F1, taking
  the best score over all acceptable gold answers (NQ-style multi-answer).
- **context precision** = qrels-based precision@k of the retrieved documents
  (fraction of the supplied chunks whose document is gold-relevant).
- **faithfulness** = fraction of the answer's content tokens grounded in the
  retrieved context (a model-free proxy).

All tests are deterministic — no model download, no GPU, no LLM judge.
"""

from __future__ import annotations

import pytest

from eval.rag_metrics import (
    evaluate_rag,
    normalize_answer,
    score_answer_correctness,
    score_context_precision,
    score_faithfulness,
)


# ---------------------------------------------------------------------------
# normalize_answer (SQuAD canonical normalisation)
# ---------------------------------------------------------------------------
class TestNormalizeAnswer:
    def test_lowercases_and_trims(self) -> None:
        assert normalize_answer("  PARIS ") == "paris"

    def test_strips_punctuation(self) -> None:
        assert normalize_answer("Paris, France!") == "paris france"

    def test_removes_articles(self) -> None:
        assert normalize_answer("the United States") == "united states"

    def test_keeps_accented_letters(self) -> None:
        # [a-z0-9]+ would have dropped the accent and split the token; SQuAD
        # normalisation keeps unicode letters.
        assert normalize_answer("Beyoncé") == "beyoncé"


# ---------------------------------------------------------------------------
# score_answer_correctness  (fuzzy=False -> EM, fuzzy=True -> token-F1)
# ---------------------------------------------------------------------------
class TestAnswerCorrectness:
    def test_exact_match_identical(self) -> None:
        assert score_answer_correctness("Paris", "Paris", fuzzy=False) == 1.0

    def test_exact_match_case_and_punctuation_insensitive(self) -> None:
        assert score_answer_correctness("paris.", "Paris", fuzzy=False) == 1.0

    def test_exact_match_article_insensitive(self) -> None:
        assert score_answer_correctness("the Louvre", "Louvre", fuzzy=False) == 1.0

    def test_exact_match_no_match(self) -> None:
        assert score_answer_correctness("Paris", "London", fuzzy=False) == 0.0

    def test_multiple_gold_answers_takes_best(self) -> None:
        # NQ-style: any acceptable alias counts.
        assert (
            score_answer_correctness("NYC", ["New York City", "NYC"], fuzzy=False)
            == 1.0
        )

    def test_f1_partial_overlap(self) -> None:
        score = score_answer_correctness(
            "The capital of France is Paris",
            "The capital of Italy is Rome",
            fuzzy=True,
        )
        assert 0.0 < score < 1.0

    def test_f1_no_overlap_is_zero(self) -> None:
        assert score_answer_correctness("Paris", "elephant", fuzzy=True) == 0.0

    def test_verbose_correct_answer_is_not_a_false_full_match(self) -> None:
        # A verbose answer that merely *contains* the gold string must NOT score
        # 1.0 (the old substring rule did); F1 gives partial credit instead.
        score = score_answer_correctness(
            "The capital of France is Paris.", "Paris", fuzzy=True
        )
        assert 0.0 < score < 1.0

    def test_substring_token_is_not_a_false_match(self) -> None:
        # "cat" must not match "category" (character-substring false positive).
        assert score_answer_correctness("category", "cat", fuzzy=True) == 0.0

    def test_both_empty(self) -> None:
        assert score_answer_correctness("", "", fuzzy=False) == 1.0
        assert score_answer_correctness("", "", fuzzy=True) == 1.0

    def test_one_empty(self) -> None:
        assert score_answer_correctness("Paris", "", fuzzy=False) == 0.0
        assert score_answer_correctness("", "Paris", fuzzy=True) == 0.0


# ---------------------------------------------------------------------------
# score_context_precision  (qrels-based precision@k)
# ---------------------------------------------------------------------------
class TestContextPrecision:
    def test_all_retrieved_docs_relevant(self) -> None:
        assert score_context_precision(["d1", "d2"], {"d1", "d2"}) == 1.0

    def test_half_relevant(self) -> None:
        assert score_context_precision(["d1", "d2"], {"d1"}) == 0.5

    def test_none_relevant(self) -> None:
        assert score_context_precision(["d1", "d2"], {"d9"}) == 0.0

    def test_empty_retrieved(self) -> None:
        assert score_context_precision([], {"d1"}) == 0.0

    def test_duplicate_docs_counted_once(self) -> None:
        # two chunks from the same relevant doc + one irrelevant doc -> 1/2.
        assert score_context_precision(["d1", "d1", "d2"], {"d1"}) == 0.5

    def test_accepts_qrels_dict_relevance_map(self) -> None:
        # convenience: a {doc_id: relevance} mapping is treated as the relevant
        # set (relevance > 0).
        assert score_context_precision(["d1", "d2"], {"d1": 1, "d2": 0}) == 0.5


# ---------------------------------------------------------------------------
# score_faithfulness  (token coverage of the answer by the context)
# ---------------------------------------------------------------------------
class TestFaithfulness:
    def test_fully_grounded(self) -> None:
        context = "Paris is the capital of France. It is a major European city."
        answer = "Paris is the capital of France."
        assert score_faithfulness(answer, context) == pytest.approx(1.0)

    def test_fully_hallucinated(self) -> None:
        context = "Paris is the capital of France."
        answer = "Elephants eat bananas in the jungle."
        assert score_faithfulness(answer, context) == 0.0

    def test_partially_grounded(self) -> None:
        context = "Paris is the capital of France. France is a country in Europe."
        answer = "Paris is the capital but elephants fly there."
        assert 0.2 < score_faithfulness(answer, context) < 0.8

    def test_empty_answer_is_trivially_faithful(self) -> None:
        assert score_faithfulness("", "Paris is the capital of France.") == 1.0

    def test_empty_context_is_unfaithful(self) -> None:
        assert score_faithfulness("Paris is the capital.", "") == 0.0


# ---------------------------------------------------------------------------
# evaluate_rag
# ---------------------------------------------------------------------------
class TestEvaluateRag:
    def _inputs(self):
        predictions = {
            "q1": "Paris is the capital of France.",
            "q2": "London is the capital of England.",
        }
        references = {"q1": "Paris", "q2": "London"}
        contexts = {
            "q1": ["Paris is the capital of France.", "France is in Europe."],
            "q2": ["London is a major city in England.", "Elephants are large."],
        }
        retrieved_doc_ids = {"q1": ["d1", "d2"], "q2": ["d3", "d4"]}
        qrels = {"q1": {"d1": 1}, "q2": {"d3": 1}}
        return predictions, references, contexts, retrieved_doc_ids, qrels

    def test_returns_standard_metric_keys(self) -> None:
        result = evaluate_rag(*self._inputs())
        assert set(result) == {
            "answer_em",
            "answer_f1",
            "context_precision",
            "faithfulness",
        }
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_context_precision_uses_qrels(self) -> None:
        # each query retrieved one relevant + one irrelevant doc -> 0.5.
        result = evaluate_rag(*self._inputs())
        assert result["context_precision"] == pytest.approx(0.5)

    def test_perfect_em_when_predictions_equal_normalized_gold(self) -> None:
        predictions = {"q1": "Paris."}
        references = {"q1": "paris"}
        contexts = {"q1": ["Paris is the capital of France."]}
        retrieved_doc_ids = {"q1": ["d1"]}
        qrels = {"q1": {"d1": 1}}
        result = evaluate_rag(
            predictions, references, contexts, retrieved_doc_ids, qrels
        )
        assert result["answer_em"] == 1.0
        assert result["answer_f1"] == 1.0
        assert result["context_precision"] == 1.0

    def test_no_overlapping_ids_returns_zero(self) -> None:
        result = evaluate_rag({"q1": "a"}, {"q2": "b"}, {}, {}, {})
        assert result == {
            "answer_em": 0.0,
            "answer_f1": 0.0,
            "context_precision": 0.0,
            "faithfulness": 0.0,
        }

    def test_empty_inputs(self) -> None:
        result = evaluate_rag({}, {}, {}, {}, {})
        assert result == {
            "answer_em": 0.0,
            "answer_f1": 0.0,
            "context_precision": 0.0,
            "faithfulness": 0.0,
        }
