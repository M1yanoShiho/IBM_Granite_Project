"""Tests for answer-provenance citations (``src.explainability.citations``).

All tests are deterministic and model-free — they verify that sentences are
assigned to the correct supporting chunks via token overlap, and that the
threshold and filtering logic behaves as expected.
"""

from __future__ import annotations

import pytest

from src.explainability.citations import Citation, attribute_answer
from src.retrieval.base import RetrievedChunk


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _chunk(doc_id: str, text: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(doc_id=doc_id, text=text, score=score)


# ---------------------------------------------------------------------------
# attribute_answer
# ---------------------------------------------------------------------------
class TestAttributeAnswer:
    """Tests for ``attribute_answer``."""

    def test_exact_sentence_match_to_one_chunk(self) -> None:
        chunks = [_chunk("d1", "Paris is the capital of France.")]
        result = attribute_answer("Paris is the capital of France.", chunks)

        assert len(result) == 1
        assert result[0].answer_span == "Paris is the capital of France."
        assert result[0].source_chunk_id == "d1"
        assert result[0].score > 0.5

    def test_multiple_sentences_split_across_chunks(self) -> None:
        chunks = [
            _chunk("d1", "Paris is the capital of France."),
            _chunk("d2", "London is the capital of England."),
        ]
        answer = "Paris is the capital of France. London is the capital of England."
        result = attribute_answer(answer, chunks)

        assert len(result) == 2
        assert result[0].source_chunk_id == "d1"
        assert result[1].source_chunk_id == "d2"

    def test_partial_overlap_attributed_to_best_chunk(self) -> None:
        chunks = [
            _chunk("d1", "Elephants live in Africa."),
            _chunk("d2", "Paris is a beautiful city in France."),
        ]
        answer = "Paris is the capital of France."
        result = attribute_answer(answer, chunks)

        # Should match d2 better than d1
        assert len(result) == 1
        assert result[0].source_chunk_id == "d2"

    def test_no_match_below_threshold(self) -> None:
        chunks = [_chunk("d1", "Elephants live in Africa.")]
        answer = "Paris is the capital of France."
        result = attribute_answer(answer, chunks, token_overlap_threshold=0.3)

        # No meaningful token overlap — nothing attributed
        assert result == []

    def test_empty_answer(self) -> None:
        chunks = [_chunk("d1", "Paris is the capital of France.")]
        assert attribute_answer("", chunks) == []

    def test_empty_chunks(self) -> None:
        assert attribute_answer("Paris is the capital.", []) == []

    def test_both_empty(self) -> None:
        assert attribute_answer("", []) == []

    def test_citation_is_dataclass_instance(self) -> None:
        chunks = [_chunk("d1", "Paris is the capital of France.")]
        result = attribute_answer("Paris is the capital of France.", chunks)
        assert isinstance(result[0], Citation)

    def test_score_rounded_to_6_decimal_places(self) -> None:
        chunks = [_chunk("d1", "Paris is the capital of France.")]
        result = attribute_answer("Paris is the capital of France.", chunks)
        # score is a rounded float
        assert isinstance(result[0].score, float)
        # Should not have more than 6 decimal digits
        assert result[0].score == round(result[0].score, 6)

    def test_order_preserved_by_answer_sentence_order(self) -> None:
        chunks = [
            _chunk("d1", "London is the capital of England."),
            _chunk("d2", "Paris is the capital of France."),
        ]
        answer = "Paris is the capital of France. London is the capital of England."
        result = attribute_answer(answer, chunks)

        assert len(result) == 2
        # Paris sentence comes first in answer, so it must be first citation
        assert result[0].answer_span == "Paris is the capital of France."
        assert result[1].answer_span == "London is the capital of England."
