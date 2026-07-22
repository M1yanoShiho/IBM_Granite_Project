"""Shared answer-extraction substrate (spec §6.1).

One extraction pass per query window feeds BOTH the tolerant rerank stage and the
strict gate stage — the gate adds no LLM calls.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from evidence_rag.contracts.models import EvidenceCandidate, Query

EXTRACT_PROMPT = (
    "Using ONLY the passage below, answer the question with the shortest exact answer "
    "(a name, place, date, or number). If the passage does not answer it, reply NONE.\n"
    "Question: {question}\n"
    "Passage: {passage}\n"
    "Answer:"
)

PARAMETRIC_PROMPT = (
    "Answer the question with the shortest exact answer from your own knowledge. "
    "If you are not sure, reply NONE.\n"
    "Question: {question}\n"
    "Answer:"
)


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class ExtractedAnswers:
    answers: tuple[str, ...]
    parametric: str | None


class AnswerExtractionEngine:
    def __init__(
        self,
        answer_extractor: TextGenerator,
        *,
        passage_chars: int = 600,
        use_parametric: bool = True,
    ) -> None:
        if passage_chars <= 0:
            raise ValueError("passage_chars must be positive")
        self.answer_extractor = answer_extractor
        self.passage_chars = passage_chars
        self.use_parametric = use_parametric

    def _extract_answer(self, query: Query, candidate: EvidenceCandidate) -> str:
        prompt = EXTRACT_PROMPT.format(
            question=query.text,
            passage=candidate.text[: self.passage_chars],
        )
        return self.answer_extractor.generate(prompt).strip()

    def extract(self, query: Query, window: Sequence[EvidenceCandidate]) -> ExtractedAnswers:
        answers = tuple(self._extract_answer(query, candidate) for candidate in window)
        parametric = (
            self.answer_extractor.generate(PARAMETRIC_PROMPT.format(question=query.text)).strip()
            if self.use_parametric
            else None
        )
        return ExtractedAnswers(answers=answers, parametric=parametric)
