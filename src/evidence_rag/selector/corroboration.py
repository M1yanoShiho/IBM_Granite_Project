import re
from collections.abc import Sequence
from typing import Protocol

from evidence_rag.contracts.models import (
    CandidateSet,
    EvidenceCandidate,
    Query,
    SelectionItem,
    SelectionResult,
)

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

MIN_ANSWER_LENGTH = 3
STOPWORDS = {"the", "a", "an", "none", "n/a", "unknown", "it", "yes", "no"}


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


def normalize_answer(answer: str) -> str:
    normalized = answer.strip().lower()
    normalized = re.sub(r"^(the|a|an)\s+", "", normalized)
    return normalized.strip(" \t\n.,;:!?\"'()[]")


def is_valid_answer(answer: str) -> bool:
    normalized = normalize_answer(answer)
    if not normalized or normalized in STOPWORDS:
        return False
    if len(normalized) < MIN_ANSWER_LENGTH and not normalized.isdigit():
        return False
    return True


def corroboration_scores(
    answers: Sequence[str],
    parametric_answer: str | None = None,
) -> tuple[float, ...]:
    normalized_answers = tuple(
        normalize_answer(answer) if is_valid_answer(answer) else None
        for answer in answers
    )
    parametric = (
        normalize_answer(parametric_answer)
        if parametric_answer is not None and is_valid_answer(parametric_answer)
        else None
    )
    scores: list[float] = []
    for index, answer in enumerate(normalized_answers):
        if answer is None:
            scores.append(0.0)
            continue
        votes = sum(
            1
            for other_index, other in enumerate(normalized_answers)
            if other_index != index and other == answer
        )
        if parametric is not None and parametric == answer:
            votes += 1
        scores.append(float(votes))
    return tuple(scores)


def minmax(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        return ()
    low = min(values)
    high = max(values)
    if low == high:
        return tuple(0.0 for _ in values)
    return tuple((value - low) / (high - low) for value in values)


class CorroborationSelector:
    """Selector that blends retrieval relevance with cross-evidence answer support."""

    def __init__(
        self,
        answer_extractor: TextGenerator,
        *,
        alpha: float = 0.6,
        top_n: int = 20,
        use_parametric: bool = True,
        passage_chars: int = 600,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        if passage_chars <= 0:
            raise ValueError("passage_chars must be positive")
        self.answer_extractor = answer_extractor
        self.alpha = alpha
        self.top_n = top_n
        self.use_parametric = use_parametric
        self.passage_chars = passage_chars

    def _extract_answer(self, query: Query, candidate: EvidenceCandidate) -> str:
        prompt = EXTRACT_PROMPT.format(
            question=query.text,
            passage=candidate.text[: self.passage_chars],
        )
        return self.answer_extractor.generate(prompt).strip()

    def _parametric_answer(self, query: Query) -> str | None:
        if not self.use_parametric:
            return None
        return self.answer_extractor.generate(
            PARAMETRIC_PROMPT.format(question=query.text)
        ).strip()

    def select(
        self,
        query: Query,
        candidates: CandidateSet,
        max_selected: int,
    ) -> SelectionResult:
        if query.query_id != candidates.query_id:
            raise ValueError("query and candidates query IDs differ")
        if max_selected <= 0:
            raise ValueError("max_selected must be positive")
        if not candidates.candidates:
            return SelectionResult(query_id=query.query_id, items=())

        ranked_candidates = tuple(
            sorted(candidates.candidates, key=lambda item: item.retrieval_rank)
        )
        window = ranked_candidates[: self.top_n]
        answers = tuple(self._extract_answer(query, candidate) for candidate in window)
        corroboration = minmax(
            corroboration_scores(answers, self._parametric_answer(query))
        )
        relevance = minmax(tuple(candidate.retrieval_score for candidate in window))
        blended = tuple(
            self.alpha * relevance[index] + (1.0 - self.alpha) * corroboration[index]
            for index in range(len(window))
        )
        reranked_window = tuple(
            window[index]
            for index in sorted(
                range(len(window)),
                key=lambda i: (-blended[i], window[i].retrieval_rank, window[i].evidence_id),
            )
        )
        score_by_id = {
            candidate.evidence_id: blended[index]
            for index, candidate in enumerate(window)
        }
        output = (reranked_window + ranked_candidates[self.top_n :])[:max_selected]
        return SelectionResult(
            query_id=query.query_id,
            items=tuple(
                SelectionItem(
                    evidence_id=candidate.evidence_id,
                    selection_score=score_by_id.get(
                        candidate.evidence_id,
                        candidate.retrieval_score,
                    ),
                    selection_rank=rank,
                )
                for rank, candidate in enumerate(output, start=1)
            ),
        )
