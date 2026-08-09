"""ReliabilityRAG-inspired contradiction filtering for retrieved evidence.

The paper supplies the useful operational idea: answer the same question from
each passage independently, detect contradictions between those answers, then
retain a mutually compatible set.  This implementation adapts that idea to the
project's frozen contracts: passage IDs and retrieval scores are preserved,
source-parent provenance breaks independent-set ties, and every external model
failure atomically falls back to the existing Top-K selector.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal, Protocol

from evidence_rag.contracts.models import (
    CandidateSet,
    EvidenceCandidate,
    Query,
    SelectionItem,
    SelectionResult,
)
from evidence_rag.selector.extraction import AnswerExtractionEngine, TextGenerator
from evidence_rag.selector.top_k import TopKSelector

LOGGER = logging.getLogger(__name__)

MAX_WINDOW_SIZE = 20
DEFAULT_CONTRADICTION_THRESHOLD = 0.5
PROMPT_VERSION = "mis-answer-v1+mis-statement-v1"

_ANSWER_PREFIX = re.compile(r"^\s*answer\s*:\s*", flags=re.IGNORECASE)
_TRAILING_PUNCTUATION = re.compile(r"[\s.!?]+$")
_UNANSWERABLE_EXACT = frozenset(
    {
        "none",
        "unknown",
        "i don't know",
        "i do not know",
        "not in passage",
        "not in the passage",
        "not in evidence",
        "not in the evidence",
    }
)
_UNANSWERABLE_PREFIXES = (
    "the answer is not in the passage",
    "the answer is not in the evidence",
    "the passage does not answer",
    "the evidence does not answer",
)


class SelectorBackendError(RuntimeError):
    """An external answer-extraction or NLI backend could not produce a result."""


class ContradictionScorer(Protocol):
    """Scores ordered ``(premise, hypothesis)`` pairs as contradiction probabilities."""

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> tuple[float, ...]: ...


class LazyAnswerGenerator:
    """Create an answer backend only when the Selector first needs it.

    Keeping this adapter in the Selector module prevents construction or runtime
    failures in an optional model from escaping the Selector's conservative
    Top-K fallback boundary.
    """

    def __init__(self, factory: Callable[[], TextGenerator]) -> None:
        self._factory = factory
        self._backend: TextGenerator | None = None

    def generate(self, prompt: str) -> str:
        try:
            if self._backend is None:
                self._backend = self._factory()
            return self._backend.generate(prompt)
        except SelectorBackendError:
            raise
        except Exception as exc:
            raise SelectorBackendError("answer-extraction backend failed") from exc


SelectionMode = Literal["mis", "topk_no_conflict", "topk_backend_fallback"]
FallbackStage = Literal["answer", "nli"]


@dataclass(frozen=True)
class MISSelectionEvent:
    """Metadata-only diagnostic event; raw passages and extracted answers are excluded."""

    query_id: str
    mode: SelectionMode
    candidate_count: int
    window_count: int
    valid_answer_count: int
    unanswerable_count: int
    pair_count: int
    contradiction_edge_count: int
    independent_parent_count: int
    unresolved_parent_count: int
    selected_ids: tuple[str, ...]
    dropped_ids: tuple[str, ...]
    model_id: str
    model_revision: str
    threshold: float
    prompt_version: str = PROMPT_VERSION
    fallback_stage: FallbackStage | None = None
    error_type: str | None = None


def canonical_candidates(candidates: Sequence[EvidenceCandidate]) -> tuple[EvidenceCandidate, ...]:
    """Use the one ordering already frozen by :class:`TopKSelector`."""

    return tuple(sorted(candidates, key=lambda item: (-item.retrieval_score, item.evidence_id)))


def clean_isolated_answer(answer: str) -> str:
    """Remove an optional ``Answer:`` label without rewriting answer content."""

    return _ANSWER_PREFIX.sub("", answer, count=1).strip()


def is_unanswerable(answer: str) -> bool:
    """Recognise the small set of abstention forms allowed by the extraction prompt."""

    normalized = clean_isolated_answer(answer).replace("’", "'").casefold()
    normalized = _TRAILING_PUNCTUATION.sub("", normalized)
    return (
        normalized == ""
        or normalized in _UNANSWERABLE_EXACT
        or any(normalized.startswith(prefix) for prefix in _UNANSWERABLE_PREFIXES)
    )


def answer_statement(question: str, answer: str) -> str:
    """Convert an isolated answer into the fixed NLI statement form."""

    return f"The answer to the question: {question}\nis {clean_isolated_answer(answer)}."


def build_pair_batch(
    question: str,
    answers: Sequence[str],
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[int, int], ...]]:
    """Build one deterministic i<j batch from answerable passages only."""

    valid = tuple(index for index, answer in enumerate(answers) if not is_unanswerable(answer))
    statements = {index: answer_statement(question, answers[index]) for index in valid}
    pair_indices = tuple(combinations(valid, 2))
    pairs = tuple((statements[left], statements[right]) for left, right in pair_indices)
    return pairs, pair_indices


def build_conflict_graph(
    vertex_count: int,
    pair_indices: Sequence[tuple[int, int]],
    probabilities: Sequence[float],
    *,
    threshold: float,
) -> tuple[tuple[int, ...], int]:
    """Return adjacency bitmasks and the number of undirected conflict edges."""

    if vertex_count < 0:
        raise ValueError("vertex_count must not be negative")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    if len(pair_indices) != len(probabilities):
        raise SelectorBackendError("NLI backend returned the wrong number of probabilities")

    adjacency = [0] * vertex_count
    edge_count = 0
    for (left, right), raw_probability in zip(pair_indices, probabilities, strict=True):
        if left < 0 or right < 0 or left >= vertex_count or right >= vertex_count or left >= right:
            raise ValueError("pair indices must satisfy 0 <= left < right < vertex_count")
        try:
            probability = float(raw_probability)
        except (TypeError, ValueError) as exc:
            raise SelectorBackendError("NLI backend returned a non-numeric probability") from exc
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise SelectorBackendError("NLI backend returned an invalid probability")
        if probability >= threshold:
            adjacency[left] |= 1 << right
            adjacency[right] |= 1 << left
            edge_count += 1
    return tuple(adjacency), edge_count


def _is_independent(vertices: Sequence[int], adjacency: Sequence[int]) -> bool:
    selected_mask = 0
    for vertex in vertices:
        if adjacency[vertex] & selected_mask:
            return False
        selected_mask |= 1 << vertex
    return True


def select_capacity_limited_independent_set(
    adjacency: Sequence[int],
    parent_keys: Sequence[str],
    *,
    capacity: int,
) -> tuple[int, ...]:
    """Choose the deterministic best independent subset of at most ``capacity`` nodes.

    Objective order:

    1. most distinct source-parent keys;
    2. most passages;
    3. lexicographically smallest canonical Top-K index tuple.

    At the frozen maximum of twenty vertices exhaustive enumeration is small
    enough to keep this logic exact, inspectable, and dependency-free.
    """

    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if len(adjacency) != len(parent_keys):
        raise ValueError("adjacency and parent_keys must have equal lengths")

    best: tuple[int, ...] = ()
    best_parent_count = 0
    best_size = 0
    for size in range(1, min(capacity, len(adjacency)) + 1):
        for vertices in combinations(range(len(adjacency)), size):
            if not _is_independent(vertices, adjacency):
                continue
            parent_count = len({parent_keys[index] for index in vertices})
            objective = (parent_count, size)
            best_objective = (best_parent_count, best_size)
            if objective > best_objective or (objective == best_objective and vertices < best):
                best = vertices
                best_parent_count = parent_count
                best_size = size
    return best


class ReliabilityMISSelector:
    """Filter explicit cross-passage answer conflicts within the Retriever's Top-N window."""

    def __init__(
        self,
        answer_extractor: TextGenerator,
        contradiction_scorer: ContradictionScorer,
        parent_by_document: Mapping[str, str],
        *,
        top_n: int = MAX_WINDOW_SIZE,
        contradiction_threshold: float = DEFAULT_CONTRADICTION_THRESHOLD,
        passage_chars: int = 600,
        on_event: Callable[[MISSelectionEvent], None] | None = None,
    ) -> None:
        if not 1 <= top_n <= MAX_WINDOW_SIZE:
            raise ValueError(f"top_n must be between 1 and {MAX_WINDOW_SIZE}")
        if not 0.0 <= contradiction_threshold <= 1.0:
            raise ValueError("contradiction_threshold must be between zero and one")
        self._answer_engine = AnswerExtractionEngine(
            answer_extractor,
            passage_chars=passage_chars,
            use_parametric=False,
        )
        self._contradiction_scorer = contradiction_scorer
        self._parent_by_document = dict(parent_by_document)
        self._top_n = top_n
        self._contradiction_threshold = contradiction_threshold
        self._on_event = on_event
        self._top_k = TopKSelector()

    def _scorer_metadata(self) -> tuple[str, str]:
        return (
            str(getattr(self._contradiction_scorer, "model_id", "injected")),
            str(getattr(self._contradiction_scorer, "model_revision", "unknown")),
        )

    def _emit(self, event: MISSelectionEvent) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(event)
        except Exception:
            LOGGER.warning("reliability-mis event sink failed", exc_info=True)

    @staticmethod
    def _selection_from_candidates(
        query_id: str, chosen: Sequence[EvidenceCandidate]
    ) -> SelectionResult:
        return SelectionResult(
            query_id=query_id,
            items=tuple(
                SelectionItem(
                    evidence_id=candidate.evidence_id,
                    selection_score=candidate.retrieval_score,
                    selection_rank=rank,
                )
                for rank, candidate in enumerate(chosen, start=1)
            ),
        )

    @staticmethod
    def _dropped_ids(
        ranked: Sequence[EvidenceCandidate],
        selected_ids: Sequence[str],
    ) -> tuple[str, ...]:
        selected = frozenset(selected_ids)
        return tuple(
            candidate.evidence_id for candidate in ranked if candidate.evidence_id not in selected
        )

    def _event(
        self,
        *,
        query_id: str,
        mode: SelectionMode,
        ranked: Sequence[EvidenceCandidate],
        window: Sequence[EvidenceCandidate],
        answers: Sequence[str],
        pair_count: int,
        edge_count: int,
        selected_ids: tuple[str, ...],
        parent_keys: Sequence[str],
        unresolved_parent_count: int,
        fallback_stage: FallbackStage | None = None,
        error_type: str | None = None,
    ) -> MISSelectionEvent:
        model_id, model_revision = self._scorer_metadata()
        selected_id_set = frozenset(selected_ids)
        independent_parent_count = len(
            {
                parent_keys[index]
                for index, candidate in enumerate(window)
                if candidate.evidence_id in selected_id_set
            }
        )
        valid_answer_count = sum(not is_unanswerable(answer) for answer in answers)
        return MISSelectionEvent(
            query_id=query_id,
            mode=mode,
            candidate_count=len(ranked),
            window_count=len(window),
            valid_answer_count=valid_answer_count,
            unanswerable_count=len(answers) - valid_answer_count,
            pair_count=pair_count,
            contradiction_edge_count=edge_count,
            independent_parent_count=independent_parent_count,
            unresolved_parent_count=unresolved_parent_count,
            selected_ids=selected_ids,
            dropped_ids=self._dropped_ids(ranked, selected_ids),
            model_id=model_id,
            model_revision=model_revision,
            threshold=self._contradiction_threshold,
            fallback_stage=fallback_stage,
            error_type=error_type,
        )

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

        ranked = canonical_candidates(candidates.candidates)
        window = ranked[: self._top_n]
        parent_keys = tuple(
            self._parent_by_document.get(candidate.document_id, candidate.document_id)
            for candidate in window
        )
        unresolved_parent_count = sum(
            candidate.document_id not in self._parent_by_document for candidate in window
        )
        top_k = self._top_k.select(query, candidates, max_selected)

        if not window:
            self._emit(
                self._event(
                    query_id=query.query_id,
                    mode="topk_no_conflict",
                    ranked=ranked,
                    window=window,
                    answers=(),
                    pair_count=0,
                    edge_count=0,
                    selected_ids=tuple(item.evidence_id for item in top_k.items),
                    parent_keys=parent_keys,
                    unresolved_parent_count=unresolved_parent_count,
                )
            )
            return top_k

        try:
            extracted = self._answer_engine.extract(query, window)
        except SelectorBackendError as exc:
            self._emit(
                self._event(
                    query_id=query.query_id,
                    mode="topk_backend_fallback",
                    ranked=ranked,
                    window=window,
                    answers=(),
                    pair_count=0,
                    edge_count=0,
                    selected_ids=tuple(item.evidence_id for item in top_k.items),
                    parent_keys=parent_keys,
                    unresolved_parent_count=unresolved_parent_count,
                    fallback_stage="answer",
                    error_type=type(exc).__name__,
                )
            )
            return top_k

        answers = tuple(clean_isolated_answer(answer) for answer in extracted.answers)
        pairs, pair_indices = build_pair_batch(query.text, answers)
        try:
            probabilities = self._contradiction_scorer.score_pairs(pairs) if pairs else ()
            adjacency, edge_count = build_conflict_graph(
                len(window),
                pair_indices,
                probabilities,
                threshold=self._contradiction_threshold,
            )
        except SelectorBackendError as exc:
            self._emit(
                self._event(
                    query_id=query.query_id,
                    mode="topk_backend_fallback",
                    ranked=ranked,
                    window=window,
                    answers=answers,
                    pair_count=len(pairs),
                    edge_count=0,
                    selected_ids=tuple(item.evidence_id for item in top_k.items),
                    parent_keys=parent_keys,
                    unresolved_parent_count=unresolved_parent_count,
                    fallback_stage="nli",
                    error_type=type(exc).__name__,
                )
            )
            return top_k

        if edge_count == 0:
            self._emit(
                self._event(
                    query_id=query.query_id,
                    mode="topk_no_conflict",
                    ranked=ranked,
                    window=window,
                    answers=answers,
                    pair_count=len(pairs),
                    edge_count=0,
                    selected_ids=tuple(item.evidence_id for item in top_k.items),
                    parent_keys=parent_keys,
                    unresolved_parent_count=unresolved_parent_count,
                )
            )
            return top_k

        chosen_indices = select_capacity_limited_independent_set(
            adjacency,
            parent_keys,
            capacity=max_selected,
        )
        chosen = tuple(window[index] for index in chosen_indices)
        result = self._selection_from_candidates(query.query_id, chosen)
        self._emit(
            self._event(
                query_id=query.query_id,
                mode="mis",
                ranked=ranked,
                window=window,
                answers=answers,
                pair_count=len(pairs),
                edge_count=edge_count,
                selected_ids=tuple(item.evidence_id for item in result.items),
                parent_keys=parent_keys,
                unresolved_parent_count=unresolved_parent_count,
            )
        )
        return result
