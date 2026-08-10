"""Conservative three-class beam selection over one frozen Top-20 pool.

The scorer receives text only.  This module never imports or calls a Generator and every returned
ID is copied from the input CandidateSet.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from evidence_rag.contracts.models import (
    CandidateSet,
    EvidenceCandidate,
    Query,
    SelectionItem,
    SelectionResult,
)


@dataclass(frozen=True)
class BeamCandidate:
    """The only candidate fields allowed to cross into the model-side pipeline."""

    evidence_id: str
    document_id: str
    text: str
    retrieval_rank: int
    retrieval_score: float


@dataclass(frozen=True)
class ClassProbabilities:
    irrelevant: float
    required: float
    harmful: float

    def __post_init__(self) -> None:
        values = (self.irrelevant, self.required, self.harmful)
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
            raise ValueError("class probabilities must be finite values in [0, 1]")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-5):
            raise ValueError("class probabilities must sum to one")


class BeamTextScorer(Protocol):
    def score(
        self,
        *,
        question: str,
        selected_passages: Sequence[str],
        candidate_passages: Sequence[str],
        hop: int,
    ) -> tuple[ClassProbabilities, ...]: ...


class FirstHopCachingScorer:
    """Reuse identical model calls during a frozen development-threshold sweep.

    The historical class name is retained for compatibility.  Every cache key contains the full
    text-only model input and hop, so memoization changes neither probabilities nor selector
    decisions; it only avoids recomputing a call already made by another threshold configuration.
    """

    def __init__(self, base: BeamTextScorer) -> None:
        self.base = base
        self._cache: dict[
            tuple[str, tuple[str, ...], tuple[str, ...], int],
            tuple[ClassProbabilities, ...],
        ] = {}

    def score(
        self,
        *,
        question: str,
        selected_passages: Sequence[str],
        candidate_passages: Sequence[str],
        hop: int,
    ) -> tuple[ClassProbabilities, ...]:
        key = (
            question,
            tuple(selected_passages),
            tuple(candidate_passages),
            hop,
        )
        cached = self._cache.get(key)
        if cached is None:
            cached = self.base.score(
                question=question,
                selected_passages=selected_passages,
                candidate_passages=candidate_passages,
                hop=hop,
            )
            self._cache[key] = cached
        return cached


@dataclass(frozen=True)
class BeamSelectorEvent:
    query_id: str
    selected_ids: tuple[str, ...]
    proposed_required_ids: tuple[str, ...]
    excluded_ids: tuple[str, ...]
    probabilities_by_id: dict[str, ClassProbabilities]


@dataclass(frozen=True)
class _BeamPath:
    indices: tuple[int, ...]
    log_score: float
    chosen_probabilities: tuple[ClassProbabilities, ...]


def _model_candidate(item: EvidenceCandidate) -> BeamCandidate:
    # The caller passes EvidenceCandidate, but keeping this adapter field-by-field prevents
    # metadata/source identifiers from crossing into the text scorer by accident.
    return BeamCandidate(
        evidence_id=item.evidence_id,
        document_id=item.document_id,
        text=item.text,
        retrieval_rank=item.retrieval_rank,
        retrieval_score=float(item.retrieval_score),
    )


class ThreeClassBeamSelector:
    def __init__(
        self,
        scorer: BeamTextScorer,
        *,
        beam_size: int = 2,
        required_threshold: float = 0.5,
        reject_threshold: float = 0.8,
        on_event: Callable[[BeamSelectorEvent], None] | None = None,
    ) -> None:
        if beam_size < 1:
            raise ValueError("beam_size must be positive")
        if not 0.0 <= required_threshold <= 1.0:
            raise ValueError("required_threshold must be in [0, 1]")
        if not 0.0 <= reject_threshold <= 1.0:
            raise ValueError("reject_threshold must be in [0, 1]")
        self.scorer = scorer
        self.beam_size = beam_size
        self.required_threshold = required_threshold
        self.reject_threshold = reject_threshold
        self.on_event = on_event

    def _score_remaining(
        self,
        question: str,
        ranked: Sequence[BeamCandidate],
        path: _BeamPath,
    ) -> tuple[tuple[int, ClassProbabilities], ...]:
        selected = set(path.indices)
        remaining = tuple(index for index in range(len(ranked)) if index not in selected)
        if not remaining:
            return ()
        probabilities = self.scorer.score(
            question=question,
            selected_passages=tuple(ranked[index].text for index in path.indices),
            candidate_passages=tuple(ranked[index].text for index in remaining),
            hop=len(path.indices),
        )
        if len(probabilities) != len(remaining):
            raise ValueError("scorer returned the wrong number of probability rows")
        return tuple(zip(remaining, probabilities, strict=True))

    def _beam(self, question: str, ranked: Sequence[BeamCandidate], limit: int) -> _BeamPath:
        beams: tuple[_BeamPath, ...] = (
            _BeamPath(indices=(), log_score=0.0, chosen_probabilities=()),
        )
        completed: list[_BeamPath] = []
        for _hop in range(limit):
            expanded: list[_BeamPath] = []
            for path in beams:
                path_expansions: list[_BeamPath] = []
                for index, probabilities in self._score_remaining(question, ranked, path):
                    confident_reject = (
                        max(probabilities.harmful, probabilities.irrelevant)
                        >= self.reject_threshold
                        and probabilities.required < self.required_threshold
                    )
                    if probabilities.required < self.required_threshold or confident_reject:
                        continue
                    path_expansions.append(
                        _BeamPath(
                            indices=(*path.indices, index),
                            log_score=path.log_score + math.log(max(probabilities.required, 1e-12)),
                            chosen_probabilities=(*path.chosen_probabilities, probabilities),
                        )
                    )
                if path_expansions:
                    expanded.extend(path_expansions)
                else:
                    completed.append(path)
            if not expanded:
                break
            expanded.sort(
                key=lambda path: (
                    -path.log_score,
                    tuple(ranked[index].retrieval_rank for index in path.indices),
                    tuple(ranked[index].evidence_id for index in path.indices),
                )
            )
            beams = tuple(expanded[: self.beam_size])
        finalists = [*completed, *beams]
        finalists.sort(
            key=lambda path: (
                -path.log_score,
                tuple(ranked[index].retrieval_rank for index in path.indices),
                tuple(ranked[index].evidence_id for index in path.indices),
            )
        )
        return finalists[0]

    def select(self, query: Query, candidates: CandidateSet, max_selected: int) -> SelectionResult:
        if query.query_id != candidates.query_id:
            raise ValueError("query and candidates query IDs differ")
        if max_selected < 1:
            raise ValueError("max_selected must be positive")
        ranked = tuple(
            _model_candidate(item)
            for item in sorted(
                candidates.candidates,
                key=lambda candidate: (candidate.retrieval_rank, candidate.evidence_id),
            )
        )
        if not ranked:
            return SelectionResult(query_id=query.query_id, items=())

        proposed = self._beam(query.text, ranked, max_selected)
        proposed_set = set(proposed.indices)
        probability_by_index = {
            index: probability
            for index, probability in zip(
                proposed.indices, proposed.chosen_probabilities, strict=True
            )
        }
        remaining_scores = self._score_remaining(query.text, ranked, proposed)
        probability_by_index.update(remaining_scores)

        excluded: set[int] = set()
        for index, probabilities in remaining_scores:
            if (
                max(probabilities.harmful, probabilities.irrelevant) >= self.reject_threshold
                and probabilities.required < self.required_threshold
            ):
                excluded.add(index)

        output = list(proposed.indices)
        for index in range(len(ranked)):
            if len(output) == max_selected:
                break
            if index not in proposed_set and index not in excluded:
                output.append(index)

        items = tuple(
            SelectionItem(
                evidence_id=ranked[index].evidence_id,
                selection_score=probability_by_index[index].required,
                selection_rank=rank,
            )
            for rank, index in enumerate(output, 1)
        )
        if self.on_event is not None:
            self.on_event(
                BeamSelectorEvent(
                    query_id=query.query_id,
                    selected_ids=tuple(ranked[index].evidence_id for index in output),
                    proposed_required_ids=tuple(
                        ranked[index].evidence_id for index in proposed.indices
                    ),
                    excluded_ids=tuple(ranked[index].evidence_id for index in sorted(excluded)),
                    probabilities_by_id={
                        ranked[index].evidence_id: probability
                        for index, probability in probability_by_index.items()
                    },
                )
            )
        return SelectionResult(query_id=query.query_id, items=items)
