from collections.abc import Sequence

from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate, Query
from evidence_rag.selector.beam_three_class import (
    BeamSelectorEvent,
    ClassProbabilities,
    FirstHopCachingScorer,
    ThreeClassBeamSelector,
)


def _pool() -> CandidateSet:
    return CandidateSet(
        query_id="q",
        candidates=tuple(
            EvidenceCandidate(
                evidence_id=f"e{rank}",
                document_id=f"d{rank}",
                chunk_id=f"c{rank}",
                text=f"passage {rank}",
                source_uri="source",
                retrieval_score=1.0 / rank,
                retrieval_rank=rank,
            )
            for rank in range(1, 21)
        ),
    )


class UsefulScorer:
    def score(
        self,
        *,
        question: str,
        selected_passages: Sequence[str],
        candidate_passages: Sequence[str],
        hop: int,
    ) -> tuple[ClassProbabilities, ...]:
        del question, selected_passages, hop
        values = []
        for text in candidate_passages:
            if text == "passage 15":
                values.append(ClassProbabilities(0.005, 0.99, 0.005))
            elif text == "passage 2":
                values.append(ClassProbabilities(0.005, 0.005, 0.99))
            else:
                values.append(ClassProbabilities(0.6, 0.2, 0.2))
        return tuple(values)


class UncertainScorer:
    def score(
        self,
        *,
        question: str,
        selected_passages: Sequence[str],
        candidate_passages: Sequence[str],
        hop: int,
    ) -> tuple[ClassProbabilities, ...]:
        del question, selected_passages, hop
        return tuple(ClassProbabilities(0.4, 0.3, 0.3) for _ in candidate_passages)


class PartiallyStoppingScorer:
    def score(
        self,
        *,
        question: str,
        selected_passages: Sequence[str],
        candidate_passages: Sequence[str],
        hop: int,
    ) -> tuple[ClassProbabilities, ...]:
        del question, hop
        values = []
        for text in candidate_passages:
            required = 0.01
            if not selected_passages and text == "passage 1":
                required = 0.9
            elif not selected_passages and text == "passage 2":
                required = 0.8
            elif selected_passages == ("passage 2",) and text == "passage 3":
                required = 0.6
            values.append(ClassProbabilities(1.0 - required, required, 0.0))
        return tuple(values)


class CountingScorer:
    def __init__(self) -> None:
        self.calls: list[tuple[int, tuple[str, ...]]] = []

    def score(
        self,
        *,
        question: str,
        selected_passages: Sequence[str],
        candidate_passages: Sequence[str],
        hop: int,
    ) -> tuple[ClassProbabilities, ...]:
        del question
        self.calls.append((hop, tuple(selected_passages)))
        return tuple(ClassProbabilities(0.4, 0.3, 0.3) for _ in candidate_passages)


class CountingPathSensitiveScorer(PartiallyStoppingScorer):
    def __init__(self) -> None:
        self.calls: list[tuple[int, tuple[str, ...], tuple[str, ...]]] = []

    def score(
        self,
        *,
        question: str,
        selected_passages: Sequence[str],
        candidate_passages: Sequence[str],
        hop: int,
    ) -> tuple[ClassProbabilities, ...]:
        self.calls.append((hop, tuple(selected_passages), tuple(candidate_passages)))
        return super().score(
            question=question,
            selected_passages=selected_passages,
            candidate_passages=candidate_passages,
            hop=hop,
        )


def test_selector_replaces_confident_harm_with_required_outside_top10() -> None:
    events: list[BeamSelectorEvent] = []
    selector = ThreeClassBeamSelector(
        UsefulScorer(),
        required_threshold=0.5,
        reject_threshold=0.8,
        on_event=events.append,
    )
    result = selector.select(Query(query_id="q", text="question"), _pool(), 10)
    selected = {item.evidence_id for item in result.items}
    assert "e15" in selected
    assert "e2" not in selected
    assert len(result.items) == 10
    assert selected <= {item.evidence_id for item in _pool().candidates}
    assert events[0].proposed_required_ids == ("e15",)
    assert "e2" in events[0].excluded_ids


def test_uncertain_model_fails_open_to_exact_topk_order() -> None:
    selector = ThreeClassBeamSelector(
        UncertainScorer(), required_threshold=0.5, reject_threshold=0.8
    )
    result = selector.select(Query(query_id="q", text="question"), _pool(), 10)
    assert [item.evidence_id for item in result.items] == [f"e{rank}" for rank in range(1, 11)]


def test_fail_open_uses_frozen_rank_even_when_scores_disagree() -> None:
    pool = _pool()
    reversed_scores = CandidateSet(
        query_id="q",
        candidates=tuple(
            item.model_copy(update={"retrieval_score": float(item.retrieval_rank)})
            for item in pool.candidates
        ),
    )
    selector = ThreeClassBeamSelector(UncertainScorer())
    result = selector.select(Query(query_id="q", text="question"), reversed_scores, 3)
    assert [item.evidence_id for item in result.items] == ["e1", "e2", "e3"]


def test_completed_high_score_path_is_not_discarded_by_an_expandable_path() -> None:
    events: list[BeamSelectorEvent] = []
    selector = ThreeClassBeamSelector(
        PartiallyStoppingScorer(),
        beam_size=2,
        required_threshold=0.5,
        reject_threshold=0.95,
        on_event=events.append,
    )
    selector.select(Query(query_id="q", text="question"), _pool(), 10)
    assert events[0].proposed_required_ids == ("e1",)


def test_threshold_sweep_cache_reuses_identical_calls_at_every_hop() -> None:
    base = CountingScorer()
    scorer = FirstHopCachingScorer(base)
    first = scorer.score(
        question="question",
        candidate_passages=("one", "two"),
        selected_passages=(),
        hop=0,
    )
    assert (
        scorer.score(
            question="question",
            candidate_passages=("one", "two"),
            selected_passages=(),
            hop=0,
        )
        == first
    )
    for _ in range(2):
        scorer.score(
            question="question",
            candidate_passages=("one", "two"),
            selected_passages=("one",),
            hop=1,
        )
    assert base.calls == [(0, ()), (1, ("one",))]


def test_threshold_sweep_cache_preserves_every_selection_while_reducing_calls() -> None:
    query = Query(query_id="q", text="question")
    pool = _pool()
    thresholds = tuple(
        (required, reject)
        for required in (0.5, 0.7, 0.85, 0.95)
        for reject in (0.8, 0.95, 0.995)
    )

    uncached = CountingPathSensitiveScorer()
    expected = [
        ThreeClassBeamSelector(
            uncached, required_threshold=required, reject_threshold=reject
        ).select(query, pool, 10)
        for required, reject in thresholds
    ]

    base = CountingPathSensitiveScorer()
    cached = FirstHopCachingScorer(base)
    observed = [
        ThreeClassBeamSelector(
            cached, required_threshold=required, reject_threshold=reject
        ).select(query, pool, 10)
        for required, reject in thresholds
    ]

    assert observed == expected
    assert len({tuple(item.evidence_id for item in result.items) for result in observed}) > 1
    assert len(base.calls) < len(uncached.calls) / 2


def test_selector_rejects_mismatched_query() -> None:
    selector = ThreeClassBeamSelector(UncertainScorer())
    try:
        selector.select(Query(query_id="other", text="question"), _pool(), 10)
    except ValueError as error:
        assert "query IDs differ" in str(error)
    else:
        raise AssertionError("mismatched query ID was accepted")
