from types import SimpleNamespace

import pytest

from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate, Query
from evidence_rag.selector.threshold_only import DEFAULT_THRESHOLD, NliThresholdOnlySelector


def _candidates() -> CandidateSet:
    return CandidateSet(
        query_id="q1",
        candidates=tuple(
            EvidenceCandidate(
                evidence_id=f"e{rank}",
                document_id=f"d{rank}",
                chunk_id=f"c{rank}",
                text=f"passage {rank}",
                source_uri=f"sealed://e{rank}",
                retrieval_score=1.0 / rank,
                retrieval_rank=rank,
            )
            for rank in range(1, 11)
        ),
    )


class _Model:
    def __init__(self, protect: list[float], harm: list[float]) -> None:
        self.protect = protect
        self.harm = harm

    def __call__(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(protect_scores=self.protect, harm_scores=self.harm)


def test_threshold_predicate_includes_both_boundaries_and_has_no_delete_cap() -> None:
    protect = [1.0 - DEFAULT_THRESHOLD] * 10
    harm = [DEFAULT_THRESHOLD] * 10
    selector = NliThresholdOnlySelector(model=_Model(protect, harm))
    result, trace = selector.select_with_trace(Query(query_id="q1", text="question"), _candidates(), 10)

    assert result.items == ()
    assert trace.dropped_evidence_ids == tuple(f"e{rank}" for rank in range(1, 11))
    assert trace.status == "normal"


def test_threshold_requires_both_conditions() -> None:
    threshold = DEFAULT_THRESHOLD
    selector = NliThresholdOnlySelector(
        model=_Model(
            [1.0 - threshold + 1e-6, 0.0] + [1.0] * 8,
            [1.0, threshold - 1e-6] + [0.0] * 8,
        )
    )
    result, trace = selector.select_with_trace(Query(query_id="q1", text="question"), _candidates(), 10)

    assert len(result.items) == 10
    assert trace.dropped_evidence_ids == ()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_any_invalid_score_fails_open_for_whole_query(bad: float) -> None:
    selector = NliThresholdOnlySelector(
        model=_Model([bad] + [0.5] * 9, [1.0] * 10)
    )
    result, trace = selector.select_with_trace(Query(query_id="q1", text="question"), _candidates(), 10)

    assert len(result.items) == 10
    assert trace.status == "fail_open_all"
    assert trace.dropped_evidence_ids == ()


def test_missing_score_fails_open_for_whole_query() -> None:
    selector = NliThresholdOnlySelector(model=_Model([0.5] * 9, [0.5] * 9))
    result, trace = selector.select_with_trace(Query(query_id="q1", text="question"), _candidates(), 10)

    assert len(result.items) == 10
    assert trace.status == "fail_open_all"
