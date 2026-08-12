import math
from dataclasses import FrozenInstanceError

import pytest

from evidence_rag.contracts.models import (
    CandidateSet,
    EvidenceCandidate,
    Query,
    SelectionResult,
)
from evidence_rag.selector.models import (
    CandidateRiskScore,
    SelectorAction,
    SelectorDecisionReason,
    SelectorDecisionTrace,
    SelectorFallbackReason,
    deletion_priority_key,
)
from evidence_rag.selector.risk_controlled import RiskControlledSelector


def _candidate(rank: int, *, evidence_id: str | None = None) -> EvidenceCandidate:
    candidate_id = evidence_id or f"ev-{rank:02d}"
    return EvidenceCandidate(
        evidence_id=candidate_id,
        document_id=f"doc-{candidate_id}",
        chunk_id=f"chunk-{candidate_id}",
        text=f"candidate {candidate_id}",
        source_uri=f"fixture://{candidate_id}",
        retrieval_score=1.0 - rank / 100.0,
        retrieval_rank=rank,
    )


def _candidate_set(size: int = 10) -> CandidateSet:
    return CandidateSet(
        query_id="q1",
        candidates=tuple(_candidate(rank) for rank in range(1, size + 1)),
    )


def _scores(
    size: int = 10,
    *,
    protect: float = 0.9,
    harm: float = 0.1,
) -> dict[str, dict[str, CandidateRiskScore]]:
    return {
        "q1": {
            f"ev-{rank:02d}": CandidateRiskScore(
                protect_score=protect,
                harm_score=harm,
            )
            for rank in range(1, size + 1)
        }
    }


def _selector(
    scores: dict[str, dict[str, CandidateRiskScore]],
    *,
    threshold: float = 0.8,
    max_delete: int = 3,
    policy_enabled: bool = True,
    dependency_ready: bool = True,
    dependency_reason: str | None = None,
) -> RiskControlledSelector:
    return RiskControlledSelector(
        scores_by_query=scores,
        safe_threshold=threshold,
        max_delete=max_delete,
        policy_enabled=policy_enabled,
        dependency_ready=dependency_ready,
        dependency_reason=dependency_reason,
    )


def _decision(trace: SelectorDecisionTrace, evidence_id: str):
    return next(item for item in trace.decisions if item.evidence_id == evidence_id)


def test_safe_score_requires_high_harm_and_low_protect() -> None:
    scores = _scores()
    scores["q1"]["ev-10"] = CandidateRiskScore(protect_score=0.1, harm_score=0.95)

    result, trace = _selector(scores, max_delete=1).select_with_trace(
        Query(query_id="q1", text="question"), _candidate_set(), max_selected=10
    )

    assert isinstance(result, SelectionResult)
    assert [item.evidence_id for item in result.items] == [
        f"ev-{rank:02d}" for rank in range(1, 10)
    ]
    dropped = _decision(trace, "ev-10")
    assert dropped.safe_score == pytest.approx(0.9)
    assert dropped.action is SelectorAction.DROP_HARM
    assert dropped.reason is SelectorDecisionReason.SAFE_SCORE_THRESHOLD_MET
    assert trace.dropped_evidence_ids == ("ev-10",)


def test_safe_score_exactly_equal_to_threshold_is_eligible_for_deletion() -> None:
    scores = _scores()
    scores["q1"]["ev-10"] = CandidateRiskScore(protect_score=0.2, harm_score=0.95)

    _, trace = _selector(scores, threshold=0.8, max_delete=1).select_with_trace(
        Query(query_id="q1", text="question"), _candidate_set(), max_selected=10
    )

    boundary = _decision(trace, "ev-10")
    assert boundary.safe_score == 0.8
    assert boundary.action is SelectorAction.DROP_HARM
    assert boundary.reason is SelectorDecisionReason.SAFE_SCORE_THRESHOLD_MET


def test_zero_deletion_is_a_valid_adaptive_outcome() -> None:
    result, trace = _selector(_scores()).select_with_trace(
        Query(query_id="q1", text="question"), _candidate_set(), max_selected=10
    )

    assert len(result.items) == 10
    assert trace.dropped_evidence_ids == ()
    assert trace.fallback_reason is None
    assert {decision.action for decision in trace.decisions} == {SelectorAction.KEEP}


def test_high_harm_high_protect_conflict_abstains_and_stays_selected() -> None:
    scores = _scores()
    scores["q1"]["ev-10"] = CandidateRiskScore(protect_score=0.9, harm_score=0.95)

    result, trace = _selector(scores).select_with_trace(
        Query(query_id="q1", text="question"), _candidate_set(), max_selected=10
    )

    conflict = _decision(trace, "ev-10")
    assert conflict.safe_score == pytest.approx(0.1)
    assert conflict.action is SelectorAction.ABSTAIN_KEEP
    assert conflict.reason is SelectorDecisionReason.PROTECT_HARM_CONFLICT
    assert "ev-10" in {item.evidence_id for item in result.items}


def test_conflict_reason_requires_harm_to_meet_the_same_frozen_threshold() -> None:
    scores = _scores()
    # Both have high protect, but only ev-10's harm head reaches the frozen action threshold.
    scores["q1"]["ev-10"] = CandidateRiskScore(protect_score=0.95, harm_score=0.8)
    scores["q1"]["ev-09"] = CandidateRiskScore(protect_score=0.95, harm_score=0.79)

    _, trace = _selector(scores, threshold=0.8).select_with_trace(
        Query(query_id="q1", text="question"), _candidate_set(), max_selected=10
    )

    assert _decision(trace, "ev-10").reason is SelectorDecisionReason.PROTECT_HARM_CONFLICT
    assert _decision(trace, "ev-09").reason is SelectorDecisionReason.BELOW_SAFE_THRESHOLD


@pytest.mark.parametrize("cap", (1, 2, 3))
def test_actual_deletion_count_is_zero_to_configured_cap(cap: int) -> None:
    scores = _scores()
    for rank in range(1, 11):
        scores["q1"][f"ev-{rank:02d}"] = CandidateRiskScore(
            protect_score=0.0,
            harm_score=1.0,
        )

    result, trace = _selector(scores, max_delete=cap).select_with_trace(
        Query(query_id="q1", text="question"), _candidate_set(), max_selected=10
    )

    assert len(trace.dropped_evidence_ids) == cap
    assert len(result.items) == 10 - cap
    assert all(
        _decision(trace, evidence_id).action is SelectorAction.DROP_HARM
        for evidence_id in trace.dropped_evidence_ids
    )
    assert any(
        decision.reason is SelectorDecisionReason.MAX_DELETE_REACHED for decision in trace.decisions
    )


def test_equal_safe_scores_prefer_later_retrieval_rank() -> None:
    scores = _scores()
    scores["q1"]["ev-03"] = CandidateRiskScore(protect_score=0.05, harm_score=0.9)
    scores["q1"]["ev-09"] = CandidateRiskScore(protect_score=0.1, harm_score=0.95)

    _, trace = _selector(scores, max_delete=1).select_with_trace(
        Query(query_id="q1", text="question"), _candidate_set(), max_selected=10
    )

    assert trace.dropped_evidence_ids == ("ev-09",)
    assert _decision(trace, "ev-03").reason is SelectorDecisionReason.MAX_DELETE_REACHED


def test_evidence_id_is_the_final_ascending_policy_tie_breaker() -> None:
    tied = (
        {"safe_score": 0.9, "retrieval_rank": 8, "evidence_id": "ev-z"},
        {"safe_score": 0.9, "retrieval_rank": 8, "evidence_id": "ev-a"},
    )

    ordered = sorted(tied, key=lambda row: deletion_priority_key(**row))

    assert [row["evidence_id"] for row in ordered] == ["ev-a", "ev-z"]


def test_kept_items_preserve_retrieval_order_after_a_middle_deletion() -> None:
    scores = _scores()
    scores["q1"]["ev-03"] = CandidateRiskScore(protect_score=0.0, harm_score=1.0)

    result, _ = _selector(scores, max_delete=1).select_with_trace(
        Query(query_id="q1", text="question"), _candidate_set(), max_selected=10
    )

    assert [item.evidence_id for item in result.items] == [
        "ev-01",
        "ev-02",
        "ev-04",
        "ev-05",
        "ev-06",
        "ev-07",
        "ev-08",
        "ev-09",
        "ev-10",
    ]
    assert [item.selection_rank for item in result.items] == list(range(1, 10))


def test_min_keep_blocks_an_eligible_second_deletion() -> None:
    scores = _scores(size=8)
    scores["q1"]["ev-08"] = CandidateRiskScore(protect_score=0.0, harm_score=1.0)
    scores["q1"]["ev-07"] = CandidateRiskScore(protect_score=0.0, harm_score=0.99)

    result, trace = _selector(scores, max_delete=3).select_with_trace(
        Query(query_id="q1", text="question"), _candidate_set(size=8), max_selected=10
    )

    assert len(result.items) == 7
    assert trace.dropped_evidence_ids == ("ev-08",)
    blocked = _decision(trace, "ev-07")
    assert blocked.action is SelectorAction.ABSTAIN_KEEP
    assert blocked.reason is SelectorDecisionReason.MIN_KEEP_REACHED


def test_exactly_seven_candidates_run_active_policy_but_delete_nothing() -> None:
    scores = _scores(size=7, protect=0.0, harm=1.0)

    result, trace = _selector(scores, max_delete=3).select_with_trace(
        Query(query_id="q1", text="question"), _candidate_set(size=7), max_selected=10
    )

    assert len(result.items) == 7
    assert trace.fallback_reason is None
    assert trace.dropped_evidence_ids == ()
    assert {decision.action for decision in trace.decisions} == {SelectorAction.ABSTAIN_KEEP}
    assert {decision.reason for decision in trace.decisions} == {
        SelectorDecisionReason.MIN_KEEP_REACHED
    }


def test_top20_input_is_restricted_to_top10_without_rank11_replacement() -> None:
    scores = _scores(size=10)
    scores["q1"]["ev-10"] = CandidateRiskScore(protect_score=0.0, harm_score=1.0)
    # There is intentionally no score for ranks 11--20: they are outside the action domain.
    candidates = _candidate_set(size=20)

    result, trace = _selector(scores, max_delete=1).select_with_trace(
        Query(query_id="q1", text="question"), candidates, max_selected=10
    )

    selected_ids = {item.evidence_id for item in result.items}
    assert len(result.items) == 9
    assert not any(f"ev-{rank:02d}" in selected_ids for rank in range(11, 21))
    assert len(trace.baseline_evidence_ids) == 10


def test_rank11_to_20_scores_cannot_change_top10_selection_or_trace() -> None:
    top10_scores = _scores(size=10)
    top10_scores["q1"]["ev-10"] = CandidateRiskScore(protect_score=0.0, harm_score=1.0)
    extra_scores = _scores(size=20)
    extra_scores["q1"].update(top10_scores["q1"])
    for rank in range(11, 21):
        extra_scores["q1"][f"ev-{rank:02d}"] = CandidateRiskScore(
            protect_score=0.0,
            harm_score=1.0,
        )
    query = Query(query_id="q1", text="question")
    candidates = _candidate_set(size=20)

    without_extras = _selector(top10_scores, max_delete=1).select_with_trace(
        query, candidates, max_selected=10
    )
    with_extras = _selector(extra_scores, max_delete=1).select_with_trace(
        query, candidates, max_selected=10
    )

    assert with_extras == without_extras


def test_missing_one_score_forces_whole_query_abstain_fallback() -> None:
    scores = _scores()
    scores["q1"]["ev-10"] = CandidateRiskScore(protect_score=0.0, harm_score=1.0)
    del scores["q1"]["ev-04"]

    result, trace = _selector(scores).select_with_trace(
        Query(query_id="q1", text="question"), _candidate_set(), max_selected=10
    )

    assert len(result.items) == 10
    assert trace.fallback_reason is SelectorFallbackReason.MISSING_SCORE
    assert {item.action for item in trace.decisions} == {SelectorAction.ABSTAIN_KEEP}
    assert {item.reason for item in trace.decisions} == {
        SelectorDecisionReason.FALLBACK_MISSING_SCORE
    }


@pytest.mark.parametrize("bad_value", (math.nan, math.inf, -math.inf, -0.1, 1.1))
def test_nonfinite_or_out_of_range_score_forces_whole_query_fallback(
    bad_value: float,
) -> None:
    scores = _scores()
    scores["q1"]["ev-04"] = CandidateRiskScore(
        protect_score=bad_value,
        harm_score=0.95,
    )

    result, trace = _selector(scores).select_with_trace(
        Query(query_id="q1", text="question"), _candidate_set(), max_selected=10
    )

    assert len(result.items) == 10
    assert trace.fallback_reason is SelectorFallbackReason.INVALID_SCORE
    assert _decision(trace, "ev-04").protect_score is None
    # The trace itself remains strict JSON even when the untrusted input was NaN/Inf.
    assert "NaN" not in trace.model_dump_json()
    assert "Infinity" not in trace.model_dump_json()


def test_fewer_than_seven_candidates_forces_whole_query_fallback() -> None:
    result, trace = _selector(_scores(size=6)).select_with_trace(
        Query(query_id="q1", text="question"), _candidate_set(size=6), max_selected=10
    )

    assert len(result.items) == 6
    assert trace.fallback_reason is SelectorFallbackReason.TOO_FEW_CANDIDATES
    assert {item.action for item in trace.decisions} == {SelectorAction.ABSTAIN_KEEP}


def test_missing_dependency_forces_whole_query_fallback_with_reason() -> None:
    result, trace = _selector(
        _scores(),
        dependency_ready=False,
        dependency_reason="ParentIndex hash mismatch",
    ).select_with_trace(Query(query_id="q1", text="question"), _candidate_set(), max_selected=10)

    assert len(result.items) == 10
    assert trace.fallback_reason is SelectorFallbackReason.MISSING_DEPENDENCY
    assert trace.dependency_reason == "ParentIndex hash mismatch"
    assert {item.action for item in trace.decisions} == {SelectorAction.ABSTAIN_KEEP}


def test_explicit_disabled_p0_keeps_top10_without_scores_or_dependencies() -> None:
    selector = _selector(
        {},
        policy_enabled=False,
        dependency_ready=False,
        dependency_reason="not required by structural P0",
    )

    result, trace = selector.select_with_trace(
        Query(query_id="q1", text="question"), _candidate_set(size=20), max_selected=10
    )

    assert len(result.items) == 10
    assert trace.fallback_reason is None
    assert trace.dropped_evidence_ids == ()
    assert {item.action for item in trace.decisions} == {SelectorAction.KEEP}
    assert {item.reason for item in trace.decisions} == {SelectorDecisionReason.POLICY_DISABLED_P0}


def test_select_matches_select_with_trace_and_preserves_retrieval_scores() -> None:
    scores = _scores()
    scores["q1"]["ev-10"] = CandidateRiskScore(protect_score=0.0, harm_score=1.0)
    selector = _selector(scores, max_delete=1)
    query = Query(query_id="q1", text="question")
    candidates = _candidate_set()

    result = selector.select(query, candidates, max_selected=10)
    traced_result, trace = selector.select_with_trace(query, candidates, max_selected=10)

    assert result == traced_result
    assert [item.selection_rank for item in result.items] == list(range(1, 10))
    assert result.items[0].selection_score == pytest.approx(0.99)
    assert SelectorDecisionTrace.model_validate_json(trace.model_dump_json()) == trace


def test_trace_rejects_a_tampered_candidate_retrieval_rank() -> None:
    _, trace = _selector(_scores()).select_with_trace(
        Query(query_id="q1", text="question"), _candidate_set(), max_selected=10
    )
    payload = trace.model_dump(mode="python")
    payload["decisions"][0]["retrieval_rank"] = payload["decisions"][1]["retrieval_rank"]

    with pytest.raises(ValueError, match="retrieval ranks must be unique"):
        SelectorDecisionTrace.model_validate(payload)


def test_constructor_freezes_nested_score_mapping() -> None:
    scores = _scores()
    selector = _selector(scores, max_delete=1)
    scores["q1"]["ev-10"] = CandidateRiskScore(protect_score=0.0, harm_score=1.0)
    scores["q1"]["new"] = CandidateRiskScore(protect_score=0.0, harm_score=1.0)

    result, trace = selector.select_with_trace(
        Query(query_id="q1", text="question"), _candidate_set(), max_selected=10
    )

    assert len(result.items) == 10
    assert trace.dropped_evidence_ids == ()
    with pytest.raises(FrozenInstanceError):
        selector.max_delete = 3  # type: ignore[misc]


@pytest.mark.parametrize("max_selected", (1, 2, 3, 5, 6, 9, 11))
def test_non_top10_max_selected_fails_fast(max_selected: int) -> None:
    with pytest.raises(ValueError, match="requires max_selected=10"):
        _selector(_scores()).select(
            Query(query_id="q1", text="question"),
            _candidate_set(),
            max_selected=max_selected,
        )


@pytest.mark.parametrize("cap", (0, 4, True, 1.0))
def test_only_caps_one_two_three_are_legal(cap: object) -> None:
    with pytest.raises(ValueError, match="max_delete"):
        _selector(_scores(), max_delete=cap)  # type: ignore[arg-type]


@pytest.mark.parametrize("threshold", (math.nan, math.inf, -0.01, 1.01))
def test_threshold_must_be_a_finite_probability(threshold: float) -> None:
    with pytest.raises(ValueError, match="safe_threshold"):
        _selector(_scores(), threshold=threshold)


def test_query_candidate_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="query IDs differ"):
        _selector(_scores()).select(
            Query(query_id="other", text="question"),
            _candidate_set(),
            max_selected=10,
        )
