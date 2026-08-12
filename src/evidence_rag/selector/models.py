"""Typed, JSON-safe records for the conservative Selector policy.

The production :class:`~evidence_rag.contracts.models.SelectionResult` contract intentionally
stays unchanged.  Model scores and the reason for every keep/drop decision live in the separate
``SelectorDecisionTrace`` sidecar defined here, so research diagnostics cannot accidentally leak
into the Generator-facing evidence contract.

``CandidateRiskScore`` is an *untrusted policy input*.  Its two fields deliberately use plain
``float`` rather than Pydantic's ``FiniteFloat``: the policy boundary must be able to receive a
NaN/Inf from a failed model run and turn the whole query into a safe keep-all fallback.  Invalid
values are never copied into the JSON-safe trace.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StrictBool,
    model_validator,
)

BASELINE_K: Literal[10] = 10
MIN_KEEP: Literal[7] = 7
POLICY_PROTOCOL_VERSION: Literal["selector-risk-controlled-v1"] = "selector-risk-controlled-v1"

NonEmpty = Annotated[str, Field(min_length=1)]
Probability = Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
PositiveRank = Annotated[int, Field(ge=1)]


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class SelectorAction(StrEnum):
    """The only three candidate-level actions in the v2 experiment."""

    KEEP = "KEEP"
    DROP_HARM = "DROP_HARM"
    ABSTAIN_KEEP = "ABSTAIN_KEEP"


class SelectorDecisionReason(StrEnum):
    """Auditable reason for one candidate action."""

    POLICY_DISABLED_P0 = "POLICY_DISABLED_P0"
    SAFE_SCORE_THRESHOLD_MET = "SAFE_SCORE_THRESHOLD_MET"
    BELOW_SAFE_THRESHOLD = "BELOW_SAFE_THRESHOLD"
    PROTECT_HARM_CONFLICT = "PROTECT_HARM_CONFLICT"
    MAX_DELETE_REACHED = "MAX_DELETE_REACHED"
    MIN_KEEP_REACHED = "MIN_KEEP_REACHED"
    FALLBACK_MISSING_DEPENDENCY = "FALLBACK_MISSING_DEPENDENCY"
    FALLBACK_TOO_FEW_CANDIDATES = "FALLBACK_TOO_FEW_CANDIDATES"
    FALLBACK_MISSING_SCORE = "FALLBACK_MISSING_SCORE"
    FALLBACK_INVALID_SCORE = "FALLBACK_INVALID_SCORE"


class SelectorFallbackReason(StrEnum):
    """Whole-query conditions that force the TopK keep-all fallback."""

    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
    TOO_FEW_CANDIDATES = "TOO_FEW_CANDIDATES"
    MISSING_SCORE = "MISSING_SCORE"
    INVALID_SCORE = "INVALID_SCORE"


_FALLBACK_DECISION_REASON: dict[SelectorFallbackReason, SelectorDecisionReason] = {
    SelectorFallbackReason.MISSING_DEPENDENCY: (SelectorDecisionReason.FALLBACK_MISSING_DEPENDENCY),
    SelectorFallbackReason.TOO_FEW_CANDIDATES: (SelectorDecisionReason.FALLBACK_TOO_FEW_CANDIDATES),
    SelectorFallbackReason.MISSING_SCORE: SelectorDecisionReason.FALLBACK_MISSING_SCORE,
    SelectorFallbackReason.INVALID_SCORE: SelectorDecisionReason.FALLBACK_INVALID_SCORE,
}


def deletion_priority_key(
    *, safe_score: float, retrieval_rank: int, evidence_id: str
) -> tuple[float, int, str]:
    """Total ordering for conservative deletion candidates.

    Python sorts ascending, hence the two numeric negations implement descending safe score and
    descending retrieval rank.  ``evidence_id`` is the final ascending tie-breaker.  Frozen
    candidate sets already require unique retrieval ranks, but keeping the final key makes the
    policy deterministic even before/while an upstream set is being validated.
    """

    return (-safe_score, -retrieval_rank, evidence_id)


class CandidateRiskScore(_FrozenStrictModel):
    """Untrusted independent-sigmoid outputs for one candidate.

    Range and finiteness checks belong to ``RiskControlledSelector`` because failing those checks
    is a defined runtime state (keep the whole query), not a schema-construction exception.
    """

    protect_score: float
    harm_score: float

    def is_valid_probability_pair(self) -> bool:
        return all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in (self.protect_score, self.harm_score)
        )

    def safe_score(self) -> float | None:
        if not self.is_valid_probability_pair():
            return None
        return min(self.harm_score, 1.0 - self.protect_score)


class SelectorCandidateDecision(_FrozenStrictModel):
    """One baseline-candidate row in ``decision_trace.jsonl``."""

    evidence_id: NonEmpty
    retrieval_rank: PositiveRank
    protect_score: Probability | None
    harm_score: Probability | None
    safe_score: Probability | None
    action: SelectorAction
    reason: SelectorDecisionReason

    @model_validator(mode="after")
    def score_and_action_are_coherent(self) -> Self:
        if (self.protect_score is None or self.harm_score is None) and self.safe_score is not None:
            raise ValueError("safe_score requires both protect_score and harm_score")
        if self.protect_score is not None and self.harm_score is not None:
            expected = min(self.harm_score, 1.0 - self.protect_score)
            if self.safe_score is None or not math.isclose(
                self.safe_score, expected, rel_tol=0.0, abs_tol=1e-15
            ):
                raise ValueError("safe_score must equal min(harm_score, 1-protect_score)")
        if self.action is SelectorAction.DROP_HARM:
            if self.reason is not SelectorDecisionReason.SAFE_SCORE_THRESHOLD_MET:
                raise ValueError("DROP_HARM requires SAFE_SCORE_THRESHOLD_MET")
            if self.safe_score is None:
                raise ValueError("DROP_HARM requires a valid safe_score")
        if self.action is SelectorAction.KEEP and self.reason not in {
            SelectorDecisionReason.POLICY_DISABLED_P0,
            SelectorDecisionReason.BELOW_SAFE_THRESHOLD,
        }:
            raise ValueError("KEEP has an incompatible decision reason")
        if self.action is SelectorAction.ABSTAIN_KEEP and self.reason not in {
            SelectorDecisionReason.PROTECT_HARM_CONFLICT,
            SelectorDecisionReason.MAX_DELETE_REACHED,
            SelectorDecisionReason.MIN_KEEP_REACHED,
            *_FALLBACK_DECISION_REASON.values(),
        }:
            raise ValueError("ABSTAIN_KEEP has an incompatible decision reason")
        return self


class SelectorDecisionTrace(_FrozenStrictModel):
    """One query's independent, JSON-safe policy audit sidecar."""

    schema_version: Literal["1.0"] = "1.0"
    protocol_version: Literal["selector-risk-controlled-v1"] = POLICY_PROTOCOL_VERSION
    query_id: NonEmpty
    policy_enabled: StrictBool
    baseline_k: Literal[10] = BASELINE_K
    min_keep: Literal[7] = MIN_KEEP
    max_delete: Literal[1, 2, 3]
    safe_threshold: Probability
    dependency_ready: StrictBool
    dependency_reason: NonEmpty | None = None
    fallback_reason: SelectorFallbackReason | None = None
    baseline_evidence_ids: tuple[NonEmpty, ...] = Field(max_length=BASELINE_K)
    selected_evidence_ids: tuple[NonEmpty, ...] = Field(max_length=BASELINE_K)
    dropped_evidence_ids: tuple[NonEmpty, ...] = Field(max_length=3)
    decisions: tuple[SelectorCandidateDecision, ...] = Field(max_length=BASELINE_K)

    @model_validator(mode="after")
    def trace_is_internally_consistent(self) -> Self:
        baseline = self.baseline_evidence_ids
        selected = self.selected_evidence_ids
        dropped = self.dropped_evidence_ids
        decision_ids = tuple(decision.evidence_id for decision in self.decisions)
        decision_ranks = tuple(decision.retrieval_rank for decision in self.decisions)
        for label, values in (
            ("baseline", baseline),
            ("selected", selected),
            ("dropped", dropped),
            ("decision", decision_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} evidence IDs must be unique")
        if decision_ids != baseline:
            raise ValueError("decision rows must follow the complete baseline retrieval order")
        if len(decision_ranks) != len(set(decision_ranks)):
            raise ValueError("decision retrieval ranks must be unique")
        if decision_ranks != tuple(sorted(decision_ranks)):
            raise ValueError("decision rows must be ordered by increasing retrieval rank")
        if set(selected) | set(dropped) != set(baseline):
            raise ValueError("selected and dropped IDs must partition the baseline IDs")
        if set(selected) & set(dropped):
            raise ValueError("an evidence ID cannot be both selected and dropped")
        expected_selected = tuple(
            evidence_id for evidence_id in baseline if evidence_id not in set(dropped)
        )
        if selected != expected_selected:
            raise ValueError("selected IDs must preserve the baseline retrieval order")

        action_by_id = {decision.evidence_id: decision.action for decision in self.decisions}
        action_drops = {
            evidence_id
            for evidence_id, action in action_by_id.items()
            if action is SelectorAction.DROP_HARM
        }
        if action_drops != set(dropped):
            raise ValueError("DROP_HARM actions must exactly match dropped_evidence_ids")
        if len(dropped) > self.max_delete:
            raise ValueError("dropped candidates exceed max_delete")
        if len(baseline) >= self.min_keep and len(selected) < self.min_keep:
            raise ValueError("selected candidates violate min_keep")

        if self.dependency_ready and self.dependency_reason is not None:
            raise ValueError("dependency_reason is only valid when dependency_ready is false")
        if not self.dependency_ready and self.dependency_reason is None:
            raise ValueError("a missing dependency requires dependency_reason")

        if not self.policy_enabled:
            if self.fallback_reason is not None:
                raise ValueError("explicit P0 is not a fallback")
            if selected != baseline or dropped:
                raise ValueError("explicit P0 must keep the complete TopK baseline")
            if any(
                decision.action is not SelectorAction.KEEP
                or decision.reason is not SelectorDecisionReason.POLICY_DISABLED_P0
                for decision in self.decisions
            ):
                raise ValueError("explicit P0 requires KEEP/POLICY_DISABLED_P0 for every item")
            return self

        if self.fallback_reason is not None:
            expected_reason = _FALLBACK_DECISION_REASON[self.fallback_reason]
            if (
                self.fallback_reason is SelectorFallbackReason.MISSING_DEPENDENCY
                and self.dependency_ready
            ):
                raise ValueError("missing-dependency fallback requires dependency_ready=false")
            if (
                self.fallback_reason is not SelectorFallbackReason.MISSING_DEPENDENCY
                and not self.dependency_ready
            ):
                raise ValueError("a missing dependency must take fallback precedence")
            if (
                self.fallback_reason is SelectorFallbackReason.TOO_FEW_CANDIDATES
                and len(baseline) >= self.min_keep
            ):
                raise ValueError("too-few-candidates fallback requires fewer than min_keep")
            if len(baseline) < self.min_keep and self.fallback_reason not in {
                SelectorFallbackReason.MISSING_DEPENDENCY,
                SelectorFallbackReason.TOO_FEW_CANDIDATES,
            }:
                raise ValueError("too-few-candidates fallback must precede score fallback")
            if selected != baseline or dropped:
                raise ValueError("whole-query fallback must keep the complete TopK baseline")
            if any(
                decision.action is not SelectorAction.ABSTAIN_KEEP
                or decision.reason is not expected_reason
                for decision in self.decisions
            ):
                raise ValueError("whole-query fallback must mark every item ABSTAIN_KEEP")
        elif any(
            decision.reason in _FALLBACK_DECISION_REASON.values() for decision in self.decisions
        ):
            raise ValueError("fallback item reason requires a whole-query fallback_reason")
        else:
            if not self.dependency_ready or len(baseline) < self.min_keep:
                raise ValueError(
                    "active policy requires ready dependencies and min_keep candidates"
                )
            if any(
                decision.protect_score is None
                or decision.harm_score is None
                or decision.safe_score is None
                for decision in self.decisions
            ):
                raise ValueError("active policy requires valid scores for every baseline item")
            ordered = sorted(
                self.decisions,
                key=lambda decision: deletion_priority_key(
                    safe_score=cast(
                        float, decision.safe_score
                    ),  # proven non-None immediately above
                    retrieval_rank=decision.retrieval_rank,
                    evidence_id=decision.evidence_id,
                ),
            )
            expected_actions: dict[str, tuple[SelectorAction, SelectorDecisionReason]] = {}
            expected_dropped: list[str] = []
            for decision in ordered:
                safe_score = cast(float, decision.safe_score)
                harm_score = cast(float, decision.harm_score)
                if safe_score < self.safe_threshold:
                    if harm_score >= self.safe_threshold:
                        expected = (
                            SelectorAction.ABSTAIN_KEEP,
                            SelectorDecisionReason.PROTECT_HARM_CONFLICT,
                        )
                    else:
                        expected = (
                            SelectorAction.KEEP,
                            SelectorDecisionReason.BELOW_SAFE_THRESHOLD,
                        )
                elif len(expected_dropped) >= self.max_delete:
                    expected = (
                        SelectorAction.ABSTAIN_KEEP,
                        SelectorDecisionReason.MAX_DELETE_REACHED,
                    )
                elif len(baseline) - len(expected_dropped) - 1 < self.min_keep:
                    expected = (
                        SelectorAction.ABSTAIN_KEEP,
                        SelectorDecisionReason.MIN_KEEP_REACHED,
                    )
                else:
                    expected = (
                        SelectorAction.DROP_HARM,
                        SelectorDecisionReason.SAFE_SCORE_THRESHOLD_MET,
                    )
                    expected_dropped.append(decision.evidence_id)
                expected_actions[decision.evidence_id] = expected
            for decision in self.decisions:
                if (decision.action, decision.reason) != expected_actions[decision.evidence_id]:
                    raise ValueError(
                        f"decision for {decision.evidence_id!r} does not match the frozen policy"
                    )
            if dropped != tuple(expected_dropped):
                raise ValueError("dropped IDs must follow the frozen policy evaluation order")
        return self


def fallback_decision_reason(
    fallback_reason: SelectorFallbackReason,
) -> SelectorDecisionReason:
    """Return the candidate-level reason paired with a whole-query fallback."""

    return _FALLBACK_DECISION_REASON[fallback_reason]
