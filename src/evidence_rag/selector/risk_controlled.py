"""Delete-only, risk-controlled Selector over the frozen TopK10 baseline.

This module is deliberately pure Python and has no Torch dependency.  A scorer or experiment
runner supplies already-computed independent protect/harm scores through the constructor.  The
normal ``select`` method still satisfies the production Selector protocol; ``select_with_trace``
returns the same result plus the separate research sidecar.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

from evidence_rag.contracts.models import (
    CandidateSet,
    EvidenceCandidate,
    Query,
    SelectionItem,
    SelectionResult,
)
from evidence_rag.selector.models import (
    BASELINE_K,
    MIN_KEEP,
    CandidateRiskScore,
    SelectorAction,
    SelectorCandidateDecision,
    SelectorDecisionReason,
    SelectorDecisionTrace,
    SelectorFallbackReason,
    deletion_priority_key,
    fallback_decision_reason,
)

ScoreTable = Mapping[str, Mapping[str, CandidateRiskScore]]


def _probability_or_none(value: float) -> float | None:
    if math.isfinite(value) and 0.0 <= value <= 1.0:
        return value
    return None


@dataclass(frozen=True, init=False, slots=True)
class RiskControlledSelector:
    """Conservative 0--cap deletion policy with a structural P0 mode.

    ``scores_by_query`` is copied into immutable nested mappings at construction time.  A caller
    cannot change a completed run by mutating the dictionaries it originally passed in.
    """

    _scores_by_query: Mapping[str, Mapping[str, CandidateRiskScore]]
    safe_threshold: float
    max_delete: Literal[1, 2, 3]
    policy_enabled: bool
    dependency_ready: bool
    dependency_reason: str | None

    def __init__(
        self,
        *,
        scores_by_query: ScoreTable | None = None,
        safe_threshold: float,
        max_delete: Literal[1, 2, 3] | int,
        policy_enabled: bool = True,
        dependency_ready: bool = True,
        dependency_reason: str | None = None,
    ) -> None:
        if isinstance(safe_threshold, bool):
            raise TypeError("safe_threshold must be a real number, not bool")
        threshold = float(safe_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("safe_threshold must be finite and in [0, 1]")
        if type(max_delete) is not int or max_delete not in (1, 2, 3):
            raise ValueError("max_delete must be one of 1, 2, or 3")
        if type(policy_enabled) is not bool:  # noqa: E721 - reject truthy integers explicitly
            raise TypeError("policy_enabled must be bool")
        if type(dependency_ready) is not bool:  # noqa: E721 - reject truthy integers explicitly
            raise TypeError("dependency_ready must be bool")
        if dependency_ready and dependency_reason is not None:
            raise ValueError("dependency_reason is only valid when dependency_ready is false")
        if not dependency_ready and (
            not isinstance(dependency_reason, str) or not dependency_reason.strip()
        ):
            raise ValueError("a missing dependency requires a non-blank dependency_reason")

        copied_scores: dict[str, Mapping[str, CandidateRiskScore]] = {}
        for query_id, candidate_scores in (scores_by_query or {}).items():
            if not isinstance(query_id, str) or not query_id:
                raise ValueError("score query IDs must be non-empty strings")
            copied_candidates: dict[str, CandidateRiskScore] = {}
            for evidence_id, score in candidate_scores.items():
                if not isinstance(evidence_id, str) or not evidence_id:
                    raise ValueError("score evidence IDs must be non-empty strings")
                if not isinstance(score, CandidateRiskScore):
                    raise TypeError("every score value must be a CandidateRiskScore")
                copied_candidates[evidence_id] = score
            copied_scores[query_id] = MappingProxyType(copied_candidates)

        object.__setattr__(self, "_scores_by_query", MappingProxyType(copied_scores))
        object.__setattr__(self, "safe_threshold", threshold)
        object.__setattr__(self, "max_delete", cast(Literal[1, 2, 3], max_delete))
        object.__setattr__(self, "policy_enabled", policy_enabled)
        object.__setattr__(self, "dependency_ready", dependency_ready)
        object.__setattr__(self, "dependency_reason", dependency_reason)

    def select(
        self,
        query: Query,
        candidates: CandidateSet,
        max_selected: int,
    ) -> SelectionResult:
        """Satisfy the unchanged production Selector protocol."""

        result, _ = self.select_with_trace(query, candidates, max_selected)
        return result

    def select_with_trace(
        self,
        query: Query,
        candidates: CandidateSet,
        max_selected: int,
    ) -> tuple[SelectionResult, SelectorDecisionTrace]:
        """Return the Generator-facing result and an independent decision sidecar."""

        if max_selected != BASELINE_K:
            raise ValueError(
                f"risk-controlled Selector requires max_selected={BASELINE_K}, got {max_selected}"
            )
        if query.query_id != candidates.query_id:
            raise ValueError("query and candidates query IDs differ")

        baseline = tuple(
            sorted(
                candidates.candidates,
                key=lambda item: (item.retrieval_rank, item.evidence_id),
            )[:BASELINE_K]
        )
        raw_scores = self._scores_by_query.get(query.query_id, MappingProxyType({}))

        if not self.policy_enabled:
            decisions = self._p0_decisions(baseline, raw_scores)
            return self._outputs(
                query_id=query.query_id,
                baseline=baseline,
                decisions=decisions,
                fallback_reason=None,
            )

        if not self.dependency_ready:
            return self._fallback_outputs(
                query_id=query.query_id,
                baseline=baseline,
                raw_scores=raw_scores,
                fallback_reason=SelectorFallbackReason.MISSING_DEPENDENCY,
            )
        if len(baseline) < MIN_KEEP:
            return self._fallback_outputs(
                query_id=query.query_id,
                baseline=baseline,
                raw_scores=raw_scores,
                fallback_reason=SelectorFallbackReason.TOO_FEW_CANDIDATES,
            )

        missing_score = any(item.evidence_id not in raw_scores for item in baseline)
        if missing_score:
            return self._fallback_outputs(
                query_id=query.query_id,
                baseline=baseline,
                raw_scores=raw_scores,
                fallback_reason=SelectorFallbackReason.MISSING_SCORE,
            )
        invalid_score = any(
            not raw_scores[item.evidence_id].is_valid_probability_pair() for item in baseline
        )
        if invalid_score:
            return self._fallback_outputs(
                query_id=query.query_id,
                baseline=baseline,
                raw_scores=raw_scores,
                fallback_reason=SelectorFallbackReason.INVALID_SCORE,
            )

        scores = {item.evidence_id: raw_scores[item.evidence_id] for item in baseline}
        decisions_by_id = self._active_decisions(baseline, scores)
        decisions = tuple(decisions_by_id[item.evidence_id] for item in baseline)
        return self._outputs(
            query_id=query.query_id,
            baseline=baseline,
            decisions=decisions,
            fallback_reason=None,
        )

    def _p0_decisions(
        self,
        baseline: tuple[EvidenceCandidate, ...],
        raw_scores: Mapping[str, CandidateRiskScore],
    ) -> tuple[SelectorCandidateDecision, ...]:
        return tuple(
            self._decision(
                candidate,
                raw_scores.get(candidate.evidence_id),
                action=SelectorAction.KEEP,
                reason=SelectorDecisionReason.POLICY_DISABLED_P0,
            )
            for candidate in baseline
        )

    def _fallback_outputs(
        self,
        *,
        query_id: str,
        baseline: tuple[EvidenceCandidate, ...],
        raw_scores: Mapping[str, CandidateRiskScore],
        fallback_reason: SelectorFallbackReason,
    ) -> tuple[SelectionResult, SelectorDecisionTrace]:
        decision_reason = fallback_decision_reason(fallback_reason)
        decisions = tuple(
            self._decision(
                candidate,
                raw_scores.get(candidate.evidence_id),
                action=SelectorAction.ABSTAIN_KEEP,
                reason=decision_reason,
            )
            for candidate in baseline
        )
        return self._outputs(
            query_id=query_id,
            baseline=baseline,
            decisions=decisions,
            fallback_reason=fallback_reason,
        )

    def _active_decisions(
        self,
        baseline: tuple[EvidenceCandidate, ...],
        scores: Mapping[str, CandidateRiskScore],
    ) -> dict[str, SelectorCandidateDecision]:
        policy_order = sorted(
            baseline,
            key=lambda item: deletion_priority_key(
                safe_score=self._valid_safe_score(scores[item.evidence_id]),
                retrieval_rank=item.retrieval_rank,
                evidence_id=item.evidence_id,
            ),
        )
        decisions: dict[str, SelectorCandidateDecision] = {}
        dropped = 0
        for candidate in policy_order:
            score = scores[candidate.evidence_id]
            safe_score = self._valid_safe_score(score)
            if safe_score < self.safe_threshold:
                if score.harm_score >= self.safe_threshold:
                    action = SelectorAction.ABSTAIN_KEEP
                    reason = SelectorDecisionReason.PROTECT_HARM_CONFLICT
                else:
                    action = SelectorAction.KEEP
                    reason = SelectorDecisionReason.BELOW_SAFE_THRESHOLD
            elif dropped >= self.max_delete:
                action = SelectorAction.ABSTAIN_KEEP
                reason = SelectorDecisionReason.MAX_DELETE_REACHED
            elif len(baseline) - dropped - 1 < MIN_KEEP:
                action = SelectorAction.ABSTAIN_KEEP
                reason = SelectorDecisionReason.MIN_KEEP_REACHED
            else:
                action = SelectorAction.DROP_HARM
                reason = SelectorDecisionReason.SAFE_SCORE_THRESHOLD_MET
                dropped += 1
            decisions[candidate.evidence_id] = self._decision(
                candidate,
                score,
                action=action,
                reason=reason,
            )
        return decisions

    @staticmethod
    def _valid_safe_score(score: CandidateRiskScore) -> float:
        safe_score = score.safe_score()
        if safe_score is None:  # protected by the whole-query validation above
            raise RuntimeError("invalid score reached the active deletion policy")
        return safe_score

    @staticmethod
    def _decision(
        candidate: EvidenceCandidate,
        score: CandidateRiskScore | None,
        *,
        action: SelectorAction,
        reason: SelectorDecisionReason,
    ) -> SelectorCandidateDecision:
        if score is None:
            protect_score = None
            harm_score = None
            safe_score = None
        else:
            protect_score = _probability_or_none(score.protect_score)
            harm_score = _probability_or_none(score.harm_score)
            safe_score = score.safe_score()
        return SelectorCandidateDecision(
            evidence_id=candidate.evidence_id,
            retrieval_rank=candidate.retrieval_rank,
            protect_score=protect_score,
            harm_score=harm_score,
            safe_score=safe_score,
            action=action,
            reason=reason,
        )

    def _outputs(
        self,
        *,
        query_id: str,
        baseline: tuple[EvidenceCandidate, ...],
        decisions: tuple[SelectorCandidateDecision, ...],
        fallback_reason: SelectorFallbackReason | None,
    ) -> tuple[SelectionResult, SelectorDecisionTrace]:
        action_by_id = {decision.evidence_id: decision.action for decision in decisions}
        selected = tuple(
            candidate
            for candidate in baseline
            if action_by_id[candidate.evidence_id] is not SelectorAction.DROP_HARM
        )
        dropped_ids = tuple(
            candidate.evidence_id
            for candidate in sorted(
                baseline,
                key=lambda item: deletion_priority_key(
                    safe_score=(
                        next(
                            decision.safe_score
                            for decision in decisions
                            if decision.evidence_id == item.evidence_id
                        )
                        or 0.0
                    ),
                    retrieval_rank=item.retrieval_rank,
                    evidence_id=item.evidence_id,
                ),
            )
            if action_by_id[candidate.evidence_id] is SelectorAction.DROP_HARM
        )
        result = SelectionResult(
            query_id=query_id,
            items=tuple(
                SelectionItem(
                    evidence_id=candidate.evidence_id,
                    selection_score=candidate.retrieval_score,
                    selection_rank=selection_rank,
                )
                for selection_rank, candidate in enumerate(selected, start=1)
            ),
        )
        trace = SelectorDecisionTrace(
            query_id=query_id,
            policy_enabled=self.policy_enabled,
            max_delete=self.max_delete,
            safe_threshold=self.safe_threshold,
            dependency_ready=self.dependency_ready,
            dependency_reason=self.dependency_reason,
            fallback_reason=fallback_reason,
            baseline_evidence_ids=tuple(item.evidence_id for item in baseline),
            selected_evidence_ids=tuple(item.evidence_id for item in selected),
            dropped_evidence_ids=dropped_ids,
            decisions=decisions,
        )
        return result, trace
