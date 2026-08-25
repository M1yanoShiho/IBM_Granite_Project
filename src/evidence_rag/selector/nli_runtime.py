"""Live NLI scoring adapter for the frozen conservative Selector policy."""

from __future__ import annotations

import importlib
from contextlib import nullcontext
from typing import Any

from evidence_rag.contracts.models import CandidateSet, Query, SelectionResult
from evidence_rag.selector.models import CandidateRiskScore, SelectorDecisionTrace
from evidence_rag.selector.risk_controlled import RiskControlledSelector


def _float_values(raw: Any, *, name: str, expected: int) -> list[float]:
    value = raw
    for method_name in ("detach", "float", "cpu"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Selector model {name} must be a one-dimensional sequence")
    values = [float(item) for item in value]
    if len(values) != expected:
        raise ValueError(
            f"Selector model {name} returned {len(values)} values for {expected} candidates"
        )
    return values


class NliRiskControlledSelector:
    """Score the current TopK10, then apply the frozen harm/protect delete policy."""

    def __init__(self, *, model: Any, safe_threshold: float, max_delete: int) -> None:
        self.model = model
        self.safe_threshold = safe_threshold
        self.max_delete = max_delete
        self.last_trace: SelectorDecisionTrace | None = None

    def select(
        self,
        query: Query,
        candidates: CandidateSet,
        max_selected: int,
    ) -> SelectionResult:
        result, _trace = self.select_with_trace(query, candidates, max_selected)
        return result

    def select_with_trace(
        self,
        query: Query,
        candidates: CandidateSet,
        max_selected: int,
    ) -> tuple[SelectionResult, SelectorDecisionTrace]:
        baseline = tuple(
            sorted(
                candidates.candidates,
                key=lambda item: (item.retrieval_rank, item.evidence_id),
            )[:10]
        )
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            inference_context = nullcontext()
        else:
            inference_context = torch.inference_mode()
        with inference_context:
            output = self.model(
                question=[query.text] * len(baseline),
                candidate_text=[candidate.text for candidate in baseline],
            )
        protect = _float_values(
            output.protect_scores,
            name="protect_scores",
            expected=len(baseline),
        )
        harm = _float_values(
            output.harm_scores,
            name="harm_scores",
            expected=len(baseline),
        )
        scores = {
            query.query_id: {
                candidate.evidence_id: CandidateRiskScore(
                    protect_score=protect_score,
                    harm_score=harm_score,
                )
                for candidate, protect_score, harm_score in zip(
                    baseline,
                    protect,
                    harm,
                    strict=True,
                )
            }
        }
        policy = RiskControlledSelector(
            scores_by_query=scores,
            safe_threshold=self.safe_threshold,
            max_delete=self.max_delete,
        )
        result, trace = policy.select_with_trace(query, candidates, max_selected)
        self.last_trace = trace
        return result, trace
