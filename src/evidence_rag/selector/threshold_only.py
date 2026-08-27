"""Threshold-only Selector frozen for Experiment 05."""

from __future__ import annotations

import importlib
import math
from contextlib import nullcontext
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from evidence_rag.contracts.models import (
    CandidateSet,
    Query,
    SelectionItem,
    SelectionResult,
)

DEFAULT_THRESHOLD = 0.9212157130241394


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ThresholdDecision(_FrozenModel):
    evidence_id: str = Field(min_length=1)
    retrieval_rank: int = Field(ge=1)
    protect_score: FiniteFloat | None = Field(default=None, ge=0.0, le=1.0)
    harm_score: FiniteFloat | None = Field(default=None, ge=0.0, le=1.0)
    action: Literal["KEEP", "DROP", "FAIL_OPEN_KEEP"]


class ThresholdSelectionTrace(_FrozenModel):
    schema_version: Literal["experiment05.threshold_selector.v1"] = (
        "experiment05.threshold_selector.v1"
    )
    query_id: str = Field(min_length=1)
    threshold: FiniteFloat = Field(ge=0.0, le=1.0)
    status: Literal["normal", "fail_open_all"]
    failure_reason: str | None = None
    baseline_evidence_ids: tuple[str, ...]
    selected_evidence_ids: tuple[str, ...]
    dropped_evidence_ids: tuple[str, ...]
    decisions: tuple[ThresholdDecision, ...]

    @model_validator(mode="after")
    def coherent_partition(self) -> ThresholdSelectionTrace:
        baseline = self.baseline_evidence_ids
        selected = self.selected_evidence_ids
        dropped = self.dropped_evidence_ids
        if len(baseline) != len(set(baseline)):
            raise ValueError("baseline evidence IDs must be unique")
        if set(selected) | set(dropped) != set(baseline) or set(selected) & set(dropped):
            raise ValueError("selected and dropped IDs must partition the baseline")
        if selected != tuple(item for item in baseline if item not in set(dropped)):
            raise ValueError("selected IDs must preserve retrieval order")
        if tuple(item.evidence_id for item in self.decisions) != baseline:
            raise ValueError("decisions must follow the complete retrieval order")
        if self.status == "fail_open_all":
            if dropped or selected != baseline or not self.failure_reason:
                raise ValueError("fail-open must keep all evidence and record a reason")
            if any(item.action != "FAIL_OPEN_KEEP" for item in self.decisions):
                raise ValueError("fail-open decisions must all be FAIL_OPEN_KEEP")
        elif self.failure_reason is not None:
            raise ValueError("normal selector trace cannot contain a failure reason")
        return self


def _values(raw: Any, *, name: str, expected: int) -> list[float]:
    value = raw
    for method_name in ("detach", "float", "cpu"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a one-dimensional sequence")
    values = [float(item) for item in value]
    if len(values) != expected:
        raise ValueError(f"{name} returned {len(values)} values for {expected} candidates")
    return values


class NliThresholdOnlySelector:
    """Delete every item satisfying the frozen two-score predicate.

    Any model, parsing, missing-value, non-finite, or range failure is handled at
    query level by keeping the complete Top10 input.
    """

    def __init__(self, *, model: Any, threshold: float = DEFAULT_THRESHOLD) -> None:
        if isinstance(threshold, bool):
            raise TypeError("threshold must be a real number")
        self.threshold = float(threshold)
        if not math.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be finite and in [0, 1]")
        self.model = model
        self.last_trace: ThresholdSelectionTrace | None = None

    def select(
        self, query: Query, candidates: CandidateSet, max_selected: int
    ) -> SelectionResult:
        result, _trace = self.select_with_trace(query, candidates, max_selected)
        return result

    def select_with_trace(
        self, query: Query, candidates: CandidateSet, max_selected: int
    ) -> tuple[SelectionResult, ThresholdSelectionTrace]:
        if query.query_id != candidates.query_id:
            raise ValueError("query and candidates query IDs differ")
        if max_selected != 10:
            raise ValueError("Experiment 05 Selector requires max_selected=10")
        baseline = tuple(
            sorted(
                candidates.candidates,
                key=lambda item: (item.retrieval_rank, item.evidence_id),
            )[:10]
        )
        try:
            try:
                torch = importlib.import_module("torch")
            except ImportError:
                inference_context = nullcontext()
            else:
                inference_context = torch.inference_mode()
            with inference_context:
                output = self.model(
                    question=[query.text] * len(baseline),
                    candidate_text=[item.text for item in baseline],
                )
            protect = _values(
                output.protect_scores, name="protect_scores", expected=len(baseline)
            )
            harm = _values(output.harm_scores, name="harm_scores", expected=len(baseline))
            if any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in (*protect, *harm)
            ):
                raise ValueError("selector scores must be finite probabilities")
        except Exception as error:  # noqa: BLE001 - query-level keep-all is the protocol
            return self._fail_open(query, baseline, type(error).__name__)

        decisions = tuple(
            ThresholdDecision(
                evidence_id=item.evidence_id,
                retrieval_rank=item.retrieval_rank,
                protect_score=protect_score,
                harm_score=harm_score,
                action=(
                    "DROP"
                    if harm_score >= self.threshold
                    and protect_score <= 1.0 - self.threshold
                    else "KEEP"
                ),
            )
            for item, protect_score, harm_score in zip(
                baseline, protect, harm, strict=True
            )
        )
        dropped = tuple(item.evidence_id for item in decisions if item.action == "DROP")
        selected = tuple(
            item for item in baseline if item.evidence_id not in set(dropped)
        )
        result = SelectionResult(
            query_id=query.query_id,
            items=tuple(
                SelectionItem(
                    evidence_id=item.evidence_id,
                    selection_score=float(item.retrieval_score),
                    selection_rank=rank,
                )
                for rank, item in enumerate(selected, start=1)
            ),
        )
        trace = ThresholdSelectionTrace(
            query_id=query.query_id,
            threshold=self.threshold,
            status="normal",
            baseline_evidence_ids=tuple(item.evidence_id for item in baseline),
            selected_evidence_ids=tuple(item.evidence_id for item in selected),
            dropped_evidence_ids=dropped,
            decisions=decisions,
        )
        self.last_trace = trace
        return result, trace

    def _fail_open(
        self, query: Query, baseline: tuple[Any, ...], reason: str
    ) -> tuple[SelectionResult, ThresholdSelectionTrace]:
        result = SelectionResult(
            query_id=query.query_id,
            items=tuple(
                SelectionItem(
                    evidence_id=item.evidence_id,
                    selection_score=float(item.retrieval_score),
                    selection_rank=rank,
                )
                for rank, item in enumerate(baseline, start=1)
            ),
        )
        trace = ThresholdSelectionTrace(
            query_id=query.query_id,
            threshold=self.threshold,
            status="fail_open_all",
            failure_reason=reason,
            baseline_evidence_ids=tuple(item.evidence_id for item in baseline),
            selected_evidence_ids=tuple(item.evidence_id for item in baseline),
            dropped_evidence_ids=(),
            decisions=tuple(
                ThresholdDecision(
                    evidence_id=item.evidence_id,
                    retrieval_rank=item.retrieval_rank,
                    action="FAIL_OPEN_KEEP",
                )
                for item in baseline
            ),
        )
        self.last_trace = trace
        return result, trace
