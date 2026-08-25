from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from evidence_rag.contracts.models import (
    CandidateSet,
    GenerationResult,
    PipelineRun,
    SelectedEvidenceSet,
    SelectionResult,
)
from evidence_rag.infrastructure.datasets import GoldCase as GoldCase

NonEmpty = Annotated[str, Field(min_length=1)]
NonNegativeCount = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[FiniteFloat, Field(ge=0.0)]
MetricDirection = Literal["higher", "lower"]
StageName = Literal["retriever", "selector", "generator"]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


MetricCompute: TypeAlias = Callable[[PipelineRun, GoldCase], float | None]


@dataclass(frozen=True)
class MetricSpec:
    key: str
    direction: MetricDirection
    compute: MetricCompute
    version: str = "1.0"

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("metric version must not be empty")


class MetricValue(FrozenModel):
    value: FiniteFloat | None
    reason: str | None = None


class CaseEvaluation(FrozenModel):
    query_id: NonEmpty
    trace: PipelineRun
    retriever: dict[str, MetricValue]
    selector: dict[str, MetricValue]
    generator: dict[str, MetricValue]
    system: dict[str, MetricValue]

    def flattened(self) -> dict[str, MetricValue]:
        return {
            **self.retriever,
            **self.selector,
            **self.generator,
            **self.system,
        }


class AggregateMetric(FrozenModel):
    mean: FiniteFloat | None
    n_scored: NonNegativeCount
    n_total: NonNegativeCount

    @model_validator(mode="after")
    def counts_match_mean(self) -> "AggregateMetric":
        if self.n_scored > self.n_total:
            raise ValueError("n_scored cannot exceed n_total")
        if (self.n_scored == 0) != (self.mean is None):
            raise ValueError("mean must be absent exactly when no cases were scored")
        return self


class StageCaseEvaluation(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    query_id: NonEmpty
    metrics: dict[str, MetricValue]


class StageEvaluationReport(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    stage: StageName
    dataset_signature: NonEmpty
    metric_registry_signature: NonEmpty
    case_ids: tuple[NonEmpty, ...]
    per_case: tuple[StageCaseEvaluation, ...]
    aggregate: dict[str, AggregateMetric]
    directions: dict[str, MetricDirection]

    @model_validator(mode="after")
    def cases_and_registry_match(self) -> "StageEvaluationReport":
        if self.case_ids != tuple(case.query_id for case in self.per_case):
            raise ValueError("case IDs must match per-case evaluations")
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("evaluation case IDs must be unique")
        if set(self.aggregate) != set(self.directions):
            raise ValueError("aggregate metrics and directions must have the same keys")
        if any(not key.startswith(f"{self.stage}.") for key in self.directions):
            raise ValueError(f"metric keys must belong to {self.stage} stage")
        for case in self.per_case:
            if set(case.metrics) != set(self.directions):
                raise ValueError("per-case metrics and directions must have the same keys")
        return self


class RetrieverStageRun(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    candidate_sets: tuple[CandidateSet, ...]
    report: StageEvaluationReport

    @model_validator(mode="after")
    def report_matches_artifacts(self) -> "RetrieverStageRun":
        if self.report.stage != "retriever":
            raise ValueError("retriever stage run requires a retriever report")
        artifact_ids = tuple(item.query_id for item in self.candidate_sets)
        if artifact_ids != self.report.case_ids:
            raise ValueError("candidate set query order must match report")
        return self


class SelectorStageRun(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    selection_results: tuple[SelectionResult, ...]
    selected_sets: tuple[SelectedEvidenceSet, ...]
    report: StageEvaluationReport

    @model_validator(mode="after")
    def report_and_artifacts_match(self) -> "SelectorStageRun":
        if self.report.stage != "selector":
            raise ValueError("selector stage run requires a selector report")
        selection_ids = tuple(item.query_id for item in self.selection_results)
        selected_ids = tuple(item.query_id for item in self.selected_sets)
        if selection_ids != self.report.case_ids or selected_ids != self.report.case_ids:
            raise ValueError("selector artifact query order must match report")
        for selection, selected in zip(
            self.selection_results,
            self.selected_sets,
            strict=True,
        ):
            selection_evidence_ids = tuple(
                item.evidence_id for item in selection.items
            )
            selected_evidence_ids = tuple(
                item.evidence_id for item in selected.evidence
            )
            if selection_evidence_ids != selected_evidence_ids:
                raise ValueError("selected evidence IDs must match selection")
        return self


class GeneratorStageRun(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    generation_results: tuple[GenerationResult, ...]
    report: StageEvaluationReport

    @model_validator(mode="after")
    def report_matches_artifacts(self) -> "GeneratorStageRun":
        if self.report.stage != "generator":
            raise ValueError("generator stage run requires a generator report")
        artifact_ids = tuple(item.query_id for item in self.generation_results)
        if artifact_ids != self.report.case_ids:
            raise ValueError("generation result query order must match report")
        return self


class EvaluationReport(FrozenModel):
    dataset_signature: NonEmpty
    evaluation_set_signature: NonEmpty
    metric_registry_signature: NonEmpty
    case_ids: tuple[NonEmpty, ...]
    per_case: tuple[CaseEvaluation, ...]
    aggregate: dict[str, AggregateMetric]
    directions: dict[str, MetricDirection]

    @model_validator(mode="after")
    def cases_and_registry_match(self) -> "EvaluationReport":
        if self.case_ids != tuple(case.query_id for case in self.per_case):
            raise ValueError("case IDs must match per-case evaluations")
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("evaluation case IDs must be unique")
        if set(self.aggregate) != set(self.directions):
            raise ValueError("aggregate metrics and directions must have the same keys")
        return self


class RegressionGuard(FrozenModel):
    metric_key: NonEmpty
    allowed_regression: NonNegativeFloat = 0.0


class RegressionReport(FrozenModel):
    passed: bool
    deltas: dict[str, FiniteFloat]
    failures: tuple[str, ...]
