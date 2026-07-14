from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from evidence_rag.contracts.models import PipelineRun

NonEmpty = Annotated[str, Field(min_length=1)]
NonNegativeCount = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[FiniteFloat, Field(ge=0.0)]
MetricDirection = Literal["higher", "lower"]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GoldCase(FrozenModel):
    query_id: NonEmpty
    relevant_document_ids: tuple[NonEmpty, ...] | None = None
    reference_answers: tuple[NonEmpty, ...] | None = None

    @model_validator(mode="after")
    def labels_are_unique(self) -> "GoldCase":
        for values in (self.relevant_document_ids, self.reference_answers):
            if values is not None and len(values) != len(set(values)):
                raise ValueError("gold labels must be unique")
        return self


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


class EvaluationReport(FrozenModel):
    dataset_signature: NonEmpty
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
