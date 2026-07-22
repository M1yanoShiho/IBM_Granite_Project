"""Harmful-in-context metric and pool-hit diagnostic (spec §1-§5).

Offline over dumped selected sets + the injector's provenance sidecar; the shared
ExperimentWorkflow and frozen contracts are untouched.
"""

import random
from collections.abc import Iterable, Mapping

from pydantic import BaseModel, ConfigDict

from evidence_rag.contracts.models import CandidateSet, SelectedEvidenceSet
from evidence_rag.evaluation.models import (
    MetricDirection,
    MetricValue,
    StageCaseEvaluation,
    StageEvaluationReport,
)
from evidence_rag.evaluation.scoring import (
    aggregate_metrics,
    document_ids,
    signature,
    unscored,
)
from evidence_rag.materializer.provenance import MutationRecord

# The harm metric is selector-owned and kept out of the shared core registry
# (scoring.CORE_DIRECTIONS) so it never perturbs the full evaluator or the frozen
# reference-artifact signatures. Its direction/version live here.
HARM_METRIC = "selector.core.harmful_in_context"
HARM_DIRECTION: MetricDirection = "lower"
HARM_VERSION = "1.0"


def provenance_harm_map(records: Iterable[MutationRecord]) -> dict[str, str]:
    return {record.query_id: record.counterfactual_document_id for record in records}


def harmful_in_context(
    selected_document_ids: set[str],
    counterfactual_document_id: str | None,
) -> MetricValue:
    if counterfactual_document_id is None:
        return unscored("query has no injected counterfactual")
    return MetricValue(value=float(counterfactual_document_id in selected_document_ids))


def evaluate_selector_harm(
    selected_sets: Iterable[SelectedEvidenceSet],
    harm_map: Mapping[str, str],
    *,
    dataset_signature: str,
) -> StageEvaluationReport:
    selected = tuple(selected_sets)
    if not selected:
        raise ValueError("evaluation dataset must not be empty")
    directions = {HARM_METRIC: HARM_DIRECTION}
    per_case = tuple(
        StageCaseEvaluation(
            query_id=item.query_id,
            metrics={
                HARM_METRIC: harmful_in_context(
                    document_ids(item.evidence), harm_map.get(item.query_id)
                )
            },
        )
        for item in selected
    )
    return StageEvaluationReport(
        stage="selector",
        dataset_signature=dataset_signature,
        metric_registry_signature=signature(
            {"metric": HARM_METRIC, "direction": HARM_DIRECTION, "version": HARM_VERSION}
        ),
        case_ids=tuple(item.query_id for item in selected),
        per_case=per_case,
        aggregate=aggregate_metrics(tuple(case.metrics for case in per_case), directions),
        directions=directions,
    )


def counterfactual_pool_hit_rate(
    candidate_sets: Iterable[CandidateSet],
    harm_map: Mapping[str, str],
) -> float | None:
    hits = 0
    total = 0
    for candidate_set in candidate_sets:
        counterfactual = harm_map.get(candidate_set.query_id)
        if counterfactual is None:
            continue
        total += 1
        if counterfactual in document_ids(candidate_set.candidates):
            hits += 1
    return None if total == 0 else hits / total


class HarmComparison(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    harm_on: float
    harm_off: float
    delta: float
    p_value: float
    ci_low: float
    ci_high: float
    n_paired: int


def compare_harm(
    on_report: StageEvaluationReport,
    off_report: StageEvaluationReport,
    *,
    seed: int = 13,
    iterations: int = 10000,
) -> HarmComparison:
    on = {case.query_id: case.metrics[HARM_METRIC].value for case in on_report.per_case}
    off = {case.query_id: case.metrics[HARM_METRIC].value for case in off_report.per_case}
    diffs: list[float] = []
    on_values: list[float] = []
    off_values: list[float] = []
    for query_id, on_value in on.items():
        off_value = off.get(query_id)
        if on_value is None or off_value is None:
            continue
        on_values.append(on_value)
        off_values.append(off_value)
        diffs.append(on_value - off_value)
    if not diffs:
        raise ValueError("no paired scored queries")
    n = len(diffs)
    harm_on = sum(on_values) / n
    harm_off = sum(off_values) / n
    delta = harm_on - harm_off
    observed = abs(delta)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(iterations):
        permuted = sum(d if rng.random() < 0.5 else -d for d in diffs) / n
        if abs(permuted) >= observed - 1e-12:
            extreme += 1
    p_value = extreme / iterations
    boot: list[float] = []
    for _ in range(iterations):
        boot.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    boot.sort()
    ci_low = boot[int(0.025 * iterations)]
    ci_high = boot[int(0.975 * iterations)]
    return HarmComparison(
        harm_on=harm_on,
        harm_off=harm_off,
        delta=delta,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        n_paired=n,
    )
