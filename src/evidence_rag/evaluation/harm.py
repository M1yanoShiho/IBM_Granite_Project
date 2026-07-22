"""Harmful-in-context metric and pool-hit diagnostic (spec §1-§5).

Offline over dumped selected sets + the injector's provenance sidecar; the shared
ExperimentWorkflow and frozen contracts are untouched.
"""

from collections.abc import Iterable, Mapping

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
