from collections.abc import Iterable

from evidence_rag.contracts.models import PipelineRun
from evidence_rag.contracts.validation import validate_generation
from evidence_rag.evaluation.models import (
    CaseEvaluation,
    EvaluationReport,
    GoldCase,
    MetricSpec,
    MetricValue,
    RegressionGuard,
    RegressionReport,
)
from evidence_rag.evaluation.scoring import (
    CORE_DIRECTIONS,
    CORE_METRIC_VERSION,
    CORE_METRIC_VERSIONS,
    RECALL_AT_K,
    aggregate_metrics,
    answer_match,
    compose_generator_metrics,
    conditional_recall,
    document_ids,
    precision,
    recall,
    recall_at_k,
    reciprocal_rank,
    signature,
)


def _validate_extra_metrics(extra_metrics: tuple[MetricSpec, ...]) -> None:
    seen: set[str] = set()
    for spec in extra_metrics:
        parts = spec.key.split(".")
        if len(parts) < 3 or any(not part for part in parts):
            raise ValueError(
                "extra metric key must use stage.namespace.metric format"
            )
        if parts[0] not in {"retriever", "selector", "generator", "system"}:
            raise ValueError(f"metric key has unknown stage: {spec.key}")
        if parts[1] == "core" or spec.key in CORE_DIRECTIONS:
            raise ValueError("extra metric cannot replace a core metric")
        if spec.key in seen:
            raise ValueError(f"duplicate extra metric: {spec.key}")
        seen.add(spec.key)


def evaluate_case(
    run: PipelineRun,
    gold: GoldCase,
    *,
    extra_metrics: tuple[MetricSpec, ...] = (),
) -> CaseEvaluation:
    _validate_extra_metrics(extra_metrics)
    if run.query.query_id != gold.query_id:
        raise ValueError("pipeline run and gold case query IDs differ")

    relevant = (
        None
        if gold.relevant_document_ids is None
        else set(gold.relevant_document_ids)
    )
    retrieved = document_ids(run.candidates.candidates)
    selected = document_ids(run.selected.evidence)
    validate_generation(run.selected, run.generation)
    by_evidence_id = {
        item.evidence_id: item for item in run.candidates.candidates
    }
    cited = {
        by_evidence_id[evidence_id].document_id
        for evidence_id in run.generation.cited_evidence_ids
    }
    generator_answer, generator_citations, generator_citation_validity = (
        compose_generator_metrics(
            selected_evidence_ids={
                item.evidence_id for item in run.selected.evidence
            },
            cited_evidence_ids=set(run.generation.cited_evidence_ids),
            selected_document_ids=selected,
            cited_document_ids=cited,
            relevant_document_ids=relevant,
            answer=run.generation.answer,
            reference_answers=gold.reference_answers,
        )
    )

    stages: dict[str, dict[str, MetricValue]] = {
        "retriever": {
            "retriever.core.document_recall": recall(retrieved, relevant),
            "retriever.core.document_mrr": reciprocal_rank(
                run.candidates.candidates, relevant
            ),
            **{
                f"retriever.core.document_recall_at_{k}": recall_at_k(
                    run.candidates.candidates, relevant, k
                )
                for k in RECALL_AT_K
            },
        },
        "selector": {
            "selector.core.conditional_document_recall": conditional_recall(
                selected,
                retrieved,
                relevant,
            ),
            "selector.core.document_precision": precision(selected, relevant),
        },
        "generator": {
            "generator.core.conditional_answer_match": generator_answer,
            "generator.core.conditional_cited_document_precision": generator_citations,
            "generator.core.citation_validity": generator_citation_validity,
        },
        "system": {
            "system.core.final_document_recall": recall(cited, relevant),
            "system.core.cited_document_precision": precision(cited, relevant),
            "system.core.answer_match": answer_match(
                run.generation.answer,
                gold.reference_answers,
            ),
        },
    }

    for spec in extra_metrics:
        stage = spec.key.partition(".")[0]
        if stage not in stages:
            raise ValueError(f"metric key has unknown stage: {spec.key}")
        stages[stage][spec.key] = MetricValue(value=spec.compute(run, gold))

    return CaseEvaluation(
        query_id=gold.query_id,
        trace=run,
        retriever=stages["retriever"],
        selector=stages["selector"],
        generator=stages["generator"],
        system=stages["system"],
    )


def evaluate_dataset(
    pairs: Iterable[tuple[PipelineRun, GoldCase]],
    *,
    dataset_signature: str | None = None,
    extra_metrics: tuple[MetricSpec, ...] = (),
) -> EvaluationReport:
    dataset = tuple(pairs)
    if not dataset:
        raise ValueError("evaluation dataset must not be empty")
    per_case = tuple(
        evaluate_case(run, gold, extra_metrics=extra_metrics)
        for run, gold in dataset
    )

    directions = dict(CORE_DIRECTIONS)
    directions.update({spec.key: spec.direction for spec in extra_metrics})
    aggregate = aggregate_metrics(
        tuple(case.flattened() for case in per_case),
        directions,
    )

    evaluation_set_signature = signature(
        tuple(
            {
                "query": run.query.model_dump(mode="json"),
                "gold": gold.model_dump(mode="json"),
            }
            for run, gold in dataset
        )
    )
    return EvaluationReport(
        dataset_signature=(
            evaluation_set_signature
            if dataset_signature is None
            else dataset_signature
        ),
        evaluation_set_signature=evaluation_set_signature,
        metric_registry_signature=signature(
            {
                "core_version": CORE_METRIC_VERSION,
                "core_metrics": tuple(
                    (
                        key,
                        CORE_DIRECTIONS[key],
                        CORE_METRIC_VERSIONS[key],
                    )
                    for key in sorted(CORE_DIRECTIONS)
                ),
                "metrics": tuple(
                    sorted(
                        (
                            spec.key,
                            spec.direction,
                            spec.version,
                        )
                        for spec in extra_metrics
                    )
                ),
            }
        ),
        case_ids=tuple(case.query_id for case in per_case),
        per_case=per_case,
        aggregate=aggregate,
        directions=directions,
    )


def compare_reports(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
    *,
    guards: tuple[RegressionGuard, ...] = (),
) -> RegressionReport:
    if baseline.case_ids != candidate.case_ids:
        raise ValueError("reports must contain the same evaluation cases")
    if baseline.dataset_signature != candidate.dataset_signature:
        raise ValueError("reports must use the same evaluation dataset")
    if baseline.evaluation_set_signature != candidate.evaluation_set_signature:
        raise ValueError("reports must contain the same evaluation cases")
    if baseline.metric_registry_signature != candidate.metric_registry_signature:
        raise ValueError("reports must use the same metric registry version")
    if baseline.directions != candidate.directions:
        raise ValueError("reports must use the same metric registry")

    deltas: dict[str, float] = {}
    for key, baseline_metric in baseline.aggregate.items():
        candidate_metric = candidate.aggregate[key]
        if baseline_metric.mean is not None and candidate_metric.mean is not None:
            deltas[key] = candidate_metric.mean - baseline_metric.mean

    guard_by_key = {
        key: RegressionGuard(metric_key=key)
        for key in baseline.directions
        if key.startswith("system.core.")
        and baseline.aggregate[key].n_scored > 0
    }
    for guard in guards:
        if not guard.metric_key.startswith("system."):
            raise ValueError("regression guards must protect system metrics")
        if guard.metric_key not in baseline.aggregate:
            raise ValueError(f"unknown guarded metric: {guard.metric_key}")
        guard_by_key[guard.metric_key] = guard

    failures: list[str] = []
    for guard in guard_by_key.values():
        baseline_metric = baseline.aggregate[guard.metric_key]
        candidate_metric = candidate.aggregate[guard.metric_key]
        if candidate_metric.n_scored < baseline_metric.n_scored:
            failures.append(f"{guard.metric_key}: scoring coverage decreased")
            continue
        if baseline_metric.mean is None or candidate_metric.mean is None:
            failures.append(f"{guard.metric_key}: score is unavailable")
            continue

        delta = candidate_metric.mean - baseline_metric.mean
        direction = baseline.directions[guard.metric_key]
        regressed = (
            delta < -guard.allowed_regression
            if direction == "higher"
            else delta > guard.allowed_regression
        )
        if regressed:
            failures.append(
                f"{guard.metric_key}: regressed by {abs(delta):.6g}"
            )

    return RegressionReport(
        passed=not failures,
        deltas=deltas,
        failures=tuple(failures),
    )
