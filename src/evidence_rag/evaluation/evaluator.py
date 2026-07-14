import hashlib
import json
import re
from collections.abc import Iterable

from evidence_rag.contracts.models import EvidenceCandidate, PipelineRun
from evidence_rag.evaluation.models import (
    AggregateMetric,
    CaseEvaluation,
    EvaluationReport,
    GoldCase,
    MetricDirection,
    MetricSpec,
    MetricValue,
    RegressionGuard,
    RegressionReport,
)

CORE_DIRECTIONS: dict[str, MetricDirection] = {
    "retriever.core.document_recall": "higher",
    "selector.core.conditional_document_recall": "higher",
    "selector.core.document_precision": "higher",
    "generator.core.conditional_answer_match": "higher",
    "generator.core.conditional_cited_document_precision": "higher",
    "system.core.final_document_recall": "higher",
    "system.core.cited_document_precision": "higher",
    "system.core.answer_match": "higher",
}
CORE_METRIC_VERSION = "1.0"


def _signature(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _unscored(reason: str) -> MetricValue:
    return MetricValue(value=None, reason=reason)


def _recall(predicted: set[str], relevant: set[str] | None) -> MetricValue:
    if relevant is None:
        return _unscored("relevant documents were not labelled")
    if not relevant:
        return _unscored("no relevant documents were labelled")
    return MetricValue(value=len(predicted & relevant) / len(relevant))


def _precision(predicted: set[str], relevant: set[str] | None) -> MetricValue:
    if relevant is None:
        return _unscored("relevant documents were not labelled")
    if not relevant:
        return _unscored("no relevant documents were labelled")
    if not predicted:
        return MetricValue(value=0.0)
    return MetricValue(value=len(predicted & relevant) / len(predicted))


def _conditional_recall(
    selected: set[str],
    retrieved: set[str],
    relevant: set[str] | None,
) -> MetricValue:
    if relevant is None:
        return _unscored("relevant documents were not labelled")
    if not relevant:
        return _unscored("no relevant documents were labelled")
    reachable = retrieved & relevant
    if not reachable:
        return _unscored("retriever supplied no labelled-relevant document")
    return MetricValue(value=len(selected & reachable) / len(reachable))


def _normalise(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def _answer_match(answer: str, references: tuple[str, ...] | None) -> MetricValue:
    if references is None:
        return _unscored("reference answers were not labelled")
    if not references:
        return _unscored("no reference answers were labelled")
    normalised_answer = _normalise(answer)
    matched = any(_normalise(reference) in normalised_answer for reference in references)
    return MetricValue(value=float(matched))


def _generator_is_eligible(
    selected: set[str],
    relevant: set[str] | None,
) -> MetricValue | None:
    if relevant is None:
        return _unscored("relevant documents were not labelled")
    if not relevant:
        return _unscored("no relevant documents were labelled")
    if not selected & relevant:
        return _unscored("selector supplied no labelled-relevant document")
    return None


def _document_ids(items: tuple[EvidenceCandidate, ...]) -> set[str]:
    return {item.document_id for item in items}


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
    retrieved = _document_ids(run.candidates.candidates)
    selected = _document_ids(run.selected.evidence)
    by_evidence_id = {
        item.evidence_id: item for item in run.candidates.candidates
    }
    cited = {
        by_evidence_id[evidence_id].document_id
        for evidence_id in run.generation.cited_evidence_ids
    }
    generator_ineligible = _generator_is_eligible(selected, relevant)
    generator_answer = (
        generator_ineligible
        if generator_ineligible is not None
        else _answer_match(run.generation.answer, gold.reference_answers)
    )
    generator_citations = (
        generator_ineligible
        if generator_ineligible is not None
        else _precision(cited, relevant)
    )

    stages: dict[str, dict[str, MetricValue]] = {
        "retriever": {
            "retriever.core.document_recall": _recall(retrieved, relevant),
        },
        "selector": {
            "selector.core.conditional_document_recall": _conditional_recall(
                selected,
                retrieved,
                relevant,
            ),
            "selector.core.document_precision": _precision(selected, relevant),
        },
        "generator": {
            "generator.core.conditional_answer_match": generator_answer,
            "generator.core.conditional_cited_document_precision": generator_citations,
        },
        "system": {
            "system.core.final_document_recall": _recall(cited, relevant),
            "system.core.cited_document_precision": _precision(cited, relevant),
            "system.core.answer_match": _answer_match(
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
    aggregate: dict[str, AggregateMetric] = {}
    for key in directions:
        values = [
            metric.value
            for case in per_case
            if (metric := case.flattened()[key]).value is not None
        ]
        aggregate[key] = AggregateMetric(
            mean=None if not values else sum(values) / len(values),
            n_scored=len(values),
            n_total=len(per_case),
        )

    return EvaluationReport(
        dataset_signature=_signature(
            tuple(
                {
                    "query": run.query.model_dump(mode="json"),
                    "gold": gold.model_dump(mode="json"),
                }
                for run, gold in dataset
            )
        ),
        metric_registry_signature=_signature(
            {
                "core_version": CORE_METRIC_VERSION,
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
