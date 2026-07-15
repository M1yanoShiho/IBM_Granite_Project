from collections.abc import Callable, Iterable
from typing import TypeVar

from evidence_rag.contracts.models import (
    CandidateSet,
    GenerationResult,
    SelectedEvidenceSet,
)
from evidence_rag.contracts.validation import validate_generation
from evidence_rag.evaluation.models import (
    GoldCase,
    StageCaseEvaluation,
    StageEvaluationReport,
    StageName,
)
from evidence_rag.evaluation.scoring import (
    CORE_DIRECTIONS,
    CORE_METRIC_VERSION,
    CORE_METRIC_VERSIONS,
    aggregate_metrics,
    compose_generator_metrics,
    conditional_recall,
    document_ids,
    precision,
    recall,
    signature,
)

RecordT = TypeVar("RecordT")

RETRIEVER_METRICS = ("retriever.core.document_recall",)
SELECTOR_METRICS = (
    "selector.core.conditional_document_recall",
    "selector.core.document_precision",
)
GENERATOR_METRICS = (
    "generator.core.conditional_answer_match",
    "generator.core.conditional_cited_document_precision",
    "generator.core.citation_validity",
)


def _unique_index(
    records: tuple[RecordT, ...],
    query_id: Callable[[RecordT], str],
    label: str,
) -> dict[str, RecordT]:
    indexed: dict[str, RecordT] = {}
    for record in records:
        record_query_id = query_id(record)
        if record_query_id in indexed:
            raise ValueError(f"duplicate {label} query ID: {record_query_id}")
        indexed[record_query_id] = record
    return indexed


def _aligned(
    primary_ids: tuple[str, ...],
    records: tuple[RecordT, ...],
    query_id: Callable[[RecordT], str],
    primary_label: str,
    record_label: str,
) -> tuple[RecordT, ...]:
    indexed = _unique_index(records, query_id, record_label)
    if set(primary_ids) != set(indexed):
        raise ValueError(f"{primary_label} and {record_label} query IDs differ")
    return tuple(indexed[value] for value in primary_ids)


def _report(
    stage: StageName,
    dataset_signature: str,
    metric_keys: tuple[str, ...],
    per_case: tuple[StageCaseEvaluation, ...],
) -> StageEvaluationReport:
    directions = {key: CORE_DIRECTIONS[key] for key in metric_keys}
    return StageEvaluationReport(
        stage=stage,
        dataset_signature=dataset_signature,
        metric_registry_signature=signature(
            {
                "core_version": CORE_METRIC_VERSION,
                "metrics": tuple(
                    (key, directions[key], CORE_METRIC_VERSIONS[key])
                    for key in metric_keys
                ),
            }
        ),
        case_ids=tuple(case.query_id for case in per_case),
        per_case=per_case,
        aggregate=aggregate_metrics(
            tuple(case.metrics for case in per_case),
            directions,
        ),
        directions=directions,
    )


def _relevant_documents(gold: GoldCase) -> set[str] | None:
    return (
        None
        if gold.relevant_document_ids is None
        else set(gold.relevant_document_ids)
    )


def evaluate_retriever_stage(
    candidate_sets: Iterable[CandidateSet],
    gold_cases: Iterable[GoldCase],
    *,
    dataset_signature: str,
) -> StageEvaluationReport:
    candidates = tuple(candidate_sets)
    if not candidates:
        raise ValueError("evaluation dataset must not be empty")
    _unique_index(candidates, lambda item: item.query_id, "candidate set")
    case_ids = tuple(item.query_id for item in candidates)
    gold = _aligned(case_ids, tuple(gold_cases), lambda item: item.query_id, "candidate set", "gold case")
    per_case = tuple(
        StageCaseEvaluation(
            query_id=candidate_set.query_id,
            metrics={
                RETRIEVER_METRICS[0]: recall(
                    document_ids(candidate_set.candidates),
                    _relevant_documents(gold_case),
                )
            },
        )
        for candidate_set, gold_case in zip(candidates, gold, strict=True)
    )
    return _report("retriever", dataset_signature, RETRIEVER_METRICS, per_case)


def evaluate_selector_stage(
    candidate_sets: Iterable[CandidateSet],
    selected_sets: Iterable[SelectedEvidenceSet],
    gold_cases: Iterable[GoldCase],
    *,
    dataset_signature: str,
) -> StageEvaluationReport:
    candidates = tuple(candidate_sets)
    if not candidates:
        raise ValueError("evaluation dataset must not be empty")
    _unique_index(candidates, lambda item: item.query_id, "candidate set")
    case_ids = tuple(item.query_id for item in candidates)
    selected = _aligned(
        case_ids,
        tuple(selected_sets),
        lambda item: item.query_id,
        "candidate set",
        "selected evidence set",
    )
    gold = _aligned(case_ids, tuple(gold_cases), lambda item: item.query_id, "candidate set", "gold case")
    per_case: list[StageCaseEvaluation] = []
    for candidate_set, selected_set, gold_case in zip(candidates, selected, gold, strict=True):
        candidates_by_id = {
            item.evidence_id: item for item in candidate_set.candidates
        }
        if any(
            candidates_by_id.get(item.evidence_id) != item
            for item in selected_set.evidence
        ):
            raise ValueError("selected evidence does not match candidates")
        relevant = _relevant_documents(gold_case)
        retrieved_ids = document_ids(candidate_set.candidates)
        selected_ids = document_ids(selected_set.evidence)
        per_case.append(
            StageCaseEvaluation(
                query_id=candidate_set.query_id,
                metrics={
                    SELECTOR_METRICS[0]: conditional_recall(
                        selected_ids,
                        retrieved_ids,
                        relevant,
                    ),
                    SELECTOR_METRICS[1]: precision(selected_ids, relevant),
                },
            )
        )
    return _report("selector", dataset_signature, SELECTOR_METRICS, tuple(per_case))


def evaluate_generator_stage(
    selected_sets: Iterable[SelectedEvidenceSet],
    generation_results: Iterable[GenerationResult],
    gold_cases: Iterable[GoldCase],
    *,
    dataset_signature: str,
) -> StageEvaluationReport:
    selected = tuple(selected_sets)
    if not selected:
        raise ValueError("evaluation dataset must not be empty")
    _unique_index(selected, lambda item: item.query_id, "selected evidence set")
    case_ids = tuple(item.query_id for item in selected)
    generations = _aligned(
        case_ids,
        tuple(generation_results),
        lambda item: item.query_id,
        "selected evidence set",
        "generation result",
    )
    gold = _aligned(
        case_ids,
        tuple(gold_cases),
        lambda item: item.query_id,
        "selected evidence set",
        "gold case",
    )
    per_case: list[StageCaseEvaluation] = []
    for selected_set, generation, gold_case in zip(selected, generations, gold, strict=True):
        validate_generation(selected_set, generation)
        relevant = _relevant_documents(gold_case)
        selected_ids = document_ids(selected_set.evidence)
        by_evidence_id = {
            item.evidence_id: item.document_id for item in selected_set.evidence
        }
        cited_ids = {
            by_evidence_id[evidence_id]
            for evidence_id in generation.cited_evidence_ids
        }
        generator_answer, generator_citations, generator_citation_validity = (
            compose_generator_metrics(
                selected_evidence_ids={
                    item.evidence_id for item in selected_set.evidence
                },
                cited_evidence_ids=set(generation.cited_evidence_ids),
                selected_document_ids=selected_ids,
                cited_document_ids=cited_ids,
                relevant_document_ids=relevant,
                answer=generation.answer,
                reference_answers=gold_case.reference_answers,
            )
        )
        per_case.append(
            StageCaseEvaluation(
                query_id=selected_set.query_id,
                metrics={
                    GENERATOR_METRICS[0]: generator_answer,
                    GENERATOR_METRICS[1]: generator_citations,
                    GENERATOR_METRICS[2]: generator_citation_validity,
                },
            )
        )
    return _report("generator", dataset_signature, GENERATOR_METRICS, tuple(per_case))
