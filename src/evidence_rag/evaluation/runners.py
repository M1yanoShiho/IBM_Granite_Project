from collections.abc import Callable, Iterable
from typing import TypeVar

from evidence_rag.contracts.models import (
    CandidateSet,
    Query,
    SelectedEvidenceSet,
)
from evidence_rag.contracts.protocols import Generator, Retriever, Selector
from evidence_rag.contracts.validation import resolve_selection, validate_generation
from evidence_rag.evaluation.models import (
    GeneratorStageRun,
    GoldCase,
    RetrieverStageRun,
    SelectorStageRun,
)
from evidence_rag.evaluation.stage_evaluators import (
    evaluate_generator_stage,
    evaluate_retriever_stage,
    evaluate_selector_stage,
)

RecordT = TypeVar("RecordT")


def _validate_queries(queries: tuple[Query, ...]) -> None:
    if not queries:
        raise ValueError("queries must not be empty")
    seen: set[str] = set()
    for query in queries:
        if query.query_id in seen:
            raise ValueError(f"duplicate query ID: {query.query_id}")
        seen.add(query.query_id)


def _ordered_records(
    queries: tuple[Query, ...],
    records: tuple[RecordT, ...],
    query_id: Callable[[RecordT], str],
    label: str,
) -> tuple[RecordT, ...]:
    indexed: dict[str, RecordT] = {}
    for record in records:
        record_query_id = query_id(record)
        if record_query_id in indexed:
            raise ValueError(f"duplicate {label} query ID: {record_query_id}")
        indexed[record_query_id] = record

    query_ids = {query.query_id for query in queries}
    for query in queries:
        if query.query_id not in indexed:
            raise ValueError(f"missing {label} for query ID: {query.query_id}")
    for record in records:
        record_query_id = query_id(record)
        if record_query_id not in query_ids:
            raise ValueError(f"unknown {label} query ID: {record_query_id}")
    return tuple(indexed[query.query_id] for query in queries)


def run_retriever_stage(
    retriever: Retriever,
    queries: Iterable[Query],
    gold_cases: Iterable[GoldCase],
    *,
    dataset_signature: str,
    top_k: int,
) -> RetrieverStageRun:
    ordered_queries = tuple(queries)
    _validate_queries(ordered_queries)
    if top_k < 1:
        raise ValueError("top_k must be positive")
    ordered_gold = _ordered_records(
        ordered_queries,
        tuple(gold_cases),
        lambda item: item.query_id,
        "gold case",
    )
    candidate_sets: list[CandidateSet] = []
    for query in ordered_queries:
        candidates = retriever.retrieve(query, top_k)
        if candidates.query_id != query.query_id:
            raise ValueError("retriever returned the wrong query ID")
        if len(candidates.candidates) > top_k:
            raise ValueError("candidate count exceeds top_k")
        candidate_sets.append(candidates)
    candidates_tuple = tuple(candidate_sets)
    return RetrieverStageRun(
        candidate_sets=candidates_tuple,
        report=evaluate_retriever_stage(
            candidates_tuple,
            ordered_gold,
            dataset_signature=dataset_signature,
        ),
    )


def run_selector_stage(
    selector: Selector,
    queries: Iterable[Query],
    candidate_sets: Iterable[CandidateSet],
    gold_cases: Iterable[GoldCase],
    *,
    dataset_signature: str,
    max_selected: int,
) -> SelectorStageRun:
    ordered_queries = tuple(queries)
    _validate_queries(ordered_queries)
    if max_selected < 1:
        raise ValueError("max_selected must be positive")
    ordered_candidates = _ordered_records(
        ordered_queries,
        tuple(candidate_sets),
        lambda item: item.query_id,
        "candidate set",
    )
    ordered_gold = _ordered_records(
        ordered_queries,
        tuple(gold_cases),
        lambda item: item.query_id,
        "gold case",
    )
    selections = []
    selected_sets: list[SelectedEvidenceSet] = []
    for query, candidates in zip(ordered_queries, ordered_candidates, strict=True):
        selection = selector.select(query, candidates, max_selected)
        if selection.query_id != query.query_id:
            raise ValueError("selector returned the wrong query ID")
        if len(selection.items) > max_selected:
            raise ValueError("selection count exceeds max_selected")
        selections.append(selection)
        selected_sets.append(resolve_selection(candidates, selection))
    selections_tuple = tuple(selections)
    selected_tuple = tuple(selected_sets)
    return SelectorStageRun(
        selection_results=selections_tuple,
        selected_sets=selected_tuple,
        report=evaluate_selector_stage(
            ordered_candidates,
            selected_tuple,
            ordered_gold,
            dataset_signature=dataset_signature,
        ),
    )


def run_generator_stage(
    generator: Generator,
    queries: Iterable[Query],
    selected_sets: Iterable[SelectedEvidenceSet],
    gold_cases: Iterable[GoldCase],
    *,
    dataset_signature: str,
) -> GeneratorStageRun:
    ordered_queries = tuple(queries)
    _validate_queries(ordered_queries)
    ordered_selected = _ordered_records(
        ordered_queries,
        tuple(selected_sets),
        lambda item: item.query_id,
        "selected evidence set",
    )
    ordered_gold = _ordered_records(
        ordered_queries,
        tuple(gold_cases),
        lambda item: item.query_id,
        "gold case",
    )
    generations = []
    for query, selected in zip(ordered_queries, ordered_selected, strict=True):
        result = generator.generate(query, selected)
        validate_generation(selected, result)
        generations.append(result)
    generations_tuple = tuple(generations)
    return GeneratorStageRun(
        generation_results=generations_tuple,
        report=evaluate_generator_stage(
            ordered_selected,
            generations_tuple,
            ordered_gold,
            dataset_signature=dataset_signature,
        ),
    )
