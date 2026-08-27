from collections.abc import Callable

import pytest

from evidence_rag.contracts.models import (
    CandidateSet,
    EvidenceCandidate,
    GenerationResult,
    Query,
    QueryChecklist,
    RetrieverProvenance,
    SelectedEvidenceSet,
    SelectionItem,
    SelectionResult,
)
from evidence_rag.evaluation.models import (
    GeneratorStageRun,
    GoldCase,
    RetrieverStageRun,
    SelectorStageRun,
)
from evidence_rag.evaluation.runners import (
    run_generator_stage,
    run_retriever_stage,
    run_selector_stage,
)


def _query(query_id: str) -> Query:
    return Query(query_id=query_id, text=f"Question {query_id}?")


def _candidate_set(query_id: str) -> CandidateSet:
    candidate = EvidenceCandidate(
        evidence_id=f"{query_id}-evidence",
        document_id=f"{query_id}-document",
        chunk_id=f"{query_id}-chunk",
        text=f"Original text for {query_id}",
        source_uri=f"fixture://{query_id}",
        retrieval_score=1.0,
        retrieval_rank=1,
    )
    return CandidateSet(query_id=query_id, candidates=(candidate,))


def _selected_set(query_id: str) -> SelectedEvidenceSet:
    return SelectedEvidenceSet(
        query_id=query_id,
        evidence=_candidate_set(query_id).candidates,
    )


def _gold(query_id: str) -> GoldCase:
    return GoldCase(
        query_id=query_id,
        relevant_document_ids=(f"{query_id}-document",),
        reference_answers=(f"Answer {query_id}",),
    )


BM25 = RetrieverProvenance(
    name="bm25", implementation_version="bm25-v1", parameters_sha256="1" * 64
)


class RetrieverSpy:
    def __init__(self, provenance: RetrieverProvenance | None = None) -> None:
        self.calls: list[tuple[Query, int]] = []
        self.provenance = provenance

    def retrieve(self, query: Query, top_k: int) -> CandidateSet:
        self.calls.append((query, top_k))
        candidates = _candidate_set(query.query_id)
        if self.provenance is None:
            return candidates
        return candidates.model_copy(update={"retriever": self.provenance})


class SelectorSpy:
    def __init__(
        self,
        result_factory: Callable[[Query, CandidateSet], SelectionResult] | None = None,
    ) -> None:
        self.calls: list[tuple[Query, CandidateSet, int]] = []
        self.result_factory = result_factory

    def select(
        self,
        query: Query,
        candidates: CandidateSet,
        max_selected: int,
    ) -> SelectionResult:
        self.calls.append((query, candidates, max_selected))
        if self.result_factory is not None:
            return self.result_factory(query, candidates)
        return SelectionResult(
            query_id=query.query_id,
            items=(
                SelectionItem(
                    evidence_id=candidates.candidates[0].evidence_id,
                    selection_score=0.75,
                    selection_rank=1,
                ),
            ),
        )


class GeneratorSpy:
    def __init__(
        self,
        result_factory: Callable[[Query, SelectedEvidenceSet], GenerationResult] | None = None,
    ) -> None:
        self.calls: list[tuple[Query, SelectedEvidenceSet]] = []
        self.result_factory = result_factory

    def generate(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> GenerationResult:
        self.calls.append((query, selected))
        if self.result_factory is not None:
            return self.result_factory(query, selected)
        return GenerationResult(
            query_id=query.query_id,
            answer=f"Answer {query.query_id}",
            cited_evidence_ids=(selected.evidence[0].evidence_id,),
        )


def test_retriever_runner_uses_only_retriever_protocol_and_query_order() -> None:
    queries = (_query("q-1"), _query("q-2"))
    retriever = RetrieverSpy()

    run = run_retriever_stage(
        retriever,
        queries,
        (_gold("q-2"), _gold("q-1")),
        dataset_signature="dataset-signature",
        top_k=4,
        retriever_provenance=BM25,
    )

    assert tuple(item.query_id for item in run.candidate_sets) == ("q-1", "q-2")
    assert tuple(query.query_id for query, _ in retriever.calls) == ("q-1", "q-2")
    assert all(top_k == 4 for _, top_k in retriever.calls)
    assert run.report.case_ids == ("q-1", "q-2")


def test_every_candidate_set_leaves_the_runner_naming_its_producer() -> None:
    """The stamp is applied by the loop that calls `retrieve`, so a pool cannot exist without
    saying what built it. `retriever_provenance` is keyword-ONLY and has no default: an
    unstamped pool is not something a caller can produce by forgetting an argument."""
    run = run_retriever_stage(
        RetrieverSpy(),
        (_query("q-1"), _query("q-2")),
        (_gold("q-1"), _gold("q-2")),
        dataset_signature="dataset-signature",
        top_k=1,
        retriever_provenance=BM25,
    )

    assert [item.retriever for item in run.candidate_sets] == [BM25, BM25]


def test_a_retriever_that_stamps_itself_differently_stops_the_run() -> None:
    """The two statements of the producer are the config the index was built from and whatever
    the retriever wrote on its own output. If they disagree, one of them is wrong and there is
    no way to tell which from the artefact afterwards -- which is the entire failure this field
    exists to prevent, so it raises instead of picking a winner."""
    other = RetrieverProvenance(
        name="strong-bm25", implementation_version="strong-bm25-v1", parameters_sha256="2" * 64
    )
    with pytest.raises(ValueError, match="disagrees"):
        run_retriever_stage(
            RetrieverSpy(provenance=other),
            (_query("q-1"),),
            (_gold("q-1"),),
            dataset_signature="dataset-signature",
            top_k=1,
            retriever_provenance=BM25,
        )


def test_a_retriever_that_stamps_itself_identically_is_accepted() -> None:
    run = run_retriever_stage(
        RetrieverSpy(provenance=BM25),
        (_query("q-1"),),
        (_gold("q-1"),),
        dataset_signature="dataset-signature",
        top_k=1,
        retriever_provenance=BM25,
    )
    assert run.candidate_sets[0].retriever == BM25


def test_selector_runner_reorders_artifacts_and_resolves_original_evidence() -> None:
    queries = (_query("q-1"), _query("q-2"))
    q1_candidates = _candidate_set("q-1")
    q2_candidates = _candidate_set("q-2")
    selector = SelectorSpy()

    run = run_selector_stage(
        selector,
        queries,
        (q2_candidates, q1_candidates),
        (_gold("q-2"), _gold("q-1")),
        dataset_signature="dataset-signature",
        max_selected=1,
    )

    assert tuple(item.query_id for item in run.selection_results) == ("q-1", "q-2")
    assert tuple(item.query_id for item in run.selected_sets) == ("q-1", "q-2")
    assert tuple(candidates.query_id for _, candidates, _ in selector.calls) == (
        "q-1",
        "q-2",
    )
    assert run.selected_sets[0].evidence[0] == q1_candidates.candidates[0]
    assert run.selected_sets[0].evidence[0].text == "Original text for q-1"
    assert q1_candidates == _candidate_set("q-1")


def test_generator_runner_reorders_artifacts_and_uses_only_generator_protocol() -> None:
    queries = (_query("q-1"), _query("q-2"))
    generator = GeneratorSpy()

    run = run_generator_stage(
        generator,
        queries,
        (_selected_set("q-2"), _selected_set("q-1")),
        (_gold("q-2"), _gold("q-1")),
        dataset_signature="dataset-signature",
    )

    assert tuple(item.query_id for item in run.generation_results) == ("q-1", "q-2")
    assert tuple(selected.query_id for _, selected in generator.calls) == ("q-1", "q-2")
    assert run.report.case_ids == ("q-1", "q-2")


@pytest.mark.parametrize(
    ("candidate_sets", "message"),
    (
        (
            (_candidate_set("q-1"), _candidate_set("q-1")),
            "duplicate candidate set query ID",
        ),
        ((_candidate_set("q-1"),), "missing candidate set for query ID: q-2"),
        (
            (_candidate_set("q-1"), _candidate_set("q-2"), _candidate_set("q-3")),
            "unknown candidate set query ID: q-3",
        ),
    ),
)
def test_selector_runner_rejects_duplicate_missing_and_extra_artifacts(
    candidate_sets: tuple[CandidateSet, ...],
    message: str,
) -> None:
    selector = SelectorSpy()

    with pytest.raises(ValueError, match=message):
        run_selector_stage(
            selector,
            (_query("q-1"), _query("q-2")),
            candidate_sets,
            (_gold("q-1"), _gold("q-2")),
            dataset_signature="dataset-signature",
            max_selected=1,
        )
    assert selector.calls == []


def test_runner_rejects_duplicate_queries_and_mismatched_gold() -> None:
    with pytest.raises(ValueError, match="duplicate query ID"):
        run_retriever_stage(
            RetrieverSpy(),
            (_query("q-1"), _query("q-1")),
            (_gold("q-1"),),
            dataset_signature="dataset-signature",
            top_k=1,
            retriever_provenance=BM25,
        )

    with pytest.raises(ValueError, match="missing gold case for query ID: q-2"):
        run_generator_stage(
            GeneratorSpy(),
            (_query("q-1"), _query("q-2")),
            (_selected_set("q-1"), _selected_set("q-2")),
            (_gold("q-1"),),
            dataset_signature="dataset-signature",
        )


def test_selector_runner_rejects_mismatched_module_output() -> None:
    selector = SelectorSpy(
        lambda _query, candidates: SelectionResult(
            query_id="wrong-query",
            items=(
                SelectionItem(
                    evidence_id=candidates.candidates[0].evidence_id,
                    selection_score=1.0,
                    selection_rank=1,
                ),
            ),
        )
    )

    with pytest.raises(ValueError, match="wrong query ID"):
        run_selector_stage(
            selector,
            (_query("q-1"),),
            (_candidate_set("q-1"),),
            (_gold("q-1"),),
            dataset_signature="dataset-signature",
            max_selected=1,
        )


def test_generator_runner_rejects_unselected_citation() -> None:
    generator = GeneratorSpy(
        lambda query, _selected: GenerationResult(
            query_id=query.query_id,
            answer="Unsupported answer",
            cited_evidence_ids=("unselected",),
        )
    )

    with pytest.raises(ValueError, match="unselected evidence"):
        run_generator_stage(
            generator,
            (_query("q-1"),),
            (_selected_set("q-1"),),
            (_gold("q-1"),),
            dataset_signature="dataset-signature",
        )


def test_retriever_stage_run_rejects_report_artifact_query_order_mismatch() -> None:
    run = run_retriever_stage(
        RetrieverSpy(),
        (_query("q-1"), _query("q-2")),
        (_gold("q-1"), _gold("q-2")),
        dataset_signature="dataset-signature",
        top_k=1,
        retriever_provenance=BM25,
    )

    with pytest.raises(ValueError, match="candidate set query order must match report"):
        RetrieverStageRun(
            candidate_sets=tuple(reversed(run.candidate_sets)),
            report=run.report,
        )


def test_selector_stage_run_rejects_report_artifact_query_order_mismatch() -> None:
    run = run_selector_stage(
        SelectorSpy(),
        (_query("q-1"), _query("q-2")),
        (_candidate_set("q-1"), _candidate_set("q-2")),
        (_gold("q-1"), _gold("q-2")),
        dataset_signature="dataset-signature",
        max_selected=1,
    )

    with pytest.raises(ValueError, match="selector artifact query order must match report"):
        SelectorStageRun(
            selection_results=tuple(reversed(run.selection_results)),
            selected_sets=tuple(reversed(run.selected_sets)),
            report=run.report,
        )


def test_generator_stage_run_rejects_report_artifact_query_order_mismatch() -> None:
    run = run_generator_stage(
        GeneratorSpy(),
        (_query("q-1"), _query("q-2")),
        (_selected_set("q-1"), _selected_set("q-2")),
        (_gold("q-1"), _gold("q-2")),
        dataset_signature="dataset-signature",
    )

    with pytest.raises(ValueError, match="generation result query order must match report"):
        GeneratorStageRun(
            generation_results=tuple(reversed(run.generation_results)),
            report=run.report,
        )


@pytest.mark.parametrize(
    ("model", "run_factory", "message"),
    (
        (
            RetrieverStageRun,
            lambda: run_retriever_stage(
                RetrieverSpy(),
                (_query("q-1"),),
                (_gold("q-1"),),
                dataset_signature="dataset-signature",
                top_k=1,
                retriever_provenance=BM25,
            ),
            "retriever stage run requires a retriever report",
        ),
        (
            SelectorStageRun,
            lambda: run_selector_stage(
                SelectorSpy(),
                (_query("q-1"),),
                (_candidate_set("q-1"),),
                (_gold("q-1"),),
                dataset_signature="dataset-signature",
                max_selected=1,
            ),
            "selector stage run requires a selector report",
        ),
        (
            GeneratorStageRun,
            lambda: run_generator_stage(
                GeneratorSpy(),
                (_query("q-1"),),
                (_selected_set("q-1"),),
                (_gold("q-1"),),
                dataset_signature="dataset-signature",
            ),
            "generator stage run requires a generator report",
        ),
    ),
)
def test_stage_run_rejects_report_from_another_stage(
    model: type[RetrieverStageRun] | type[SelectorStageRun] | type[GeneratorStageRun],
    run_factory: Callable[[], RetrieverStageRun | SelectorStageRun | GeneratorStageRun],
    message: str,
) -> None:
    run = run_factory()
    generator_report = run_generator_stage(
        GeneratorSpy(),
        (_query("q-1"),),
        (_selected_set("q-1"),),
        (_gold("q-1"),),
        dataset_signature="dataset-signature",
    ).report
    retriever_report = run_retriever_stage(
        RetrieverSpy(),
        (_query("q-1"),),
        (_gold("q-1"),),
        dataset_signature="dataset-signature",
        top_k=1,
        retriever_provenance=BM25,
    ).report
    wrong_report = retriever_report if isinstance(run, GeneratorStageRun) else generator_report

    with pytest.raises(ValueError, match=message):
        model.model_validate({**run.model_dump(), "report": wrong_report})


def test_selector_stage_run_rejects_selection_selected_mismatch() -> None:
    run = run_selector_stage(
        SelectorSpy(),
        (_query("q-1"),),
        (_candidate_set("q-1"),),
        (_gold("q-1"),),
        dataset_signature="dataset-signature",
        max_selected=1,
    )
    mismatched = SelectedEvidenceSet(
        query_id="q-1",
        evidence=(
            run.selected_sets[0].evidence[0].model_copy(
                update={"evidence_id": "different-evidence"}
            ),
        ),
    )

    with pytest.raises(ValueError, match="selected evidence IDs must match selection"):
        SelectorStageRun(
            selection_results=run.selection_results,
            selected_sets=(mismatched,),
            report=run.report,
        )
