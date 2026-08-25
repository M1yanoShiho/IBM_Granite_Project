from pathlib import Path

import pytest
from pydantic import ValidationError

from evidence_rag.composition import build_baseline
from evidence_rag.contracts.models import (
    CandidateSet,
    Document,
    EvidenceCandidate,
    GenerationResult,
    Query,
    SelectedEvidenceSet,
)
from evidence_rag.evaluation.evaluator import evaluate_dataset
from evidence_rag.evaluation.models import GoldCase, StageEvaluationReport
from evidence_rag.evaluation.stage_evaluators import (
    evaluate_generator_stage,
    evaluate_retriever_stage,
    evaluate_selector_stage,
)
from evidence_rag.infrastructure.artifacts import ArtifactStore, RunManifest
from evidence_rag.infrastructure.config import ModuleConfig


def _candidate(query_id: str, evidence_id: str, document_id: str) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=document_id,
        chunk_id=f"{document_id}:0",
        text=f"Text for {document_id}",
        source_uri=f"fixture://{document_id}",
        retrieval_score=1.0,
        retrieval_rank=1,
    )


def _stage_artifacts(
    query_id: str = "q-1",
) -> tuple[CandidateSet, SelectedEvidenceSet, GenerationResult, GoldCase]:
    candidate = _candidate(query_id, "e-1", "doc-1")
    candidates = CandidateSet(query_id=query_id, candidates=(candidate,))
    selected = SelectedEvidenceSet(query_id=query_id, evidence=(candidate,))
    generation = GenerationResult(
        query_id=query_id,
        answer="The reference answer.",
        cited_evidence_ids=(candidate.evidence_id,),
    )
    gold = GoldCase(
        query_id=query_id,
        relevant_document_ids=(candidate.document_id,),
        reference_answers=("reference answer",),
    )
    return candidates, selected, generation, gold


def _manifest() -> RunManifest:
    return RunManifest(
        dataset_id="dataset",
        dataset_version="1",
        split="test",
        dataset_signature="dataset-signature",
        corpus_signature="corpus-signature",
        chunker_name="WordChunker",
        chunker_version="word-v1",
        chunk_size=120,
        overlap=20,
        index_implementation="bm25",
        index_implementation_version="bm25-v1",
        index_signature="1" * 64,
        retriever=ModuleConfig(name="retriever"),
        selector=ModuleConfig(name="selector"),
        generator=ModuleConfig(name="generator"),
        top_k=3,
        max_selected=2,
        seed=0,
        git_commit="abc123",
        git_dirty=False,
        source_tree_signature="2" * 64,
    )


def test_stage_metrics_match_full_evaluator_values_reasons_and_coverage() -> None:
    from evidence_rag.evaluation.scoring import CORE_METRIC_VERSIONS

    pipeline = build_baseline(
        (
            Document(
                document_id="annual-report",
                text="Revenue increased by ten percent.",
                source_uri="fixture://annual-report",
            ),
        )
    )
    queries = (
        Query(query_id="q-found", text="revenue increase"),
        Query(query_id="q-missed", text="volcanic eruption"),
        Query(query_id="q-unlabelled", text="revenue increase"),
    )
    runs = tuple(pipeline.run_with_trace(query) for query in queries)
    gold_cases = (
        GoldCase(
            query_id="q-found",
            relevant_document_ids=("annual-report",),
            reference_answers=("Revenue increased by ten percent.",),
        ),
        GoldCase(
            query_id="q-missed",
            relevant_document_ids=("annual-report",),
            reference_answers=("Revenue increased by ten percent.",),
        ),
        GoldCase(
            query_id="q-unlabelled",
            relevant_document_ids=None,
            reference_answers=None,
        ),
    )
    full = evaluate_dataset(
        zip(runs, gold_cases, strict=True),
        dataset_signature="dataset-signature",
    )
    retriever = evaluate_retriever_stage(
        tuple(run.candidates for run in runs),
        tuple(reversed(gold_cases)),
        dataset_signature="dataset-signature",
    )
    selector = evaluate_selector_stage(
        tuple(run.candidates for run in runs),
        tuple(run.selected for run in reversed(runs)),
        tuple(reversed(gold_cases)),
        dataset_signature="dataset-signature",
    )
    generator = evaluate_generator_stage(
        tuple(run.selected for run in runs),
        tuple(run.generation for run in reversed(runs)),
        tuple(reversed(gold_cases)),
        dataset_signature="dataset-signature",
    )

    reports = (retriever, selector, generator)
    for report in reports:
        assert report.schema_version == "1.0"
        assert report.dataset_signature == full.dataset_signature
        assert report.case_ids == tuple(query.query_id for query in queries)
        for stage_case, full_case in zip(report.per_case, full.per_case, strict=True):
            assert stage_case.metrics == {
                key: full_case.flattened()[key] for key in report.directions
            }
        for key, aggregate in report.aggregate.items():
            assert aggregate == full.aggregate[key]

    missed_selector = selector.per_case[1].metrics["selector.core.conditional_document_recall"]
    unlabelled_retriever = retriever.per_case[2].metrics["retriever.core.document_recall"]
    assert missed_selector.reason == "retriever supplied no labelled-relevant document"
    assert unlabelled_retriever.reason == "relevant documents were not labelled"
    assert generator.aggregate["generator.core.conditional_answer_match"].n_scored == 1
    citation_validity = generator.aggregate["generator.core.citation_validity"]
    assert citation_validity.n_scored == citation_validity.n_total == 3
    assert CORE_METRIC_VERSIONS["generator.core.citation_validity"] == "1.0"


def test_empty_gold_labels_remain_unscored() -> None:
    candidates, selected, generation, _ = _stage_artifacts()
    gold = GoldCase(query_id="q-1", relevant_document_ids=(), reference_answers=())

    retriever = evaluate_retriever_stage(
        (candidates,), (gold,), dataset_signature="dataset-signature"
    )
    selector = evaluate_selector_stage(
        (candidates,), (selected,), (gold,), dataset_signature="dataset-signature"
    )
    generator = evaluate_generator_stage(
        (selected,), (generation,), (gold,), dataset_signature="dataset-signature"
    )

    for report in (retriever, selector):
        assert all(metric.value is None for metric in report.per_case[0].metrics.values())
        assert all(metric.n_scored == 0 for metric in report.aggregate.values())
    generator_gold_metrics = {
        key: metric
        for key, metric in generator.per_case[0].metrics.items()
        if key != "generator.core.citation_validity"
    }
    assert all(metric.value is None for metric in generator_gold_metrics.values())
    validity = generator.aggregate["generator.core.citation_validity"]
    assert (validity.mean, validity.n_scored, validity.n_total) == (1.0, 1, 1)
    assert (
        retriever.per_case[0].metrics["retriever.core.document_recall"].reason
        == "no relevant documents were labelled"
    )


def test_generator_evaluator_rejects_citations_outside_selected_evidence() -> None:
    _, selected, _, gold = _stage_artifacts()
    invalid = GenerationResult(
        query_id="q-1",
        answer="Unsupported answer",
        cited_evidence_ids=("not-selected",),
    )

    with pytest.raises(ValueError, match="unselected evidence"):
        evaluate_generator_stage(
            (selected,), (invalid,), (gold,), dataset_signature="dataset-signature"
        )


@pytest.mark.parametrize("replacement", ("fabricated", "substituted"))
def test_selector_evaluator_rejects_evidence_not_identical_to_candidate(
    replacement: str,
) -> None:
    candidates, selected, _, gold = _stage_artifacts()
    original = selected.evidence[0]
    invalid_evidence = (
        original.model_copy(update={"evidence_id": "fabricated-evidence"})
        if replacement == "fabricated"
        else original.model_copy(update={"text": "Substituted text"})
    )
    invalid = SelectedEvidenceSet(query_id="q-1", evidence=(invalid_evidence,))

    with pytest.raises(ValueError, match="selected evidence does not match candidates"):
        evaluate_selector_stage(
            (candidates,),
            (invalid,),
            (gold,),
            dataset_signature="dataset-signature",
        )


@pytest.mark.parametrize(
    ("candidate_sets", "gold_cases", "message"),
    (
        (
            (
                CandidateSet(query_id="q-1", candidates=()),
                CandidateSet(query_id="q-1", candidates=()),
            ),
            (GoldCase(query_id="q-1"),),
            "duplicate candidate set query ID",
        ),
        (
            (CandidateSet(query_id="q-1", candidates=()),),
            (GoldCase(query_id="q-2"),),
            "query IDs differ",
        ),
    ),
)
def test_retriever_evaluator_rejects_invalid_query_id_sets(
    candidate_sets: tuple[CandidateSet, ...],
    gold_cases: tuple[GoldCase, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_retriever_stage(
            candidate_sets,
            gold_cases,
            dataset_signature="dataset-signature",
        )


def test_stage_report_validates_case_order_uniqueness_and_registry() -> None:
    candidates, _, _, gold = _stage_artifacts()
    report = evaluate_retriever_stage((candidates,), (gold,), dataset_signature="dataset-signature")
    payload = report.model_dump()

    with pytest.raises(ValidationError, match="case IDs must match"):
        StageEvaluationReport.model_validate({**payload, "case_ids": ("other",)})
    with pytest.raises(ValidationError, match="case IDs must be unique"):
        StageEvaluationReport.model_validate(
            {
                **payload,
                "case_ids": ("q-1", "q-1"),
                "per_case": (report.per_case[0], report.per_case[0]),
            }
        )
    with pytest.raises(ValidationError, match="same keys"):
        StageEvaluationReport.model_validate({**payload, "directions": {}})
    with pytest.raises(ValidationError, match="Extra inputs"):
        StageEvaluationReport.model_validate({**payload, "unexpected": True})


def test_stage_report_rejects_metric_prefix_from_another_stage() -> None:
    candidates, _, _, gold = _stage_artifacts()
    report = evaluate_retriever_stage((candidates,), (gold,), dataset_signature="dataset-signature")
    payload = report.model_dump()
    wrong_key = "selector.core.document_precision"
    metric = next(iter(report.per_case[0].metrics.values()))
    aggregate = next(iter(report.aggregate.values()))

    with pytest.raises(ValidationError, match="metric keys must belong to retriever stage"):
        StageEvaluationReport.model_validate(
            {
                **payload,
                "per_case": ({"query_id": "q-1", "metrics": {wrong_key: metric}},),
                "aggregate": {wrong_key: aggregate},
                "directions": {wrong_key: "higher"},
            }
        )


@pytest.mark.parametrize(
    ("selected_document_ids", "relevant_document_ids", "expected_reason"),
    (
        ({"doc-1"}, {"doc-1"}, None),
        (set(), {"doc-1"}, "selector supplied no labelled-relevant document"),
        ({"doc-1"}, None, "relevant documents were not labelled"),
    ),
)
def test_shared_generator_metric_composition_scores_conditional_and_validity_metrics(
    selected_document_ids: set[str],
    relevant_document_ids: set[str] | None,
    expected_reason: str | None,
) -> None:
    from evidence_rag.evaluation.scoring import compose_generator_metrics

    answer_metric, citation_metric, citation_validity = compose_generator_metrics(
        selected_evidence_ids={"e-1"},
        cited_evidence_ids={"e-1"},
        selected_document_ids=selected_document_ids,
        cited_document_ids={"doc-1"},
        relevant_document_ids=relevant_document_ids,
        answer="The reference answer.",
        reference_answers=("reference answer",),
    )

    assert answer_metric.reason == expected_reason
    assert citation_metric.reason == expected_reason
    assert citation_validity.value == 1.0
    assert citation_validity.reason is None
    if expected_reason is None:
        assert answer_metric.value == 1.0
        assert citation_metric.value == 1.0


def test_stage_reports_round_trip_through_artifact_store(tmp_path: Path) -> None:
    candidates, selected, generation, gold = _stage_artifacts()
    reports = (
        evaluate_retriever_stage((candidates,), (gold,), dataset_signature="dataset-signature"),
        evaluate_selector_stage(
            (candidates,), (selected,), (gold,), dataset_signature="dataset-signature"
        ),
        evaluate_generator_stage(
            (selected,), (generation,), (gold,), dataset_signature="dataset-signature"
        ),
    )
    store = ArtifactStore(tmp_path / "run")
    store.write_manifest(_manifest())

    for report in reports:
        filename = f"{report.stage}-report.json"
        store.write_json(filename, report)
        restored = store.read_json(
            filename,
            StageEvaluationReport,
            expected_dataset_signature="dataset-signature",
            expected_corpus_signature="corpus-signature",
        )
        assert restored == report
