import pytest

from evidence_rag.composition import build_baseline
from evidence_rag.contracts.models import Document, Query


def test_real_pipeline_report_shows_each_stage_and_extra_metrics() -> None:
    try:
        from evidence_rag.evaluation.evaluator import evaluate_case
        from evidence_rag.evaluation.models import GoldCase, MetricSpec
    except ImportError as error:
        pytest.fail(f"pipeline evaluator is missing: {error}")

    pipeline = build_baseline(
        (
            Document(
                document_id="annual-report",
                text="Revenue increased by ten percent.",
                source_uri="fixture://annual-report",
            ),
        )
    )
    run = pipeline.run_with_trace(
        Query(query_id="q-1", text="revenue increase"),
        top_k=3,
        max_selected=2,
    )
    gold = GoldCase(
        query_id="q-1",
        relevant_document_ids=("annual-report",),
        reference_answers=("Revenue increased by ten percent.",),
    )
    extra = MetricSpec(
        key="selector.extra.mean_selection_score",
        direction="higher",
        compute=lambda trace, _: sum(
            item.selection_score for item in trace.selection.items
        )
        / len(trace.selection.items),
    )

    report = evaluate_case(run, gold, extra_metrics=(extra,))

    assert report.retriever["retriever.core.document_recall"].value == 1.0
    assert report.selector["selector.core.conditional_document_recall"].value == 1.0
    assert report.selector["selector.core.document_precision"].value == 1.0
    assert report.generator["generator.core.conditional_answer_match"].value == 1.0
    assert (
        report.generator["generator.core.conditional_cited_document_precision"].value
        == 1.0
    )
    assert report.system["system.core.final_document_recall"].value == 1.0
    assert report.system["system.core.cited_document_precision"].value == 1.0
    assert report.selector["selector.extra.mean_selection_score"].value is not None

    # One report exposes both the metrics and what every module actually produced.
    assert report.trace == run
    assert report.trace.candidates.candidates[0].retrieval_score > 0
    assert report.trace.selection.items[0].selection_score > 0
    assert report.trace.generation.cited_evidence_ids


def test_dataset_report_aggregates_scores_and_keeps_scoring_coverage() -> None:
    try:
        from evidence_rag.evaluation.evaluator import evaluate_dataset
        from evidence_rag.evaluation.models import GoldCase
    except ImportError as error:
        pytest.fail(f"dataset evaluator is missing: {error}")

    pipeline = build_baseline(
        (
            Document(
                document_id="annual-report",
                text="Revenue increased by ten percent.",
                source_uri="fixture://annual-report",
            ),
        )
    )
    found = pipeline.run_with_trace(
        Query(query_id="q-found", text="revenue increase")
    )
    missed = pipeline.run_with_trace(
        Query(query_id="q-missed", text="volcanic eruption")
    )

    report = evaluate_dataset(
        (
            (
                found,
                GoldCase(
                    query_id="q-found",
                    relevant_document_ids=("annual-report",),
                    reference_answers=("Revenue increased by ten percent.",),
                ),
            ),
            (
                missed,
                GoldCase(
                    query_id="q-missed",
                    relevant_document_ids=("annual-report",),
                    reference_answers=("Revenue increased by ten percent.",),
                ),
            ),
        )
    )

    retriever = report.aggregate["retriever.core.document_recall"]
    selector = report.aggregate["selector.core.conditional_document_recall"]
    generator = report.aggregate["generator.core.conditional_answer_match"]
    system = report.aggregate["system.core.final_document_recall"]
    system_answer = report.aggregate["system.core.answer_match"]
    assert (retriever.mean, retriever.n_scored, retriever.n_total) == (0.5, 2, 2)
    assert (selector.mean, selector.n_scored, selector.n_total) == (1.0, 1, 2)
    assert (generator.mean, generator.n_scored, generator.n_total) == (1.0, 1, 2)
    assert (system.mean, system.n_scored, system.n_total) == (0.5, 2, 2)
    assert (system_answer.mean, system_answer.n_scored, system_answer.n_total) == (
        0.5,
        2,
        2,
    )
    assert report.case_ids == ("q-found", "q-missed")


def test_module_improvement_cannot_hide_system_regression() -> None:
    try:
        from evidence_rag.contracts.models import (
            CandidateSet,
            GenerationResult,
            SelectedEvidenceSet,
            SelectionItem,
            SelectionResult,
        )
        from evidence_rag.evaluation.evaluator import compare_reports, evaluate_dataset
        from evidence_rag.evaluation.models import GoldCase, MetricSpec
        from evidence_rag.pipeline.service import EvidenceRAGPipeline
        from evidence_rag.retriever.bm25 import BM25Retriever
    except ImportError as error:
        pytest.fail(f"regression guard is missing: {error}")

    class HigherScoreSelector:
        def select(
            self,
            query: Query,
            candidates: CandidateSet,
            max_selected: int,
        ) -> SelectionResult:
            chosen = candidates.candidates[:max_selected]
            return SelectionResult(
                query_id=query.query_id,
                items=tuple(
                    SelectionItem(
                        evidence_id=item.evidence_id,
                        selection_score=item.retrieval_score + 1.0,
                        selection_rank=rank,
                    )
                    for rank, item in enumerate(chosen, start=1)
                ),
            )

    class WrongGenerator:
        def generate(
            self,
            query: Query,
            selected: SelectedEvidenceSet,
        ) -> GenerationResult:
            if not selected.evidence:
                return GenerationResult(
                    query_id=query.query_id,
                    answer="",
                    cited_evidence_ids=(),
                )
            return GenerationResult(
                query_id=query.query_id,
                answer="An unrelated answer.",
                cited_evidence_ids=(selected.evidence[0].evidence_id,),
            )

    documents = (
        Document(
            document_id="annual-report",
            text="Revenue increased by ten percent.",
            source_uri="fixture://annual-report",
        ),
    )
    query = Query(query_id="q-guard", text="revenue increase")
    gold = GoldCase(
        query_id=query.query_id,
        relevant_document_ids=("annual-report",),
        reference_answers=("Revenue increased by ten percent.",),
    )
    extra = MetricSpec(
        key="selector.extra.mean_selection_score",
        direction="higher",
        compute=lambda trace, _: sum(
            item.selection_score for item in trace.selection.items
        )
        / len(trace.selection.items),
    )
    baseline = evaluate_dataset(
        ((build_baseline(documents).run_with_trace(query), gold),),
        extra_metrics=(extra,),
    )
    candidate_pipeline = EvidenceRAGPipeline(
        retriever=BM25Retriever(documents),
        selector=HigherScoreSelector(),
        generator=WrongGenerator(),
    )
    candidate = evaluate_dataset(
        ((candidate_pipeline.run_with_trace(query), gold),),
        extra_metrics=(extra,),
    )

    comparison = compare_reports(
        baseline,
        candidate,
    )

    selector_key = "selector.extra.mean_selection_score"
    assert candidate.aggregate[selector_key].mean > baseline.aggregate[selector_key].mean
    assert comparison.passed is False
    assert any("system.core.answer_match" in failure for failure in comparison.failures)


def test_extra_metrics_cannot_replace_fixed_core_metrics() -> None:
    from evidence_rag.evaluation.evaluator import evaluate_case
    from evidence_rag.evaluation.models import GoldCase, MetricSpec

    pipeline = build_baseline(
        (
            Document(
                document_id="annual-report",
                text="Revenue increased by ten percent.",
                source_uri="fixture://annual-report",
            ),
        )
    )
    run = pipeline.run_with_trace(Query(query_id="q-core", text="revenue increase"))
    gold = GoldCase(
        query_id="q-core",
        relevant_document_ids=("annual-report",),
        reference_answers=("Revenue increased by ten percent.",),
    )
    replacement = MetricSpec(
        key="retriever.core.document_recall",
        direction="higher",
        compute=lambda _trace, _gold: 0.0,
    )

    with pytest.raises(ValueError, match="core metric"):
        evaluate_case(run, gold, extra_metrics=(replacement,))


def test_report_comparison_rejects_changed_gold_labels() -> None:
    from evidence_rag.evaluation.evaluator import compare_reports, evaluate_dataset
    from evidence_rag.evaluation.models import GoldCase, RegressionGuard

    pipeline = build_baseline(
        (
            Document(
                document_id="annual-report",
                text="Revenue increased by ten percent.",
                source_uri="fixture://annual-report",
            ),
        )
    )
    run = pipeline.run_with_trace(Query(query_id="q-same", text="revenue increase"))
    baseline = evaluate_dataset(
        (
            (
                run,
                GoldCase(
                    query_id="q-same",
                    relevant_document_ids=("annual-report",),
                    reference_answers=("Revenue increased by ten percent.",),
                ),
            ),
        )
    )
    changed_gold = evaluate_dataset(
        (
            (
                run,
                GoldCase(
                    query_id="q-same",
                    relevant_document_ids=("different-document",),
                    reference_answers=("A different reference answer.",),
                ),
            ),
        )
    )

    with pytest.raises(ValueError, match="dataset"):
        compare_reports(
            baseline,
            changed_gold,
            guards=(RegressionGuard(metric_key="system.core.answer_match"),),
        )


def test_default_system_guards_catch_noisy_final_citations() -> None:
    from evidence_rag.contracts.models import GenerationResult, SelectedEvidenceSet
    from evidence_rag.evaluation.evaluator import compare_reports, evaluate_dataset
    from evidence_rag.evaluation.models import EvaluationReport, GoldCase
    from evidence_rag.pipeline.service import EvidenceRAGPipeline
    from evidence_rag.retriever.bm25 import BM25Retriever
    from evidence_rag.selector.top_k import TopKSelector

    class FixedAnswerGenerator:
        def __init__(self, cite_everything: bool) -> None:
            self.cite_everything = cite_everything

        def generate(
            self,
            query: Query,
            selected: SelectedEvidenceSet,
        ) -> GenerationResult:
            relevant = next(
                item for item in selected.evidence if item.document_id == "annual-report"
            )
            citations = (
                tuple(item.evidence_id for item in selected.evidence)
                if self.cite_everything
                else (relevant.evidence_id,)
            )
            return GenerationResult(
                query_id=query.query_id,
                answer="Revenue increased by ten percent.",
                cited_evidence_ids=citations,
            )

    documents = (
        Document(
            document_id="annual-report",
            text="Revenue increased by ten percent.",
            source_uri="fixture://annual-report",
        ),
        Document(
            document_id="noise",
            text="Revenue forecasts may increase later.",
            source_uri="fixture://noise",
        ),
    )
    query = Query(query_id="q-citations", text="revenue increase")
    gold = GoldCase(
        query_id=query.query_id,
        relevant_document_ids=("annual-report",),
        reference_answers=("Revenue increased by ten percent.",),
    )

    def report(cite_everything: bool) -> EvaluationReport:
        pipeline = EvidenceRAGPipeline(
            BM25Retriever(documents),
            TopKSelector(),
            FixedAnswerGenerator(cite_everything),
        )
        trace = pipeline.run_with_trace(query, top_k=2, max_selected=2)
        return evaluate_dataset(((trace, gold),))

    baseline = report(False)
    candidate = report(True)
    comparison = compare_reports(baseline, candidate)

    assert baseline.aggregate["system.core.answer_match"].mean == 1.0
    assert candidate.aggregate["system.core.answer_match"].mean == 1.0
    assert candidate.aggregate["system.core.cited_document_precision"].mean == 0.5
    assert comparison.passed is False
    assert any(
        "system.core.cited_document_precision" in failure
        for failure in comparison.failures
    )


def test_default_guards_allow_intentionally_unlabelled_metrics() -> None:
    from evidence_rag.evaluation.evaluator import compare_reports, evaluate_dataset
    from evidence_rag.evaluation.models import GoldCase

    pipeline = build_baseline(
        (
            Document(
                document_id="annual-report",
                text="Revenue increased by ten percent.",
                source_uri="fixture://annual-report",
            ),
        )
    )
    trace = pipeline.run_with_trace(Query(query_id="q-partial", text="revenue increase"))
    report = evaluate_dataset(
        (
            (
                trace,
                GoldCase(
                    query_id="q-partial",
                    relevant_document_ids=("annual-report",),
                    reference_answers=None,
                ),
            ),
        )
    )

    comparison = compare_reports(report, report)

    assert report.aggregate["system.core.answer_match"].n_scored == 0
    assert comparison.passed is True
