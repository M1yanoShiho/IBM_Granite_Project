from evidence_rag.contracts.models import Document, PipelineRun, Query
from evidence_rag.contracts.protocols import Generator, Retriever, Selector
from evidence_rag.evaluation.evaluator import (
    compare_reports,
    evaluate_case,
    evaluate_dataset,
)
from evidence_rag.evaluation.models import (
    CaseEvaluation,
    EvaluationReport,
    GoldCase,
    RegressionReport,
)
from evidence_rag.generator.extractive import ExtractiveGenerator
from evidence_rag.pipeline.service import EvidenceRAGPipeline
from evidence_rag.retriever.bm25 import BM25Retriever
from evidence_rag.selector.top_k import TopKSelector

documents = (
    Document(document_id="doc", text="text", source_uri="fixture://doc"),
)
retriever: Retriever = BM25Retriever(documents)
selector: Selector = TopKSelector()
generator: Generator = ExtractiveGenerator()
pipeline = EvidenceRAGPipeline(retriever, selector, generator)
trace: PipelineRun = pipeline.run_with_trace(Query(query_id="q", text="text"))
gold = GoldCase(
    query_id="q",
    relevant_document_ids=("doc",),
    reference_answers=("text",),
)
case_report: CaseEvaluation = evaluate_case(trace, gold)
dataset_report: EvaluationReport = evaluate_dataset(
    ((trace, gold),),
    dataset_signature="dataset-signature",
)
legacy_dataset_report: EvaluationReport = evaluate_dataset(((trace, gold),))
evaluation_set_signature: str = dataset_report.evaluation_set_signature
comparison: RegressionReport = compare_reports(dataset_report, dataset_report)
