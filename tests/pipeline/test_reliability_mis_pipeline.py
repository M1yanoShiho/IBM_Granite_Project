from __future__ import annotations

import json
from collections.abc import Sequence

from evidence_rag.composition import build_selector
from evidence_rag.contracts.models import Document, Query
from evidence_rag.generator.extractive import ExtractiveGenerator
from evidence_rag.infrastructure.config import ModuleConfig
from evidence_rag.pipeline.service import EvidenceRAGPipeline
from evidence_rag.retriever.bm25 import BM25Retriever
from evidence_rag.selector.reliability_mis import SelectorBackendError
from evidence_rag.selector.top_k import TopKSelector


class PassageAnswerLLM:
    def generate(self, prompt: str) -> str:
        if "Bristol" in prompt:
            return "Bristol"
        if "London" in prompt:
            return "London"
        return "NONE"


class AlwaysContradictionScorer:
    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> tuple[float, ...]:
        return tuple(0.99 for _pair in pairs)


class NeverContradictionScorer:
    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> tuple[float, ...]:
        return tuple(0.0 for _pair in pairs)


class FailingContradictionScorer:
    def score_pairs(self, _pairs: Sequence[tuple[str, str]]) -> tuple[float, ...]:
        raise SelectorBackendError("NLI unavailable")


def documents() -> tuple[Document, ...]:
    return (
        Document(
            document_id="doc-bristol",
            text="The launch city is Bristol.",
            source_uri="fixture://bristol",
        ),
        Document(
            document_id="doc-london",
            text="The launch city is London.",
            source_uri="fixture://london",
        ),
    )


def install_sidecar(tmp_path, monkeypatch, source_documents: Sequence[Document]) -> None:
    sidecar = tmp_path / "source_parent.jsonl"
    sidecar.write_text(
        "\n".join(
            json.dumps(
                {"document_id": document.document_id, "source_parent_id": document.document_id}
            )
            for document in source_documents
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SOURCE_PARENT_INDEX", str(sidecar))


def test_retriever_selector_generator_interfaces_run_end_to_end(tmp_path, monkeypatch) -> None:
    source_documents = documents()
    install_sidecar(tmp_path, monkeypatch, source_documents)

    pipeline = EvidenceRAGPipeline(
        retriever=BM25Retriever(source_documents),
        selector=build_selector(
            ModuleConfig(name="reliability-mis"),
            llm=PassageAnswerLLM(),
            contradiction_scorer=AlwaysContradictionScorer(),
        ),
        generator=ExtractiveGenerator(),
    )
    trace = pipeline.run_with_trace(
        Query(query_id="q-interface", text="What is the launch city?"),
        top_k=2,
        max_selected=2,
    )

    assert len(trace.candidates.candidates) == 2
    assert len(trace.selection.items) == 1
    assert trace.selection.items[0].evidence_id == trace.selected.evidence[0].evidence_id
    assert trace.generation.cited_evidence_ids == (trace.selected.evidence[0].evidence_id,)
    assert trace.selected.evidence[0].text in trace.generation.answer
    assert "The answer to the question" not in trace.generation.answer


def test_no_conflict_pipeline_is_identical_to_top_k_pipeline(tmp_path, monkeypatch) -> None:
    source_documents = documents()
    install_sidecar(tmp_path, monkeypatch, source_documents)
    query = Query(query_id="q-no-conflict", text="What is the launch city?")
    reliability_pipeline = EvidenceRAGPipeline(
        retriever=BM25Retriever(source_documents),
        selector=build_selector(
            ModuleConfig(name="reliability-mis"),
            llm=PassageAnswerLLM(),
            contradiction_scorer=NeverContradictionScorer(),
        ),
        generator=ExtractiveGenerator(),
    )
    top_k_pipeline = EvidenceRAGPipeline(
        retriever=BM25Retriever(source_documents),
        selector=TopKSelector(),
        generator=ExtractiveGenerator(),
    )

    reliability_trace = reliability_pipeline.run_with_trace(query, top_k=2, max_selected=2)
    top_k_trace = top_k_pipeline.run_with_trace(query, top_k=2, max_selected=2)
    assert reliability_trace.selection == top_k_trace.selection
    assert reliability_trace.selected == top_k_trace.selected
    assert reliability_trace.generation == top_k_trace.generation


def test_nli_failure_pipeline_completes_with_exact_top_k_evidence(tmp_path, monkeypatch) -> None:
    source_documents = documents()
    install_sidecar(tmp_path, monkeypatch, source_documents)
    query = Query(query_id="q-fallback", text="What is the launch city?")
    pipeline = EvidenceRAGPipeline(
        retriever=BM25Retriever(source_documents),
        selector=build_selector(
            ModuleConfig(name="reliability-mis"),
            llm=PassageAnswerLLM(),
            contradiction_scorer=FailingContradictionScorer(),
        ),
        generator=ExtractiveGenerator(),
    )
    top_k_candidates = BM25Retriever(source_documents).retrieve(query, 2)
    expected = TopKSelector().select(query, top_k_candidates, 2)

    trace = pipeline.run_with_trace(query, top_k=2, max_selected=2)
    assert trace.selection == expected
    assert trace.generation.cited_evidence_ids == tuple(item.evidence_id for item in expected.items)
