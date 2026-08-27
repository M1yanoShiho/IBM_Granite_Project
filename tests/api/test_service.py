from __future__ import annotations

from pathlib import Path

import evidence_rag.api.service as api_service
from evidence_rag.api.schemas import QueryRequest
from evidence_rag.api.service import PipelineApiService, load_runtime_pipeline
from evidence_rag.cli.smoke import (
    DEFAULT_CPU_SMOKE_CONFIG,
    build_cpu_smoke_pipeline,
)


def test_api_service_loads_pipeline_once_and_exposes_module_boundary() -> None:
    pipeline, _config = build_cpu_smoke_pipeline()
    loads = 0

    def load_pipeline():
        nonlocal loads
        loads += 1
        return pipeline

    service = PipelineApiService(load_pipeline)
    assert service.health().model_loaded is False

    request = QueryRequest(
        query_id="frontend-q1",
        query="What company did IBM acquire in 2019?",
    )
    first = service.query(request)
    second = service.query(request.model_copy(update={"query_id": "frontend-q2"}))

    assert loads == 1
    assert service.health().model_loaded is True
    assert first.query_id == "frontend-q1"
    assert second.query_id == "frontend-q2"
    assert first.answer == "IBM acquired Red Hat in 2019."
    assert first.diagnostics.retriever == "hybrid"
    assert first.diagnostics.selector == "nli-risk-controlled"
    assert first.diagnostics.generator == "grounded-grc"
    assert first.diagnostics.candidate_count == 10
    assert first.diagnostics.selected_count == 9
    assert first.diagnostics.dropped_count == 1
    assert any("poison" in item.text for item in first.candidates)
    assert all("poison" not in item.text for item in first.selected_evidence)
    assert {item.evidence_id for item in first.citations} <= {
        item.evidence_id for item in first.selected_evidence
    }


def test_runtime_loader_builds_configured_corpus_before_model_loading(
    monkeypatch,
) -> None:
    expected_pipeline, expected_config = build_cpu_smoke_pipeline()
    captured: dict[str, object] = {}

    def fake_build_pipeline(config, corpus, *, index_directory: Path | None = None):
        captured["config"] = config
        captured["corpus"] = corpus
        captured["index_directory"] = index_directory
        return expected_pipeline

    monkeypatch.delenv("EVIDENCE_RAG_INDEX_DIR", raising=False)
    monkeypatch.setattr(api_service, "build_pipeline_from_config", fake_build_pipeline)

    actual = load_runtime_pipeline(DEFAULT_CPU_SMOKE_CONFIG)

    assert actual is expected_pipeline
    assert captured["config"] == expected_config
    assert captured["index_directory"] is None
    assert captured["corpus"].manifest.chunker_name == "WordChunker"
