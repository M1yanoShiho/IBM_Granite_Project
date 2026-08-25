"""Application service that adapts the final pipeline to the public API schema."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from uuid import uuid4

from evidence_rag.api.schemas import (
    EvidenceItem,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RuntimeDiagnostics,
)
from evidence_rag.composition import build_pipeline_from_config
from evidence_rag.contracts.models import EvidenceCandidate, Query
from evidence_rag.infrastructure.config import load_experiment_config
from evidence_rag.infrastructure.corpus import CorpusBuilder, build_chunker
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.pipeline.service import EvidenceRAGPipeline

PipelineLoader = Callable[[], EvidenceRAGPipeline]


def _api_evidence(candidate: EvidenceCandidate) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=candidate.evidence_id,
        document_id=candidate.document_id,
        text=candidate.text,
        source_uri=candidate.source_uri,
        retrieval_score=candidate.retrieval_score,
        retrieval_rank=candidate.retrieval_rank,
    )


class PipelineApiService:
    """Load one pipeline lazily and reuse it for every frontend request."""

    def __init__(self, load_pipeline: PipelineLoader) -> None:
        self._load_pipeline = load_pipeline
        self._pipeline: EvidenceRAGPipeline | None = None
        self._load_lock = Lock()

    def _pipeline_instance(self) -> EvidenceRAGPipeline:
        if self._pipeline is None:
            with self._load_lock:
                if self._pipeline is None:
                    self._pipeline = self._load_pipeline()
        return self._pipeline

    def health(self) -> HealthResponse:
        return HealthResponse(model_loaded=self._pipeline is not None)

    def query(self, request: QueryRequest) -> QueryResponse:
        query_id = request.query_id or f"query-{uuid4().hex}"
        run = self._pipeline_instance().run_with_trace(
            Query(query_id=query_id, text=request.query),
            top_k=request.top_k,
            max_selected=request.max_selected,
        )

        candidates = tuple(_api_evidence(item) for item in run.candidates.candidates)
        selected = tuple(_api_evidence(item) for item in run.selected.evidence)
        selected_by_id = {item.evidence_id: item for item in selected}
        citations = tuple(
            selected_by_id[evidence_id]
            for evidence_id in run.generation.cited_evidence_ids
        )
        return QueryResponse(
            query_id=query_id,
            session_id=request.session_id,
            answer=run.generation.answer,
            candidates=candidates,
            selected_evidence=selected,
            citations=citations,
            diagnostics=RuntimeDiagnostics(
                candidate_count=len(candidates),
                selected_count=len(selected),
                dropped_count=len(candidates) - len(selected),
                citation_count=len(citations),
            ),
        )


def load_runtime_pipeline(config_path: Path) -> EvidenceRAGPipeline:
    """Build the real final pipeline from a portable runtime config."""

    config = load_experiment_config(config_path)
    dataset = JsonlDatasetAdapter.load(config.dataset_manifest_path)
    chunker = build_chunker(
        config.chunker.name,
        chunk_size=config.chunker.chunk_size,
        overlap=config.chunker.overlap,
    )
    corpus = CorpusBuilder(chunker).build(
        dataset.documents,
        dataset.dataset_signature,
    )
    raw_index_directory = os.environ.get("EVIDENCE_RAG_INDEX_DIR")
    index_directory = (
        Path(raw_index_directory).expanduser().resolve()
        if raw_index_directory
        else None
    )
    return build_pipeline_from_config(
        config,
        corpus,
        index_directory=index_directory,
    )
