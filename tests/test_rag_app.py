"""Tests for the demo service seam (``src/rag_app.py``).

``build_rag_pipeline_from_text`` is the thin, testable composition the Streamlit
demo calls: it turns a pasted document + an LLM into a ready ``RAGPipeline`` by
reusing the retriever factory. The embedding backend is monkeypatched with a
bag-of-words fake (no download) and the LLM is injected as a fake, so the
retrieve-then-generate wiring is tested without loading any real model.
"""

from __future__ import annotations

import pytest

from src.rag_app import build_rag_pipeline_from_text
from src.rag_pipeline import (
    AstuteRAGPipeline,
    CorrectiveRAGPipeline,
    RAGPipeline,
    RAGResult,
)
from src.retrieval.base import RetrievedChunk


class FakeSentenceTransformer:
    """Deterministic SentenceTransformer stand-in (no model download)."""

    _VOCAB = ("granite", "retrieval", "banana", "cake")

    def __init__(self, model_id, cache_folder=None) -> None:
        pass

    def encode(
        self,
        texts,
        convert_to_numpy: bool = False,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
    ):
        if isinstance(texts, str):
            texts = [texts]
        return [[float(t.lower().split().count(w)) for w in self._VOCAB] for t in texts]


class FakeLLM:
    """Stand-in for LLMClient: records prompts, returns a canned answer."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "a grounded answer"


@pytest.fixture(autouse=True)
def _fake_embedding_backend(monkeypatch):
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSentenceTransformer
    )


def test_pipeline_retrieves_from_the_document_then_generates() -> None:
    # End-to-end wiring: the pasted document is indexed and retrieved, and the
    # retrieved context is fed to the (injected) generator.
    llm = FakeLLM()

    pipeline = build_rag_pipeline_from_text(
        "granite retrieval granite retrieval", llm, top_k=2
    )
    result = pipeline.query("granite retrieval")

    assert isinstance(result, RAGResult)
    assert result.answer == "a grounded answer"
    assert result.retrieved_chunks
    assert all(isinstance(c, RetrievedChunk) for c in result.retrieved_chunks)
    # chunks carry the pasted document's id (default "document"), not a chunk id
    assert all(c.doc_id == "document" for c in result.retrieved_chunks)
    # the retrieved context actually reached the generator's prompt
    assert "granite retrieval" in llm.prompts[0]


def test_threads_top_k_into_the_pipeline() -> None:
    pipeline = build_rag_pipeline_from_text("granite retrieval", FakeLLM(), top_k=3)

    assert pipeline.top_k == 3


def test_builder_defaults_to_corrective_pipeline() -> None:
    pipeline = build_rag_pipeline_from_text("granite retrieval", FakeLLM(), top_k=2)

    assert isinstance(pipeline, CorrectiveRAGPipeline)
    assert pipeline.top_k == 2


def test_builder_can_create_plain_pipeline() -> None:
    pipeline = build_rag_pipeline_from_text(
        "granite retrieval",
        FakeLLM(),
        top_k=2,
        pipeline_type="plain",
    )

    assert isinstance(pipeline, RAGPipeline)
    assert not isinstance(pipeline, CorrectiveRAGPipeline)


def test_builder_can_create_astute_pipeline() -> None:
    pipeline = build_rag_pipeline_from_text(
        "granite retrieval",
        FakeLLM(),
        top_k=2,
        pipeline_type="astute",
    )

    assert isinstance(pipeline, AstuteRAGPipeline)


def test_builder_threads_corrective_parameters() -> None:
    pipeline = build_rag_pipeline_from_text(
        "granite retrieval",
        FakeLLM(),
        top_k=2,
        pipeline_type="corrective",
        confidence_threshold=0.25,
        fallback_top_k=6,
    )

    assert isinstance(pipeline, CorrectiveRAGPipeline)
    assert pipeline.confidence_threshold == 0.25
    assert pipeline.fallback_top_k == 6


def test_builder_defaults_to_plain_dense_retriever() -> None:
    from src.retrieval.query_transform import TransformingRetriever

    pipeline = build_rag_pipeline_from_text("granite retrieval", FakeLLM(), top_k=2)

    assert not isinstance(pipeline.retriever, TransformingRetriever)  # plain dense


def test_builder_wraps_retriever_with_q2d_transform() -> None:
    # The certified NIAH raiser: Query2Doc expands the query with the shared LLM, then
    # the dense retriever searches with it. The transform must REUSE the injected LLM.
    from src.retrieval.query_transform import Query2DocTransform, TransformingRetriever

    llm = FakeLLM()
    pipeline = build_rag_pipeline_from_text(
        "granite retrieval", llm, top_k=2, retriever_type="q2d"
    )

    assert isinstance(pipeline.retriever, TransformingRetriever)
    assert isinstance(pipeline.retriever.transform, Query2DocTransform)
    assert pipeline.retriever.transform.llm is llm       # reuse, no second model load
    assert pipeline.query("granite retrieval").answer == "a grounded answer"


def test_builder_rejects_unknown_retriever_type() -> None:
    with pytest.raises(ValueError, match="retriever_type"):
        build_rag_pipeline_from_text(
            "granite retrieval", FakeLLM(), top_k=2, retriever_type="bogus"
        )
