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
from src.rag_pipeline import RAGResult
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
