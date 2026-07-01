"""Tests for LLM query-side augmentation (HyDE / Query2Doc) as a Retriever wrapper.

A query transform is ``str -> str``; wrapping a retriever with it keeps CONTRACT 1
so the SAME object is scored on nDCG (run_benchmark) and cover-EM (run_rag). The
LLM is injected so these tests need no model download.
"""

from __future__ import annotations

from src.retrieval.base import RetrievedChunk, Retriever
from src.retrieval.query_transform import (
    HyDETransform,
    Query2DocTransform,
    TransformingRetriever,
)


class FakeLLM:
    def __init__(self, output: str = "a hypothetical passage") -> None:
        self.output = output
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.output


class RecordingRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, query: str):
        self.queries.append(query)
        return [RetrievedChunk(doc_id="d1", text=query, score=1.0)]


def test_hyde_transform_returns_generated_pseudo_document() -> None:
    llm = FakeLLM("granite is an ibm model family")
    transform = HyDETransform(llm)

    out = transform("what is granite?")

    assert out == "granite is an ibm model family"
    assert "what is granite?" in llm.prompts[0]  # the query is embedded in the prompt


def test_query2doc_keeps_original_query_terms() -> None:
    # Query2Doc concatenates the pseudo-doc with the ORIGINAL query, so exact query
    # terms are preserved (helps the lexical arm) rather than replaced.
    transform = Query2DocTransform(FakeLLM("pseudo document text"))

    out = transform("granite")

    assert "granite" in out
    assert "pseudo document text" in out


def test_transforming_retriever_retrieves_with_transformed_query() -> None:
    base = RecordingRetriever()
    transform_retriever = TransformingRetriever(base, lambda q: f"{q} EXPANDED")

    transform_retriever.retrieve("granite")

    assert base.queries == ["granite EXPANDED"]


def test_transforming_retriever_satisfies_retriever_contract() -> None:
    transform_retriever = TransformingRetriever(RecordingRetriever(), lambda q: q)

    assert isinstance(transform_retriever, Retriever)
