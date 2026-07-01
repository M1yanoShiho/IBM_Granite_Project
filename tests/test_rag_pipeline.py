"""Tests for the corrective RAG pipeline (confidence-gated re-retrieval).

``CorrectiveRAGPipeline`` turns the static single-shot pipeline into an adaptive
loop: retrieve -> score retrieval confidence -> if low, re-retrieve with a
rewritten query -> generate. This is a lightweight, model-free gate (a top-1
margin), NOT the learned evaluator of CRAG (Yan et al., 2024); it is named for the
family of methods, not a reimplementation.
"""

from __future__ import annotations

from src.rag_pipeline import CorrectiveRAGPipeline, RAGResult
from src.retrieval.base import RetrievedChunk


class ScriptedRetriever:
    """Returns canned results per query text and records the queries it saw."""

    def __init__(self, responses) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def retrieve(self, query: str):
        self.calls.append(query)
        return self.responses.get(query, [])


class EchoLLM:
    def generate(self, prompt: str) -> str:
        return "ANSWER"


def test_confidence_is_high_when_top_result_dominates() -> None:
    chunks = [RetrievedChunk("d1", "", 0.9), RetrievedChunk("d2", "", 0.1)]

    assert CorrectiveRAGPipeline._confidence(chunks) > 0.5


def test_confidence_is_zero_when_top_scores_tie() -> None:
    # Regression for the degenerate signal: (s0 - min)/(max - min) was ALWAYS 1.0
    # for best-first chunks, so the corrective branch never fired. A real signal
    # must be ~0 when the top results are indistinguishable (ambiguous retrieval).
    chunks = [
        RetrievedChunk("d1", "", 0.5),
        RetrievedChunk("d2", "", 0.5),
        RetrievedChunk("d3", "", 0.5),
    ]

    assert CorrectiveRAGPipeline._confidence(chunks) == 0.0


def test_low_confidence_triggers_corrective_reretrieval() -> None:
    retriever = ScriptedRetriever(
        {
            "q": [RetrievedChunk("d1", "", 0.5), RetrievedChunk("d2", "", 0.5)],
            "q REWRITTEN": [
                RetrievedChunk("d9", "good", 0.9),
                RetrievedChunk("d8", "", 0.1),
            ],
        }
    )
    pipeline = CorrectiveRAGPipeline(
        retriever,
        EchoLLM(),
        top_k=1,
        query_rewriter=lambda q: f"{q} REWRITTEN",
        confidence_threshold=0.5,
        fallback_top_k=2,
    )

    result = pipeline.query("q")

    assert retriever.calls == ["q", "q REWRITTEN"]  # re-retrieved with the rewrite
    assert [c.doc_id for c in result.retrieved_chunks] == ["d9", "d8"]  # fallback depth


def test_high_confidence_skips_correction() -> None:
    retriever = ScriptedRetriever(
        {"q": [RetrievedChunk("d1", "", 0.9), RetrievedChunk("d2", "", 0.1)]}
    )
    rewriter_calls: list[str] = []
    pipeline = CorrectiveRAGPipeline(
        retriever,
        EchoLLM(),
        top_k=1,
        query_rewriter=lambda q: rewriter_calls.append(q) or f"{q}x",
        confidence_threshold=0.5,
    )

    result = pipeline.query("q")

    assert retriever.calls == ["q"]  # no re-retrieval
    assert rewriter_calls == []  # rewriter never invoked
    assert [c.doc_id for c in result.retrieved_chunks] == ["d1"]  # top_k


def test_no_rewriter_cannot_correct_and_returns_first_pass() -> None:
    retriever = ScriptedRetriever(
        {"q": [RetrievedChunk("d1", "", 0.5), RetrievedChunk("d2", "", 0.5)]}
    )
    pipeline = CorrectiveRAGPipeline(retriever, EchoLLM(), top_k=1)  # no rewriter

    result = pipeline.query("q")

    assert retriever.calls == ["q"]
    assert [c.doc_id for c in result.retrieved_chunks] == ["d1"]


def test_query_returns_ragresult_with_generated_answer() -> None:
    retriever = ScriptedRetriever({"q": [RetrievedChunk("d1", "ctx", 0.9)]})
    pipeline = CorrectiveRAGPipeline(retriever, EchoLLM(), top_k=1)

    result = pipeline.query("q")

    assert isinstance(result, RAGResult)
    assert result.answer == "ANSWER"
