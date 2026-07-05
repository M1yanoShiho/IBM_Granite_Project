"""Tests for the corrective RAG pipeline (confidence-gated re-retrieval).

``CorrectiveRAGPipeline`` turns the static single-shot pipeline into an adaptive
loop: retrieve -> score retrieval confidence -> if low, re-retrieve with a
rewritten query -> generate. This is a lightweight, model-free gate (a top-1
margin), NOT the learned evaluator of CRAG (Yan et al., 2024); it is named for the
family of methods, not a reimplementation.
"""

from __future__ import annotations

from src.rag_pipeline import (
    AstuteRAGPipeline,
    CorrectiveRAGPipeline,
    RAGPipeline,
    RAGResult,
)
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


class AnswerLLM:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def generate(self, prompt: str) -> str:
        return self.answer


class ScriptedLLM:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.outputs.pop(0)


def test_plain_pipeline_returns_citations_for_supported_answer() -> None:
    retriever = ScriptedRetriever(
        {"q": [RetrievedChunk("d1", "Paris is the capital of France.", 0.9)]}
    )
    pipeline = RAGPipeline(
        retriever, AnswerLLM("Paris is the capital of France."), top_k=1
    )

    result = pipeline.query("q")

    assert result.abstained is False
    assert result.citations
    assert result.citations[0].source_chunk_id == "d1"


def test_plain_pipeline_abstains_when_model_says_it_does_not_know() -> None:
    retriever = ScriptedRetriever(
        {"q": [RetrievedChunk("d1", "Paris is the capital of France.", 0.9)]}
    )
    pipeline = RAGPipeline(retriever, AnswerLLM("I don't know."), top_k=1)

    result = pipeline.query("q")

    assert result.abstained is True
    assert result.abstain_reason == "model_reported_unknown"


def test_plain_pipeline_abstains_when_no_chunks_are_retrieved() -> None:
    retriever = ScriptedRetriever({"q": []})
    pipeline = RAGPipeline(retriever, AnswerLLM("Some answer"), top_k=1)

    result = pipeline.query("q")

    assert result.retrieved_chunks == []
    assert result.abstained is True
    assert result.abstain_reason == "no_retrieved_context"


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


def test_corrective_pipeline_marks_when_correction_was_used() -> None:
    retriever = ScriptedRetriever(
        {
            "q": [RetrievedChunk("d1", "weak", 0.5), RetrievedChunk("d2", "tie", 0.5)],
            "q REWRITTEN": [
                RetrievedChunk("d9", "Paris is the capital of France.", 0.9)
            ],
        }
    )
    pipeline = CorrectiveRAGPipeline(
        retriever,
        AnswerLLM("Paris is the capital of France."),
        top_k=1,
        query_rewriter=lambda q: f"{q} REWRITTEN",
        confidence_threshold=0.5,
        fallback_top_k=1,
    )

    result = pipeline.query("q")

    assert result.used_corrective_retrieval is True
    assert result.confidence == 0.0
    assert result.abstained is False


def test_corrective_pipeline_abstains_on_low_confidence_without_rewriter() -> None:
    retriever = ScriptedRetriever(
        {
            "q": [
                RetrievedChunk("d1", "Paris is the capital of France.", 0.5),
                RetrievedChunk("d2", "Paris is in Texas.", 0.5),
            ]
        }
    )
    pipeline = CorrectiveRAGPipeline(
        retriever,
        AnswerLLM("Paris is the capital of France."),
        top_k=1,
        query_rewriter=None,
        confidence_threshold=0.5,
    )

    result = pipeline.query("q")

    assert result.confidence == 0.0
    assert result.abstained is True
    assert result.abstain_reason == "low_retrieval_confidence"


def test_astute_pipeline_returns_shared_rag_result_metadata() -> None:
    retriever = ScriptedRetriever(
        {"q": [RetrievedChunk("d1", "Paris is the capital of France.", 0.9)]}
    )
    llm = ScriptedLLM(
        [
            "Paris is the capital of France.",
            "The reliable consolidated answer is Paris.",
            "Paris is the capital of France.",
        ]
    )
    pipeline = AstuteRAGPipeline(retriever, llm, top_k=1)

    result = pipeline.query("q")

    assert result.abstained is False
    assert result.citations
    assert result.citations[0].source_chunk_id == "d1"
    assert result.used_corrective_retrieval is False
