from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from eval.benchmarks.loader import BenchmarkData
from eval.run_rag import RAGEvalConfig, run
from src.retrieval.base import RetrievedChunk


class FakeRetriever:
    def retrieve(self, query: str):
        return [
            RetrievedChunk(doc_id="doc1", text="Paris is the capital of France.", score=0.9),
            RetrievedChunk(doc_id="doc2", text="France is a country in Europe.", score=0.7),
        ]


class FakeLLM:
    def generate(self, prompt: str) -> str:
        return "Paris"


def _make_data(with_answers: bool = True) -> BenchmarkData:
    return BenchmarkData(
        corpus={"doc1": "Paris is the capital of France."},
        queries={"q1": "What is the capital of France?"},
        qrels={"q1": {"doc1": 1}},
        answers={"q1": "Paris"} if with_answers else None,
    )


def test_run_returns_metrics():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = RAGEvalConfig(results_path=Path(tmpdir) / "rag_results.csv")
        metrics = run(config, data=_make_data(), retriever=FakeRetriever(), llm=FakeLLM())
    assert "answer_correctness" in metrics
    assert "context_precision" in metrics
    assert "faithfulness" in metrics


def test_run_writes_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "rag_results.csv"
        config = RAGEvalConfig(results_path=out)
        run(config, data=_make_data(), retriever=FakeRetriever(), llm=FakeLLM())
        assert out.exists()


def test_run_raises_without_answers():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = RAGEvalConfig(results_path=Path(tmpdir) / "rag_results.csv")
        with pytest.raises(ValueError, match="no gold answers"):
            run(config, data=_make_data(with_answers=False), retriever=FakeRetriever(), llm=FakeLLM())


def test_run_correctness_score_is_high_for_exact_match():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = RAGEvalConfig(results_path=Path(tmpdir) / "rag_results.csv")
        metrics = run(config, data=_make_data(), retriever=FakeRetriever(), llm=FakeLLM())
    assert metrics["answer_correctness"] == 1.0