from __future__ import annotations

import csv
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
        qrels={"q1": {"doc1": 1}},  # doc1 relevant, doc2 not
        answers={"q1": ["Paris"]} if with_answers else None,
    )


def test_run_returns_metric_suite():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = RAGEvalConfig(results_path=Path(tmpdir) / "rag_results.csv")
        metrics = run(config, data=_make_data(), retriever=FakeRetriever(), llm=FakeLLM())
    assert set(metrics) == {
        "answer_em",
        "answer_f1",
        "context_precision",
        "faithfulness",
    }


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


def test_exact_match_for_correct_answer():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = RAGEvalConfig(results_path=Path(tmpdir) / "rag_results.csv")
        metrics = run(config, data=_make_data(), retriever=FakeRetriever(), llm=FakeLLM())
    assert metrics["answer_em"] == 1.0
    assert metrics["answer_f1"] == 1.0


def test_context_precision_reflects_qrels():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = RAGEvalConfig(results_path=Path(tmpdir) / "rag_results.csv")
        metrics = run(config, data=_make_data(), retriever=FakeRetriever(), llm=FakeLLM())
    # retrieved doc1 (relevant) + doc2 (not relevant) -> precision 0.5
    assert metrics["context_precision"] == 0.5


def test_append_accumulates_one_row_per_retriever():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "rag_results.csv"
        run(
            RAGEvalConfig(retriever="granite_dense", results_path=out, append=True),
            data=_make_data(), retriever=FakeRetriever(), llm=FakeLLM(),
        )
        run(
            RAGEvalConfig(retriever="bm25", results_path=out, append=True),
            data=_make_data(), retriever=FakeRetriever(), llm=FakeLLM(),
        )
        with open(out, newline="") as f:
            rows = list(csv.DictReader(f))
    assert [r["retriever"] for r in rows] == ["granite_dense", "bm25"]


def test_no_append_overwrites():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "rag_results.csv"
        run(
            RAGEvalConfig(retriever="granite_dense", results_path=out),
            data=_make_data(), retriever=FakeRetriever(), llm=FakeLLM(),
        )
        run(
            RAGEvalConfig(retriever="bm25", results_path=out),
            data=_make_data(), retriever=FakeRetriever(), llm=FakeLLM(),
        )
        with open(out, newline="") as f:
            rows = list(csv.DictReader(f))
    assert [r["retriever"] for r in rows] == ["bm25"]
