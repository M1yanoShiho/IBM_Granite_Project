from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pytest

from eval.benchmarks.loader import BenchmarkData
from eval.run_rag import RAGEvalConfig, _parse_args, run
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
        "answer_cover",
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


def test_per_query_out_merges_retriever_columns():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "rag.csv"
        pq = Path(tmpdir) / "perq"
        run(
            RAGEvalConfig(retriever="granite_dense", results_path=out, per_query_out=pq, append=True),
            data=_make_data(), retriever=FakeRetriever(), llm=FakeLLM(),
        )
        run(
            RAGEvalConfig(retriever="bm25", results_path=out, per_query_out=pq, append=True),
            data=_make_data(), retriever=FakeRetriever(), llm=FakeLLM(),
        )
        f1_file = Path(tmpdir) / "perq_answer_f1.csv"
        assert f1_file.exists()
        with open(f1_file, newline="") as f:
            rows = list(csv.reader(f))
    # wide format: qid + one column per retriever (significance-ready)
    assert rows[0] == ["qid", "granite_dense", "bm25"]
    assert rows[1][0] == "q1"
    assert len(rows) == 2  # header + one query


def test_predictions_out_dumps_jsonl():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "rag.csv"
        pred = Path(tmpdir) / "pred"
        run(
            RAGEvalConfig(retriever="granite_dense", results_path=out, predictions_out=pred),
            data=_make_data(), retriever=FakeRetriever(), llm=FakeLLM(),
        )
        f = Path(tmpdir) / "pred_granite_dense.jsonl"
        assert f.exists()
        rec = json.loads(f.read_text(encoding="utf-8").splitlines()[0])
    assert rec["qid"] == "q1"
    assert rec["prediction"] == "Paris"
    assert rec["gold"] == ["Paris"]
    assert "capital of France" in rec["question"]


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


def test_cli_parses_subset_and_append_flags():
    cfg = _parse_args(
        [
            "--dataset", "nq", "--retriever", "bm25",
            "--max-queries", "300", "--max-docs", "50000",
            "--top-k", "6", "--append",
        ]
    )
    assert cfg.dataset == "nq"
    assert cfg.retriever == "bm25"
    assert cfg.max_queries == 300
    assert cfg.max_docs == 50000
    assert cfg.top_k == 6
    assert cfg.append is True


def test_cli_defaults_to_full_set():
    cfg = _parse_args([])
    assert cfg.max_queries is None
    assert cfg.max_docs is None
    assert cfg.append is False
