"""Tests for the P6 benchmark orchestration (``eval/run_benchmark.py``).

Covers the parts of P6 that are unblocked by the shared contracts and need no
real retriever (P3/P4) or index (P5): the chunk->doc aggregation (``build_run``,
contract 3), the per-retriever scoring glue, the comparison-table assembly, and
the CSV output. A ``FakeRetriever`` (canned per-query responses) stands in for
the real retrievers — it satisfies the ``Retriever`` Protocol structurally.
"""

from __future__ import annotations

import csv
from typing import Dict, List

import pytest

from src.retrieval.base import RetrievedChunk
from eval.benchmarks.loader import BenchmarkData
from eval.run_benchmark import (
    BenchmarkConfig,
    build_run,
    evaluate_one,
    run,
    write_results_csv,
)


class FakeRetriever:
    """A retriever with canned responses, keyed by query text.

    Satisfies the ``Retriever`` Protocol just by defining ``retrieve`` — no
    subclassing needed (the Protocol is structural / ``runtime_checkable``).
    """

    def __init__(self, responses: Dict[str, List[RetrievedChunk]]) -> None:
        self._responses = responses

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        return self._responses.get(query, [])


def test_build_run_max_pools_chunks_of_same_doc() -> None:
    # d1 surfaces via two chunks (0.4 and 0.9); contract 3 says the doc's score
    # is the MAX of its chunk scores, not the first/last/sum. d2 has one chunk.
    retriever = FakeRetriever(
        {
            "some query": [
                RetrievedChunk(doc_id="d1", text="chunk a", score=0.4),
                RetrievedChunk(doc_id="d1", text="chunk b", score=0.9),
                RetrievedChunk(doc_id="d2", text="other", score=0.5),
            ],
        }
    )

    run = build_run(retriever, {"q1": "some query"})

    assert run == {"q1": {"d1": 0.9, "d2": 0.5}}


def test_evaluate_one_scores_a_perfect_retriever_as_one() -> None:
    # Each query has a single relevant doc; the retriever ranks it first. So the
    # rank-based metrics are exactly 1.0. (precision@k for k>1 is not — only one
    # relevant doc exists — so we don't assert on it: that would be flaky.)
    data = BenchmarkData(
        corpus={"d1": "doc one", "d2": "doc two", "d3": "doc three"},
        queries={"q1": "qtext1", "q2": "qtext2"},
        qrels={"q1": {"d1": 1}, "q2": {"d2": 1}},
    )
    retriever = FakeRetriever(
        {
            "qtext1": [
                RetrievedChunk(doc_id="d1", text="doc one", score=0.9),
                RetrievedChunk(doc_id="d3", text="doc three", score=0.5),
            ],
            "qtext2": [
                RetrievedChunk(doc_id="d2", text="doc two", score=0.9),
                RetrievedChunk(doc_id="d3", text="doc three", score=0.5),
            ],
        }
    )

    metrics = evaluate_one(retriever, data, k_values=[1, 10])

    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["recall@10"] == pytest.approx(1.0)
    assert metrics["ndcg@10"] == pytest.approx(1.0)
    assert metrics["precision@1"] == pytest.approx(1.0)


def test_write_results_csv_round_trips(tmp_path) -> None:
    # One row per retriever; the retriever name is a column; values survive the
    # write+read round-trip. Read back with DictReader so column order is free.
    results = {
        "bm25": {"precision@1": 0.5, "recall@10": 0.8, "ndcg@10": 0.6, "mrr": 0.7},
        "granite_dense": {"precision@1": 0.9, "recall@10": 0.95, "ndcg@10": 0.92, "mrr": 0.93},
    }
    path = tmp_path / "benchmark_results.csv"

    write_results_csv(results, path)

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    by_name = {row["retriever"]: row for row in rows}
    assert set(by_name) == {"bm25", "granite_dense"}
    assert float(by_name["granite_dense"]["mrr"]) == 0.93
    assert float(by_name["bm25"]["precision@1"]) == 0.5


def test_run_writes_one_row_per_injected_retriever(tmp_path) -> None:
    # End-to-end orchestration with injected fakes (no real loader / retriever):
    # run() scores each retriever and writes the comparison CSV. "perfect" ranks
    # the relevant doc first (mrr 1.0); "wrong" only finds an irrelevant one (0.0).
    data = BenchmarkData(
        corpus={"d1": "doc one", "d2": "doc two"},
        queries={"q1": "qtext1"},
        qrels={"q1": {"d1": 1}},
    )
    retrievers = {
        "perfect": FakeRetriever({"qtext1": [RetrievedChunk("d1", "doc one", 0.9)]}),
        "wrong": FakeRetriever({"qtext1": [RetrievedChunk("d2", "doc two", 0.9)]}),
    }
    config = BenchmarkConfig(k_values=[1], results_path=tmp_path / "out.csv")

    run(config, retrievers=retrievers, data=data)

    with open(config.results_path, newline="") as f:
        rows = {row["retriever"]: row for row in csv.DictReader(f)}
    assert set(rows) == {"perfect", "wrong"}
    assert float(rows["perfect"]["mrr"]) == 1.0
    assert float(rows["wrong"]["mrr"]) == 0.0


def test_run_builds_real_bm25_and_writes_scored_row(tmp_path) -> None:
    # No injected retrievers -> run() builds the real BM25Retriever itself
    # (via _build_retrievers). The query terms appear only in d1, the relevant
    # doc, so BM25 ranks it first and the scored row is a perfect result.
    data = BenchmarkData(
        corpus={
            "d1": "granite dense retrieval embeddings",
            "d2": "banana cake recipe with sugar",
            "d3": "weather forecast tomorrow rain",
        },
        queries={"q1": "granite retrieval"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(
        retrievers=["bm25"],
        k_values=[1],
        results_path=tmp_path / "out.csv",
    )

    run(config, data=data)

    with open(config.results_path, newline="") as f:
        rows = {row["retriever"]: row for row in csv.DictReader(f)}
    assert set(rows) == {"bm25"}
    assert float(rows["bm25"]["mrr"]) == 1.0


def test_run_dense_retriever_pending_until_p5(tmp_path) -> None:
    # Dense retrievers need P5's vector index (contract 5). Until it lands,
    # building one must fail loudly, not silently produce an empty result.
    data = BenchmarkData(
        corpus={"d1": "granite retrieval"},
        queries={"q1": "granite retrieval"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(
        retrievers=["granite_dense"],
        k_values=[1],
        results_path=tmp_path / "out.csv",
    )

    with pytest.raises(NotImplementedError):
        run(config, data=data)
