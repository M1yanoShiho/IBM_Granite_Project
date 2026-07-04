"""Tests for eval/run_niah.py — the NIAH needle-found evaluation core."""
from __future__ import annotations

from src.niah.types import NiahExample, NiahTask
from src.retrieval.base import RetrievedChunk
from eval.run_niah import evaluate_retriever_on_task


class _MultiRetriever:
    """Returns a per-query ranked list of doc ids (best-first) as RetrievedChunks."""

    def __init__(self, by_query):
        self._by = by_query

    def retrieve(self, query):
        return [RetrievedChunk(doc_id=i, text="", score=1.0) for i in self._by[query]]


def _task(examples):
    return NiahTask(
        corpus={}, queries={e.query_id: e.query for e in examples}, qrels={}, examples=examples
    )


def test_evaluate_scores_designated_needle_hit_and_mrr() -> None:
    task = _task([
        NiahExample(query_id="q1", query="a", needle_ids=["n1"], needle_id="n1"),
        NiahExample(query_id="q2", query="b", needle_ids=["n2"], needle_id="n2"),
    ])
    # q1: needle n1 at rank 2 -> found, rr=0.5 ; q2: n2 absent -> not found, rr=0
    retriever = _MultiRetriever({"a": ["d0", "n1"], "b": ["d1", "d2"]})
    out = evaluate_retriever_on_task(retriever, task, k=10)
    assert out["needle_found"] == {"q1": 1.0, "q2": 0.0}
    assert out["mrr"]["q1"] == 0.5
    assert out["mrr"]["q2"] == 0.0


def test_evaluate_hit_respects_k() -> None:
    task = _task([NiahExample(query_id="q1", query="a", needle_ids=["n1"], needle_id="n1")])
    retriever = _MultiRetriever({"a": ["d0", "n1"]})   # n1 at rank 2
    assert evaluate_retriever_on_task(retriever, task, k=1)["needle_found"] == {"q1": 0.0}
    assert evaluate_retriever_on_task(retriever, task, k=2)["needle_found"] == {"q1": 1.0}


def test_evaluate_dedups_chunks_to_doc_level() -> None:
    # multiple chunks of the same doc collapse to one rank (best-first)
    task = _task([NiahExample(query_id="q1", query="a", needle_ids=["n1"], needle_id="n1")])
    retriever = _MultiRetriever({"a": ["d0", "d0", "d0", "n1"]})   # n1 is doc-rank 2, not 4
    assert evaluate_retriever_on_task(retriever, task, k=2)["needle_found"] == {"q1": 1.0}


def test_evaluate_skips_queries_without_designated_needle() -> None:
    task = _task([NiahExample(query_id="q1", query="a", needle_ids=[], needle_id=None)])
    retriever = _MultiRetriever({"a": ["d0"]})
    assert evaluate_retriever_on_task(retriever, task, k=10)["needle_found"] == {}
