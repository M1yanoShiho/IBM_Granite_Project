"""Tests for eval/run_niah.py — the NIAH needle-found evaluation core."""
from __future__ import annotations

import json
from pathlib import Path

from src.niah.types import NiahExample, NiahTask
from src.retrieval.base import RetrievedChunk
from eval.run_niah import (
    evaluate_retriever_on_task,
    load_niah_run,
    main as run_niah_main,
)
from eval.run_benchmark import retrievers_need_llm


class _MultiRetriever:
    """Returns a per-query ranked list of doc ids (best-first) as RetrievedChunks.

    Entries may be plain doc ids (score defaults to 1.0) or ``(doc_id, score)``."""

    def __init__(self, by_query):
        self._by = by_query

    def retrieve(self, query):
        chunks = []
        for item in self._by[query]:
            doc_id, score = item if isinstance(item, tuple) else (item, 1.0)
            chunks.append(RetrievedChunk(doc_id=doc_id, text="", score=score))
        return chunks


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


def test_evaluate_collects_doc_level_run_with_best_scores() -> None:
    # WS-0 item 2: the FULL ranking (doc_id -> score) survives, not just hit@k --
    # WS-2/3/4/5 (burial, migration, margin cascade, oracle) all consume it. A doc's
    # duplicate chunks collapse to its best (first-seen) score.
    task = _task([NiahExample(query_id="q1", query="a", needle_ids=["n1"], needle_id="n1")])
    retriever = _MultiRetriever({"a": [("d0", 0.9), ("d0", 0.4), ("n1", 0.7)]})

    out = evaluate_retriever_on_task(retriever, task, k=10)

    assert out["run"] == {"q1": {"d0": 0.9, "n1": 0.7}}


def test_run_niah_dump_runs_writes_one_loadable_file_per_retriever(monkeypatch, tmp_path) -> None:
    # --dump-runs <path> fans out to <stem>_<retriever><suffix>, one self-contained
    # JSON per retriever: {"retriever", "k", "needles", "run"} -- the WS-0 dump.
    def fake_build_retrievers(config, data, llm=None):
        return {
            "granite_dense": _MultiRetriever({"a": [("cf", 0.99), ("n1", 0.5)]}),
            "q2d_granite": _MultiRetriever({"a": [("n1", 0.8), ("cf", 0.6)]}),
        }

    task = _task([NiahExample(query_id="q1", query="a", needle_ids=["n1"], needle_id="n1")])
    monkeypatch.setattr("eval.build_niah_task.load_niah_task", lambda *a, **k: task)
    monkeypatch.setattr("eval.run_benchmark._build_retrievers", fake_build_retrievers)
    monkeypatch.setattr(
        "eval.run_benchmark.retrievers_need_llm", lambda names: False
    )

    dump_base = tmp_path / "nq300_runs.json"
    run_niah_main([
        "--task", str(tmp_path / "t.json"),
        "--retrievers", "granite_dense", "q2d_granite",
        "--out", str(tmp_path / "o.csv"),
        "--dump-runs", str(dump_base),
    ])

    dense = load_niah_run(tmp_path / "nq300_runs_granite_dense.json")
    q2d = load_niah_run(tmp_path / "nq300_runs_q2d_granite.json")
    assert dense["retriever"] == "granite_dense"
    assert dense["run"] == {"q1": {"cf": 0.99, "n1": 0.5}}
    assert dense["needles"] == {"q1": "n1"}
    assert dense["k"] == 10
    assert q2d["run"]["q1"] == {"n1": 0.8, "cf": 0.6}


def test_retrievers_need_llm_flags_transforms_and_llm_rerankers() -> None:
    # Transforms, LLM listwise rerankers, and rerank-of-a-transform need an LLM;
    # pure dense / sparse / cross-encoder / fusion do not.
    assert retrievers_need_llm(["hyde_granite"])
    assert retrievers_need_llm(["q2d_granite"])
    assert retrievers_need_llm(["granite_listrank"])
    assert retrievers_need_llm(["q2d_granite_rerank"])       # cross-encoder over a transform
    assert retrievers_need_llm(["hyde_granite_rerank"])
    assert retrievers_need_llm(["granite_dense", "q2d_granite"])  # any-of
    assert not retrievers_need_llm(["granite_dense", "splade", "granite_rerank"])
    assert not retrievers_need_llm(["convex_hybrid_granite_splade"])


def test_run_niah_builds_one_shared_llm_for_multiple_transforms(monkeypatch, tmp_path) -> None:
    # The OOM fix: a run scoring several transforms must construct the 3B ONCE and
    # inject it, not once per transform. Everything heavy is faked so nothing loads.
    import eval.run_niah as rn

    built = []

    class FakeLLMClient:
        def __init__(self, *a, **k):
            built.append(1)

    captured = {}

    def fake_build_retrievers(config, data, llm=None):
        captured["llm"] = llm
        return {name: _MultiRetriever({"a": ["n1"]}) for name in config.retrievers}

    task = _task([NiahExample(query_id="q1", query="a", needle_ids=["n1"], needle_id="n1")])
    monkeypatch.setattr("eval.build_niah_task.load_niah_task", lambda *a, **k: task)
    monkeypatch.setattr("eval.run_benchmark._build_retrievers", fake_build_retrievers)
    monkeypatch.setattr("src.llm_client.LLMClient", FakeLLMClient)

    run_niah_main([
        "--task", str(tmp_path / "t.json"),
        "--retrievers", "q2d_granite", "hyde_granite", "granite_listrank",
        "--out", str(tmp_path / "o.csv"),
    ])

    assert built == [1]                       # exactly one 3B load for all three
    assert captured["llm"] is not None         # and it was injected into the builder


def test_run_niah_builds_no_llm_for_pure_dense(monkeypatch, tmp_path) -> None:
    # A dense/sparse-only run must not load an LLM at all.
    built = []

    class FakeLLMClient:
        def __init__(self, *a, **k):
            built.append(1)

    captured = {}

    def fake_build_retrievers(config, data, llm=None):
        captured["llm"] = llm
        return {name: _MultiRetriever({"a": ["n1"]}) for name in config.retrievers}

    task = _task([NiahExample(query_id="q1", query="a", needle_ids=["n1"], needle_id="n1")])
    monkeypatch.setattr("eval.build_niah_task.load_niah_task", lambda *a, **k: task)
    monkeypatch.setattr("eval.run_benchmark._build_retrievers", fake_build_retrievers)
    monkeypatch.setattr("src.llm_client.LLMClient", FakeLLMClient)

    run_niah_main([
        "--task", str(tmp_path / "t.json"),
        "--retrievers", "granite_dense", "splade",
        "--out", str(tmp_path / "o.csv"),
    ])

    assert built == []                # no LLM constructed
    assert captured["llm"] is None
