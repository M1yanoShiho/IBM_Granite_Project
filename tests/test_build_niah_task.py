# tests/test_build_niah_task.py
"""Tests for eval/build_niah_task.py — task-builder wiring (with fakes)."""
from __future__ import annotations

import json

from src.niah.types import NiahTask
from src.retrieval.base import RetrievedChunk
from eval.build_niah_task import build_task, compute_retrieval_signals, write_task_json


class _FakeLLM:
    def generate(self, prompt: str) -> str:
        # wrong-entity proposer: return a fixed alternative; answerability judge: 'NO'
        return "NO" if prompt.strip().endswith("Answer:") else "Mary Jones"


def _rank(ids):
    return {doc_id: i + 1 for i, doc_id in enumerate(ids)}


def test_build_task_makes_counterfactual_distractor_and_keeps_qrels_clean() -> None:
    corpus = {"d1": "Linda Davis won the 1994 award."}
    queries = {"q1": "who won the 1994 award?"}
    qrels = {"q1": {"d1": 1}}
    answers = {"q1": ["Linda Davis"]}
    # both retrievers rank the distractor highly (hard); scores below the gold
    dense_rank = {"q1": _rank(["d1", "q1__d1__cf0"])}
    sparse_rank = {"q1": _rank(["d1", "q1__d1__cf0"])}
    scores = {"q1": {"q1__d1__cf0": 0.4}}

    task = build_task(
        corpus=corpus, queries=queries, qrels=qrels, answers=answers,
        llm=_FakeLLM(), judge=_FakeLLM(),
        dense_rank=dense_rank, sparse_rank=sparse_rank, cand_scores=scores,
        positive_scores={"q1": 0.9}, margin=0.05, rank_threshold=10,
    )

    assert isinstance(task, NiahTask)
    assert "q1__d1__cf0" in task.corpus                 # distractor injected
    assert task.corpus["q1__d1__cf0"] == "Mary Jones won the 1994 award."
    assert "q1__d1__cf0" not in task.qrels["q1"]        # distractor NOT relevant
    assert task.qrels["q1"] == {"d1": 1}                # needle still the only gold
    assert task.examples[0].distractors[0].source == "counterfactual"


def test_build_task_ignores_non_positive_qrels() -> None:
    # a judged-negative (rel=0) qrels entry must NOT be treated as a needle
    corpus = {"d1": "Linda Davis won the 1994 award.", "d2": "Unrelated judged-negative doc."}
    queries = {"q1": "who won the 1994 award?"}
    qrels = {"q1": {"d1": 1, "d2": 0}}
    answers = {"q1": ["Linda Davis"]}
    dense_rank = {"q1": _rank(["d1", "q1__d1__cf0"])}
    sparse_rank = {"q1": _rank(["d1", "q1__d1__cf0"])}
    scores = {"q1": {"q1__d1__cf0": 0.4}}

    task = build_task(
        corpus=corpus, queries=queries, qrels=qrels, answers=answers,
        llm=_FakeLLM(), judge=_FakeLLM(),
        dense_rank=dense_rank, sparse_rank=sparse_rank, cand_scores=scores,
        positive_scores={"q1": 0.9}, margin=0.05, rank_threshold=10,
    )
    assert task.examples[0].needle_ids == ["d1"]     # d2 (rel=0) excluded
    assert "q1__d2__cf0" not in task.corpus           # no distractor built from d2


def test_build_task_wires_mined_source_c_without_polluting_qrels() -> None:
    # a mined topical negative (Source C) is recorded as a distractor but stays
    # out of qrels; the needle itself is never mined as a distractor.
    corpus = {
        "d1": "Linda Davis won the 1994 award.",
        "m1": "A topically adjacent passage that does not answer the question.",
    }
    queries = {"q1": "who won the 1994 award?"}
    qrels = {"q1": {"d1": 1}}
    answers = {"q1": ["Linda Davis"]}
    dense_rank = {"q1": _rank(["d1", "m1"])}
    sparse_rank = {"q1": _rank(["d1", "m1"])}
    scores = {"q1": {"m1": 0.4}}   # mined m1 scored below the positive anchor

    task = build_task(
        corpus=corpus, queries=queries, qrels=qrels, answers=answers,
        llm=_FakeLLM(), judge=_FakeLLM(),
        dense_rank=dense_rank, sparse_rank=sparse_rank, cand_scores=scores,
        positive_scores={"q1": 0.9}, margin=0.05, rank_threshold=10,
        mined_ids={"q1": ["m1", "d1"]},   # d1 is the needle -> must be skipped
    )
    sources = {d.doc_id: d.source for d in task.examples[0].distractors}
    assert sources.get("m1") == "mined"       # mined distractor recorded
    assert "d1" not in sources                 # the needle is never a distractor
    assert "m1" not in task.qrels["q1"]        # mined stays non-relevant


class _AnswerLeakLLM:
    # wrong-entity proposer returns an alternative, but the answerability judge
    # says the (counterfactual) passage still answers the query -> it is dropped.
    def generate(self, prompt: str) -> str:
        return "YES" if prompt.strip().endswith("Answer:") else "Mary Jones"


def test_build_task_drops_counterfactual_that_still_answers() -> None:
    corpus = {"d1": "Linda Davis won the 1994 award."}
    queries = {"q1": "who won the 1994 award?"}
    qrels = {"q1": {"d1": 1}}
    answers = {"q1": ["Linda Davis"]}

    task = build_task(
        corpus=corpus, queries=queries, qrels=qrels, answers=answers,
        llm=_AnswerLeakLLM(), judge=_AnswerLeakLLM(),
        dense_rank={}, sparse_rank={}, cand_scores={},
        positive_scores={"q1": 0.9}, margin=0.05, rank_threshold=10,
    )
    assert task.examples[0].distractors == []          # answer-leaking A dropped
    assert "q1__d1__cf0" not in task.corpus


class _FakeRetriever:
    def __init__(self, ranking):
        self._ranking = ranking  # [(doc_id, score)], best-first

    def retrieve(self, query: str):
        return [RetrievedChunk(doc_id=i, text="", score=s) for i, s in self._ranking]


def test_compute_retrieval_signals_derives_ranks_positive_and_mined() -> None:
    dense = _FakeRetriever([("d1", 0.9), ("m1", 0.5)])   # needle d1 top, m1 next
    sparse = _FakeRetriever([("m1", 3.0), ("d1", 2.0)])
    queries = {"q1": "q"}
    qrels = {"q1": {"d1": 1}}

    sig = compute_retrieval_signals(dense, sparse, queries, qrels, top_n=10, mine_k=10)

    assert sig.dense_rank["q1"] == {"d1": 1, "m1": 2}
    assert sig.sparse_rank["q1"] == {"m1": 1, "d1": 2}
    assert sig.cand_scores["q1"]["m1"] == 0.5            # dense-scale candidate score
    assert sig.positive_scores["q1"] == 0.9             # needle d1's own dense score
    assert set(sig.mined_ids["q1"]) == {"m1"}           # d1 (needle) excluded


def test_write_task_json_roundtrips(tmp_path) -> None:
    task = build_task(
        corpus={"d1": "Linda Davis won the 1994 award."},
        queries={"q1": "who won the 1994 award?"},
        qrels={"q1": {"d1": 1}},
        answers={"q1": ["Linda Davis"]},
        llm=_FakeLLM(), judge=_FakeLLM(),
        dense_rank={}, sparse_rank={}, cand_scores={},
        positive_scores={"q1": 0.9}, margin=0.05, rank_threshold=10,
    )
    out = tmp_path / "task.json"
    write_task_json(task, out)

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert "q1__d1__cf0" in loaded["corpus"]
    assert loaded["qrels"]["q1"] == {"d1": 1}
    assert loaded["examples"][0]["distractors"][0]["source"] == "counterfactual"
