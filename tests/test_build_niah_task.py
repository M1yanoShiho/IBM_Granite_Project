# tests/test_build_niah_task.py
"""Tests for eval/build_niah_task.py — task-builder wiring (with fakes)."""
from __future__ import annotations

from src.niah.types import NiahTask
from eval.build_niah_task import build_task


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
