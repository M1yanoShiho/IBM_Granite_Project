# tests/test_build_niah_task.py
"""Tests for eval/build_niah_task.py — task-builder wiring (with fakes)."""
from __future__ import annotations

import json

from src.niah.types import NiahTask
from src.retrieval.base import RetrievedChunk
from eval.benchmarks.loader import BenchmarkData
from eval.build_niah_task import (
    build_task,
    compute_retrieval_signals,
    designate_needle,
    load_niah_task,
    write_task_json,
)


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
    assert task.examples[0].needle_id == "d1"           # the single designated needle
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


def _sample_task():
    return build_task(
        corpus={"d1": "Linda Davis won the 1994 award."},
        queries={"q1": "who won the 1994 award?"},
        qrels={"q1": {"d1": 1}},
        answers={"q1": ["Linda Davis"]},
        llm=_FakeLLM(), judge=_FakeLLM(),
        dense_rank={}, sparse_rank={}, cand_scores={},
        positive_scores={"q1": 0.9}, margin=0.05, rank_threshold=10,
    )


_RECIPE = {"dataset": "nq", "split": "dev", "max_queries": 1, "max_docs": 10}


def test_write_task_json_stores_recipe_and_distractor_text_not_corpus(tmp_path) -> None:
    out = tmp_path / "task.json"
    write_task_json(_sample_task(), out, recipe=_RECIPE)

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert "corpus" not in loaded                        # no full-corpus dump
    assert loaded["recipe"]["dataset"] == "nq"
    d0 = loaded["examples"][0]["distractors"][0]
    assert d0["source"] == "counterfactual"
    assert d0["text"] == "Mary Jones won the 1994 award."   # generated text preserved


def test_load_niah_task_reconstructs_from_recipe(tmp_path) -> None:
    out = tmp_path / "task.json"
    write_task_json(_sample_task(), out, recipe=_RECIPE)

    def fake_loader(name, split="test", max_queries=None, max_docs=None):
        # background haystack + the needle d1, deterministically rebuilt from recipe
        return BenchmarkData(
            corpus={"d1": "Linda Davis won the 1994 award.", "bg1": "unrelated hay"},
            queries={"q1": "who won the 1994 award?"},
            qrels={"q1": {"d1": 1}},
        )

    task = load_niah_task(out, loader=fake_loader)
    assert task.corpus["d1"] == "Linda Davis won the 1994 award."          # needle
    assert "bg1" in task.corpus                                             # background
    assert task.corpus["q1__d1__cf0"] == "Mary Jones won the 1994 award."  # distractor re-injected
    assert task.examples[0].needle_id == "d1"
    assert task.qrels["q1"] == {"d1": 1}


def test_load_niah_task_max_docs_override_drives_the_scale_sweep(tmp_path) -> None:
    out = tmp_path / "task.json"
    write_task_json(_sample_task(), out, recipe=_RECIPE)   # recipe max_docs = 10
    seen = {}

    def fake_loader(name, split="test", max_queries=None, max_docs=None):
        seen["max_docs"] = max_docs
        return BenchmarkData(corpus={"d1": "x"}, queries={"q1": "q"}, qrels={"q1": {"d1": 1}})

    load_niah_task(out, loader=fake_loader, max_docs=5000)
    assert seen["max_docs"] == 5000     # override wins (a scale-sweep point)
    load_niah_task(out, loader=fake_loader)
    assert seen["max_docs"] == 10        # falls back to the recipe


def test_designate_needle_prefers_answer_bearing_smallest_id() -> None:
    corpus = {"d1": "no answer here", "d2": "Linda Davis won", "d3": "Linda Davis too"}
    assert designate_needle(["d1", "d2", "d3"], "Linda Davis", corpus) == "d2"


def test_designate_needle_falls_back_to_smallest_gold() -> None:
    corpus = {"d1": "x", "d2": "y"}
    assert designate_needle(["d2", "d1"], "absent answer", corpus) == "d1"


def test_designate_needle_none_when_no_gold_in_corpus() -> None:
    assert designate_needle(["d9"], "x", {"d1": "y"}) is None


def test_build_task_designates_one_needle_from_multiple_golds() -> None:
    corpus = {
        "d1": "Linda Davis won the 1994 award.",
        "d2": "Linda Davis performed at the 1994 award show.",
    }
    queries = {"q1": "who won the 1994 award?"}
    qrels = {"q1": {"d1": 1, "d2": 1}}          # two golds -> exactly one designated
    answers = {"q1": ["Linda Davis"]}
    task = build_task(
        corpus=corpus, queries=queries, qrels=qrels, answers=answers,
        llm=_FakeLLM(), judge=_FakeLLM(),
        dense_rank={}, sparse_rank={}, cand_scores={},
        positive_scores={"q1": 0.9}, margin=0.05, rank_threshold=10,
    )
    ex = task.examples[0]
    assert set(ex.needle_ids) == {"d1", "d2"}        # both golds recorded (honest qrels)
    assert ex.needle_id == "d1"                       # one designated (answer-bearing, smallest id)
    cfs = [d for d in ex.distractors if d.source == "counterfactual"]
    assert len(cfs) == 1                              # exactly ONE counterfactual, not one per gold
    assert cfs[0].parent_needle_id == "d1"


def test_build_task_wires_source_b_generative() -> None:
    corpus = {"d1": "Linda Davis won the 1994 award."}
    queries = {"q1": "who won the 1994 award?"}
    qrels = {"q1": {"d1": 1}}
    answers = {"q1": ["Linda Davis"]}
    task = build_task(
        corpus=corpus, queries=queries, qrels=qrels, answers=answers,
        llm=_FakeLLM(), judge=_FakeLLM(),
        dense_rank={}, sparse_rank={}, cand_scores={},
        positive_scores={"q1": 0.9}, margin=0.05, rank_threshold=10,
    )
    sources = {d.source for d in task.examples[0].distractors}
    assert "counterfactual" in sources       # Source A
    assert "generative" in sources           # Source B now wired in -> distractor diversity
    gen = [d for d in task.examples[0].distractors if d.source == "generative"]
    assert gen[0].doc_id == "q1__gen0"
    assert gen[0].doc_id in task.corpus      # injected into the haystack
