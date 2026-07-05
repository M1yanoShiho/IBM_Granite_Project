"""Tests for the offline lambda-sweep of the corroboration reranker."""
from eval.tune_corroboration import (
    _parse_args,
    best_alpha,
    build_corroboration_runs,
    dedup_docs,
    needle_found_at_k,
    sweep_corroboration,
)
from src.niah.types import NiahExample, NiahTask
from src.retrieval.base import RetrievedChunk


def _task(examples):
    return NiahTask(
        corpus={}, queries={e.query_id: e.query for e in examples}, qrels={}, examples=examples
    )


class FakeRetriever:
    def __init__(self, by_query):
        self._by = by_query

    def retrieve(self, query):
        return self._by[query]


class FakeScorer:
    """Stand-in for CorroborationReranker.score_docs: scripted (relevance, corroboration)."""

    def __init__(self, rel_by_id, corr_by_id):
        self.rel_by_id = rel_by_id
        self.corr_by_id = corr_by_id

    def score_docs(self, query, docs):
        return (
            [self.rel_by_id[d.doc_id] for d in docs],
            [self.corr_by_id[d.doc_id] for d in docs],
        )


def test_dedup_docs_keeps_first_per_doc_capped_at_top_n():
    chunks = [
        RetrievedChunk("a", "", 0.9),
        RetrievedChunk("a", "", 0.5),   # duplicate doc -> dropped (best-first keeps the first)
        RetrievedChunk("b", "", 0.4),
        RetrievedChunk("c", "", 0.3),
    ]
    assert [c.doc_id for c in dedup_docs(chunks, top_n=2)] == ["a", "b"]


def test_build_corroboration_runs_extracts_two_runs_and_needles():
    task = _task([NiahExample(query_id="q1", query="a", needle_ids=["n1"], needle_id="n1")])
    retr = FakeRetriever({"a": [RetrievedChunk("cf", "", 0.99), RetrievedChunk("n1", "", 0.5)]})
    scorer = FakeScorer(rel_by_id={"cf": 0.99, "n1": 0.5}, corr_by_id={"cf": 0.0, "n1": 1.0})

    rel, corr, needles = build_corroboration_runs(task, retr, scorer, top_n=20)

    assert rel["q1"] == {"cf": 0.99, "n1": 0.5}
    assert corr["q1"] == {"cf": 0.0, "n1": 1.0}
    assert needles == {"q1": "n1"}


def test_build_corroboration_runs_skips_queries_without_a_needle():
    task = _task([NiahExample(query_id="q1", query="a", needle_ids=[], needle_id=None)])
    retr = FakeRetriever({"a": [RetrievedChunk("d0", "", 0.9)]})
    scorer = FakeScorer(rel_by_id={"d0": 0.9}, corr_by_id={"d0": 0.0})

    rel, corr, needles = build_corroboration_runs(task, retr, scorer, top_n=20)

    assert needles == {} and rel == {} and corr == {}


def test_needle_found_at_k_checks_the_designated_needle():
    fused = {"q1": {"n1": 0.9, "d0": 0.5}, "q2": {"d0": 0.9, "n2": 0.1}}
    needles = {"q1": "n1", "q2": "n2"}
    assert needle_found_at_k(fused, needles, k=1) == 0.5   # q1 hit (n1 top), q2 miss (n2 rank2)


def test_sweep_shows_corroboration_recovers_a_buried_needle():
    # relevance ranks the counterfactual above the needle; corroboration ranks the needle first.
    rel = {"q1": {"cf": 0.99, "n1": 0.5}}
    corr = {"q1": {"cf": 0.0, "n1": 1.0}}
    needles = {"q1": "n1"}

    curve = dict(sweep_corroboration(rel, corr, needles, grid=[0.0, 0.5, 1.0], k=1))

    assert curve[1.0] == 0.0   # alpha=1 pure relevance -> counterfactual wins -> needle missed
    assert curve[0.0] == 1.0   # alpha=0 pure corroboration -> needle wins -> found


def test_best_alpha_breaks_ties_toward_larger_alpha():
    # tie -> prefer more relevance (less reliance on the novel corroboration signal)
    assert best_alpha([(0.0, 0.5), (0.5, 0.5), (1.0, 0.5)]) == (1.0, 0.5)
    assert best_alpha([(0.0, 0.9), (0.5, 0.5)]) == (0.0, 0.9)


def test_parse_args_defaults():
    args = _parse_args(["--task", "results/x.json"])
    assert args.task.name == "x.json"
    assert args.first_stage == "q2d_granite"
    assert args.k == 10
