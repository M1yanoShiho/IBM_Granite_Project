"""Tests for the offline lambda-sweep of the corroboration reranker."""
from eval.tune_corroboration import (
    _parse_args,
    best_alpha,
    build_corroboration_runs,
    dedup_docs,
    dump_runs,
    load_runs,
    needle_found_at_k,
    per_query_hits,
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


def test_build_corroboration_runs_caps_at_max_queries():
    task = _task([
        NiahExample(query_id="q1", query="a", needle_ids=["n1"], needle_id="n1"),
        NiahExample(query_id="q2", query="b", needle_ids=["n2"], needle_id="n2"),
    ])
    retr = FakeRetriever({"a": [RetrievedChunk("n1", "", 0.9)], "b": [RetrievedChunk("n2", "", 0.9)]})
    scorer = FakeScorer(rel_by_id={"n1": 0.9, "n2": 0.9}, corr_by_id={"n1": 0.0, "n2": 0.0})

    rel, corr, needles = build_corroboration_runs(task, retr, scorer, top_n=20, max_queries=1)

    assert set(needles) == {"q1"}          # only the first query processed
    assert set(rel) == {"q1"} and set(corr) == {"q1"}


def test_parse_args_defaults():
    args = _parse_args(["--task", "results/x.json"])
    assert args.task.name == "x.json"
    assert args.first_stage == "q2d_granite"
    assert args.k == 10
    assert args.max_queries is None


def test_parse_args_max_queries():
    assert _parse_args(["--task", "x.json", "--max-queries", "100"]).max_queries == 100


def test_per_query_hits_flags_each_designated_needle():
    # Per-query 0/1 needle-found (the input to a paired significance test), vs the
    # scalar mean that needle_found_at_k returns.
    fused = {"q1": {"n1": 0.9, "d0": 0.5}, "q2": {"d0": 0.9, "n2": 0.1}}
    needles = {"q1": "n1", "q2": "n2"}
    assert per_query_hits(fused, needles, k=1) == {"q1": 1.0, "q2": 0.0}


def test_dump_and_load_runs_round_trip(tmp_path):
    # The extraction is the expensive step; dumping the two runs lets significance /
    # re-sweeps run offline (--from-runs) without re-extracting.
    rel = {"q1": {"n1": 0.9, "cf": 0.99}}
    corr = {"q1": {"n1": 1.0, "cf": 0.0}}
    needles = {"q1": "n1"}
    path = tmp_path / "runs.json"

    dump_runs(rel, corr, needles, path)

    assert load_runs(path) == (rel, corr, needles)


def test_per_query_hits_tiebreak_is_deterministic_by_doc_id():
    # An exact score tie must resolve identically regardless of dict insertion order
    # (it previously fell back to hash-seed-dependent order). Rule: score desc, doc_id asc.
    a_first = {"q": {"a": 1.0, "z": 1.0}}
    z_first = {"q": {"z": 1.0, "a": 1.0}}
    # 'z' loses the tie to 'a' (smaller doc_id ranks first) -> needle 'z' missed at k=1
    assert per_query_hits(a_first, {"q": "z"}, k=1) == {"q": 0.0}
    assert per_query_hits(z_first, {"q": "z"}, k=1) == {"q": 0.0}
    # 'a' wins the tie -> needle 'a' found
    assert per_query_hits(a_first, {"q": "a"}, k=1) == {"q": 1.0}
