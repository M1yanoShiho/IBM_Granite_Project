"""Tests for eval/injection_analysis.py — V2 injection-removal + V3 natural conflict rate.

V2: arithmetically remove the INJECTED distractors (counterfactual + generative; mined
docs are natural corpus passages) from a dumped ranking and see where the needle lands
-> splits burial into injection-caused vs natural. V3: among the NATURAL docs of each
query's extraction pool, how often do valid extracted answers disagree -> does the
conflicting-evidence pathology exist without our construction?
"""
import pytest

from eval.injection_analysis import (
    classify_query,
    filter_run,
    injected_ids,
    natural_conflict,
)

RECIPE = {
    "recipe": {"dataset": "nq"},
    "examples": [
        {
            "query_id": "q1", "query": "a", "needle_id": "n1", "needle_ids": ["n1"],
            "distractors": [
                {"doc_id": "q1__n1__cf0", "text": "x", "source": "counterfactual",
                 "parent_needle_id": "n1"},
                {"doc_id": "q1__gen0", "text": "y", "source": "generative",
                 "parent_needle_id": "n1"},
                {"doc_id": "mined77", "text": "z", "source": "mined",
                 "parent_needle_id": ""},
            ],
        }
    ],
}


def test_injected_ids_collects_cf_and_gen_but_not_mined():
    # Mined distractors are real corpus docs (never injected) — removing them would
    # distort the natural corpus, so they stay.
    assert injected_ids(RECIPE) == {"q1__n1__cf0", "q1__gen0"}


def test_filter_run_removes_only_injected_and_preserves_scores():
    run = {"q1": {"q1__n1__cf0": 0.99, "n1": 0.5, "mined77": 0.4}}
    out = filter_run(run, {"q1__n1__cf0", "q1__gen0"})
    assert out == {"q1": {"n1": 0.5, "mined77": 0.4}}


def test_classify_found_injection_buried_natural_buried_unreachable():
    injected = {"cf"}
    # found: needle already in top-1
    assert classify_query({"n": 0.9, "cf": 0.1}, "n", injected, k=1) == "found"
    # injection_buried: cf outranks needle; removal rescues it into top-1
    assert classify_query({"cf": 0.9, "n": 0.8}, "n", injected, k=1) == "injection_buried"
    # natural_buried: a NATURAL doc outranks the needle even after removal
    assert (
        classify_query({"nat": 0.9, "cf": 0.85, "n": 0.8}, "n", injected, k=1)
        == "natural_buried"
    )
    # unreachable: needle not in the dumped ranking at all
    assert classify_query({"cf": 0.9}, "n", injected, k=1) == "unreachable"


def test_natural_conflict_counts_distinct_valid_answers_of_natural_docs_only():
    answers = {
        "d1": "Paris",          # natural, valid
        "d2": "the paris",      # natural, same answer after normalisation
        "cf": "Berlin",         # INJECTED -> must be excluded from the natural pool
        "d3": "NONE",           # invalid extraction -> ignored
    }
    res = natural_conflict(answers, injected={"cf"})
    assert res["n_natural_valid"] == 2
    assert res["n_distinct"] == 1
    assert res["conflict"] is False


def test_natural_conflict_true_when_natural_docs_disagree():
    answers = {"d1": "Paris", "d2": "Lyon", "d3": "Paris"}
    res = natural_conflict(answers, injected=set())
    assert res["n_distinct"] == 2
    assert res["conflict"] is True
    assert res["majority_share"] == pytest.approx(2 / 3)
