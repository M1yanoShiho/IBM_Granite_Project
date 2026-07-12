"""Tests for eval/anygold_rescore.py — V1 any-gold sensitivity re-score.

Answers the metric critique: does the certified story survive when a query counts
as a hit if ANY gold (needle_ids), not only the designated needle, reaches top-k?
"""
from eval.anygold_rescore import any_gold_hits, golds_from_recipe


def test_golds_from_recipe_maps_qid_to_full_gold_set():
    payload = {
        "recipe": {"dataset": "nq"},
        "examples": [
            {"query_id": "q1", "query": "a", "needle_id": "n1",
             "needle_ids": ["n1", "g2", "g3"], "distractors": []},
            {"query_id": "q2", "query": "b", "needle_id": None,
             "needle_ids": [], "distractors": []},
        ],
    }
    golds = golds_from_recipe(payload)
    assert golds["q1"] == {"n1", "g2", "g3"}
    assert golds["q2"] == set()


def test_any_gold_hit_when_non_designated_gold_is_in_top_k():
    # The exact scenario the critique describes: designated needle buried, but a
    # DIFFERENT gold made top-k -> designated metric says miss, any-gold says hit.
    run = {"q1": {"g2": 0.9, "d0": 0.8, "n1": 0.1}}
    golds = {"q1": {"n1", "g2"}}
    assert any_gold_hits(run, golds, k=2) == {"q1": 1.0}


def test_any_gold_miss_when_no_gold_in_top_k():
    run = {"q1": {"d0": 0.9, "d1": 0.8, "n1": 0.1}}
    golds = {"q1": {"n1", "g2"}}   # g2 not retrieved at all, n1 at rank 3
    assert any_gold_hits(run, golds, k=2) == {"q1": 0.0}


def test_any_gold_tiebreak_matches_designated_metric_rule():
    # Same deterministic tie-break as per_query_hits: score desc, then doc_id asc.
    # Gold 'z' ties with non-gold 'a' at k=1 -> 'a' wins the tie -> miss.
    run = {"q1": {"z": 1.0, "a": 1.0}}
    assert any_gold_hits(run, {"q1": {"z"}}, k=1) == {"q1": 0.0}
    assert any_gold_hits(run, {"q1": {"a"}}, k=1) == {"q1": 1.0}


def test_queries_missing_from_golds_score_zero_not_crash():
    run = {"q9": {"d0": 0.9}}
    assert any_gold_hits(run, {}, k=10) == {"q9": 0.0}
