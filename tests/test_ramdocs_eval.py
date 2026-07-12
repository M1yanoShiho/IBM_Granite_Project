"""Tests for eval/ramdocs_eval.py — V4 external validation on RAMDocs (COLM 2025).

RAMDocs supplies per-query document SETS (correct / misinfo / noise labelled), so
corroboration reranks them directly — no index. The pre-registered question: does the
frozen alpha=0.6 relevance+corroboration blend rank correct-evidence docs above
misinformation better than relevance alone, on a benchmark WE did not construct?
"""
import json

import pytest

from eval.ramdocs_eval import (
    correct_at_k,
    load_ramdocs,
    misinfo_at_1,
    rank_order,
    rr_first_correct,
)


def test_load_ramdocs_parses_jsonl_schema(tmp_path):
    line = {
        "question": "who is X?",
        "documents": [
            {"text": "doc a", "type": "correct", "answer": "Alice"},
            {"text": "doc b", "type": "misinfo", "answer": "Bob"},
            {"text": "doc c", "type": "noise", "answer": "unknown"},
        ],
        "disambig_entity": ["X (person)"],
        "gold_answers": ["Alice"],
        "wrong_answers": ["Bob"],
    }
    p = tmp_path / "RAMDocs_test.jsonl"
    p.write_text(json.dumps(line) + "\n", encoding="utf-8")

    examples = load_ramdocs(p)

    assert len(examples) == 1
    assert examples[0]["question"] == "who is X?"
    assert [d["type"] for d in examples[0]["documents"]] == ["correct", "misinfo", "noise"]


def test_load_ramdocs_fails_loud_on_missing_keys(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"question": "q"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_ramdocs(p)


def test_rank_order_alpha1_is_pure_relevance():
    # alpha=1.0 must reproduce the relevance order (consistency check, same as the
    # tune_corroboration convention: alpha = relevance weight).
    order = rank_order(relevance=[0.2, 0.9, 0.5], votes=[5.0, 0.0, 1.0], alpha=1.0)
    assert order == [1, 2, 0]


def test_rank_order_blend_demotes_lone_high_relevance_misinfo():
    # The mechanism claim, ported to RAMDocs: a lone misinfo doc with the HIGHEST
    # relevance but zero corroboration falls below two agreeing correct docs.
    relevance = [0.99, 0.90, 0.80]     # doc0 = misinfo, doc1/doc2 = correct (agree)
    votes = [0.0, 1.0, 1.0]            # corroboration_scores counts of agreeing others
    order = rank_order(relevance, votes, alpha=0.6)
    assert order[0] in {1, 2}          # a corroborated correct doc leads
    assert order.index(0) > 0          # the misinfo doc lost the top spot


def test_metrics_over_ranked_types():
    ranked = ["misinfo", "correct", "noise"]
    assert correct_at_k(ranked, 1) == 0.0
    assert correct_at_k(ranked, 3) == 1.0
    assert misinfo_at_1(ranked) == 1.0
    assert rr_first_correct(ranked) == pytest.approx(0.5)
    assert rr_first_correct(["noise", "noise"]) == 0.0
