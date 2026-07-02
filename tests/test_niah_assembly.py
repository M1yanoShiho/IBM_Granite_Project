# tests/test_niah_assembly.py
"""Tests for src/niah/assembly.py — haystack assembly + scale capping."""
from __future__ import annotations

from src.niah.assembly import cap_haystack, inject
from src.niah.types import Distractor


def test_inject_adds_distractor_docs() -> None:
    corpus = {"d1": "gold"}
    ds = [Distractor(doc_id="d1__cf0", text="wrong", source="counterfactual", parent_needle_id="d1")]
    out = inject(corpus, ds)
    assert out == {"d1": "gold", "d1__cf0": "wrong"}
    assert corpus == {"d1": "gold"}  # pure: input not mutated


def test_cap_haystack_always_keeps_needles_and_distractors() -> None:
    corpus = {f"bg{i}": "hay" for i in range(100)}
    corpus["gold1"] = "needle"
    corpus["gold1__cf0"] = "distractor"
    out = cap_haystack(
        corpus, keep_ids={"gold1", "gold1__cf0"}, max_docs=10, seed=0
    )
    assert "gold1" in out and "gold1__cf0" in out
    assert len(out) == 12  # 10 background + the 2 kept


def test_cap_haystack_is_deterministic_under_seed() -> None:
    corpus = {f"bg{i}": "hay" for i in range(100)}
    a = cap_haystack(corpus, keep_ids=set(), max_docs=5, seed=42)
    b = cap_haystack(corpus, keep_ids=set(), max_docs=5, seed=42)
    assert a == b


def test_cap_haystack_none_keeps_everything() -> None:
    corpus = {f"bg{i}": "hay" for i in range(10)}
    assert cap_haystack(corpus, keep_ids=set(), max_docs=None, seed=0) == corpus
