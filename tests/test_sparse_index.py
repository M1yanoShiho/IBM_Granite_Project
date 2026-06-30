"""Tests for src/retrieval/sparse_index.py — scipy CSR sparse index + search.

Pure sparse-math on hand-built term-weight vectors; no model. Pins the dot-product
ranking, score>0 filtering, top_k, and the deterministic tie-break.
"""
from __future__ import annotations

import pytest

from src.retrieval.sparse_index import SparseIndex


def _index() -> SparseIndex:
    # vocab size 5; 3 docs.
    docs = [
        {0: 1.0, 1: 2.0},   # d0
        {1: 1.0, 2: 3.0},   # d1
        {3: 5.0},           # d2 (disjoint from a {0,1,2} query)
    ]
    return SparseIndex.build(docs, ["d0", "d1", "d2"], vocab_size=5)


def test_build_sets_shape_and_doc_ids() -> None:
    idx = _index()
    assert idx.matrix.shape == (3, 5)
    assert idx.doc_ids == ["d0", "d1", "d2"]


def test_search_ranks_by_sparse_dot_product() -> None:
    # query {1:1, 2:1}: d0 -> 2 ; d1 -> 1 + 3 = 4 ; d2 -> 0 (filtered)
    out = _index().search({1: 1.0, 2: 1.0}, top_k=10)
    assert out == [(1, pytest.approx(4.0)), (0, pytest.approx(2.0))]


def test_search_filters_zero_score_docs() -> None:
    out = _index().search({3: 1.0}, top_k=10)  # only d2 overlaps
    assert out == [(2, pytest.approx(5.0))]


def test_search_empty_query_returns_empty() -> None:
    assert _index().search({}, top_k=5) == []


def test_search_no_overlap_returns_empty() -> None:
    assert _index().search({4: 1.0}, top_k=5) == []  # term 4 in no doc


def test_search_respects_top_k() -> None:
    # d0 -> 3 ; d1 -> 4 ; top 1 = d1
    out = _index().search({0: 1.0, 1: 1.0, 2: 1.0}, top_k=1)
    assert out == [(1, pytest.approx(4.0))]


def test_search_deterministic_tie_break_to_smaller_index() -> None:
    idx = SparseIndex.build([{0: 1.0}, {0: 1.0}], ["a", "b"], vocab_size=2)
    out = idx.search({0: 1.0}, top_k=2)
    assert out == [(0, pytest.approx(1.0)), (1, pytest.approx(1.0))]


def test_build_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        SparseIndex.build([{0: 1.0}], ["a", "b"], vocab_size=2)


def test_search_rejects_bad_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        _index().search({0: 1.0}, top_k=0)


def test_save_load_round_trip(tmp_path) -> None:
    idx = _index()
    path = tmp_path / "splade_idx"
    idx.save(path)
    loaded = SparseIndex.load(path)
    assert loaded.doc_ids == idx.doc_ids
    assert loaded.vocab_size == idx.vocab_size
    query = {1: 1.0, 2: 1.0}
    assert loaded.search(query, top_k=10) == idx.search(query, top_k=10)
