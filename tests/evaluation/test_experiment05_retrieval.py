from __future__ import annotations

import numpy as np
import pytest

from evidence_rag.evaluation.experiment05_retrieval import (
    RapidRetrievalConfig,
    build_rapid_retrieval_trace,
    protocol_identity,
    validate_rapid_retrieval_trace,
)


class FakeEncoder:
    def encode(self, texts, **kwargs):  # noqa: ANN001, ANN003
        vectors = {
            "question": [1.0, 0.0],
            "alpha": [1.0, 0.0],
            "beta": [0.8, 0.2],
            "gamma": [0.0, 1.0],
        }
        return np.asarray([vectors[text] for text in texts], dtype="float32")


def _candidate(identifier: str, contents: str, rank: int) -> dict[str, object]:
    return {
        "id": identifier,
        "contents": contents,
        "title": identifier,
        "source_kind": "test",
        "source_id": identifier,
        "start_unit": None,
        "end_unit": None,
        "bm25_rank": rank,
        "bm25_score": float(4 - rank),
    }


def test_two_stage_trace_freezes_bm25_and_hybrid_inputs() -> None:
    config = RapidRetrievalConfig(
        candidate_pool_size=3,
        final_top_k=2,
        reranker_pool_size=3,
        rrf_k=60,
        dense_batch_size=2,
    )
    trace = build_rapid_retrieval_trace(
        dataset="kilt-nq",
        query_id="q1",
        question="question",
        bm25_candidates=(
            _candidate("a", "gamma", 1),
            _candidate("b", "alpha", 2),
            _candidate("c", "beta", 3),
        ),
        encoder=FakeEncoder(),
        config=config,
    )

    assert trace["bm25_top10_ids"] == ["a", "b"]
    assert trace["hybrid_top40_ids"] == ["b", "a", "c"]
    assert set(trace["materialized_evidence"]) == {"a", "b", "c"}
    validate_rapid_retrieval_trace(trace)


def test_trace_rejects_noncontiguous_bm25_ranks() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        build_rapid_retrieval_trace(
            dataset="kilt-nq",
            query_id="q1",
            question="question",
            bm25_candidates=(_candidate("a", "alpha", 2),),
            encoder=FakeEncoder(),
            config=RapidRetrievalConfig(
                candidate_pool_size=2,
                final_top_k=1,
                reranker_pool_size=1,
                dense_batch_size=1,
            ),
        )


def test_protocol_identity_changes_with_index_identity() -> None:
    config = RapidRetrievalConfig()
    first = protocol_identity(
        config=config,
        corpus_manifest_sha256="a" * 64,
        bm25_index_tree_sha256="b" * 64,
        granite_model_snapshot_sha256="c" * 64,
    )
    second = protocol_identity(
        config=config,
        corpus_manifest_sha256="a" * 64,
        bm25_index_tree_sha256="d" * 64,
        granite_model_snapshot_sha256="c" * 64,
    )
    assert first.startswith("rapid-retrieval-sha256:")
    assert first != second
