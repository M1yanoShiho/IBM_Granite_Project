"""Two-stage full-corpus retrieval for Experiment 05 rapid execution.

The sparse first stage searches the complete frozen BM25 index.  Granite embeddings are
computed only for that query's frozen candidate set; no gold data enters this module.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

RAPID_RETRIEVAL_SCHEMA_VERSION = "experiment05.rapid_retrieval.v1"


@dataclass(frozen=True)
class RapidRetrievalConfig:
    candidate_pool_size: int = 1_000
    final_top_k: int = 10
    reranker_pool_size: int = 40
    rrf_k: int = 60
    dense_batch_size: int = 512

    def validate(self) -> None:
        if min(asdict(self).values()) <= 0:
            raise ValueError("rapid retrieval parameters must be positive")
        if self.final_top_k > self.reranker_pool_size:
            raise ValueError("final_top_k cannot exceed reranker_pool_size")
        if self.reranker_pool_size > self.candidate_pool_size:
            raise ValueError("reranker_pool_size cannot exceed candidate_pool_size")


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def protocol_identity(
    *,
    config: RapidRetrievalConfig,
    corpus_manifest_sha256: str,
    bm25_index_tree_sha256: str,
    granite_model_snapshot_sha256: str,
) -> str:
    config.validate()
    return "rapid-retrieval-sha256:" + canonical_sha256(
        {
            "schema_version": RAPID_RETRIEVAL_SCHEMA_VERSION,
            "config": asdict(config),
            "corpus_manifest_sha256": corpus_manifest_sha256,
            "bm25_index_tree_sha256": bm25_index_tree_sha256,
            "granite_model_snapshot_sha256": granite_model_snapshot_sha256,
            "dense_metric": "cosine_on_l2_normalized_vectors",
            "candidate_source": "full_corpus_bm25_only_no_gold_injection",
        }
    )


def _normalise(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype="float32")
    if values.ndim != 2:
        raise ValueError("encoder output must be a matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("encoder returned a zero vector")
    return np.ascontiguousarray(values / norms, dtype="float32")


def _encode(encoder: Any, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
    values = encoder.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    matrix = _normalise(np.asarray(values, dtype="float32"))
    if matrix.shape[0] != len(texts):
        raise ValueError("encoder returned a different number of rows than requested")
    return matrix


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    value = candidate.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError("candidate has no non-empty id")
    return value


def _candidate_contents(candidate: Mapping[str, Any]) -> str:
    value = candidate.get("contents")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("candidate has no non-empty contents")
    return value


def build_rapid_retrieval_trace(
    *,
    dataset: str,
    query_id: str,
    question: str,
    bm25_candidates: Sequence[Mapping[str, Any]],
    encoder: Any,
    config: RapidRetrievalConfig | None = None,
) -> dict[str, Any]:
    """Dense-score a BM25 candidate set and freeze BM25/hybrid system inputs."""

    config = config or RapidRetrievalConfig()
    config.validate()
    if not dataset or not query_id or not question.strip():
        raise ValueError("dataset, query_id, and question must be non-empty")
    if not bm25_candidates:
        raise ValueError("BM25 returned no candidates")
    if len(bm25_candidates) > config.candidate_pool_size:
        raise ValueError("BM25 candidate set exceeds the frozen pool size")

    candidate_ids = [_candidate_id(candidate) for candidate in bm25_candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("BM25 candidate set contains duplicate IDs")
    for expected_rank, candidate in enumerate(bm25_candidates, start=1):
        if int(candidate.get("bm25_rank", -1)) != expected_rank:
            raise ValueError("BM25 ranks must be contiguous and ordered")
        _candidate_contents(candidate)

    query_vector = _encode(encoder, (question,), batch_size=1)[0]
    document_vectors = _encode(
        encoder,
        tuple(_candidate_contents(candidate) for candidate in bm25_candidates),
        batch_size=config.dense_batch_size,
    )
    dense_scores = document_vectors @ query_vector
    dense_order = sorted(
        range(len(bm25_candidates)),
        key=lambda index: (
            -float(dense_scores[index]),
            int(bm25_candidates[index]["bm25_rank"]),
            candidate_ids[index],
        ),
    )
    dense_rank_by_index = {index: rank for rank, index in enumerate(dense_order, start=1)}
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(bm25_candidates):
        bm25_rank = int(candidate["bm25_rank"])
        dense_rank = dense_rank_by_index[index]
        rows.append(
            {
                "evidence_id": candidate_ids[index],
                "contents_sha256": hashlib.sha256(
                    _candidate_contents(candidate).encode("utf-8")
                ).hexdigest(),
                "bm25_rank": bm25_rank,
                "bm25_score": float(candidate["bm25_score"]),
                "dense_rank": dense_rank,
                "dense_score": float(dense_scores[index]),
                "rrf_score": (1.0 / (config.rrf_k + bm25_rank))
                + (1.0 / (config.rrf_k + dense_rank)),
            }
        )

    hybrid_rows = sorted(
        rows,
        key=lambda row: (
            -float(row["rrf_score"]),
            int(row["bm25_rank"]),
            int(row["dense_rank"]),
            str(row["evidence_id"]),
        ),
    )
    bm25_top10_ids = candidate_ids[: config.final_top_k]
    hybrid_top40_ids = [
        str(row["evidence_id"])
        for row in hybrid_rows[: config.reranker_pool_size]
    ]
    materialized_ids = set(bm25_top10_ids) | set(hybrid_top40_ids)
    materialized = {
        candidate_ids[index]: {
            key: candidate.get(key)
            for key in (
                "id",
                "contents",
                "title",
                "source_kind",
                "source_id",
                "start_unit",
                "end_unit",
            )
        }
        for index, candidate in enumerate(bm25_candidates)
        if candidate_ids[index] in materialized_ids
    }
    candidate_audit = sorted(rows, key=lambda row: int(row["bm25_rank"]))
    return {
        "schema_version": RAPID_RETRIEVAL_SCHEMA_VERSION,
        "dataset": dataset,
        "query_id": query_id,
        "config": asdict(config),
        "candidate_count": len(candidate_audit),
        "candidate_pool_sha256": canonical_sha256(candidate_audit),
        "candidate_audit": candidate_audit,
        "bm25_top10_ids": bm25_top10_ids,
        "hybrid_top40_ids": hybrid_top40_ids,
        "materialized_evidence": materialized,
    }


def validate_rapid_retrieval_trace(trace: Mapping[str, Any]) -> None:
    if trace.get("schema_version") != RAPID_RETRIEVAL_SCHEMA_VERSION:
        raise ValueError("rapid retrieval trace schema differs")
    audit = trace.get("candidate_audit")
    if not isinstance(audit, list) or len(audit) != trace.get("candidate_count"):
        raise ValueError("rapid retrieval candidate count differs")
    if canonical_sha256(audit) != trace.get("candidate_pool_sha256"):
        raise ValueError("rapid retrieval candidate pool hash differs")
    ids = [str(row.get("evidence_id", "")) for row in audit]
    if not ids or len(ids) != len(set(ids)) or any(not item for item in ids):
        raise ValueError("rapid retrieval candidate IDs are invalid")
    materialized = trace.get("materialized_evidence")
    if not isinstance(materialized, Mapping):
        raise ValueError("rapid retrieval materialized evidence is missing")
    required_ids = set(trace.get("bm25_top10_ids", ())) | set(
        trace.get("hybrid_top40_ids", ())
    )
    if not required_ids <= set(materialized):
        raise ValueError("system input evidence is not fully materialized")
