"""Runtime-safe system preparation contracts for Experiment 05."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate, Query
from evidence_rag.evaluation.experiment05_data import validate_runtime_record
from evidence_rag.evaluation.experiment05_retrieval import validate_rapid_retrieval_trace
from evidence_rag.retriever.rerank import PairReranker
from evidence_rag.selector.provence import PassagePruner
from evidence_rag.selector.threshold_only import (
    NliThresholdOnlySelector,
    ThresholdSelectionTrace,
)

MAIN_ARMS = (
    "bm25_rag",
    "hybrid_rag",
    "granite_rerank_rag",
    "provence_rag",
    "ours_seed13",
    "ours_seed42",
    "ours_seed73",
)
ABLATION_ARMS = (
    "ablation_bm25_retriever",
    "ablation_no_selector",
    "ablation_direct_generator",
)
ALL_ARMS = MAIN_ARMS + ABLATION_ARMS
PREPARED_SCHEMA_VERSION: Literal["experiment05.prepared.v1"] = "experiment05.prepared.v1"

NonEmpty = Annotated[str, Field(min_length=1)]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PreparedEvidence(_FrozenModel):
    evidence_id: NonEmpty
    text: NonEmpty
    retrieval_score: float
    retrieval_rank: int = Field(ge=1)
    source_kind: str | None = None
    source_id: str | None = None
    start_unit: int | None = None
    end_unit: int | None = None


class PreparedArm(_FrozenModel):
    retrieved_evidence_ids: tuple[NonEmpty, ...]
    selected: tuple[PreparedEvidence, ...]
    selection_status: Literal["normal", "fail_open_all"] = "normal"

    @model_validator(mode="after")
    def selected_is_retrieved(self) -> PreparedArm:
        selected_ids = tuple(item.evidence_id for item in self.selected)
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("selected evidence IDs must be unique")
        if not set(selected_ids) <= set(self.retrieved_evidence_ids):
            raise ValueError("selected evidence must be a subset of retrieved evidence")
        return self


class PreparedQuery(_FrozenModel):
    schema_version: Literal["experiment05.prepared.v1"] = PREPARED_SCHEMA_VERSION
    dataset: NonEmpty
    query_id: NonEmpty
    arms: dict[NonEmpty, PreparedArm]
    selector_traces: dict[Literal["hybrid", "bm25"], ThresholdSelectionTrace]

    @model_validator(mode="after")
    def exact_arm_contract(self) -> PreparedQuery:
        if set(self.arms) != set(ALL_ARMS):
            raise ValueError("prepared query must contain exactly ten Experiment 05 arms")
        full = self.arms["ours_seed13"].model_dump_json()
        if any(self.arms[arm].model_dump_json() != full for arm in ("ours_seed42", "ours_seed73")):
            raise ValueError("the three Ours seeds must share frozen upstream evidence")
        if self.arms["ablation_direct_generator"].model_dump_json() != full:
            raise ValueError("Direct Generator ablation must share Full upstream evidence")
        if self.arms["ablation_no_selector"].model_dump_json() != self.arms[
            "hybrid_rag"
        ].model_dump_json():
            raise ValueError("no-Selector ablation must share Hybrid keep-all evidence")
        return self


def _candidate(
    *,
    query_id: str,
    evidence_id: str,
    rank: int,
    score: float,
    materialized: Mapping[str, Any],
) -> EvidenceCandidate:
    contents = str(materialized["contents"]).strip()
    if not contents:
        raise ValueError("materialized evidence text is empty")
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=evidence_id,
        chunk_id=evidence_id,
        text=contents,
        source_uri=f"sealed-retrieval://{query_id}/{evidence_id}",
        retrieval_score=score,
        retrieval_rank=rank,
    )


def _candidate_set(
    trace: Mapping[str, Any], *, ids: Sequence[str], score_name: str
) -> CandidateSet:
    query_id = str(trace["query_id"])
    audit = {str(row["evidence_id"]): row for row in trace["candidate_audit"]}
    materialized = trace["materialized_evidence"]
    return CandidateSet(
        query_id=query_id,
        candidates=tuple(
            _candidate(
                query_id=query_id,
                evidence_id=evidence_id,
                rank=rank,
                score=float(audit[evidence_id][score_name]),
                materialized=materialized[evidence_id],
            )
            for rank, evidence_id in enumerate(ids, start=1)
        ),
    )


def _rerank(
    query: Query, candidates: CandidateSet, reranker: PairReranker
) -> CandidateSet:
    scores = tuple(float(value) for value in reranker.score(
        query.text, tuple(item.text for item in candidates.candidates)
    ))
    if len(scores) != len(candidates.candidates):
        raise ValueError("reranker returned a different number of scores than passages")
    ranked = sorted(
        zip(scores, candidates.candidates, strict=True),
        key=lambda item: (-item[0], item[1].retrieval_rank, item[1].evidence_id),
    )
    return CandidateSet(
        query_id=query.query_id,
        candidates=tuple(
            candidate.model_copy(update={"retrieval_score": score, "retrieval_rank": rank})
            for rank, (score, candidate) in enumerate(ranked[:10], start=1)
        ),
    )


def _prepared(candidate: EvidenceCandidate, materialized: Mapping[str, Any], *, text: str | None = None) -> PreparedEvidence:
    return PreparedEvidence(
        evidence_id=candidate.evidence_id,
        text=text if text is not None else candidate.text,
        retrieval_score=float(candidate.retrieval_score),
        retrieval_rank=candidate.retrieval_rank,
        source_kind=materialized.get("source_kind"),
        source_id=None if materialized.get("source_id") is None else str(materialized["source_id"]),
        start_unit=materialized.get("start_unit"),
        end_unit=materialized.get("end_unit"),
    )


def _keep_all(candidates: CandidateSet, materialized: Mapping[str, Mapping[str, Any]]) -> tuple[PreparedEvidence, ...]:
    return tuple(_prepared(item, materialized[item.evidence_id]) for item in candidates.candidates)


def _selected(
    candidates: CandidateSet,
    selected_ids: Sequence[str],
    materialized: Mapping[str, Mapping[str, Any]],
) -> tuple[PreparedEvidence, ...]:
    selected = set(selected_ids)
    return tuple(
        _prepared(item, materialized[item.evidence_id])
        for item in candidates.candidates
        if item.evidence_id in selected
    )


def _arm(
    retrieved: CandidateSet,
    selected: Sequence[PreparedEvidence],
    *,
    status: Literal["normal", "fail_open_all"] = "normal",
) -> PreparedArm:
    return PreparedArm(
        retrieved_evidence_ids=tuple(item.evidence_id for item in retrieved.candidates),
        selected=tuple(selected),
        selection_status=status,
    )


def _title_body(materialized: Mapping[str, Any]) -> tuple[str, str]:
    title = str(materialized.get("title") or "").strip()
    contents = str(materialized["contents"]).strip()
    prefix = f"{title}\n"
    body = contents[len(prefix) :] if title and contents.startswith(prefix) else contents
    return title, body


def prepare_query(
    runtime: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    reranker: PairReranker,
    provence: PassagePruner,
    selector: NliThresholdOnlySelector,
) -> PreparedQuery:
    """Prepare all ten arms from one shared gold-free rapid-retrieval trace."""

    validate_runtime_record(runtime)
    validate_rapid_retrieval_trace(trace)
    if runtime["dataset"] != trace["dataset"] or runtime["query_id"] != trace["query_id"]:
        raise ValueError("runtime and retrieval trace identities differ")
    query = Query(query_id=str(runtime["query_id"]), text=str(runtime["question"]))
    materialized = trace["materialized_evidence"]
    bm25 = _candidate_set(trace, ids=trace["bm25_top10_ids"], score_name="bm25_score")
    hybrid40 = _candidate_set(trace, ids=trace["hybrid_top40_ids"], score_name="rrf_score")
    hybrid = CandidateSet(query_id=query.query_id, candidates=hybrid40.candidates[:10])
    reranked = _rerank(query, hybrid40, reranker)

    hybrid_result, hybrid_trace = selector.select_with_trace(query, hybrid, 10)
    bm25_result, bm25_trace = selector.select_with_trace(query, bm25, 10)
    hybrid_selected = _selected(
        hybrid, tuple(item.evidence_id for item in hybrid_result.items), materialized
    )
    bm25_selected = _selected(
        bm25, tuple(item.evidence_id for item in bm25_result.items), materialized
    )

    provence_selected: list[PreparedEvidence] = []
    for item in hybrid.candidates:
        title, body = _title_body(materialized[item.evidence_id])
        pruned = provence.prune(question=query.text, title=title, text=body)
        if pruned.strip():
            provence_selected.append(
                _prepared(item, materialized[item.evidence_id], text=pruned.strip())
            )

    bm25_keep = _arm(bm25, _keep_all(bm25, materialized))
    hybrid_keep = _arm(hybrid, _keep_all(hybrid, materialized))
    rerank_keep = _arm(reranked, _keep_all(reranked, materialized))
    provence_arm = _arm(hybrid, provence_selected)
    full = _arm(hybrid, hybrid_selected, status=hybrid_trace.status)
    bm25_ablation = _arm(bm25, bm25_selected, status=bm25_trace.status)
    arms = {
        "bm25_rag": bm25_keep,
        "hybrid_rag": hybrid_keep,
        "granite_rerank_rag": rerank_keep,
        "provence_rag": provence_arm,
        "ours_seed13": full,
        "ours_seed42": full,
        "ours_seed73": full,
        "ablation_bm25_retriever": bm25_ablation,
        "ablation_no_selector": hybrid_keep,
        "ablation_direct_generator": full,
    }
    return PreparedQuery(
        dataset=str(runtime["dataset"]),
        query_id=query.query_id,
        arms=arms,
        selector_traces={"hybrid": hybrid_trace, "bm25": bm25_trace},
    )


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
