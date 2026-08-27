from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate, Query
from evidence_rag.selector.top_k import TopKSelector


def candidate(evidence_id: str, score: float, rank: int) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=f"doc-{evidence_id}",
        chunk_id=f"chunk-{evidence_id}",
        text=f"text {evidence_id}",
        source_uri=f"fixture://{evidence_id}",
        retrieval_score=score,
        retrieval_rank=rank,
    )


def test_selector_returns_only_id_score_and_rank() -> None:
    candidates = CandidateSet(
        query_id="q-1",
        candidates=(candidate("ev-low", 0.2, 2), candidate("ev-high", 0.9, 1)),
    )
    result = TopKSelector().select(
        Query(query_id="q-1", text="question"),
        candidates,
        max_selected=1,
    )
    assert result.items[0].evidence_id == "ev-high"
    assert result.items[0].selection_rank == 1
    assert not hasattr(result.items[0], "text")
    assert not hasattr(result, "status")


def test_empty_candidates_return_empty_selection() -> None:
    result = TopKSelector().select(
        Query(query_id="q-2", text="question"),
        CandidateSet(query_id="q-2", candidates=()),
        max_selected=2,
    )
    assert result.items == ()


def test_topk_preserves_frozen_rank_when_scores_disagree() -> None:
    candidates = CandidateSet(
        query_id="q-3",
        candidates=(candidate("rank-one", 0.1, 1), candidate("rank-two", 0.9, 2)),
    )
    result = TopKSelector().select(
        Query(query_id="q-3", text="question"), candidates, max_selected=2
    )
    assert [item.evidence_id for item in result.items] == ["rank-one", "rank-two"]
