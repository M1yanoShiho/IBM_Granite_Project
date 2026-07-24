import pytest

from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate, Query
from evidence_rag.retriever.fusion import (
    convex_fusion,
    min_max_normalise,
    reciprocal_rank_fusion,
)
from evidence_rag.retriever.hybrid import ConvexHybridRetriever, HybridRetriever


def candidate(evidence_id: str, doc: str, score: float, rank: int) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=doc,
        chunk_id=f"chunk-{evidence_id}",
        text=f"text for {evidence_id}",
        source_uri=f"fixture://{doc}",
        retrieval_score=score,
        retrieval_rank=rank,
    )


class StubRetriever:
    """Returns a fixed ranked list, ignoring the query text but echoing query_id."""

    def __init__(self, candidates: tuple[EvidenceCandidate, ...]) -> None:
        self._candidates = candidates
        self.received_top_k: list[int] = []

    def retrieve(self, query: Query, top_k: int) -> CandidateSet:
        self.received_top_k.append(top_k)
        return CandidateSet(
            query_id=query.query_id,
            candidates=self._candidates[:top_k],
        )


def test_rrf_rewards_agreement_across_arms() -> None:
    arm_a = StubRetriever((candidate("ev-1", "d1", 9.0, 1), candidate("ev-2", "d2", 8.0, 2)))
    arm_b = StubRetriever((candidate("ev-2", "d2", 5.0, 1), candidate("ev-3", "d3", 4.0, 2)))
    hybrid = HybridRetriever((arm_a, arm_b))

    result = hybrid.retrieve(Query(query_id="q", text="x"), top_k=3)

    # ev-2 appears in both arms so it fuses to the top; ranks are contiguous 1-based.
    assert result.candidates[0].evidence_id == "ev-2"
    assert tuple(c.retrieval_rank for c in result.candidates) == (1, 2, 3)
    assert {c.evidence_id for c in result.candidates} == {"ev-1", "ev-2", "ev-3"}


def test_hybrid_pool_size_overrides_requested_top_k() -> None:
    arm = StubRetriever((candidate("ev-1", "d1", 1.0, 1),))
    hybrid = HybridRetriever((arm,), pool_size=25)

    hybrid.retrieve(Query(query_id="q", text="x"), top_k=5)

    assert arm.received_top_k == [25]


def test_hybrid_requires_at_least_one_retriever() -> None:
    with pytest.raises(ValueError, match="at least one"):
        HybridRetriever(())


def test_hybrid_rejects_query_id_mismatch() -> None:
    class WrongIdRetriever:
        def retrieve(self, query: Query, top_k: int) -> CandidateSet:
            return CandidateSet(query_id="other", candidates=())

    hybrid = HybridRetriever((WrongIdRetriever(),))
    with pytest.raises(ValueError, match="wrong query ID"):
        hybrid.retrieve(Query(query_id="q", text="x"), top_k=3)


def test_rrf_fusion_helper_dedupes_and_reranks() -> None:
    a = CandidateSet(query_id="q", candidates=(candidate("ev-1", "d1", 9.0, 1),))
    b = CandidateSet(query_id="q", candidates=(candidate("ev-1", "d1", 1.0, 1),))
    fused = reciprocal_rank_fusion((a, b), query_id="q", top_k=5)
    assert len(fused.candidates) == 1
    assert fused.candidates[0].retrieval_rank == 1


def test_min_max_normalise_equal_scores_map_to_one() -> None:
    cands = (candidate("ev-1", "d1", 3.0, 1), candidate("ev-2", "d2", 3.0, 2))
    assert min_max_normalise(cands) == {"ev-1": 1.0, "ev-2": 1.0}


def test_min_max_normalise_spreads_scores() -> None:
    cands = (candidate("ev-1", "d1", 10.0, 1), candidate("ev-2", "d2", 0.0, 2))
    assert min_max_normalise(cands) == {"ev-1": 1.0, "ev-2": 0.0}


def test_convex_alpha_one_follows_dense_arm() -> None:
    sparse = StubRetriever((candidate("ev-1", "d1", 10.0, 1),))
    dense = StubRetriever(
        (candidate("ev-2", "d2", 10.0, 1), candidate("ev-1", "d1", 0.0, 2))
    )
    hybrid = ConvexHybridRetriever(sparse, dense, alpha=1.0)

    result = hybrid.retrieve(Query(query_id="q", text="x"), top_k=2)

    # alpha=1 ignores the sparse arm entirely, so dense's top wins.
    assert result.candidates[0].evidence_id == "ev-2"


def test_convex_alpha_zero_follows_sparse_arm() -> None:
    sparse = CandidateSet(
        query_id="q",
        candidates=(candidate("ev-1", "d1", 10.0, 1), candidate("ev-2", "d2", 0.0, 2)),
    )
    dense = CandidateSet(query_id="q", candidates=(candidate("ev-2", "d2", 10.0, 1),))
    fused = convex_fusion(sparse, dense, query_id="q", top_k=2, alpha=0.0)
    assert fused.candidates[0].evidence_id == "ev-1"


def test_convex_rejects_alpha_out_of_range() -> None:
    stub = StubRetriever((candidate("ev-1", "d1", 1.0, 1),))
    with pytest.raises(ValueError, match="alpha"):
        ConvexHybridRetriever(stub, stub, alpha=1.5)
