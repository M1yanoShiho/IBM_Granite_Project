import pytest

from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate, Query
from evidence_rag.retriever.fusion import (
    best_rank_fusion,
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


def test_best_rank_fusion_keeps_a_single_arm_specialist_on_top() -> None:
    # The measured multi-hop failure in one shape: "spec" is first in one arm and absent
    # from the others; "broad" is mid-ranked in all three. RRF's sum lets breadth win.
    spec = candidate("ev-spec", "d-spec", 9.0, 1)
    broad_ranks = (5, 5, 5)
    arms = tuple(
        CandidateSet(
            query_id="q",
            candidates=(
                (spec,) if index == 0 else ()
            )
            + (candidate("ev-broad", "d-broad", 1.0, rank),),
        )
        for index, rank in enumerate(broad_ranks)
    )

    rrf = reciprocal_rank_fusion(arms, query_id="q", top_k=5)
    best = best_rank_fusion(arms, query_id="q", top_k=5)

    assert rrf.candidates[0].evidence_id == "ev-broad"
    assert best.candidates[0].evidence_id == "ev-spec"


def test_best_rank_fusion_breaks_best_rank_ties_on_breadth_not_alphabet() -> None:
    # Two documents each placed first by some arm tie on best rank. Falling through to
    # evidence_id would make rank 1 arbitrary, so the summed RRF score decides: "ev-b"
    # is first in one arm and also present in the other, "ev-a" only in one.
    arm_a = CandidateSet(
        query_id="q",
        candidates=(candidate("ev-a", "da", 9.0, 1), candidate("ev-b", "db", 8.0, 2)),
    )
    arm_b = CandidateSet(query_id="q", candidates=(candidate("ev-b", "db", 9.0, 1),))

    fused = best_rank_fusion((arm_a, arm_b), query_id="q", top_k=5)

    assert [c.evidence_id for c in fused.candidates] == ["ev-b", "ev-a"]


def one_arm_gold_versus_two_arm_distractor() -> tuple[CandidateSet, ...]:
    # The R2 dilution shape: gold is placed first by exactly one arm (the original
    # query) and absent from the N sub-query arms, each of which places the same
    # distractor first. Summed at parity, N votes beat 1.
    return (
        CandidateSet(query_id="q", candidates=(candidate("ev-gold", "d-gold", 9.0, 1),)),
        CandidateSet(query_id="q", candidates=(candidate("ev-distractor", "d-x", 9.0, 1),)),
        CandidateSet(query_id="q", candidates=(candidate("ev-distractor", "d-x", 9.0, 1),)),
    )


def test_rrf_weights_let_one_arm_outvote_the_majority() -> None:
    arms = one_arm_gold_versus_two_arm_distractor()

    parity = reciprocal_rank_fusion(arms, query_id="q", top_k=5)
    weighted = reciprocal_rank_fusion(arms, query_id="q", top_k=5, weights=(3.0, 1.0, 1.0))

    assert parity.candidates[0].evidence_id == "ev-distractor"
    assert weighted.candidates[0].evidence_id == "ev-gold"


def test_best_rank_fusion_applies_weights_too() -> None:
    # Both fusions are dispatched through one signature, so weights must reach either.
    # At parity the two tie on best rank and the summed secondary key hands rank 1 to
    # the twice-seen distractor; weighting the gold arm makes its placement decisive.
    arms = one_arm_gold_versus_two_arm_distractor()

    parity = best_rank_fusion(arms, query_id="q", top_k=5)
    weighted = best_rank_fusion(arms, query_id="q", top_k=5, weights=(3.0, 1.0, 1.0))

    assert parity.candidates[0].evidence_id == "ev-distractor"
    assert weighted.candidates[0].evidence_id == "ev-gold"


def test_equal_weights_reproduce_the_unweighted_fusions() -> None:
    # Guards every recorded result: passing weights explicitly must not perturb the
    # default path, so an all-ones sweep point stays comparable with the baseline.
    arms = one_arm_gold_versus_two_arm_distractor()
    ones = (1.0, 1.0, 1.0)

    for fuse in (reciprocal_rank_fusion, best_rank_fusion):
        default = fuse(arms, query_id="q", top_k=5)
        explicit = fuse(arms, query_id="q", top_k=5, weights=ones)
        assert [(c.evidence_id, c.retrieval_score) for c in default.candidates] == [
            (c.evidence_id, c.retrieval_score) for c in explicit.candidates
        ]


def test_fusion_weights_must_match_the_number_of_arms() -> None:
    arms = one_arm_gold_versus_two_arm_distractor()
    with pytest.raises(ValueError, match="one entry per result list"):
        reciprocal_rank_fusion(arms, query_id="q", top_k=5, weights=(1.0, 1.0))


def test_best_rank_fusion_ordering_is_insensitive_to_k() -> None:
    # max of a monotone function of rank is equivalent to min rank, so k only rescales
    # the recorded score. Documented in best_rank_fusion; guarded here so nobody sweeps it.
    arms = (
        CandidateSet(query_id="q", candidates=(candidate("ev-1", "d1", 9.0, 1),)),
        CandidateSet(query_id="q", candidates=(candidate("ev-2", "d2", 9.0, 3),)),
    )
    low = best_rank_fusion(arms, query_id="q", top_k=5, k=1)
    high = best_rank_fusion(arms, query_id="q", top_k=5, k=1000)
    assert [c.evidence_id for c in low.candidates] == [
        c.evidence_id for c in high.candidates
    ]


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
