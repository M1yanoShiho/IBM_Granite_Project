"""Rank/score fusion primitives shared by the hybrid and decomposing retrievers.

These helpers take the ``CandidateSet`` outputs of several retrieval passes (either
different retrievers, or the same retriever over decomposed sub-queries) and merge
them into a single ranked ``CandidateSet`` with contiguous 1-based ranks.
"""

from collections.abc import Sequence

from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate

DEFAULT_RRF_K = 60


def _ranked(
    scored: dict[str, float],
    representative: dict[str, EvidenceCandidate],
    *,
    query_id: str,
    top_k: int,
) -> CandidateSet:
    # Sort by fused score desc, breaking ties on evidence_id for determinism
    # (mirrors the tie-break in BM25/dense retrievers).
    ordered = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
    candidates = tuple(
        representative[evidence_id].model_copy(
            update={"retrieval_score": score, "retrieval_rank": rank}
        )
        for rank, (evidence_id, score) in enumerate(ordered[:top_k], start=1)
    )
    return CandidateSet(query_id=query_id, candidates=candidates)


def reciprocal_rank_fusion(
    results: Sequence[CandidateSet],
    *,
    query_id: str,
    top_k: int,
    k: int = DEFAULT_RRF_K,
) -> CandidateSet:
    """Fuse ranked lists by Reciprocal Rank Fusion (Cormack et al., 2009).

    Each candidate contributes ``1 / (k + rank)`` from every list it appears in;
    ``k`` damps the weight of top ranks (larger ``k`` flattens the curve). RRF
    fuses *rankings*, so the arms' scores need not be comparable.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if k <= 0:
        raise ValueError("RRF k must be positive")
    scored: dict[str, float] = {}
    representative: dict[str, EvidenceCandidate] = {}
    for result in results:
        for candidate in result.candidates:
            scored[candidate.evidence_id] = scored.get(candidate.evidence_id, 0.0) + 1.0 / (
                k + candidate.retrieval_rank
            )
            representative.setdefault(candidate.evidence_id, candidate)
    return _ranked(scored, representative, query_id=query_id, top_k=top_k)


def min_max_normalise(candidates: Sequence[EvidenceCandidate]) -> dict[str, float]:
    """Per-list min-max normalise retrieval scores into ``[0, 1]`` by evidence_id.

    When every score is equal (including the single-candidate case), min-max is
    undefined, so each present candidate is mapped to ``1.0``.
    """

    if not candidates:
        return {}
    scores = [candidate.retrieval_score for candidate in candidates]
    low, high = min(scores), max(scores)
    span = high - low
    if span == 0.0:
        return {candidate.evidence_id: 1.0 for candidate in candidates}
    return {
        candidate.evidence_id: (candidate.retrieval_score - low) / span
        for candidate in candidates
    }


def convex_fusion(
    sparse: CandidateSet,
    dense: CandidateSet,
    *,
    query_id: str,
    top_k: int,
    alpha: float,
) -> CandidateSet:
    """Fuse two arms by a convex combination of per-arm min-max scores.

    ``score = (1 - alpha) * sparse_norm + alpha * dense_norm``; ``alpha`` is the
    dense weight. Candidates missing from an arm contribute ``0`` there.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("convex alpha must satisfy 0 <= alpha <= 1")
    sparse_norm = min_max_normalise(sparse.candidates)
    dense_norm = min_max_normalise(dense.candidates)
    representative: dict[str, EvidenceCandidate] = {}
    for candidate in sparse.candidates:
        representative.setdefault(candidate.evidence_id, candidate)
    for candidate in dense.candidates:
        representative.setdefault(candidate.evidence_id, candidate)
    scored = {
        evidence_id: (1.0 - alpha) * sparse_norm.get(evidence_id, 0.0)
        + alpha * dense_norm.get(evidence_id, 0.0)
        for evidence_id in representative
    }
    return _ranked(scored, representative, query_id=query_id, top_k=top_k)
