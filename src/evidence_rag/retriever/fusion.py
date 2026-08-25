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
    secondary: dict[str, float] | None = None,
) -> CandidateSet:
    # Sort by fused score desc, then by an optional secondary score desc, breaking
    # remaining ties on evidence_id for determinism (mirrors the tie-break in
    # BM25/dense retrievers). Callers that pass no secondary score are unaffected.
    secondary = secondary or {}
    ordered = sorted(
        scored.items(),
        key=lambda item: (-item[1], -secondary.get(item[0], 0.0), item[0]),
    )
    candidates = tuple(
        representative[evidence_id].model_copy(
            update={"retrieval_score": score, "retrieval_rank": rank}
        )
        for rank, (evidence_id, score) in enumerate(ordered[:top_k], start=1)
    )
    return CandidateSet(query_id=query_id, candidates=candidates)


def _arm_weights(
    results: Sequence[CandidateSet],
    weights: Sequence[float] | None,
) -> tuple[float, ...]:
    if weights is None:
        return (1.0,) * len(results)
    if len(weights) != len(results):
        raise ValueError("fusion weights must have one entry per result list")
    return tuple(float(weight) for weight in weights)


def reciprocal_rank_fusion(
    results: Sequence[CandidateSet],
    *,
    query_id: str,
    top_k: int,
    k: int = DEFAULT_RRF_K,
    weights: Sequence[float] | None = None,
) -> CandidateSet:
    """Fuse ranked lists by Reciprocal Rank Fusion (Cormack et al., 2009).

    Each candidate contributes ``weight / (k + rank)`` from every list it appears
    in; ``k`` damps the weight of top ranks (larger ``k`` flattens the curve). RRF
    fuses *rankings*, so the arms' scores need not be comparable.

    ``weights`` (one entry per list, default all ``1.0``) scales each list's vote.
    Equal weights make a list's influence shrink as lists are added, since every
    list votes once into the same sum — which is exactly the dilution measured in
    R2 (``docs/results-summary.md``), where re-adding the original query as one
    equal arm among N sub-queries recovered its top-20 placements but not rank 1.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if k <= 0:
        raise ValueError("RRF k must be positive")
    scored: dict[str, float] = {}
    representative: dict[str, EvidenceCandidate] = {}
    for result, weight in zip(results, _arm_weights(results, weights), strict=True):
        for candidate in result.candidates:
            scored[candidate.evidence_id] = scored.get(candidate.evidence_id, 0.0) + weight / (
                k + candidate.retrieval_rank
            )
            representative.setdefault(candidate.evidence_id, candidate)
    return _ranked(scored, representative, query_id=query_id, top_k=top_k)


def best_rank_fusion(
    results: Sequence[CandidateSet],
    *,
    query_id: str,
    top_k: int,
    k: int = DEFAULT_RRF_K,
    weights: Sequence[float] | None = None,
) -> CandidateSet:
    """Fuse ranked lists by each candidate's *best* rank, summed rank as tie-break.

    RRF (above) *sums* ``1 / (k + rank)`` across arms, which rewards appearing in
    many lists. On decomposed multi-hop queries that is the wrong incentive: a gold
    document answers exactly one hop, so it ranks highly in one arm and is absent
    from the rest, while a document ranking mediocrely in every arm accumulates more
    total mass and overtakes it (measured: R1/R2 in ``docs/results-summary.md``).
    Taking the maximum instead makes a single strong placement decisive, which is the
    property a multi-hop gold document actually has.

    Two caveats worth knowing before tuning this:

    - At equal ``weights``, ``k`` does **not** affect the ordering here. ``max`` of a
      monotonically decreasing function of rank is equivalent to ``min`` of rank, so
      ``k`` only rescales the recorded score. Sweeping it is pointless; it is accepted
      solely to keep the signature interchangeable with
      :func:`reciprocal_rank_fusion`. Unequal ``weights`` break that equivalence — a
      weighted arm's placement is compared against another arm's *through* the ``k``
      offset — so ``k`` and the weights must then be tuned together.
    - Pure ``max`` ties heavily — every document placed first by *some* arm shares the
      identical score, leaving rank 1 to an arbitrary tie-break. The summed RRF score
      is therefore kept as a secondary key, so breadth across arms still separates
      documents that are tied on their best placement instead of falling through to
      alphabetical order.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if k <= 0:
        raise ValueError("RRF k must be positive")
    best: dict[str, float] = {}
    total: dict[str, float] = {}
    representative: dict[str, EvidenceCandidate] = {}
    for result, weight in zip(results, _arm_weights(results, weights), strict=True):
        for candidate in result.candidates:
            contribution = weight / (k + candidate.retrieval_rank)
            evidence_id = candidate.evidence_id
            best[evidence_id] = max(best.get(evidence_id, 0.0), contribution)
            total[evidence_id] = total.get(evidence_id, 0.0) + contribution
            representative.setdefault(evidence_id, candidate)
    return _ranked(
        best, representative, query_id=query_id, top_k=top_k, secondary=total
    )


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
