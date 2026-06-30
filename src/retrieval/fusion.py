"""Convex-combination fusion of dense + lexical retrieval scores.

The RRF hybrid (``src/retrieval/hybrid.HybridRetriever``) fuses *rankings* and lost
to pure dense on every BEIR set (results-summary finding #4). Convex combination
fuses *normalised scores* with a tunable weight ``alpha``, which beats RRF and is
more sample-efficient (Bruch et al., 2023). This module is the pure maths — no
models, no I/O — so it is reused by both the online ``ConvexHybridRetriever`` and the
offline alpha sweep (``eval/tune_alpha.py``), keeping one definition of the fusion.

``alpha`` is the DENSE weight: ``fused = alpha * norm(dense) + (1 - alpha) *
norm(lexical)``. ``alpha = 1`` -> pure dense ranking; ``alpha = 0`` -> pure lexical.

Types are local (``Scores`` = one query's ``{doc_id: score}``; a per-query
``Dict[str, Scores]`` is the ``Run`` shape) so ``src`` keeps no dependency on
``eval``.
"""

from __future__ import annotations

from typing import Callable, Dict

Scores = Dict[str, float]  # {doc_id: score} for a single query


def minmax_normalize(scores: Scores) -> Scores:
    """Min-max scale one query's scores to [0, 1].

    Empty input -> empty output. When all scores are equal (max == min), every
    document maps to 1.0: a retrieved doc with no spread is treated as maximally
    relevant by that arm (it was retrieved at all), and the alpha weighting still
    blends it against the other arm.
    """
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi == lo:
        return {doc_id: 1.0 for doc_id in scores}
    span = hi - lo
    return {doc_id: (s - lo) / span for doc_id, s in scores.items()}


def fuse_one(
    dense: Scores,
    lexical: Scores,
    alpha: float,
    normalize: Callable[[Scores], Scores] = minmax_normalize,
) -> Scores:
    """Convex-combine one query's two arms into a fused ``{doc_id: score}``.

    Each arm is normalised independently, then over the union of doc ids
    ``fused = alpha * dense + (1 - alpha) * lexical``, with a doc absent from an arm
    contributing 0 for that arm. ``alpha`` is the dense weight, in [0, 1].
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}.")
    nd = normalize(dense)
    nl = normalize(lexical)
    doc_ids = set(nd) | set(nl)
    return {
        doc_id: alpha * nd.get(doc_id, 0.0) + (1.0 - alpha) * nl.get(doc_id, 0.0)
        for doc_id in doc_ids
    }


def convex_fuse(
    dense_run: Dict[str, Scores],
    lexical_run: Dict[str, Scores],
    alpha: float,
    normalize: Callable[[Scores], Scores] = minmax_normalize,
) -> Dict[str, Scores]:
    """Convex-combine two per-query Runs into one fused Run (the offline path).

    Fuses query-by-query over the union of query ids; a query present in only one
    run is fused against an empty other arm. See :func:`fuse_one`.
    """
    qids = set(dense_run) | set(lexical_run)
    return {
        qid: fuse_one(dense_run.get(qid, {}), lexical_run.get(qid, {}), alpha, normalize)
        for qid in qids
    }
