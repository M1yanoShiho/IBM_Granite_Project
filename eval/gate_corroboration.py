"""Offline gated dynamic-alpha analysis for the corroboration reranker.

Everything here is pure arithmetic on a runs dump produced by
``eval.tune_corroboration --dump-runs`` (no LLM, no GPU, no index): dev/test split,
per-query gate signals, gated convex blend, dev-only selection, test-only
certification, and the descriptive flip table. Two deliverables: re-certify
finding 15 under an honest dev/test protocol, and test whether a per-query gate
(blend only where consensus exists / where the first stage is unsure) beats the
global blend. Design spec: docs/superpowers/specs/2026-07-06-gated-corroboration-design.md.

A structural fact this leans on: ``minmax_normalize`` maps an all-equal score set to
all-1.0, and a convex blend with a constant is order-preserving -- so zero-consensus
queries are untouched by the global blend already; the gate's room to act is the
weak-consensus region (votes gate tau=1 must therefore reproduce the global blend).
"""
from __future__ import annotations

import random

from typing import Dict, List, NamedTuple, Optional, Tuple

from eval.ir_metrics import Run
from eval.tune_corroboration import per_query_hits
from src.retrieval.fusion import convex_fuse, fuse_one, minmax_normalize


def max_votes_signal(corroboration_run: Run) -> Dict[str, float]:
    """Per-query strongest consensus: max raw vote count (0.0 for an empty pool)."""
    return {
        qid: (max(scores.values()) if scores else 0.0)
        for qid, scores in corroboration_run.items()
    }


def margin_signal(relevance_run: Run) -> Dict[str, float]:
    """Per-query first-stage confidence: top1 - top2 of the min-max-normalised
    relevance scores (0.0 when fewer than 2 docs; all-equal scores normalise to
    all-1.0 so the margin is 0.0 and a margin gate fires -- harmless, the blend
    then decides)."""
    out: Dict[str, float] = {}
    for qid, scores in relevance_run.items():
        norm = sorted(minmax_normalize(scores).values(), reverse=True)
        out[qid] = norm[0] - norm[1] if len(norm) >= 2 else 0.0
    return out


def gate_mask(
    family: str,
    param: Optional[float],
    votes: Dict[str, float],
    margins: Dict[str, float],
) -> Dict[str, bool]:
    """Per-query blend/keep decision. ``global`` always blends; ``votes`` blends iff
    ``max_votes >= param``; ``margin`` blends iff ``margin < param`` (blend only where
    the first stage is unsure)."""
    if family == "global":
        return {qid: True for qid in votes}
    if family == "votes":
        return {qid: v >= param for qid, v in votes.items()}
    if family == "margin":
        return {qid: m < param for qid, m in margins.items()}
    raise ValueError(f"unknown gate family: {family!r}")


def gated_fuse(
    relevance_run: Run, corroboration_run: Run, alpha: float, gate: Dict[str, bool]
) -> Run:
    """Blend gated-on queries with ``fuse_one``; gated-off queries keep their raw
    first-stage scores (identical ranking to alpha=1 for that query)."""
    fused: Run = {}
    for qid, rel in relevance_run.items():
        if gate.get(qid, False):
            fused[qid] = fuse_one(rel, corroboration_run.get(qid, {}), alpha)
        else:
            fused[qid] = dict(rel)
    return fused


def split_queries(
    qids: List[str], seed: int = 0, dev_fraction: float = 0.5
) -> Tuple[List[str], List[str]]:
    """Deterministic dev/test split: sort, seeded shuffle, cut at ``dev_fraction``.

    Sorting first makes the split a function of (qid set, seed) alone -- input
    order (dict iteration, file order) cannot change who lands in test.
    """
    ordered = sorted(qids)
    random.Random(seed).shuffle(ordered)
    n_dev = round(len(ordered) * dev_fraction)
    return ordered[:n_dev], ordered[n_dev:]


def evaluate_config(
    relevance_run: Run,
    corroboration_run: Run,
    needles: Dict[str, str],
    qids: List[str],
    family: str,
    param: Optional[float],
    alpha: float,
    k: int,
) -> Dict[str, float]:
    """Per-query needle-found hits for ONE (family, param, alpha) config, restricted
    to ``qids`` (the dev or test half). The single definition of "score a config" --
    the sweep, the selection, and the test arms all go through here."""
    votes = max_votes_signal(corroboration_run)
    margins = margin_signal(relevance_run)
    mask = gate_mask(family, param, votes, margins)
    rel = {q: relevance_run[q] for q in qids}
    cor = {q: corroboration_run.get(q, {}) for q in qids}
    nee = {q: needles[q] for q in qids}
    return per_query_hits(gated_fuse(rel, cor, alpha, mask), nee, k)


def _mean(hits: Dict[str, float]) -> float:
    return sum(hits.values()) / len(hits) if hits else 0.0


# Gate grids (spec section 4). tau=1 reproduces the global blend (structural fact,
# asserted in tests); the real hypothesis space is tau >= 2.
VOTE_TAUS = [1.0, 2.0, 3.0, 4.0]
MARGIN_THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]


class CurveRow(NamedTuple):
    """One point of the dev sensitivity surface."""

    family: str
    param: Optional[float]
    alpha: float
    score: float
    n_gated: int


def sweep_gated(
    relevance_run: Run,
    corroboration_run: Run,
    needles: Dict[str, str],
    qids: List[str],
    alpha_grid: List[float],
    k: int,
) -> List[CurveRow]:
    """Score every (family, param, alpha) config on ``qids`` (the dev half).

    Pure arithmetic; publish the WHOLE surface (sensitivity artifact, no
    cherry-picking), selection happens separately in :func:`select_on_dev`.
    """
    votes = max_votes_signal(corroboration_run)
    margins = margin_signal(relevance_run)
    rows: List[CurveRow] = []
    for family, params in (
        ("global", [None]),
        ("votes", VOTE_TAUS),
        ("margin", MARGIN_THRESHOLDS),
    ):
        for param in params:
            mask = gate_mask(family, param, votes, margins)
            n_gated = sum(1 for q in qids if mask.get(q, False))
            for alpha in alpha_grid:
                hits = evaluate_config(
                    relevance_run, corroboration_run, needles, qids, family, param, alpha, k
                )
                rows.append(CurveRow(family, param, alpha, _mean(hits), n_gated))
    return rows


_FAMILY_ORDER = {"global": 0, "votes": 1, "margin": 2}  # ties -> simpler wins


def _strictness(row: CurveRow) -> float:
    """Larger = gate fires less often (spec tie-break: prefer the stricter gate)."""
    if row.family == "votes":
        return row.param
    if row.family == "margin":
        return -row.param
    return 0.0


def select_on_dev(
    rows: List[CurveRow],
) -> Tuple[Dict[str, CurveRow], CurveRow, CurveRow]:
    """``(best_per_family, overall_winner, best_gated)`` under the spec tie-breaks.

    Within a family: max score, ties to larger alpha (closer to pure relevance,
    matching ``tune_corroboration.best_alpha``), then to the stricter gate. Across
    families: max score, ties to the simpler family (global > votes > margin).
    ``best_gated`` is the winner among the votes/margin families only -- always
    certified on test so "gated vs global" is measured even when global wins dev.
    """
    best: Dict[str, CurveRow] = {}
    for family in _FAMILY_ORDER:
        candidates = [r for r in rows if r.family == family]
        best[family] = max(candidates, key=lambda r: (r.score, r.alpha, _strictness(r)))
    winner = max(
        best.values(), key=lambda r: (r.score, -_FAMILY_ORDER[r.family], r.alpha)
    )
    best_gated = max(
        (best["votes"], best["margin"]),
        key=lambda r: (r.score, -_FAMILY_ORDER[r.family], r.alpha),
    )
    return best, winner, best_gated


_VOTE_BUCKETS = ["0", "1", "2", "3", "4+"]
_MARGIN_BUCKETS = ["<0.05", "0.05-0.1", "0.1-0.2", ">=0.2"]
_STATUSES = ["fixed", "broken", "unchanged_hit", "unchanged_miss"]


def vote_bucket(votes: float) -> str:
    return "4+" if votes >= 4 else str(int(votes))


def margin_bucket(margin: float) -> str:
    if margin < 0.05:
        return "<0.05"
    if margin < 0.1:
        return "0.05-0.1"
    if margin < 0.2:
        return "0.1-0.2"
    return ">=0.2"


def flip_status(base_hit: float, blended_hit: float) -> str:
    """fixed (miss->hit) / broken (hit->miss) / unchanged_hit / unchanged_miss."""
    if blended_hit and not base_hit:
        return "fixed"
    if base_hit and not blended_hit:
        return "broken"
    return "unchanged_hit" if base_hit else "unchanged_miss"


def flip_table(
    relevance_run: Run,
    corroboration_run: Run,
    needles: Dict[str, str],
    alpha: float,
    k: int,
) -> List[Dict[str, object]]:
    """Descriptive flip analysis on the FULL query set at one alpha (no tuning):
    who does the global blend fix/break, bucketed by each gate signal. The
    evidence for/against gating's premise; goes in the report either way."""
    votes = max_votes_signal(corroboration_run)
    margins = margin_signal(relevance_run)
    base = per_query_hits(relevance_run, needles, k)
    blended = per_query_hits(convex_fuse(relevance_run, corroboration_run, alpha), needles, k)
    statuses = {qid: flip_status(base[qid], blended[qid]) for qid in needles}
    rows: List[Dict[str, object]] = []
    for signal, sig_map, buckets, bucket_fn in (
        ("max_votes", votes, _VOTE_BUCKETS, vote_bucket),
        ("margin", margins, _MARGIN_BUCKETS, margin_bucket),
    ):
        counts = {b: {s: 0 for s in _STATUSES} for b in buckets}
        for qid in needles:
            counts[bucket_fn(sig_map[qid])][statuses[qid]] += 1
        for b in buckets:
            rows.append({"signal": signal, "bucket": b, **counts[b]})
    return rows
