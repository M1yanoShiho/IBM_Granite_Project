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

from typing import Dict, List, Optional, Tuple

from eval.ir_metrics import Run
from eval.tune_corroboration import per_query_hits
from src.retrieval.fusion import fuse_one, minmax_normalize


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
