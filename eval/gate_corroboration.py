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

from typing import Dict

from eval.ir_metrics import Run
from src.retrieval.fusion import minmax_normalize


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
