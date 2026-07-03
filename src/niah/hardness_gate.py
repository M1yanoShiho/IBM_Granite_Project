# src/niah/hardness_gate.py
"""Task-level hardness gate (spec §5.1): baselines must NOT saturate.

If a baseline retriever already finds the needle almost always (mean recall above
``threshold``), the constructed task is too easy and measures nothing — raise the
distractor ratio/hardness and rebuild.
"""
from __future__ import annotations

from typing import Dict, List, Set


def recall_at_k(ranked_ids: List[str], gold_ids: Set[str], k: int) -> float:
    """Fraction of gold ids present in the top-``k`` retrieved ids (multi-gold)."""
    if not gold_ids:
        return 0.0
    topk = set(ranked_ids[:k])
    return len(topk & gold_ids) / len(gold_ids)


def hit_at_k(ranked_ids: List[str], needle_id: str, k: int) -> float:
    """1.0 if the single designated needle is in the top-``k``, else 0.0.

    The single-target NIAH metric (vs ``recall_at_k`` for the multi-gold case);
    averaged over queries it is the needle-found rate. Robust to NQ-style incomplete
    labels because it tracks one *known* target, not "any relevant doc".
    """
    return 1.0 if needle_id in ranked_ids[:k] else 0.0


def reciprocal_rank(ranked_ids: List[str], needle_id: str) -> float:
    """Reciprocal of the designated needle's 1-indexed rank; 0.0 if not present."""
    for i, doc_id in enumerate(ranked_ids):
        if doc_id == needle_id:
            return 1.0 / (i + 1)
    return 0.0


def mean_recall(per_query: Dict[str, float]) -> float:
    """Mean of per-query recall values (0.0 if empty)."""
    return sum(per_query.values()) / len(per_query) if per_query else 0.0


def is_saturated(mean_recall_value: float, threshold: float = 0.95) -> bool:
    """True if mean recall is at/above the saturation ``threshold``."""
    return mean_recall_value >= threshold


def gate_report(per_query: Dict[str, float], threshold: float = 0.95) -> dict:
    """Summarise the gate: mean recall, saturated?, and pass (= not saturated)."""
    mr = mean_recall(per_query)
    saturated = is_saturated(mr, threshold)
    return {"mean_recall": mr, "saturated": saturated, "passes_gate": not saturated}
