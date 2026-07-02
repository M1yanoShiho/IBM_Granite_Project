# src/niah/hardness_gate.py
"""Task-level hardness gate (spec §5.1): baselines must NOT saturate.

If a baseline retriever already finds the needle almost always (mean recall above
``threshold``), the constructed task is too easy and measures nothing — raise the
distractor ratio/hardness and rebuild.
"""
from __future__ import annotations

from typing import Dict, List, Set


def recall_at_k(ranked_ids: List[str], gold_ids: Set[str], k: int) -> float:
    """Fraction of gold ids present in the top-``k`` retrieved ids."""
    if not gold_ids:
        return 0.0
    topk = set(ranked_ids[:k])
    return len(topk & gold_ids) / len(gold_ids)


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
