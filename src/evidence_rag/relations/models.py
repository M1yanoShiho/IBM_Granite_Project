"""Frozen relation schema (Graph 2.0 design §2, M0 §2).

Three relations plus UNKNOWN. UNKNOWN is a predicted CLASS, never a threshold artefact: the path
that can drop a candidate must carry no absolute cross-domain scale, which is the property that
distinguishes this gate from credibility-threshold approaches.
"""

from dataclasses import dataclass
from enum import StrEnum


class RelationLabel(StrEnum):
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class RelationPair:
    """One labelled premise/hypothesis pair.

    `group` is the leakage-audit key (VitaminC page, NIAH synthetic family). It is never a
    feature — models must not see it.
    """

    premise: str
    hypothesis: str
    label: RelationLabel
    group: str


@dataclass(frozen=True)
class RelationPrediction:
    """One predicted edge. Hashes and model version are recorded so an edge can be traced back
    to the exact inputs and checkpoint that produced it (TRAINING_PLAN §3.2)."""

    label: RelationLabel
    confidence: float
    model_version: str
    premise_hash: str
    hypothesis_hash: str
