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


@dataclass(frozen=True)
class ClaimNode:
    """One claim vertex of the query-local bipartite graph (design §2.1).

    `hypothesis` is the full sentence built by `claims.py`, never the bare answer string: an NLI
    cross-encoder fed "Paul" against "Apostle Paul" is being given degenerate input (design
    §2.4 rule 2). Parametric self-answers never become claim nodes — model priors stay in the
    rerank stage and are barred from the gate (design §2.4 rule 3).
    """

    claim_id: str
    answer: str
    hypothesis: str


@dataclass(frozen=True)
class RelationEdge:
    """One passage -> claim edge of the query-local bipartite graph (design §2.1).

    Carries the frozen per-edge provenance of M0 §2.1 invariant 2 — confidence, model version,
    premise/hypothesis hash — so any edge in a per-query dump can be traced to the exact inputs
    and checkpoint that produced it. REFUTES and UNKNOWN edges are recorded exactly like
    SUPPORTS ones; under the primary `conflict_mode` the gate simply never reads them, and
    abstention rate is a first-class report item (M0 §3.6, G-AB).
    """

    passage_id: str
    claim_id: str
    label: RelationLabel
    confidence: float
    model_version: str
    premise_hash: str
    hypothesis_hash: str

    @classmethod
    def from_prediction(
        cls, passage_id: str, claim_id: str, prediction: RelationPrediction
    ) -> "RelationEdge":
        return cls(
            passage_id=passage_id,
            claim_id=claim_id,
            label=prediction.label,
            confidence=prediction.confidence,
            model_version=prediction.model_version,
            premise_hash=prediction.premise_hash,
            hypothesis_hash=prediction.hypothesis_hash,
        )
