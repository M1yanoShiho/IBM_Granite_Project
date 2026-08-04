"""Frozen relation schema (Graph 2.0 design §2, M0 §2, amended by M0 §9 / A1).

The relation model's output space is BINARY under A1 (g2-proto-2, approved 2026-08-03):
SUPPORTS or NOT_SUPPORTED, and nothing else. NOT_SUPPORTED is a predicted CLASS, never a
threshold artefact — the path that can drop a candidate must carry no absolute cross-domain
scale, which is the property distinguishing this gate from credibility-threshold approaches
(M0 §9.5a).

REFUTES and UNKNOWN remain enum members but are NO LONGER PRODUCED by the relation model:

  * REFUTES is retained because A1 §9.1 keeps `CLAIM_REFUTES` as a schema edge type "until there
    is evidence for it from outside the binary space", so the ablation arm can return if A1 is
    overturned.
  * UNKNOWN is retained because 0B-1 still reads it: `vitaminc.py` maps official NEI gold onto
    it and `external_report.non_unknown_coverage` counts it. A1 §9.1 changes the 0B-2 metric and
    says NOTHING about 0B-1, so that tier is deliberately untouched here.

Both are also the labels historical dumps and warm cache entries were written with, and
`RelationLabel(...)` has to keep parsing them.

NOT_SUPPORTED SUBSUMES UNKNOWN on the prediction side. A1 §9.3's degeneracy table lists
"all-UNKNOWN / all-NOT-SUPPORTED" as ONE strategy, and §9.5's published binary readings were
obtained by counting a three-class UNKNOWN prediction as NOT-SUPPORTED. The cost is stated in
§9.3 and is real: binary cannot separate "abstained" from "committed to the contrary", so G-AB
(M0 §3.6) loses that decomposition and keeps only the non-SUPPORTS rate.
"""

from dataclasses import dataclass
from enum import StrEnum


class RelationLabel(StrEnum):
    SUPPORTS = "SUPPORTS"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    # Schema-only from A1 onward — see the module docstring. Not emitted by the relation model.
    REFUTES = "REFUTES"
    UNKNOWN = "UNKNOWN"


# The labels the relation model may emit (A1 §9.1). Ordered, because this IS the frozen §2.4
# tie-break: a tie resolves toward NOT_SUPPORTED and never toward SUPPORTS, since a SUPPORTS edge
# is what makes a candidate droppable and a coin flip must not create one.
PREDICTED_LABELS: tuple[RelationLabel, ...] = (
    RelationLabel.NOT_SUPPORTED,
    RelationLabel.SUPPORTS,
)


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
    and checkpoint that produced it. NOT_SUPPORTED edges are recorded exactly like SUPPORTS
    ones; under the primary `conflict_mode` the gate simply never reads them, and the
    non-SUPPORTS rate is a first-class report item (M0 §3.6, G-AB).

    An edge may still carry REFUTES or UNKNOWN when it was read back from a cache or dump
    written before A1. The graph handles that unchanged, because only SUPPORTS ever votes.
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
