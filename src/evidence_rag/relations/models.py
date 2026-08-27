"""Frozen relation schema (Graph 2.0 design §2, M0 §2, amended by A1 §9 and A2 §10).

The relation model's output space is THREE-CLASS: SUPPORTS / REFUTES / UNKNOWN.

A1 (g2-proto-2) made it binary. A2 (g2-proto-3, approved 2026-08-06) NARROWED A1 back to the
reading, because A1's conclusion outran its evidence: §9.2 and §9.3 show only that the GATE and
the 0B-2 metric read two classes, and the one production-side argument (§9.4, the G module's
binary backends) cites backends that operate WITH a threshold — which §9.5a forbids on this
path. So the output space returns to three classes and the collapse happens LATER, at each
consumer:

  * `gate0b.task_report` (0B-2) scores `predicted != SUPPORTS` / `predicted == SUPPORTS`, so it
    collapses implicitly and needs no separate step;
  * `graph.RelationGraph` counts only SUPPORTS votes, so it collapses implicitly too;
  * `gate0b.external_report` (0B-1) does NOT collapse. That tier reads all three classes, which
    is exactly what makes its five frozen thresholds evaluable again AT THEIR ORIGINAL
    CALIBRATION — no silent recalibration, which was the cost of every option §9.11 listed.

NOT_SUPPORTED is a DERIVED label from A2 onward, not a predicted one. It stays in the enum for
two reasons that are not stylistic: A1-era dumps, warm cache entries and probe files were
written with it and must keep parsing, and a natively-binary checkpoint emits it directly (see
`BINARY_PREDICTED_LABELS`).

WHAT A2 DOES NOT CHANGE. The gate still reads only SUPPORTS, so `independent_support` is
untouched. The 0B-2 thresholds, the joint gate as the sole anti-gaming mechanism (§9.3), and
§9.5a's ban on introducing a threshold all stand. §9.5's published binary readings were obtained
by relabelling a three-class argmax, so they are unaffected: A2 neither produces nor voids any
experimental data (§10.7(D)).
"""

from dataclasses import dataclass
from enum import StrEnum


class RelationLabel(StrEnum):
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    UNKNOWN = "UNKNOWN"
    # Derived from A2 onward, not predicted by a three-class head: it is what the 0B-2 metric and
    # the graph read a non-SUPPORTS prediction AS. Still parsed, and still emitted directly by a
    # natively-binary checkpoint — see the module docstring.
    NOT_SUPPORTED = "NOT_SUPPORTED"


# The labels a three-class relation model may emit (M0 §2.1). Ordered, because this IS the frozen
# §2.4 tie-break: a tie resolves UNKNOWN -> REFUTES -> SUPPORTS and NEVER lands on SUPPORTS, since
# a SUPPORTS edge is what makes a candidate droppable and a coin flip must not create one.
PREDICTED_LABELS: tuple[RelationLabel, ...] = (
    RelationLabel.UNKNOWN,
    RelationLabel.REFUTES,
    RelationLabel.SUPPORTS,
)

# A natively-binary checkpoint has no third class to emit: MiniCheck-FT5 is a
# `T5ForConditionalGeneration` reading two label tokens, and inventing a REFUTES/UNKNOWN split for
# it would be a fabrication. It scores 0B-2 and builds edges exactly like a three-class arm,
# because both consumers only ever ask "is this SUPPORTS?" — but it CANNOT be certified on 0B-1,
# whose five thresholds are defined on the three-class split (A2 §10.6 cost 3). Same tie-break
# rule, same reason: never toward SUPPORTS.
BINARY_PREDICTED_LABELS: tuple[RelationLabel, ...] = (
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
