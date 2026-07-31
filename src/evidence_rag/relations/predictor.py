"""Relation prediction over (premise, hypothesis) pairs (Graph 2.0 design §2.5).

argmax is the PRIMARY head: there is no confidence threshold anywhere on the path that can drop
a candidate, so the gate carries no absolute cross-domain scale. That is the property separating
this design from credibility-threshold arbitration, and it is why UNKNOWN is a predicted class
rather than "score below tau". Threshold gating exists only as a pre-registered remedy if Gate
0B REFUTES precision falls short, and invoking it must be declared in the report.
"""

import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from evidence_rag.relations.models import RelationLabel, RelationPrediction

ScoreFn = Callable[[Sequence[tuple[str, str]]], Sequence[Mapping[str, float]]]

# Ties resolve toward UNKNOWN, never toward SUPPORTS. A SUPPORTS edge is what makes a candidate
# droppable at all, so a coin-flip tie must not create one: the design's failure direction is
# silence (spec §7.1). Ties are near-impossible with float softmax, but the rule must still be
# deterministic and stated rather than falling out of enum declaration order.
_TIE_BREAK_ORDER = (RelationLabel.UNKNOWN, RelationLabel.REFUTES, RelationLabel.SUPPORTS)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class RelationPredictor(Protocol):
    def predict(self, pairs: Sequence[tuple[str, str]]) -> tuple[RelationPrediction, ...]: ...


class NLIRelationPredictor:
    def __init__(self, *, score_fn: ScoreFn, model_version: str) -> None:
        self.score_fn = score_fn
        self.model_version = model_version

    def predict(self, pairs: Sequence[tuple[str, str]]) -> tuple[RelationPrediction, ...]:
        if not pairs:
            return ()
        scored = self.score_fn(list(pairs))
        predictions: list[RelationPrediction] = []
        for (premise, hypothesis), scores in zip(pairs, scored, strict=True):
            for label in RelationLabel:
                if label.value not in scores:
                    raise ValueError(f"missing relation class in scores: {label.value}")
            best = max(_TIE_BREAK_ORDER, key=lambda label: scores[label.value])
            predictions.append(
                RelationPrediction(
                    label=best,
                    confidence=float(scores[best.value]),
                    model_version=self.model_version,
                    premise_hash=text_hash(premise),
                    hypothesis_hash=text_hash(hypothesis),
                )
            )
        return tuple(predictions)
