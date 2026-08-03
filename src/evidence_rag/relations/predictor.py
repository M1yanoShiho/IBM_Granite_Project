"""Relation prediction over (premise, hypothesis) pairs (Graph 2.0 design §2.5).

argmax is the PRIMARY head: there is no confidence threshold anywhere on the path that can drop
a candidate, so the gate carries no absolute cross-domain scale. That is the property separating
this design from credibility-threshold arbitration, and it is why UNKNOWN is a predicted class
rather than "score below tau". Threshold gating exists only as a pre-registered remedy if Gate
0B REFUTES precision falls short, and invoking it must be declared in the report.
"""

import hashlib
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
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


_FINGERPRINTED = re.compile(r"@[0-9a-f]{16}$")


def weight_fingerprint(named_buffers: Iterable[tuple[str, bytes]]) -> str:
    """Hash a checkpoint's parameters so `model_version` cannot be forged by naming.

    The edge cache is keyed on (model_version, premise_hash, hypothesis_hash). Two of the three
    are content-addressed; `model_version` is not, and that is the one dimension in which the
    f63e905 failure mode (a cache serving a previous run's answers) survives. Two fine-tuning
    seeds share every parameter name, shape and dtype, so ONLY hashing the values separates
    them — without this, a warm cache would serve seed 13's edges under seed 42's name and
    nothing downstream could tell.

    Takes (spec, raw_bytes) rather than tensors so this stays torch-free and unit-testable; the
    caller folds name, shape and dtype into `spec`. Sorted, because `state_dict()` order is not
    contractual.
    """
    digest = hashlib.sha256()
    for spec, raw in sorted(named_buffers):
        digest.update(spec.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(raw)
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


def fingerprinted_version(model_id: str, fingerprint: str) -> str:
    """`<model id>@<fingerprint>` — readable in a dump, still bound to the weights."""
    return f"{model_id}@{fingerprint}"


def require_fingerprinted_version(model_version: str) -> str:
    """Refuse a hand-written label. Deriving the version from the weights is the real fix; this
    is the guard that stops a future caller from quietly reintroducing the hole."""
    if not _FINGERPRINTED.search(model_version):
        raise ValueError(
            f"model_version {model_version!r} is not fingerprinted: it must end in "
            "'@<16 hex>' derived from the checkpoint weights (see weight_fingerprint). "
            "A hand-written label lets two checkpoints share one cache key."
        )
    return model_version


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
