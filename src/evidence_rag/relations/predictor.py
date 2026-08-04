"""Relation prediction over (premise, hypothesis) pairs (Graph 2.0 design §2.5).

argmax is the PRIMARY head: there is no confidence threshold anywhere on the path that can drop
a candidate, so the gate carries no absolute cross-domain scale. That is the property separating
this design from credibility-threshold arbitration, and it is why NOT_SUPPORTED is a predicted
class rather than "score below tau". Threshold gating exists only as a pre-registered remedy if
Gate 0B REFUTES precision falls short, and invoking it must be declared in the report.

A1 (g2-proto-2) narrowed the output space from three classes to two and did NOT weaken the
sentence above. M0 §9.5a is explicit that A1's scope is argmax only, that a threshold may not be
introduced on the back of it, and that the pre-registered remedy's trigger — REFUTES precision
below .85 — was never met (measured .9026 / .9127). Binary argmax still contains no threshold;
introducing one requires a separate amendment.
"""

import hashlib
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Protocol

from evidence_rag.relations.models import PREDICTED_LABELS, RelationLabel, RelationPrediction

ScoreFn = Callable[[Sequence[tuple[str, str]]], Sequence[Mapping[str, float]]]

# Ties resolve toward NOT_SUPPORTED, never toward SUPPORTS. A SUPPORTS edge is what makes a
# candidate droppable at all, so a coin-flip tie must not create one: the design's failure
# direction is silence (spec §7.1, M0 §2.4). Ties are near-impossible with float softmax, but the
# rule must still be deterministic and stated rather than falling out of enum declaration order.
# `PREDICTED_LABELS` is already in tie-break order, so the two cannot drift apart.
_TIE_BREAK_ORDER = PREDICTED_LABELS

# Emitting one of these would mean a scorer is still speaking the pre-A1 three-class contract.
_SCHEMA_ONLY = (RelationLabel.REFUTES, RelationLabel.UNKNOWN)


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
            for label in PREDICTED_LABELS:
                if label.value not in scores:
                    raise ValueError(f"missing relation class in scores: {label.value}")
            for label in _SCHEMA_ONLY:
                if label.value in scores:
                    raise ValueError(
                        f"scores carry {label.value}, which is not emitted by the relation model "
                        "under A1 (g2-proto-2): collapse a three-class head into "
                        "SUPPORTS/NOT_SUPPORTED at the checkpoint adapter, not here"
                    )
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
