"""Relation prediction over (premise, hypothesis) pairs (Graph 2.0 design §2.5).

argmax is the PRIMARY head: there is no confidence threshold anywhere on the path that can drop
a candidate, so the gate carries no absolute cross-domain scale. That is the property separating
this design from credibility-threshold arbitration, and it is why UNKNOWN is a predicted class
rather than "score below tau". Threshold gating exists only as a pre-registered remedy if Gate
0B REFUTES precision falls short, and invoking it must be declared in the report.

A1 (g2-proto-2) narrowed the output space to two classes; A2 (g2-proto-3) narrowed A1 ITSELF
back to the reading, so a three-class head emits three classes here again and the collapse
happens at each consumer instead (see `relations.models`). Neither amendment weakens the
paragraph above. M0 §9.5a is explicit that a threshold may not be introduced on the back of A1,
and A2 additionally RESTORES that remedy's trigger: under A1 `refutes_precision` was
structurally unevaluable, so the one pre-registered escape hatch in the design had no successor
measurement. It is evaluable again, measured .9026 / .9127, so the trigger condition remains
unmet. Introducing a threshold still requires its own amendment.
"""

import hashlib
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Protocol

from evidence_rag.relations.models import (
    BINARY_PREDICTED_LABELS,
    PREDICTED_LABELS,
    RelationLabel,
    RelationPrediction,
)

ScoreFn = Callable[[Sequence[tuple[str, str]]], Sequence[Mapping[str, float]]]

# Ties resolve AWAY from SUPPORTS, never toward it. A SUPPORTS edge is what makes a candidate
# droppable at all, so a coin-flip tie must not create one: the design's failure direction is
# silence (spec §7.1, M0 §2.4). Ties are near-impossible with float softmax, but the rule must
# still be deterministic and stated rather than falling out of enum declaration order. Both tuples
# are already in tie-break order, so they cannot drift apart from this rule.
_OUTPUT_SPACES: tuple[tuple[RelationLabel, ...], ...] = (
    PREDICTED_LABELS,
    BINARY_PREDICTED_LABELS,
)


def _output_space(scores: Mapping[str, float]) -> tuple[RelationLabel, ...]:
    """Pick the output space this scorer speaks, by EXACT key match.

    Two shapes are legitimate after A2 and they must never be silently mixed: a three-class head
    (M0 §2.1) and a natively-binary checkpoint (M0 §10.6 cost 3). The match is exact on purpose.
    A subset check would let a three-class head that still collapses — the pre-A2 behaviour —
    pass as binary and quietly report a two-class argmax as a three-class one; a superset check
    would let a scorer offer both spaces and make the answer depend on which tuple was tried
    first. Either way the result is a plausible label rather than a crash, which is the failure
    mode this project has already paid for more than once.
    """
    keys = set(scores)
    for space in _OUTPUT_SPACES:
        if keys == {label.value for label in space}:
            return space
    raise ValueError(
        f"scores {sorted(keys)} match no output space exactly: three-class "
        f"{sorted(label.value for label in PREDICTED_LABELS)} (M0 §2.1) or natively-binary "
        f"{sorted(label.value for label in BINARY_PREDICTED_LABELS)} (M0 §10.6). A three-class "
        "head must NOT be collapsed before this point: A2 moved the collapse to the consumers, "
        "because 0B-1 reads all three classes."
    )


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
            # Re-resolved per row rather than once per batch: a scorer that changes shape
            # part-way through is exactly the kind of fault that otherwise surfaces as a
            # plausible number instead of an error.
            order = _output_space(scores)
            best = max(order, key=lambda label: scores[label.value])
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
