"""Gate 0B: two-tier relation-model acceptance (Graph 2.0 design §3.2, §3.3).

0B-1 is external validity on VitaminC official test. 0B-2 is task validity on the mutation-log
probe. Both tiers exist because either alone is insufficient: a model can clear VitaminC and be
useless on NQ passages with template hypotheses, and a model can clear the task probe by
overfitting a synthetic mutation pattern.

All thresholds are pre-registered and must not move after seeing results. A1 (g2-proto-2)
renamed the 0B-2 twin metric and changed its definition; it moved no threshold VALUE.

WHAT STOPS A MODEL FROM GAMING 0B-2, AFTER A1. Before A1, "UNKNOWN counts as failure" was the
answer: a model abstaining everywhere would otherwise score perfectly on "did not assert a false
conflict". Under the binary space that sentence has nothing to attach to — abstention and
contradiction are the same output — so per A1 §9.3 the JOINT GATE is now the sole anti-gaming
mechanism, and it still closes both degeneracies:

    predict NOT_SUPPORTED everywhere -> twin 1.00, gold_supports_recall 0.00  -> blocked
    predict SUPPORTS everywhere      -> twin 0.00, gold_supports_recall 1.00  -> blocked

Neither metric may therefore be reported or thresholded alone. §9.5 also concedes that the twin
metric is now close to saturated (.87-.998 across the four measured arms) and has degraded from
a selection criterion into a lower-bound guard, leaving `gold_supports_recall >= .85` carrying
the real selection pressure.

0B-1 IS RUNNABLE AGAIN, AND ITS THRESHOLDS NEVER MOVED. A1 §9.1 amended the 0B-2 metric and said
nothing about this tier, which left three of its five thresholds structurally unevaluable against
a binary predictor: `refutes_precision` and `refutes_coverage` read 0.0 because the model could
never predict REFUTES, `macro_f1` was capped at 0.5 for the same reason, and
`non_unknown_coverage` read a vacuous 1.0. Those readings held even for a PERFECT model, which is
what suspended the tier (§9.11).

A2 (§10) closed that gap by narrowing A1 rather than by rewriting anything here: the relation
model emits three classes again, so `external_report`, its five threshold VALUES and
`vitaminc.py` are still byte-for-byte what they were when they were calibrated. None of the three
options §9.11 listed could say that — binarising the gold would have recalibrated .85/.80/.70
silently, and cancelling the tier would have required §3.2 to be re-argued from scratch.

A natively-binary checkpoint still cannot be scored on this tier. That is now a stated cost
(§10.6 cost 3), not a defect: with no third class the thresholds have nothing to measure. Do NOT
add a gold mapping to make one fit.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from evidence_rag.relations.models import RelationLabel
from evidence_rag.relations.task_probe import GOLD_SUPPORTS, TWIN_NOT_SUPPORTED

THRESHOLDS = {
    "refutes_precision": 0.85,
    "macro_f1": 0.80,
    "non_unknown_coverage": 0.80,
    "support_coverage": 0.70,
    "refutes_coverage": 0.70,
    "twin_not_supported_accuracy": 0.70,
    "gold_supports_recall": 0.85,
}


def _precision_recall_f1(
    gold: Sequence[RelationLabel],
    predicted: Sequence[RelationLabel],
    label: RelationLabel,
) -> tuple[float, float, float]:
    """Zero denominators yield 0.0 rather than being skipped.

    A Gate 0B run containing no examples of a class is a broken evaluation — usually the wrong
    split — and should fail loudly rather than silently pass by having nothing to measure.
    """
    true_positive = sum(1 for g, p in zip(gold, predicted, strict=True) if g == p == label)
    predicted_positive = sum(1 for p in predicted if p == label)
    actual_positive = sum(1 for g in gold if g == label)
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


@dataclass(frozen=True)
class ExternalReport:
    n: int
    supports_precision: float
    supports_recall: float
    refutes_precision: float
    refutes_recall: float
    macro_f1: float
    non_unknown_coverage: float
    support_coverage: float
    refutes_coverage: float
    failures: tuple[str, ...]

    @property
    def passes(self) -> bool:
        return not self.failures


def external_report(
    *, gold: Sequence[RelationLabel], predicted: Sequence[RelationLabel]
) -> ExternalReport:
    n = len(gold)
    if n == 0:
        raise ValueError("external_report needs at least one pair")
    if RelationLabel.NOT_SUPPORTED in predicted:
        # A2 §10.6 cost 3, enforced rather than merely written down. NOT_SUPPORTED is only ever
        # emitted by a natively-binary checkpoint, and this tier's five thresholds are defined on
        # the three-class split: scored against such an arm they read refutes_precision 0.0,
        # refutes_coverage 0.0, macro_f1 0.5 and a vacuous non_unknown_coverage 1.0 EVEN FOR A
        # PERFECTLY CORRECT MODEL. Under A1 that was the whole tier's condition and a payload
        # warning flagged it; A2 makes it a property of one arm shape, which a warning field can
        # no longer express. So it fails loudly instead: the alternative is a sweep JSON carrying
        # four plausible numbers that mean nothing, which is the failure mode this project keeps
        # paying for. Run such an arm with --external-pairs omitted.
        raise ValueError(
            "0B-1 cannot certify a natively-binary arm: predictions contain NOT_SUPPORTED, so "
            "this checkpoint has no third class and the five frozen thresholds have nothing to "
            "measure (M0 §10.6 cost 3). Three of them would read a structural failure and one a "
            "vacuous pass even for a perfect model. Omit --external-pairs for this arm; do NOT "
            "add a gold mapping to make it fit."
        )
    s_precision, s_recall, s_f1 = _precision_recall_f1(gold, predicted, RelationLabel.SUPPORTS)
    r_precision, r_recall, r_f1 = _precision_recall_f1(gold, predicted, RelationLabel.REFUTES)
    macro_f1 = (s_f1 + r_f1) / 2
    non_unknown = sum(1 for p in predicted if p is not RelationLabel.UNKNOWN) / n
    values = {
        "refutes_precision": r_precision,
        "macro_f1": macro_f1,
        "non_unknown_coverage": non_unknown,
        "support_coverage": s_recall,
        "refutes_coverage": r_recall,
    }
    failures = tuple(sorted(key for key, value in values.items() if value < THRESHOLDS[key]))
    return ExternalReport(
        n=n,
        supports_precision=s_precision,
        supports_recall=s_recall,
        refutes_precision=r_precision,
        refutes_recall=r_recall,
        macro_f1=macro_f1,
        non_unknown_coverage=non_unknown,
        support_coverage=s_recall,
        refutes_coverage=r_recall,
        failures=failures,
    )


@dataclass(frozen=True)
class TaskReport:
    n_twin: int
    n_gold: int
    twin_not_supported_accuracy: float
    gold_supports_recall: float
    not_supported_rate: float
    unknown_rate: float
    failures: tuple[str, ...]

    @property
    def passes(self) -> bool:
        return not self.failures


def task_report(
    *,
    kinds: Sequence[str],
    gold: Sequence[RelationLabel],
    predicted: Sequence[RelationLabel],
) -> TaskReport:
    """Score the two rows that map onto the selector's two failure modes.

    A false SUPPORTS on the twin puts the poison in the gold cluster and harm stops falling; a
    missed SUPPORTS on a gold-bearing passage under-counts gold's votes, isolates the needle,
    and recall drops. Everything else in the probe is context, not a gate.

    Both metrics are written against SUPPORTS rather than against the gold label, exactly as A1
    §9.1 defines them ("predicted != SUPPORTED" / "predicted == SUPPORTS"). That is not a
    restatement of `predicted == gold`: it is what makes this function correct on the pre-A1
    `dump-*.jsonl` files §9.10a says R012 / R012b must be recomputed from, where a prediction
    can still be REFUTES or UNKNOWN. Comparing against gold there would silently hand back the
    OLD three-class number under the new field name.

    `gold` is therefore no longer read, but it is still required and still zipped: it is the
    arity guard that catches a caller whose gold column has drifted out of step with `kinds`.
    It is deliberately NOT validated against the output space — a pre-A1 dump carries REFUTES
    in that column, and rejecting it would break the very recomputation path above.

    THE IMPLICIT COLLAPSE IS BY MAX, NOT BY SUM, and after A2 this function is where it lives.
    Both metrics compare against SUPPORTS, so a three-class argmax relabelled is exactly what
    they score:

        max  -> argmax(SUPPORTS, NOT_SUPPORTED) == argmax(SUPPORTS, REFUTES, UNKNOWN) relabelled
        sum  -> argmax(SUPPORTS, NOT_SUPPORTED) == "P(SUPPORTS) > .5", a threshold

    That distinction is load-bearing, not stylistic. Three independent checks agree: §9.5's
    published twin readings (albert .9980 / .9871, DeBERTa .8689 / .9008) reproduce from the
    run-log confusion matrix as 1 - P(predicted == SUPPORTS); §9.1's change table pins
    `gold_supports_recall` as UNCHANGED, and summing would move it because argmax can pick
    SUPPORTS at p < .5 while the sum form cannot; and §9.5a forbids introducing a threshold,
    which the sum form is. Scoring `predicted != SUPPORTS` gets the max form for free — there is
    no arithmetic here to get wrong — which is why A2 could delete the explicit collapse rather
    than relocate it.

    G-AB REPORTING (M0 §3.6). `not_supported_rate` is the joint rate A1 left; `unknown_rate` is
    the decomposition A2 gave back. A1 §9.3 conceded that a binary space cannot separate
    "abstained" from "committed to the contrary", and under A1 a field called `unknown_rate`
    would have read a structural 0.0 forever. A three-class head emits UNKNOWN again, so
    abstention is separately measurable and both are reported: R012 already measured it (albert
    .482, DeBERTa .176), which is what shows the decomposition was always there and was lost to
    the amendment rather than to the data. A natively-binary arm reports 0.0 here truthfully —
    it has no UNKNOWN to emit — so the two fields must be read together with the arm's shape.
    """
    rows = list(zip(kinds, gold, predicted, strict=True))
    twin = [p for kind, _gold, p in rows if kind in TWIN_NOT_SUPPORTED]
    gold_rows = [p for kind, _gold, p in rows if kind in GOLD_SUPPORTS]
    twin_accuracy = (
        sum(1 for p in twin if p is not RelationLabel.SUPPORTS) / len(twin) if twin else 0.0
    )
    gold_recall = (
        sum(1 for p in gold_rows if p is RelationLabel.SUPPORTS) / len(gold_rows)
        if gold_rows
        else 0.0
    )
    not_supported_rate = (
        sum(1 for p in predicted if p is not RelationLabel.SUPPORTS) / len(predicted)
        if predicted
        else 0.0
    )
    unknown_rate = (
        sum(1 for p in predicted if p is RelationLabel.UNKNOWN) / len(predicted)
        if predicted
        else 0.0
    )
    values = {
        "twin_not_supported_accuracy": twin_accuracy,
        "gold_supports_recall": gold_recall,
    }
    failures = tuple(sorted(key for key, value in values.items() if value < THRESHOLDS[key]))
    return TaskReport(
        n_twin=len(twin),
        n_gold=len(gold_rows),
        twin_not_supported_accuracy=twin_accuracy,
        gold_supports_recall=gold_recall,
        not_supported_rate=not_supported_rate,
        unknown_rate=unknown_rate,
        failures=failures,
    )
