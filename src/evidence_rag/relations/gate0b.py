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

0B-1 IS NOT AMENDED, AND NO LONGER COMPOSES WITH 0B-2. A1 §9.1 changes the 0B-2 metric and says
nothing about this tier, so `external_report`, its five thresholds and `vitaminc.py` are left
exactly as they were. Three of those five are now undefined or vacuous against a binary
predictor — `refutes_precision` and `refutes_coverage` read 0.0 because the model can never
predict REFUTES, `macro_f1` is capped at 0.5 for the same reason, and `non_unknown_coverage`
reads 1.0 unconditionally. This is a KNOWN GAP IN THE AMENDMENT awaiting a human ruling, not an
oversight in this module, and it must not be papered over by inventing a gold mapping here. See
`tests/cli/test_gate0b.py::test_0b1_is_UNRUNNABLE_after_A1_and_this_test_records_it_rather_than_fixing_it`.
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

    `not_supported_rate` replaces the pre-A1 `unknown_rate`. It is the G-AB report item (M0
    §3.6) in the only form the binary space supports: A1 §9.3 concedes binary cannot separate
    "abstained" from "committed to the contrary", so the two are counted together, and the
    sparse-graph consequence is picked up by `gold_supports_recall` inside the joint gate. A
    field still called `unknown_rate` would have reported a structural 0.0 forever.
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
        failures=failures,
    )
