"""Gate 0B: two-tier relation-model acceptance (Graph 2.0 design §3.2, §3.3).

0B-1 is external validity on VitaminC official test. 0B-2 is task validity on the mutation-log
probe. Both tiers exist because either alone is insufficient: a model can clear VitaminC and be
useless on NQ passages with template hypotheses, and a model can clear the task probe by
overfitting a synthetic mutation pattern.

All thresholds are pre-registered and must not move after seeing results.

UNKNOWN counts as FAILURE in 0B-2. Without that, a model that abstains everywhere scores
perfectly on "did not assert a false conflict" — the same manipulability that forced the
false_conflict guardrail to be a joint gate rather than a single rate.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from evidence_rag.relations.models import RelationLabel
from evidence_rag.relations.task_probe import GOLD_SUPPORTS, TWIN_REFUTES

THRESHOLDS = {
    "refutes_precision": 0.85,
    "macro_f1": 0.80,
    "non_unknown_coverage": 0.80,
    "support_coverage": 0.70,
    "refutes_coverage": 0.70,
    "twin_refutes_accuracy": 0.70,
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
    twin_refutes_accuracy: float
    gold_supports_recall: float
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
    """
    twin = [
        (g, p) for kind, g, p in zip(kinds, gold, predicted, strict=True) if kind in TWIN_REFUTES
    ]
    gold_rows = [
        (g, p) for kind, g, p in zip(kinds, gold, predicted, strict=True) if kind in GOLD_SUPPORTS
    ]
    twin_accuracy = sum(1 for g, p in twin if p == g) / len(twin) if twin else 0.0
    gold_recall = sum(1 for g, p in gold_rows if p == g) / len(gold_rows) if gold_rows else 0.0
    unknown_rate = (
        sum(1 for p in predicted if p is RelationLabel.UNKNOWN) / len(predicted)
        if predicted
        else 0.0
    )
    values = {
        "twin_refutes_accuracy": twin_accuracy,
        "gold_supports_recall": gold_recall,
    }
    failures = tuple(sorted(key for key, value in values.items() if value < THRESHOLDS[key]))
    return TaskReport(
        n_twin=len(twin),
        n_gold=len(gold_rows),
        twin_refutes_accuracy=twin_accuracy,
        gold_supports_recall=gold_recall,
        unknown_rate=unknown_rate,
        failures=failures,
    )
