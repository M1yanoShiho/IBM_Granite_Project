import pytest

from evidence_rag.relations.gate0b import THRESHOLDS, external_report, task_report
from evidence_rag.relations.models import RelationLabel

S = RelationLabel.SUPPORTS
R = RelationLabel.REFUTES
U = RelationLabel.UNKNOWN


def test_thresholds_are_the_pre_registered_values() -> None:
    """These are pre-registered. A change here is a protocol change, not a code change."""
    assert THRESHOLDS == {
        "refutes_precision": 0.85,
        "macro_f1": 0.80,
        "non_unknown_coverage": 0.80,
        "support_coverage": 0.70,
        "refutes_coverage": 0.70,
        "twin_refutes_accuracy": 0.70,
        "gold_supports_recall": 0.85,
    }


def test_external_report_computes_per_class_precision_and_coverage() -> None:
    report = external_report(gold=[S, R, U, R], predicted=[S, R, U, U])
    assert report.refutes_precision == 1.0
    assert report.refutes_recall == 0.5
    assert report.non_unknown_coverage == 0.5
    assert report.n == 4


def test_external_report_flags_only_the_thresholds_that_failed() -> None:
    report = external_report(gold=[R, R], predicted=[R, S])
    assert report.passes is False
    assert "refutes_precision" not in report.failures
    assert "support_coverage" in report.failures
    assert "refutes_coverage" in report.failures


def test_external_report_passes_when_every_threshold_is_met() -> None:
    gold = [S] * 10 + [R] * 10
    report = external_report(gold=gold, predicted=list(gold))
    assert report.passes is True
    assert report.failures == ()
    assert report.macro_f1 == 1.0


def test_external_report_rejects_an_empty_evaluation() -> None:
    with pytest.raises(ValueError, match="at least one pair"):
        external_report(gold=[], predicted=[])


def test_absent_class_fails_rather_than_silently_passing() -> None:
    """No SUPPORTS examples means a broken split, not a satisfied coverage gate."""
    report = external_report(gold=[R, R], predicted=[R, R])
    assert "support_coverage" in report.failures


def test_task_report_counts_unknown_as_failure() -> None:
    """Otherwise a model that abstains everywhere games the gate."""
    report = task_report(
        kinds=["cf_gold", "needle_replacement", "needle_gold"],
        gold=[R, R, S],
        predicted=[R, U, U],
    )
    assert report.twin_refutes_accuracy == 0.5
    assert report.gold_supports_recall == 0.0
    assert report.unknown_rate == pytest.approx(2 / 3)
    assert report.passes is False


def test_task_report_passes_when_both_thresholds_are_met() -> None:
    report = task_report(
        kinds=["cf_gold", "needle_replacement", "needle_gold", "needle_gold"],
        gold=[R, R, S, S],
        predicted=[R, R, S, S],
    )
    assert report.twin_refutes_accuracy == 1.0
    assert report.gold_supports_recall == 1.0
    assert report.n_twin == 2
    assert report.n_gold == 2
    assert report.passes is True


def test_cf_replacement_rows_are_context_not_a_gate() -> None:
    """Only the twin-refutes and gold-supports rows are scored; cf_replacement is reported
    elsewhere but must not silently enter either threshold."""
    report = task_report(
        kinds=["cf_gold", "needle_replacement", "needle_gold", "cf_replacement"],
        gold=[R, R, S, S],
        predicted=[R, R, S, U],
    )
    assert report.n_twin == 2
    assert report.n_gold == 1
    assert report.passes is True


def test_a_model_that_abstains_everywhere_fails_both_gates() -> None:
    report = task_report(
        kinds=["cf_gold", "needle_replacement", "needle_gold"],
        gold=[R, R, S],
        predicted=[U, U, U],
    )
    assert report.twin_refutes_accuracy == 0.0
    assert report.gold_supports_recall == 0.0
    assert report.unknown_rate == 1.0
    assert set(report.failures) == {"twin_refutes_accuracy", "gold_supports_recall"}
