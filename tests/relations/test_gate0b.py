import dataclasses

import pytest

from evidence_rag.relations.gate0b import THRESHOLDS, external_report, task_report
from evidence_rag.relations.models import RelationLabel

S = RelationLabel.SUPPORTS
# Derived, not predicted, after A2 — still emitted directly by a natively-binary arm, still
# carried by A1-era dumps and by the 0B-2 probe's gold column.
N = RelationLabel.NOT_SUPPORTED
# Predicted again after A2. Also reachable in 0B-1 gold and in the pre-A1 dumps that §9.10a says
# R012 / R012b are recomputed from.
R = RelationLabel.REFUTES
U = RelationLabel.UNKNOWN


def test_r012s_published_0b1_readings_still_score_the_same_way_under_A2() -> None:
    """A2 §10.5's central claim, machine-checked: the tier reopened without a threshold moving.

    These are R012's recorded 0B-1 readings (job 18235972, n_external=55197,
    `results/gate0b/sweep-full.json`), produced 2026-08-02 by the ORIGINAL three-class
    implementation — before A1 existed. A2 restored that output space, so the same numbers must
    still land on the same side of the same thresholds. Any silent recalibration of .85/.80/.70
    surfaces here, which is what option (a) in §9.11 would have done invisibly.

    Note which arm gets which verdict. albert passes all five; DeBERTa — the arm with the best
    `gold_supports_recall` in the entire sweep (.7942) — fails three. Restoring this tier makes
    Gate 0B STRICTER, which is the asymmetry §10.0 rests on: A1 made the gate easier to pass and
    A2 makes it harder.
    """
    albert = {
        "refutes_precision": 0.9026491385320187,
        "macro_f1": 0.9215112105529577,
        "non_unknown_coverage": 0.8716959255031976,
        "support_coverage": 0.9507663389242337,
        "refutes_coverage": 0.8944979027880582,
    }
    deberta = {
        "refutes_precision": 0.9126615843553199,
        "macro_f1": 0.7582379949558993,
        "non_unknown_coverage": 0.6750185698498107,
        "support_coverage": 0.79786003470214,
        "refutes_coverage": 0.5434986429805083,
    }
    assert [key for key, value in albert.items() if value < THRESHOLDS[key]] == []
    assert sorted(key for key, value in deberta.items() if value < THRESHOLDS[key]) == [
        "macro_f1",
        "non_unknown_coverage",
        "refutes_coverage",
    ]


def test_the_0b1_report_carries_exactly_the_fields_r012_published() -> None:
    """The tier's output contract, pinned against the artefact rather than against intent.

    R012's `external` block carries these ten keys. A2 claims it RESTORED 0B-1 rather than
    rebuilding it, and a changed field set would quietly falsify that: anyone comparing a new
    sweep against the R012 artefact would be comparing different quantities without being told.
    """
    report = external_report(gold=[S, R, U], predicted=[S, R, U])
    assert set(dataclasses.asdict(report)) == {
        "failures",
        "macro_f1",
        "n",
        "non_unknown_coverage",
        "refutes_coverage",
        "refutes_precision",
        "refutes_recall",
        "support_coverage",
        "supports_precision",
        "supports_recall",
    }


def test_thresholds_are_the_pre_registered_values() -> None:
    """These are pre-registered. A change here is a protocol change, not a code change.

    A1 §9.1 renames the 0B-2 twin metric and leaves every VALUE alone, including .70 and .85.
    The 0B-1 keys are untouched — A1 says nothing about that tier.
    """
    assert THRESHOLDS == {
        "refutes_precision": 0.85,
        "macro_f1": 0.80,
        "non_unknown_coverage": 0.80,
        "support_coverage": 0.70,
        "refutes_coverage": 0.70,
        "twin_not_supported_accuracy": 0.70,
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


def test_twin_metric_counts_anything_that_is_not_supports() -> None:
    """A1 §9.1: `twin NOT-SUPPORTED accuracy` = the share predicted != SUPPORTED.

    The joint gate is now the SOLE anti-gaming mechanism (§9.3): a model that says
    NOT_SUPPORTED everywhere scores 1.0 here and 0.0 on gold recall, so it still cannot pass.
    """
    report = task_report(
        kinds=["cf_gold", "needle_replacement", "needle_gold"],
        gold=[N, N, S],
        predicted=[N, S, N],
    )
    assert report.twin_not_supported_accuracy == 0.5
    assert report.gold_supports_recall == 0.0
    assert report.not_supported_rate == pytest.approx(2 / 3)
    assert report.passes is False


def test_twin_metric_scores_a_legacy_three_class_prediction_as_not_supported() -> None:
    """THE §9.10a recomputation path. R012 / R012b are recomputed from `dump-*.jsonl`, which
    holds three-class argmax predictions, and the published binary readings were obtained by
    counting BOTH REFUTES and UNKNOWN as not-supported.

    Scoring `predicted == gold` instead would silently reproduce the OLD three-class twin number
    (here 0.5 instead of 1.0) while every field name claimed to be binary — the exact failure
    that is invisible in the aggregate.
    """
    report = task_report(
        kinds=["cf_gold", "needle_replacement"],
        gold=[N, N],
        predicted=[R, U],
    )
    assert report.twin_not_supported_accuracy == 1.0


def test_gold_recall_still_demands_the_word_supports_from_a_legacy_prediction() -> None:
    """§9.1 pins `gold_supports_recall` as UNCHANGED: predicted == SUPPORTS on needle_gold rows.
    A legacy UNKNOWN is not a support, so recomputing a dump must not credit it."""
    report = task_report(
        kinds=["needle_gold", "needle_gold"],
        gold=[S, S],
        predicted=[S, U],
    )
    assert report.gold_supports_recall == 0.5


def test_a_pre_A1_pairs_file_still_scores_and_scores_IDENTICALLY() -> None:
    """PASSED ON ARRIVAL — characterisation, not a new behaviour. It contradicts a premise of
    M0 §9.10a and is pinned so the contradiction cannot be undone silently.

    §9.10a predicted that changing the output space would make existing
    `data/gate0b/task_pairs*.jsonl` "directly fail to parse", because `cli/gate0b.py` reads gold
    back with `RelationLabel(row["label"])`. That does not happen here, for two reasons that are
    both consequences of following A1 literally:

      * A1 §9.1 RETAINS `CLAIM_REFUTES` as a schema edge type, so REFUTES stays in the enum and
        `RelationLabel("REFUTES")` still parses;
      * §9.1 defines both 0B-2 metrics against SUPPORTS, not against gold, so the gold column
        is not read by either metric.

    Consequence for sequencing: the R012b rung ladder does NOT need regenerated pair files to
    stay comparable, and rung 3 may be scored on the file it already has.
    """
    kinds = ["needle_gold", "cf_gold", "needle_replacement"]
    predicted = [S, N, N]
    pre_a1 = task_report(kinds=kinds, gold=[S, R, R], predicted=predicted)
    post_a1 = task_report(kinds=kinds, gold=[S, N, N], predicted=predicted)
    assert pre_a1 == post_a1
    assert pre_a1.twin_not_supported_accuracy == 1.0
    assert pre_a1.gold_supports_recall == 1.0


def test_task_report_passes_when_both_thresholds_are_met() -> None:
    report = task_report(
        kinds=["cf_gold", "needle_replacement", "needle_gold", "needle_gold"],
        gold=[N, N, S, S],
        predicted=[N, N, S, S],
    )
    assert report.twin_not_supported_accuracy == 1.0
    assert report.gold_supports_recall == 1.0
    assert report.n_twin == 2
    assert report.n_gold == 2
    assert report.passes is True


def test_cf_replacement_rows_are_context_not_a_gate() -> None:
    """Only the twin and gold-supports rows are scored; cf_replacement is reported elsewhere
    but must not silently enter either threshold."""
    report = task_report(
        kinds=["cf_gold", "needle_replacement", "needle_gold", "cf_replacement"],
        gold=[N, N, S, S],
        predicted=[N, N, S, N],
    )
    assert report.n_twin == 2
    assert report.n_gold == 1
    assert report.passes is True


def test_a_model_that_abstains_everywhere_is_stopped_by_the_joint_gate() -> None:
    """A1 §9.3's degeneracy table, row 1. Binary cannot tell abstention from contradiction, so
    the twin metric saturates at 1.0 — and the joint gate is what still stops it."""
    report = task_report(
        kinds=["cf_gold", "needle_replacement", "needle_gold"],
        gold=[N, N, S],
        predicted=[N, N, N],
    )
    assert report.twin_not_supported_accuracy == 1.0
    assert report.gold_supports_recall == 0.0
    assert report.not_supported_rate == 1.0
    assert report.failures == ("gold_supports_recall",)
    assert report.passes is False


def test_a_model_that_supports_everything_is_stopped_by_the_joint_gate() -> None:
    """A1 §9.3's degeneracy table, row 2. The mirror image, and the reason .70 survives as a
    lower-bound guard even though §9.5 concedes the twin metric has lost discriminating power."""
    report = task_report(
        kinds=["cf_gold", "needle_replacement", "needle_gold"],
        gold=[N, N, S],
        predicted=[S, S, S],
    )
    assert report.twin_not_supported_accuracy == 0.0
    assert report.gold_supports_recall == 1.0
    assert report.failures == ("twin_not_supported_accuracy",)
    assert report.passes is False


def test_a_value_exactly_at_the_threshold_passes() -> None:
    """Boundary semantics are >= not >. These gates decide whether weeks of training start, so
    'exactly .70' must not silently fail. A mutation from < to <= survived without this."""
    # 7 of 10 twin rows correct = .70 exactly; 17 of 20 gold rows = .85 exactly
    kinds = ["cf_gold"] * 10 + ["needle_gold"] * 20
    gold = [N] * 10 + [S] * 20
    predicted = [N] * 7 + [S] * 3 + [S] * 17 + [N] * 3
    report = task_report(kinds=kinds, gold=gold, predicted=predicted)
    assert report.twin_not_supported_accuracy == pytest.approx(0.70)
    assert report.gold_supports_recall == pytest.approx(0.85)
    assert report.passes is True


def test_a_value_just_below_the_threshold_fails() -> None:
    kinds = ["cf_gold"] * 10 + ["needle_gold"] * 20
    gold = [N] * 10 + [S] * 20
    predicted = [N] * 6 + [S] * 4 + [S] * 17 + [N] * 3
    report = task_report(kinds=kinds, gold=gold, predicted=predicted)
    assert report.twin_not_supported_accuracy == pytest.approx(0.60)
    assert report.failures == ("twin_not_supported_accuracy",)


def test_external_report_boundary_is_inclusive() -> None:
    """non_unknown_coverage exactly .80 must pass."""
    gold = [S] * 10 + [R] * 10
    predicted = [S] * 8 + [U] * 2 + [R] * 8 + [U] * 2
    report = external_report(gold=gold, predicted=predicted)
    assert report.non_unknown_coverage == pytest.approx(0.80)
    assert "non_unknown_coverage" not in report.failures


def test_macro_f1_is_the_mean_of_per_class_f1_not_of_precision() -> None:
    """Pins the VALUE, not just the pass/fail. With perfectly-scored inputs precision, recall and
    f1 all coincide, so an f1-vs-precision mix-up hides unless the arms are asymmetric."""
    report = external_report(gold=[S, S, R, R], predicted=[S, U, R, R])
    # SUPPORTS: precision 1.0, recall .5 -> f1 .667 | REFUTES: precision 1.0, recall 1.0 -> f1 1.0
    assert report.supports_precision == pytest.approx(1.0)
    assert report.supports_recall == pytest.approx(0.5)
    assert report.macro_f1 == pytest.approx((2 / 3 + 1.0) / 2)
    assert report.macro_f1 != pytest.approx(1.0)
