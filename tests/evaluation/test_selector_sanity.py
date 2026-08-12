import hashlib
import json
import math
from typing import Literal

import pytest
from pydantic import ValidationError

from evidence_rag.evaluation.selector_sanity import (
    SANITY_PROTOCOL_VERSION,
    SanityCandidateScoreRow,
    SanityLabelRecord,
    SanityPolicyQueryOutcome,
    SanityQuantileThreshold,
    SanitySourceHeadLossTrend,
    SanityTrainingPrediction,
    SanityTrainingTrace,
    canonical_model_bytes,
    compute_class_weights,
    decide_safe_corner,
    derive_train_fit_thresholds,
    evaluate_policy_metrics,
    evaluate_training_sanity,
    nearest_rank_quantiles,
    sanity_sample_digest,
    select_sanity_queries,
)


def test_sanity_sampling_uses_literal_frozen_digest_and_is_order_independent() -> None:
    ids = {
        "niah": tuple(f"n-{index:02d}" for index in range(20)),
        "2wiki": tuple(f"w-{index:02d}" for index in range(20)),
    }
    expected_digest = hashlib.sha256(b"selector-r005-sanity-v1\nniah\nn-00\n20260811").hexdigest()
    assert sanity_sample_digest("niah", "n-00") == expected_digest

    selected = select_sanity_queries(ids)
    reversed_selected = select_sanity_queries(
        {"2wiki": tuple(reversed(ids["2wiki"])), "niah": tuple(reversed(ids["niah"]))}
    )

    assert selected == reversed_selected
    assert len(selected.queries) == 32
    assert [row.dataset_kind for row in selected.queries[:16]] == ["niah"] * 16
    assert [row.selection_index for row in selected.queries[16:]] == list(range(1, 17))
    assert all(row.role == "train-fit" for row in selected.queries)


def test_sanity_sampling_rejects_wrong_sources_duplicate_ids_and_short_input() -> None:
    with pytest.raises(ValueError, match="exactly"):
        select_sanity_queries({"niah": tuple(f"n-{index}" for index in range(16))})
    with pytest.raises(ValueError, match="duplicate"):
        select_sanity_queries(
            {
                "niah": tuple(["same"] * 16),
                "2wiki": tuple(f"w-{index}" for index in range(16)),
            }
        )
    with pytest.raises(ValueError, match="fewer than 16"):
        select_sanity_queries(
            {
                "niah": tuple(f"n-{index}" for index in range(15)),
                "2wiki": tuple(f"w-{index}" for index in range(16)),
            }
        )


def _label(
    *,
    dataset_kind: str,
    query_id: str,
    evidence_id: str,
    protect: int | None,
    harm: int | None,
    role: str = "train-fit",
) -> dict[str, object]:
    return {
        # Extra fields mirror ordinary R004 label mappings and are intentionally projected out.
        "schema_version": "1.0",
        "dataset_kind": dataset_kind,
        "query_id": query_id,
        "evidence_id": evidence_id,
        "role": role,
        "protect_label": protect,
        "protect_mask": protect is not None,
        "harm_label": harm,
        "harm_mask": harm is not None,
    }


def test_class_weights_are_inverse_sqrt_and_active_example_mean_is_one() -> None:
    rows = [
        _label(
            dataset_kind="niah",
            query_id="q0",
            evidence_id="p0",
            protect=0,
            harm=1,
        )
    ]
    rows.extend(
        _label(
            dataset_kind="niah",
            query_id=f"q{index}",
            evidence_id=f"p{index}",
            protect=1,
            harm=0,
        )
        for index in range(1, 5)
    )
    rows.extend(
        _label(
            dataset_kind="2wiki",
            query_id=f"w{index}",
            evidence_id=f"s{index}",
            protect=1,
            harm=None,
        )
        for index in range(3)
    )

    weights = compute_class_weights(rows)
    by_key = {(row.dataset_kind, row.head, row.class_label): row for row in weights}

    assert set(by_key) == {
        ("niah", "protect", 0),
        ("niah", "protect", 1),
        ("niah", "harm", 0),
        ("niah", "harm", 1),
        ("2wiki", "protect", 1),
    }
    assert by_key[("niah", "protect", 0)].inverse_sqrt_frequency == pytest.approx(1.0)
    assert by_key[("niah", "protect", 1)].inverse_sqrt_frequency == pytest.approx(0.5)
    weighted_mean = (
        by_key[("niah", "protect", 0)].normalized_weight
        + 4 * by_key[("niah", "protect", 1)].normalized_weight
    ) / 5
    assert weighted_mean == pytest.approx(1.0)
    assert by_key[("2wiki", "protect", 1)].normalized_weight == pytest.approx(1.0)


def test_class_weights_reject_modelval_and_duplicate_rows_without_dividing_absent_class() -> None:
    modelval = _label(
        dataset_kind="niah",
        query_id="q",
        evidence_id="e",
        protect=1,
        harm=0,
        role="train-modelval",
    )
    with pytest.raises(ValueError, match="train-fit"):
        compute_class_weights((modelval,))
    train = {**modelval, "role": "train-fit"}
    with pytest.raises(ValueError, match="duplicate"):
        compute_class_weights((train, train))


def _score_row(
    dataset_kind: Literal["niah", "2wiki"],
    query_id: str,
    rank: int,
    *,
    protect: float,
    harm: float,
    role: Literal["train-fit", "train-modelval"] = "train-fit",
) -> SanityCandidateScoreRow:
    return SanityCandidateScoreRow(
        dataset_kind=dataset_kind,
        query_id=query_id,
        evidence_id=f"{query_id}-e{rank:02d}",
        document_id=f"{query_id}-d{rank:02d}",
        role=role,
        retrieval_rank=rank,
        score_scope=(
            "train-modelval-heldout"
            if role == "train-modelval"
            else ("sanity-sample-overfit-extra" if rank > 10 else "train-fit-quantile")
        ),
        text_pair_sha256="a" * 64,
        protect_score=protect,
        harm_score=harm,
        safe_score=min(harm, 1.0 - protect),
    )


def test_nearest_rank_quantiles_keep_conservative_order_and_duplicate_thresholds() -> None:
    thresholds = nearest_rank_quantiles([0.25] * 100)
    assert [row.quantile for row in thresholds] == [0.99, 0.975, 0.95, 0.90]
    assert [row.nearest_rank for row in thresholds] == [99, 98, 95, 90]
    assert [row.threshold for row in thresholds] == [0.25] * 4


def test_thresholds_use_both_sources_all_train_fit_top10_rows() -> None:
    source_queries: tuple[tuple[Literal["niah", "2wiki"], str], ...] = (
        ("niah", "n1"),
        ("2wiki", "w1"),
    )
    rows = tuple(
        _score_row(
            dataset_kind,
            query_id,
            rank,
            protect=0.0,
            harm=rank / 20,
        )
        for dataset_kind, query_id in source_queries
        for rank in range(1, 11)
    )

    thresholds = derive_train_fit_thresholds(rows, expected_query_counts={"niah": 1, "2wiki": 1})

    assert all(row.score_count == 20 for row in thresholds)
    assert thresholds[0].threshold == pytest.approx(0.5)
    assert thresholds[-1].threshold == pytest.approx(0.45)


def test_threshold_derivation_rejects_wrong_role_duplicate_or_incomplete_top20() -> None:
    dataset_kinds: tuple[Literal["niah", "2wiki"], ...] = ("niah", "2wiki")
    complete = [
        _score_row(kind, f"{kind}-q", rank, protect=0.1, harm=0.2)
        for kind in dataset_kinds
        for rank in range(1, 11)
    ]
    wrong_role = complete.copy()
    wrong_role[0] = _score_row("niah", "niah-q", 1, protect=0.1, harm=0.2, role="train-modelval")
    with pytest.raises(ValueError, match="train-fit"):
        derive_train_fit_thresholds(
            tuple(wrong_role), expected_query_counts={"niah": 1, "2wiki": 1}
        )
    with pytest.raises(ValueError, match="duplicate"):
        derive_train_fit_thresholds(
            tuple(complete + [complete[0]]),
            expected_query_counts={"niah": 1, "2wiki": 1},
        )
    with pytest.raises(ValueError, match="ranks 1..10"):
        derive_train_fit_thresholds(
            tuple(complete[:-1]), expected_query_counts={"niah": 1, "2wiki": 1}
        )
    with pytest.raises(ValueError, match="ranks 1..10"):
        derive_train_fit_thresholds(
            tuple(complete + [_score_row("niah", "niah-q", 11, protect=0.1, harm=0.2)]),
            expected_query_counts={"niah": 1, "2wiki": 1},
        )
    with pytest.raises(ValueError, match="expected 920"):
        derive_train_fit_thresholds(tuple(complete))


def _prediction(
    dataset_kind: Literal["niah", "2wiki"],
    query_id: str,
    evidence_id: str,
    *,
    protect_label: Literal[0, 1] | None,
    harm_label: Literal[0, 1] | None,
    protect_score: float,
    harm_score: float,
) -> SanityTrainingPrediction:
    return SanityTrainingPrediction(
        dataset_kind=dataset_kind,
        query_id=query_id,
        evidence_id=evidence_id,
        protect_label=protect_label,
        protect_mask=protect_label is not None,
        harm_label=harm_label,
        harm_mask=harm_label is not None,
        protect_score=protect_score,
        harm_score=harm_score,
        safe_score=min(harm_score, 1.0 - protect_score),
    )


def _perfect_training_predictions() -> tuple[SanityTrainingPrediction, ...]:
    rows: list[SanityTrainingPrediction] = []
    for index in range(16):
        query_id = f"n-{index:02d}"
        rows.append(
            _prediction(
                "niah",
                query_id,
                f"{query_id}-clean",
                protect_label=1,
                harm_label=0,
                protect_score=0.99,
                harm_score=0.01,
            )
        )
        rows.append(
            _prediction(
                "niah",
                query_id,
                f"{query_id}-cf",
                protect_label=0,
                harm_label=1,
                protect_score=0.01,
                harm_score=0.99,
            )
        )
    rows.extend(
        _prediction(
            "2wiki",
            f"w-{index:02d}",
            f"w-{index:02d}-support",
            protect_label=1,
            harm_label=None,
            protect_score=0.99,
            harm_score=0.1,
        )
        for index in range(16)
    )
    return tuple(rows)


def _complete_trace() -> SanityTrainingTrace:
    loss_keys: tuple[tuple[Literal["niah", "2wiki"], Literal["protect", "harm"]], ...] = (
        ("niah", "protect"),
        ("niah", "harm"),
        ("2wiki", "protect"),
    )
    return SanityTrainingTrace(
        termination_reason="completed",
        epochs_completed=30,
        final_checkpoint_epoch=30,
        epoch_mean_losses=tuple(1.0 / (index + 1) for index in range(30)),
        active_source_head_loss_trends=tuple(
            SanitySourceHeadLossTrend(
                dataset_kind=dataset_kind,
                head=head,
                initial_loss=1.0,
                final_loss=0.1,
                decreased=True,
            )
            for dataset_kind, head in loss_keys
        ),
        protect_head_parameter_l2_change=0.1,
        harm_head_parameter_l2_change=0.2,
        twowiki_harm_head_gradient_l1=0.0,
        independent_sigmoid_heads=True,
        oom_encountered=False,
        nonfinite_encountered=False,
    )


def _early_trace(
    termination_reason: str,
    *,
    epochs_completed: int,
) -> SanityTrainingTrace:
    return SanityTrainingTrace.model_validate(
        {
            "termination_reason": termination_reason,
            "epochs_completed": epochs_completed,
            "failure_epoch": (
                None if termination_reason == "sample-coverage" else epochs_completed + 1
            ),
            "final_checkpoint_epoch": epochs_completed,
            "epoch_mean_losses": tuple(1.0 / (index + 1) for index in range(epochs_completed)),
            "active_source_head_loss_trends": (),
            "protect_head_parameter_l2_change": 0.0,
            "harm_head_parameter_l2_change": 0.0,
            "twowiki_harm_head_gradient_l1": None,
            "independent_sigmoid_heads": True,
            "oom_encountered": termination_reason == "cuda-oom",
            "nonfinite_encountered": termination_reason == "nonfinite",
        }
    )


def test_training_sanity_passes_per_class_and_all_three_pair_directions() -> None:
    decision = evaluate_training_sanity(_perfect_training_predictions(), _complete_trace())

    assert decision.status == "PASS"
    assert len(decision.class_accuracy) == 5
    assert all(row.accuracy == 1.0 for row in decision.class_accuracy)
    assert decision.niah_pair_direction.paired_queries == 16
    assert decision.niah_pair_direction.all_three_accuracy == 1.0
    assert decision.failed_checks == ()


def test_training_trace_accepts_all_frozen_termination_states() -> None:
    completed = _complete_trace()
    sample_coverage = _early_trace("sample-coverage", epochs_completed=0)
    cuda_oom = _early_trace("cuda-oom", epochs_completed=0)
    nonfinite = _early_trace("nonfinite", epochs_completed=29)

    assert completed.failure_epoch is None
    assert sample_coverage.final_checkpoint_epoch == 0
    assert sample_coverage.failure_epoch is None
    assert cuda_oom.failure_epoch == 1
    assert nonfinite.failure_epoch == 30
    assert nonfinite.final_checkpoint_epoch == 29
    for trace in (sample_coverage, cuda_oom, nonfinite):
        assert trace.active_source_head_loss_trends == ()
        assert trace.twowiki_harm_head_gradient_l1 is None
        decision = evaluate_training_sanity((), trace)
        assert decision.status == "FAIL"
        assert not decision.class_coverage_pass
        assert not decision.class_accuracy_pass
        assert not decision.pair_direction_pass
        assert not decision.execution_pass


@pytest.mark.parametrize(
    ("trace", "update", "message"),
    (
        (
            _complete_trace(),
            {
                "epochs_completed": 29,
                "final_checkpoint_epoch": 29,
                "epoch_mean_losses": _complete_trace().epoch_mean_losses[:29],
            },
            "all 30 epochs",
        ),
        (_complete_trace(), {"failure_epoch": 30}, "failure epoch"),
        (_complete_trace(), {"oom_encountered": True}, "cannot record OOM"),
        (_complete_trace(), {"active_source_head_loss_trends": ()}, "exactly NIAH"),
        (_complete_trace(), {"twowiki_harm_head_gradient_l1": None}, "gradient probe"),
        (
            _early_trace("sample-coverage", epochs_completed=0),
            {"final_checkpoint_epoch": None},
            "epoch-0 checkpoint",
        ),
        (
            _early_trace("sample-coverage", epochs_completed=0),
            {"failure_epoch": 1},
            "before any training epoch",
        ),
        (
            _early_trace("sample-coverage", epochs_completed=0),
            {"nonfinite_encountered": True},
            "cannot record OOM",
        ),
        (
            _early_trace("cuda-oom", epochs_completed=0),
            {"failure_epoch": 2},
            "epoch after",
        ),
        (
            _early_trace("cuda-oom", epochs_completed=0),
            {"oom_encountered": False},
            "exactly one runtime failure flag",
        ),
        (
            _early_trace("cuda-oom", epochs_completed=0),
            {"nonfinite_encountered": True},
            "exactly one runtime failure flag",
        ),
        (
            _early_trace("nonfinite", epochs_completed=29),
            {
                "epochs_completed": 30,
                "failure_epoch": 30,
                "final_checkpoint_epoch": 30,
                "epoch_mean_losses": tuple(1.0 / (index + 1) for index in range(30)),
            },
            "0..29 completed epochs",
        ),
        (
            _early_trace("nonfinite", epochs_completed=29),
            {"twowiki_harm_head_gradient_l1": 0.0},
            "final gradient probe",
        ),
    ),
)
def test_training_trace_rejects_cross_state_combinations(
    trace: SanityTrainingTrace,
    update: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        SanityTrainingTrace.model_validate({**trace.model_dump(), **update})


def test_early_training_failure_is_conservative_even_if_predictions_exist() -> None:
    decision = evaluate_training_sanity(
        _perfect_training_predictions(),
        _early_trace("cuda-oom", epochs_completed=4),
    )

    assert decision.status == "FAIL"
    assert decision.class_coverage_pass
    assert decision.class_accuracy_pass
    assert decision.pair_direction_pass
    assert not decision.execution_pass


def test_completed_training_requires_predictions_and_exact_16_plus_16_queries() -> None:
    with pytest.raises(ValueError, match="requires final-epoch predictions"):
        evaluate_training_sanity((), _complete_trace())

    only_fifteen_twowiki = tuple(
        row
        for row in _perfect_training_predictions()
        if not (row.dataset_kind == "2wiki" and row.query_id == "w-15")
    )
    with pytest.raises(ValueError, match="exactly 16 queries per dataset"):
        evaluate_training_sanity(only_fifteen_twowiki, _complete_trace())


def test_training_sanity_treats_ties_as_pair_failures_and_requires_final_epoch() -> None:
    predictions = list(_perfect_training_predictions())
    clean = predictions[0]
    predictions[1] = predictions[1].model_copy(
        update={
            "protect_score": clean.protect_score,
            "harm_score": clean.harm_score,
            "safe_score": clean.safe_score,
        }
    )
    decision = evaluate_training_sanity(tuple(predictions), _complete_trace())
    assert decision.status == "FAIL"
    assert decision.niah_pair_direction.all_three_correct == 15
    assert not decision.pair_direction_pass

    incomplete = _complete_trace().model_copy(
        update={
            "epochs_completed": 29,
            "final_checkpoint_epoch": 29,
            "epoch_mean_losses": _complete_trace().epoch_mean_losses[:29],
        }
    )
    assert evaluate_training_sanity(_perfect_training_predictions(), incomplete).status == "FAIL"


def test_training_execution_gate_requires_loss_drop_both_heads_changed_and_zero_masked_gradient() -> (
    None
):
    predictions = _perfect_training_predictions()
    base = _complete_trace()
    for update in (
        {"protect_head_parameter_l2_change": 0.0},
        {"harm_head_parameter_l2_change": 0.0},
        {"twowiki_harm_head_gradient_l1": 1e-9},
    ):
        decision = evaluate_training_sanity(predictions, base.model_copy(update=update))
        assert decision.status == "FAIL"
        assert not decision.execution_pass

    worsening = SanitySourceHeadLossTrend(
        dataset_kind="niah",
        head="protect",
        initial_loss=0.1,
        final_loss=0.2,
        decreased=False,
    )
    loss_trace = base.model_copy(
        update={
            "active_source_head_loss_trends": (
                worsening,
                *base.active_source_head_loss_trends[1:],
            )
        }
    )
    assert not evaluate_training_sanity(predictions, loss_trace).execution_pass


def test_training_sanity_rejects_duplicate_predictions_and_nonfinite_scores() -> None:
    predictions = _perfect_training_predictions()
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_training_sanity(predictions + (predictions[0],), _complete_trace())
    with pytest.raises(ValidationError, match="finite"):
        _prediction(
            "niah",
            "q",
            "e",
            protect_label=1,
            harm_label=0,
            protect_score=math.nan,
            harm_score=0.0,
        )


def _point(index: int, threshold: float) -> SanityQuantileThreshold:
    quantiles = (0.99, 0.975, 0.95, 0.90)
    return SanityQuantileThreshold(
        point_index=index,
        quantile=quantiles[index - 1],
        threshold=threshold,
        nearest_rank=math.ceil(quantiles[index - 1] * 100),
        score_count=100,
    )


def _outcomes(*, delete_one_harm: bool) -> tuple[SanityPolicyQueryOutcome, ...]:
    rows: list[SanityPolicyQueryOutcome] = []
    for index in range(103):
        baseline = ("gold", "harm", f"n-noise-{index}")
        drops = delete_one_harm and index == 0
        rows.append(
            SanityPolicyQueryOutcome(
                dataset_kind="niah",
                query_id=f"n-{index:03d}",
                baseline_document_ids=baseline,
                selected_document_ids=(("gold", f"n-noise-{index}") if drops else baseline),
                dropped_evidence_ids=("harm-evidence",) if drops else (),
                dropped_document_ids=("harm",) if drops else (),
                required_document_ids=("gold",),
                harmful_document_id="harm",
                harmful_in_top20_pool=True,
            )
        )
    rows.extend(
        SanityPolicyQueryOutcome(
            dataset_kind="2wiki",
            query_id=f"w-{index:03d}",
            baseline_document_ids=("support-a", "support-b", f"w-noise-{index}"),
            selected_document_ids=("support-a", "support-b", f"w-noise-{index}"),
            dropped_evidence_ids=(),
            dropped_document_ids=(),
            required_document_ids=("support-a", "support-b"),
        )
        for index in range(300)
    )
    return tuple(rows)


def test_policy_metrics_compute_relative_recall_chain_harm_and_precision() -> None:
    metrics = evaluate_policy_metrics(
        point=_point(1, 0.9),
        outcomes=_outcomes(delete_one_harm=True),
        random_repeat_precisions=(0.1,) * 100,
    )

    assert metrics.niah.queries == 103
    assert metrics.twowiki.queries == 300
    assert metrics.niah.actual_deletions == 1
    assert metrics.niah.topk10_relative_recall_loss == 0.0
    assert metrics.niah.conditional_chain_loss == 0.0
    assert metrics.niah_harm.pool_conditional_harmful_reduction == pytest.approx(1 / 103)
    assert metrics.niah_harm.deletion_precision == 1.0
    assert metrics.count_matched_random.mean_precision == pytest.approx(0.1)


def test_safe_corner_uses_strict_gates_and_first_conservative_witness() -> None:
    no_drop = evaluate_policy_metrics(
        point=_point(1, 0.9),
        outcomes=_outcomes(delete_one_harm=False),
        random_repeat_precisions=(None,) * 100,
    )
    passing = tuple(
        evaluate_policy_metrics(
            point=_point(index, threshold),
            outcomes=_outcomes(delete_one_harm=True),
            random_repeat_precisions=(0.0,) * 100,
        )
        for index, threshold in ((2, 0.8), (3, 0.7), (4, 0.6))
    )

    decision = decide_safe_corner((no_drop, *passing))

    assert decision.status == "PASS"
    assert decision.witness_point_index == 2
    assert not decision.points[0].actual_nonzero_drop_pass
    assert no_drop.count_matched_random.mean_precision is None
    assert decision.points[1].passed
    assert decision.points[1].niah_recall_loss_pass
    assert decision.points[1].twowiki_recall_loss_pass
    assert decision.points[1].niah_conditional_chain_loss_pass
    assert decision.points[1].twowiki_conditional_chain_loss_pass
    assert not decision.thresholds_reused_by_r006_or_r007


def test_safe_corner_requires_precision_strictly_above_random_mean() -> None:
    metrics = tuple(
        evaluate_policy_metrics(
            point=_point(index, threshold),
            outcomes=_outcomes(delete_one_harm=True),
            random_repeat_precisions=(1.0,) * 100,
        )
        for index, threshold in ((1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6))
    )
    decision = decide_safe_corner(metrics)
    assert decision.status == "CUT"
    assert decision.witness_point_index is None
    assert all(not point.deletion_precision_pass for point in decision.points)


def test_policy_metrics_reject_wrong_query_count_duplicate_ids_and_random_count() -> None:
    outcomes = _outcomes(delete_one_harm=True)
    with pytest.raises(ValueError, match="expected 300"):
        evaluate_policy_metrics(
            point=_point(1, 0.9),
            outcomes=outcomes[:-1],
            random_repeat_precisions=(0.0,) * 100,
        )
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_policy_metrics(
            point=_point(1, 0.9),
            outcomes=outcomes + (outcomes[0],),
            random_repeat_precisions=(0.0,) * 100,
        )
    with pytest.raises(ValueError, match="exactly 100"):
        evaluate_policy_metrics(
            point=_point(1, 0.9),
            outcomes=outcomes,
            random_repeat_precisions=(0.0,) * 99,
        )


def test_random_precision_cannot_fake_zero_when_selector_deleted_nothing() -> None:
    with pytest.raises(ValueError, match="zero NIAH deletion requires 100 N/A"):
        evaluate_policy_metrics(
            point=_point(1, 0.9),
            outcomes=_outcomes(delete_one_harm=False),
            random_repeat_precisions=(0.0,) * 100,
        )


def test_models_are_strict_and_canonical_bytes_reject_undeclared_fields() -> None:
    row = SanityLabelRecord(
        dataset_kind="niah",
        query_id="q",
        evidence_id="e",
        role="train-fit",
        protect_label=1,
        protect_mask=True,
        harm_label=0,
        harm_mask=True,
    )
    payload = canonical_model_bytes(row)
    assert payload.endswith(b"\n")
    assert json.loads(payload)["query_id"] == "q"
    assert SANITY_PROTOCOL_VERSION == "selector-r005-sanity-v1"
    with pytest.raises(ValidationError, match="Extra inputs"):
        SanityLabelRecord.model_validate({**row.model_dump(), "unexpected": True})
