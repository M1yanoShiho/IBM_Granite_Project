import hashlib
import math
from pathlib import Path

import pytest

import evidence_rag.cli.run_selector_sanity as sanity_runner
from evidence_rag.cli.run_selector_preflight import _Pair
from evidence_rag.cli.run_selector_sanity import (
    _assess_sample_classes,
    _load_sanity_config,
    _NonFiniteTrainingError,
    _policy_contract_probes,
    _prepare_output_root,
    _project_query_universe,
    _quantile_policies,
    _require_empty_artifacts,
    _run_epoch_boundaries,
    _sample_rows,
    _ScoreRow,
    _validate_stored_epoch_rows,
    _validate_training_runtime_log,
    _weighted_bce_from_scores,
)
from evidence_rag.evaluation.selector_sanity import (
    SanitySampleQuery,
    SanitySourceHeadLossTrend,
    SanityTrainingTrace,
)


def _config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs/selector/adaptive_risk_r005_sanity.toml"


def test_tracked_r005_config_matches_the_runner_contract() -> None:
    _, sanity = _load_sanity_config(_config_path())

    assert sanity.batch_size == 4
    assert sanity.accumulation == 4
    assert sanity.learning_rate == pytest.approx(2e-5)


def test_r005_config_does_not_mutate_the_r004_pinned_config() -> None:
    r004_config = Path(__file__).resolve().parents[2] / "configs/selector/adaptive_risk_v1.toml"

    assert r004_config != _config_path()
    assert len(r004_config.read_bytes()) == 3921
    assert hashlib.sha256(r004_config.read_bytes()).hexdigest() == (
        "b5545848202964af77821af9ef92c360b2981c1aa6bdccc1b9d8d143218810ed"
    )


def test_source_superset_is_projected_to_the_r004_labelled_query_universe() -> None:
    projected = _project_query_universe(
        {"unlabelled": 0, "q2": 2, "q1": 1},
        {"q1", "q2"},
        label="fixture source",
    )

    assert projected == {"q1": 1, "q2": 2}
    with pytest.raises(ValueError, match="omits R004 labelled queries"):
        _project_query_universe({"q1": 1}, {"q1", "q2"}, label="fixture source")


def test_policy_contract_probes_cover_every_frozen_fallback() -> None:
    result = _policy_contract_probes()

    assert result["status"] == "PASS"
    checks = result["checks"]
    assert isinstance(checks, dict)
    assert len(checks) == 9
    assert all(checks.values())


def test_sample_rows_are_strict_core_records() -> None:
    sample = {
        "niah": tuple(f"n{index}" for index in range(16)),
        "2wiki": tuple(f"w{index}" for index in range(16)),
    }

    rows = _sample_rows(sample)

    assert len(rows) == 32
    assert tuple(SanitySampleQuery.model_validate(row) for row in rows)


def test_output_root_must_be_empty(tmp_path: Path) -> None:
    root = _prepare_output_root(tmp_path / "R005.staging")
    assert (root / "checkpoint").is_dir()
    assert (root / "sanity").is_dir()

    with pytest.raises(ValueError, match="must be empty"):
        _prepare_output_root(root)


def _full_quantile_universe() -> tuple[_ScoreRow, ...]:
    rows = []
    for kind, query_count in (("niah", 920), ("2wiki", 2700)):
        for query_index in range(query_count):
            for rank in range(1, 11):
                rows.append(
                    _ScoreRow(
                        dataset_kind=kind,
                        query_id=f"{kind}-q{query_index:04d}",
                        evidence_id=f"e{rank:02d}",
                        document_id=f"d{rank:02d}",
                        retrieval_rank=rank,
                        role="train-fit",
                        score_scope="train-fit-quantile",
                        text_pair_sha256="a" * 64,
                        protect_score=0.7,
                        harm_score=0.2,
                        safe_score=0.2,
                    )
                )
    return tuple(rows)


def test_runner_quantiles_require_the_full_unfiltered_trainfit_topk10() -> None:
    rows = _full_quantile_universe()

    policies, points = _quantile_policies(rows)

    assert [policy["policy_id"] for policy in policies] == [
        "Q990_CAP1",
        "Q975_CAP1",
        "Q950_CAP1",
        "Q900_CAP1",
    ]
    assert [point.threshold for point in points] == [0.2, 0.2, 0.2, 0.2]
    assert all(point.score_count == 36_200 for point in points)

    with pytest.raises(ValueError, match="must contain exactly ranks 1..10"):
        _quantile_policies(rows[:-1])


def _pair(
    *,
    kind: str,
    query_id: str,
    evidence_id: str,
    protect_label: int | None,
    protect_mask: bool,
    harm_label: int | None,
    harm_mask: bool,
) -> _Pair:
    return _Pair(
        dataset_kind=kind,  # type: ignore[arg-type]
        query_id=query_id,
        evidence_id=evidence_id,
        document_id=f"doc-{evidence_id}",
        retrieval_rank=1,
        question=f"question-{query_id}",
        candidate_text=f"candidate-{evidence_id}",
        role="train-fit",
        component_id="component",
        protect_label=protect_label,
        protect_mask=protect_mask,
        harm_label=harm_label,
        harm_mask=harm_mask,
    )


def _passing_active_pairs() -> dict[str, tuple[_Pair, ...]]:
    niah = []
    for index in range(12):
        query_id = f"niah-{index}"
        niah.extend(
            (
                _pair(
                    kind="niah",
                    query_id=query_id,
                    evidence_id=f"clean-{index}",
                    protect_label=1,
                    protect_mask=True,
                    harm_label=0,
                    harm_mask=True,
                ),
                _pair(
                    kind="niah",
                    query_id=query_id,
                    evidence_id=f"counterfactual-{index}",
                    protect_label=0,
                    protect_mask=True,
                    harm_label=1,
                    harm_mask=True,
                ),
            )
        )
    twowiki = (
        _pair(
            kind="2wiki",
            query_id="twowiki-0",
            evidence_id="positive-0",
            protect_label=1,
            protect_mask=True,
            harm_label=None,
            harm_mask=False,
        ),
    )
    return {"niah": tuple(niah), "2wiki": twowiki}


def test_sample_assessment_returns_structured_fail_instead_of_throwing() -> None:
    pairs = _passing_active_pairs()
    pairs["niah"] = pairs["niah"][:2]

    result = _assess_sample_classes(pairs)  # type: ignore[arg-type]

    assert result.status == "FAIL"
    assert result.strict_niah_pair_count == 1
    assert result.failed_checks == ("minimum-strict-niah-pairs",)
    assert result.as_report()["status"] == "FAIL"


def test_sample_assessment_reports_missing_classes_without_throwing() -> None:
    pairs = _passing_active_pairs()
    pairs["niah"] = tuple(
        pair for pair in pairs["niah"] if not (pair.protect_label == 0 and pair.harm_label == 1)
    )

    result = _assess_sample_classes(pairs)  # type: ignore[arg-type]

    assert result.status == "FAIL"
    assert result.missing_required_classes == (
        ("niah", "harm", 1),
        ("niah", "protect", 0),
    )
    assert "required-active-class-coverage" in result.failed_checks


def test_sample_assessment_passes_complete_coverage_and_twelve_strict_pairs() -> None:
    result = _assess_sample_classes(_passing_active_pairs())  # type: ignore[arg-type]

    assert result.status == "PASS"
    assert result.missing_required_classes == ()
    assert result.strict_niah_pair_count == 12


def test_sample_assessment_requires_both_heads_active_for_a_strict_pair() -> None:
    pairs = _passing_active_pairs()
    first = pairs["niah"][0]
    pairs["niah"] = (
        _pair(
            kind="niah",
            query_id=first.query_id,
            evidence_id=first.evidence_id,
            protect_label=1,
            protect_mask=True,
            harm_label=0,
            harm_mask=False,
        ),
        *pairs["niah"][1:],
    )

    result = _assess_sample_classes(pairs)  # type: ignore[arg-type]

    assert result.status == "FAIL"
    assert result.strict_niah_pair_count == 11


def test_sample_assessment_keeps_invalid_active_labels_as_hard_errors() -> None:
    pairs = _passing_active_pairs()
    pairs["niah"] = (
        _pair(
            kind="niah",
            query_id="bad",
            evidence_id="bad",
            protect_label=2,
            protect_mask=True,
            harm_label=None,
            harm_mask=False,
        ),
    )

    with pytest.raises(ValueError, match="active protect label is not binary"):
        _assess_sample_classes(pairs)  # type: ignore[arg-type]


def test_sample_assessment_keeps_forbidden_twowiki_labels_as_hard_errors() -> None:
    pairs = _passing_active_pairs()
    pairs["2wiki"] = (
        _pair(
            kind="2wiki",
            query_id="bad",
            evidence_id="bad",
            protect_label=0,
            protect_mask=True,
            harm_label=None,
            harm_mask=False,
        ),
    )

    with pytest.raises(ValueError, match="protocol-forbidden 2Wiki active labels"):
        _assess_sample_classes(pairs)  # type: ignore[arg-type]


class _FakeCudaOom(RuntimeError):
    pass


@pytest.mark.parametrize(
    "error",
    (_FakeCudaOom("initial OOM"), _NonFiniteTrainingError("initial non-finite")),
)
def test_initial_baseline_failures_remain_hard_errors(
    monkeypatch: pytest.MonkeyPatch, error: RuntimeError
) -> None:
    def fail_prediction(**_: object) -> object:
        raise error

    monkeypatch.setattr(sanity_runner, "_predict_pairs", fail_prediction)

    with pytest.raises(type(error)) as caught:
        sanity_runner._initial_active_losses(
            model=object(),
            torch=object(),
            pairs=(),
            weights={},
            batch_size=4,
        )

    assert caught.value is error


def _epoch_row(epoch: int) -> dict[str, object]:
    return {
        "row_type": "epoch",
        "epoch": epoch,
        "microbatches": 4,
        "optimizer_steps": 1,
        "mean_total_loss": 1.0 / epoch,
        "mean_protect_loss": 0.5,
        "mean_harm_loss": 0.5,
        "protect_effective_labels_with_deterministic_repetition": 4,
        "harm_effective_labels_with_deterministic_repetition": 4,
        "wall_time_seconds": 0.1,
        "nonfinite": False,
    }


def test_epoch_boundary_runner_rolls_oom_back_to_last_complete_epoch() -> None:
    state = {"value": 0}

    def run_epoch(epoch: int) -> tuple[dict[str, object], int]:
        state["value"] = epoch
        if epoch == 3:
            state["value"] = 999
            raise _FakeCudaOom("out of memory")
        return _epoch_row(epoch), 2

    outcome = _run_epoch_boundaries(
        run_epoch=run_epoch,
        snapshot_state=lambda: state["value"],
        restore_state=lambda value: state.__setitem__("value", value),
        oom_error_type=_FakeCudaOom,
        epochs=4,
    )

    assert outcome.termination_reason == "cuda-oom"
    assert outcome.epochs_completed == 2
    assert outcome.failure_epoch == 3
    assert outcome.optimizer_steps == 4
    assert state["value"] == 2


def test_epoch_boundary_runner_rolls_nonfinite_back_to_epoch_zero() -> None:
    state = {"value": 0}

    def run_epoch(epoch: int) -> tuple[dict[str, object], int]:
        state["value"] = epoch
        raise _NonFiniteTrainingError("non-finite")

    outcome = _run_epoch_boundaries(
        run_epoch=run_epoch,
        snapshot_state=lambda: state["value"],
        restore_state=lambda value: state.__setitem__("value", value),
        oom_error_type=_FakeCudaOom,
        epochs=4,
    )

    assert outcome.termination_reason == "nonfinite"
    assert outcome.epochs_completed == 0
    assert outcome.failure_epoch == 1
    assert state["value"] == 0


def test_epoch_boundary_runner_does_not_disguise_ordinary_runtime_error() -> None:
    state = {"value": 0}

    def run_epoch(epoch: int) -> tuple[dict[str, object], int]:
        state["value"] = epoch
        raise RuntimeError("programming defect")

    with pytest.raises(RuntimeError, match="programming defect"):
        _run_epoch_boundaries(
            run_epoch=run_epoch,
            snapshot_state=lambda: state["value"],
            restore_state=lambda value: state.__setitem__("value", value),
            oom_error_type=_FakeCudaOom,
            epochs=4,
        )

    assert state["value"] == 1


def test_weighted_bce_uses_active_count_not_sum_of_weights() -> None:
    pairs = (
        _pair(
            kind="niah",
            query_id="q0",
            evidence_id="e0",
            protect_label=0,
            protect_mask=True,
            harm_label=None,
            harm_mask=False,
        ),
        _pair(
            kind="niah",
            query_id="q1",
            evidence_id="e1",
            protect_label=1,
            protect_mask=True,
            harm_label=None,
            harm_mask=False,
        ),
    )
    scores = {
        ("niah", "q0", "e0"): (0.5, 0.5, 0.5),
        ("niah", "q1", "e1"): (0.5, 0.5, 0.5),
    }
    weights = {("niah", "protect", 0): 2.0, ("niah", "protect", 1): 4.0}

    losses = _weighted_bce_from_scores(pairs=pairs, scores=scores, weights=weights)

    assert losses["niah.protect"] == pytest.approx(3.0 * math.log(2.0))


def _early_trace() -> SanityTrainingTrace:
    return SanityTrainingTrace(
        termination_reason="cuda-oom",
        epochs_completed=1,
        failure_epoch=2,
        final_checkpoint_epoch=1,
        epoch_mean_losses=(1.0,),
        active_source_head_loss_trends=(),
        protect_head_parameter_l2_change=0.1,
        harm_head_parameter_l2_change=0.1,
        twowiki_harm_head_gradient_l1=None,
        independent_sigmoid_heads=True,
        oom_encountered=True,
        nonfinite_encountered=False,
    )


def test_verify_helpers_accept_dynamic_epoch_rows_and_matching_abort_log() -> None:
    trace = _early_trace()
    _validate_stored_epoch_rows(
        training_summary={"optimizer_steps": 1},
        epoch_rows=(_epoch_row(1),),
        trace=trace,
    )

    events = _validate_training_runtime_log(
        (
            {"event": "start"},
            {
                "event": "training-aborted",
                "termination_reason": "cuda-oom",
                "epochs_completed": 1,
                "failure_epoch": 2,
            },
            {
                "event": "training-gate-complete",
                "status": "FAIL",
                "checkpoint_weights_sha256": "a" * 64,
            },
            {"event": "complete", "status": "FAIL"},
        ),
        trace,
        expected_gate_status="FAIL",
        expected_checkpoint_sha256="a" * 64,
        expected_run_status="FAIL",
    )

    assert events == ("start", "training-aborted", "training-gate-complete", "complete")


def test_verify_helpers_reject_early_threshold_event_and_nonempty_artifact(
    tmp_path: Path,
) -> None:
    trace = _early_trace()
    with pytest.raises(ValueError, match="threshold/modelval"):
        _validate_training_runtime_log(
            (
                {"event": "start"},
                {
                    "event": "training-aborted",
                    "termination_reason": "cuda-oom",
                    "epochs_completed": 1,
                    "failure_epoch": 2,
                },
                {"event": "training-gate-complete"},
                {"event": "trainfit-thresholds-frozen-before-modelval"},
                {"event": "complete"},
            ),
            trace,
        )

    artifact = tmp_path / "candidate_scores.jsonl"
    artifact.write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="must be empty"):
        _require_empty_artifacts(
            tmp_path,
            (artifact.name,),
            context="early failure",
        )


def _completed_trace() -> SanityTrainingTrace:
    return SanityTrainingTrace(
        termination_reason="completed",
        epochs_completed=30,
        failure_epoch=None,
        final_checkpoint_epoch=30,
        epoch_mean_losses=tuple(1.0 for _ in range(30)),
        active_source_head_loss_trends=(
            SanitySourceHeadLossTrend(
                dataset_kind="niah",
                head="protect",
                initial_loss=2.0,
                final_loss=1.0,
                decreased=True,
            ),
            SanitySourceHeadLossTrend(
                dataset_kind="niah",
                head="harm",
                initial_loss=2.0,
                final_loss=1.0,
                decreased=True,
            ),
            SanitySourceHeadLossTrend(
                dataset_kind="2wiki",
                head="protect",
                initial_loss=2.0,
                final_loss=1.0,
                decreased=True,
            ),
        ),
        protect_head_parameter_l2_change=0.1,
        harm_head_parameter_l2_change=0.1,
        twowiki_harm_head_gradient_l1=0.0,
        independent_sigmoid_heads=True,
        oom_encountered=False,
        nonfinite_encountered=False,
    )


def test_runtime_log_requires_exact_pass_cut_threshold_modelval_order() -> None:
    rows = (
        {"event": "start"},
        {"event": "training-gate-complete", "status": "PASS"},
        {"event": "trainfit-thresholds-frozen-before-modelval"},
        {"event": "modelval-safe-corner-complete"},
        {"event": "complete", "status": "CUT"},
    )

    assert _validate_training_runtime_log(
        rows,
        _completed_trace(),
        expected_gate_status="PASS",
        expected_run_status="CUT",
    ) == tuple(row["event"] for row in rows)

    duplicate = (*rows[:-1], rows[2], rows[-1])
    with pytest.raises(ValueError, match="exactly one threshold and modelval"):
        _validate_training_runtime_log(
            duplicate,
            _completed_trace(),
            expected_gate_status="PASS",
            expected_run_status="CUT",
        )


def test_runtime_log_rejects_thresholds_for_completed_gate_fail() -> None:
    with pytest.raises(ValueError, match="FAIL runtime log"):
        _validate_training_runtime_log(
            (
                {"event": "start"},
                {"event": "training-gate-complete", "status": "FAIL"},
                {"event": "trainfit-thresholds-frozen-before-modelval"},
                {"event": "complete", "status": "FAIL"},
            ),
            _completed_trace(),
            expected_gate_status="FAIL",
            expected_run_status="FAIL",
        )
