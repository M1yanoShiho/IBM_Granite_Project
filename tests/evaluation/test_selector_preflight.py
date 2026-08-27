import copy
import hashlib
import math
from dataclasses import asdict
from pathlib import Path

import pytest
from pydantic import ValidationError

from evidence_rag.evaluation.selector_preflight import (
    FORWARD_TRACE_FILE,
    MANIFEST_FILE,
    OUTPUT_FILES,
    PREFLIGHT_FORWARD_PAIRS,
    PREFLIGHT_PROTOCOL_VERSION,
    PREFLIGHT_TRAINING_MICROBATCHES,
    PREFLIGHT_TRAINING_WARMUP_MICROBATCHES,
    REPORT_FILE,
    RUNTIME_LOG_FILE,
    SAMPLE_FILE,
    TOKEN_AUDIT_FILE,
    PreflightSampleQuery,
    ResourcePreflightArtifacts,
    TokenLengthAuditRow,
    UntrainedForwardTraceRow,
    build_resource_preflight_artifacts,
    estimate_seed_training_gpu_hours,
    freeze_resource_preflight_artifacts,
    nearest_rank_percentile,
    preflight_sample_digest,
    select_preflight_queries,
    summarize_token_lengths,
    verify_resource_preflight_artifacts,
)

BundleInputs = tuple[
    tuple[PreflightSampleQuery, ...],
    tuple[TokenLengthAuditRow, ...],
    tuple[UntrainedForwardTraceRow, ...],
    dict[str, object],
    tuple[dict[str, object], ...],
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, object],
]


def _token_row(*, raw: int, encoded: int) -> TokenLengthAuditRow:
    return TokenLengthAuditRow(
        dataset_kind="niah",
        query_id="q1",
        evidence_id=f"e-{raw}",
        retrieval_rank=1,
        question_tokens=2,
        candidate_tokens=raw - 5,
        special_tokens=3,
        raw_pair_tokens=raw,
        encoded_tokens=encoded,
        truncated_candidate_tokens=raw - encoded,
        truncated=raw > encoded,
    )


def _input_pin(name: str) -> dict[str, object]:
    payload = f"frozen:{name}\n".encode()
    return {
        "path": f"inputs/{name}",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _valid_bundle_inputs() -> BundleInputs:
    query_ids = {
        "niah": tuple(f"n-{index:03d}" for index in range(103)),
        "2wiki": tuple(f"w-{index:03d}" for index in range(300)),
    }
    sample = select_preflight_queries(query_ids)

    token_rows: list[TokenLengthAuditRow] = []
    token_by_query: dict[tuple[str, str], list[TokenLengthAuditRow]] = {}
    for dataset_kind in ("niah", "2wiki"):
        for query_id in query_ids[dataset_kind]:
            rows: list[TokenLengthAuditRow] = []
            for rank in range(1, 21):
                raw_tokens = 32 + rank
                row = TokenLengthAuditRow(
                    dataset_kind=dataset_kind,
                    query_id=query_id,
                    evidence_id=f"{dataset_kind}-{query_id}-e{rank:02d}",
                    retrieval_rank=rank,
                    question_tokens=8,
                    candidate_tokens=raw_tokens - 11,
                    special_tokens=3,
                    raw_pair_tokens=raw_tokens,
                    encoded_tokens=raw_tokens,
                    truncated_candidate_tokens=0,
                    truncated=False,
                )
                rows.append(row)
                token_rows.append(row)
            token_by_query[(dataset_kind, query_id)] = rows

    forward_rows = tuple(
        UntrainedForwardTraceRow(
            dataset_kind=token_row.dataset_kind,
            query_id=token_row.query_id,
            evidence_id=token_row.evidence_id,
            retrieval_rank=token_row.retrieval_rank,
            protect_score=0.75,
            harm_score=0.25,
        )
        for sample_row in sample
        for token_row in token_by_query[(sample_row.dataset_kind, sample_row.query_id)]
    )

    microbatch_seconds = [2.0 + index / 10 for index in range(12)]
    estimate = estimate_seed_training_gpu_hours(
        niah_supervised_pairs=12_000,
        twowiki_supervised_pairs=24_000,
        batch_size=8,
        gradient_accumulation_steps=4,
        epochs=3,
        microbatch_seconds=microbatch_seconds,
        warmup_microbatches=PREFLIGHT_TRAINING_WARMUP_MICROBATCHES,
    )
    input_pins = {
        name: _input_pin(name)
        for name in (
            "config",
            "model_snapshot/model.safetensors",
            "niah/dataset_manifest.json",
            "niah/source_parent.jsonl",
            "niah/labels/train-modelval.jsonl",
            "niah/assignment.jsonl",
            "niah/provenance.jsonl",
            "niah/candidate_pool.jsonl",
            "niah/pool_manifest.json",
            "niah/components/component_map.jsonl",
            "2wiki/dataset_manifest.json",
            "2wiki/source_parent.jsonl",
            "2wiki/labels/train-modelval.jsonl",
            "2wiki/candidate_pool.jsonl",
            "2wiki/pool_manifest.json",
            "2wiki/components/component_map.jsonl",
        )
    }
    expected_git: dict[str, object] = {
        "branch": "refactor/three-module-baseline",
        "commit": "a" * 40,
        "dirty": False,
    }
    expected_model: dict[str, object] = {
        "model_id": "answerdotai/ModernBERT-base",
        "revision": "frozen-revision",
        "snapshot_identity_sha256": "b" * 64,
        "snapshot_files": {"model.safetensors": input_pins["model_snapshot/model.safetensors"]},
    }
    overall_token_summary = asdict(summarize_token_lengths(tuple(token_rows)))
    by_dataset_token_summary = {
        dataset_kind: asdict(
            summarize_token_lengths(
                tuple(row for row in token_rows if row.dataset_kind == dataset_kind)
            )
        )
        for dataset_kind in ("niah", "2wiki")
    }
    forward_batch_seconds = [1.0, 2.0, 3.0, 4.0]
    forward_wall_time = sum(forward_batch_seconds)
    report: dict[str, object] = {
        "schema_version": "1.0",
        "run_id": "R004",
        "stage": "label-audit-and-200q-preflight",
        "status": "PASS",
        "interpretation": "RESOURCE_AND_LABEL_FEASIBILITY_ONLY_NOT_SELECTOR_EFFECT",
        "git": expected_git,
        "config_sha256": input_pins["config"]["sha256"],
        "model": {
            **expected_model,
            "random_head_seed": 13,
            "initialized_full_state_fingerprint": {
                "schema_version": "selector-dual-head-fingerprint-v1",
                "model_id": expected_model["model_id"],
                "revision": expected_model["revision"],
                "max_length": 512,
                "weights_sha256": "c" * 64,
            },
            "independent_linear_heads": True,
            "activation": "independent-sigmoid",
            "checkpoint": "NOT_APPLICABLE_EPHEMERAL_MODEL_DISCARDED",
        },
        "label_audit": {
            "niah": {
                "protocol_version": "selector-labels-v2",
                "dataset_kind": "niah",
                "source_split": "train",
                "status": "LABEL_AUDIT_READY",
            },
            "2wiki": {
                "protocol_version": "selector-labels-v2",
                "dataset_kind": "2wiki",
                "source_split": "train",
                "status": "LABEL_AUDIT_READY",
            },
        },
        "sampling": {
            "protocol_version": PREFLIGHT_PROTOCOL_VERSION,
            "seed": 20260811,
            "role": "train-modelval",
            "queries_per_dataset": 100,
            "sample_queries": 200,
            "candidates_per_query": 20,
            "forward_pairs": PREFLIGHT_FORWARD_PAIRS,
            "selection_uses_labels_or_lengths": False,
        },
        "token_audit": {
            "scope": "all-train-modelval-top20",
            "queries": 403,
            "pairs": 8060,
            "max_length": 512,
            "truncation": "only_second",
            "overall": overall_token_summary,
            "by_dataset": by_dataset_token_summary,
        },
        "forward_probe": {
            "interpretation": "UNTRAINED_RESOURCE_PROBE_DO_NOT_INTERPRET",
            "queries": 200,
            "pairs": PREFLIGHT_FORWARD_PAIRS,
            "batch_size": 1000,
            "warmup_batches": 4,
            "timed_batches": 4,
            "batch_seconds": forward_batch_seconds,
            "wall_time_seconds": forward_wall_time,
            "pairs_per_second": PREFLIGHT_FORWARD_PAIRS / forward_wall_time,
            "queries_per_second": 200 / forward_wall_time,
            "batch_seconds_mean": forward_wall_time / len(forward_batch_seconds),
            "batch_seconds_p50": 2.0,
            "batch_seconds_p95": 4.0,
            "max_head_score_difference": 0.5,
            "mean_head_score_difference": 0.5,
            "peak_memory_allocated_bytes": 1_000_000,
            "peak_memory_reserved_bytes": 1_500_000,
            "oom_encountered": False,
            "nonfinite_encountered": False,
        },
        "training_probe": {
            "interpretation": "EPHEMERAL_RESOURCE_PROBE_NO_CHECKPOINT",
            "source_role": "train-fit",
            "pair_order": (
                "ascending-sha256(dataset-kind,newline,query-id,newline,evidence-id,newline,seed)"
            ),
            "loss": "unweighted independent masked BCE for resource measurement only",
            "batch_size": 8,
            "gradient_accumulation_steps": 4,
            "niah_supervised_pairs": 12_000,
            "twowiki_supervised_pairs": 24_000,
            "microbatches": PREFLIGHT_TRAINING_MICROBATCHES,
            "warmup_microbatches": PREFLIGHT_TRAINING_WARMUP_MICROBATCHES,
            "timed_microbatches": (
                PREFLIGHT_TRAINING_MICROBATCHES - PREFLIGHT_TRAINING_WARMUP_MICROBATCHES
            ),
            "checkpoint": "NOT_APPLICABLE_EPHEMERAL_MODEL_DISCARDED",
            "peak_memory_allocated_bytes": 2_000_000,
            "peak_memory_reserved_bytes": 2_500_000,
            "microbatch_seconds": microbatch_seconds,
            "losses": [1.0 - index / 100 for index in range(12)],
            "protect_effective_labels": [8] * 12,
            "harm_effective_labels": [4] * 12,
            "optimizer_steps": 3,
            "oom_encountered": False,
            "nonfinite_encountered": False,
        },
        "seed13_training_gpu_estimate": estimate.model_dump(mode="python"),
        "device": {
            "kind": "cuda",
            "requested": "cuda:0",
            "name": "test-gpu",
            "total_memory_bytes": 16_000_000,
            "compute_capability": "8.0",
            "torch_version": "2.7.0",
            "cuda_runtime_version": "12.8",
            "precision": "float32",
            "peak_memory_allocated_bytes": 2_000_000,
        },
        "runtime": {
            "host": "test-host",
            "python_version": "3.11.14",
            "started_at_utc": "2026-08-12T00:00:00Z",
            "finished_at_utc": "2026-08-12T00:01:00Z",
            "wall_time_seconds": 60.0,
            "command": ["python", "-m", "selector_preflight"],
        },
        "boundaries": {
            "model_input_fields": ["question", "candidate_text"],
            "provenance_is_model_input": False,
            "unjudged_as_negative": False,
            "selector_policy_constructed": False,
            "checkpoint_written": False,
            "sealed_or_heldout_accessed": False,
            "production_default_changed": False,
        },
    }
    runtime_log = (
        {"event": "start", "time_utc": "2026-08-12T00:00:00Z"},
        {"event": "labels-verified", "time_utc": "2026-08-12T00:00:10Z"},
        {"event": "token-audit-complete", "time_utc": "2026-08-12T00:00:20Z"},
        {
            "event": "forward-complete",
            "time_utc": "2026-08-12T00:00:30Z",
            "pairs": PREFLIGHT_FORWARD_PAIRS,
        },
        {
            "event": "ephemeral-training-probe-complete",
            "time_utc": "2026-08-12T00:00:50Z",
            "microbatches": 12,
        },
        {"event": "pass", "time_utc": "2026-08-12T00:01:00Z"},
    )
    assert len(sample) == 200
    assert len(token_rows) == 8060
    assert len(forward_rows) == PREFLIGHT_FORWARD_PAIRS
    return (
        sample,
        tuple(token_rows),
        forward_rows,
        report,
        runtime_log,
        input_pins,
        expected_git,
        expected_model,
    )


@pytest.fixture(scope="module")
def valid_bundle_inputs() -> BundleInputs:
    return _valid_bundle_inputs()


def _build(
    values: BundleInputs,
    *,
    sample: tuple[PreflightSampleQuery, ...] | None = None,
    token_audit: tuple[TokenLengthAuditRow, ...] | None = None,
    forward_trace: tuple[UntrainedForwardTraceRow, ...] | None = None,
    report: dict[str, object] | None = None,
) -> ResourcePreflightArtifacts:
    (
        base_sample,
        base_tokens,
        base_forward,
        base_report,
        runtime_log,
        input_pins,
        expected_git,
        expected_model,
    ) = values
    return build_resource_preflight_artifacts(
        sample=base_sample if sample is None else sample,
        token_audit=base_tokens if token_audit is None else token_audit,
        forward_trace=base_forward if forward_trace is None else forward_trace,
        report=base_report if report is None else report,
        runtime_log=runtime_log,
        input_pins=input_pins,
        expected_git=expected_git,
        expected_model=expected_model,
    )


def test_sampling_is_equal_label_blind_and_order_independent() -> None:
    forward = {
        "niah": [f"n-{index:03d}" for index in range(103)],
        "2wiki": [f"w-{index:03d}" for index in range(300)],
    }
    reverse = {key: list(reversed(values)) for key, values in forward.items()}
    selected = select_preflight_queries(forward)
    assert selected == select_preflight_queries(reverse)
    assert sum(row.dataset_kind == "niah" for row in selected) == 100
    assert sum(row.dataset_kind == "2wiki" for row in selected) == 100
    assert len({row.query_id for row in selected if row.dataset_kind == "niah"}) == 100
    assert all(len(row.sample_digest) == 64 for row in selected)
    assert preflight_sample_digest("niah", "q") != preflight_sample_digest("2wiki", "q")


def test_sampling_rejects_missing_dataset_duplicates_and_small_pool() -> None:
    with pytest.raises(ValueError, match="exactly"):
        select_preflight_queries({"niah": ["q"]})
    with pytest.raises(ValueError, match="duplicate"):
        select_preflight_queries({"niah": ["q"] * 100, "2wiki": [f"w{i}" for i in range(100)]})
    with pytest.raises(ValueError, match="eligible"):
        select_preflight_queries(
            {"niah": [f"n{i}" for i in range(99)], "2wiki": [f"w{i}" for i in range(100)]}
        )


def test_token_rows_enforce_only_second_accounting_and_summary() -> None:
    rows = (_token_row(raw=10, encoded=10), _token_row(raw=600, encoded=512))
    summary = summarize_token_lengths(rows)
    assert summary.pairs == 2
    assert summary.truncated_pairs == 1
    assert summary.truncated_fraction == 0.5
    assert summary.raw_tokens_p50 == 10
    assert summary.raw_tokens_p95 == 600
    assert summary.encoded_tokens_max == 512
    assert summary.truncated_candidate_tokens_total == 88
    with pytest.raises(ValidationError):
        TokenLengthAuditRow(
            dataset_kind="niah",
            query_id="q",
            evidence_id="e",
            retrieval_rank=1,
            question_tokens=2,
            candidate_tokens=10,
            special_tokens=3,
            raw_pair_tokens=15,
            encoded_tokens=14,
            truncated_candidate_tokens=0,
            truncated=False,
        )


def test_nearest_rank_percentile_is_frozen_and_validated() -> None:
    values = [1, 2, 3, 4, 100]
    assert nearest_rank_percentile(values, 0) == 1
    assert nearest_rank_percentile(values, 50) == 3
    assert nearest_rank_percentile(values, 95) == 100
    assert nearest_rank_percentile(values, 100) == 100
    with pytest.raises(ValueError):
        nearest_rank_percentile([], 50)
    with pytest.raises(ValueError):
        nearest_rank_percentile([math.nan], 50)


def test_gpu_hour_estimate_uses_alternating_larger_source_and_excludes_warmup() -> None:
    estimate = estimate_seed_training_gpu_hours(
        niah_supervised_pairs=9,
        twowiki_supervised_pairs=17,
        batch_size=4,
        gradient_accumulation_steps=4,
        epochs=3,
        microbatch_seconds=[9.0, 8.0, 1.0, 2.0, 3.0, 4.0],
        warmup_microbatches=2,
    )
    # max(ceil(9/4), ceil(17/4)) = 5; alternating sources -> 10/epoch.
    assert estimate.microbatches_per_epoch == 10
    assert estimate.optimizer_steps_total == 9
    assert estimate.timed_microbatch_seconds == (1.0, 2.0, 3.0, 4.0)
    assert estimate.mean_seconds_per_microbatch == 2.5
    assert estimate.p95_seconds_per_microbatch == 4.0
    assert estimate.point_gpu_hours == pytest.approx(30 * 2.5 / 3600)
    assert estimate.conservative_gpu_hours == pytest.approx(30 * 4 / 3600)


def test_gpu_hour_estimate_rejects_nonpositive_or_all_warmup_times() -> None:
    common = dict(
        niah_supervised_pairs=1,
        twowiki_supervised_pairs=1,
        batch_size=1,
        gradient_accumulation_steps=1,
        epochs=1,
    )
    with pytest.raises(ValueError, match="leave at least one"):
        estimate_seed_training_gpu_hours(**common, microbatch_seconds=[1.0], warmup_microbatches=1)
    with pytest.raises(ValueError, match="finite and positive"):
        estimate_seed_training_gpu_hours(
            **common, microbatch_seconds=[0.0, 1.0], warmup_microbatches=1
        )


def test_complete_resource_bundle_builds_freezes_and_verifies(
    tmp_path: Path, valid_bundle_inputs: BundleInputs
) -> None:
    sample, _, _, _, _, input_pins, expected_git, expected_model = valid_bundle_inputs
    artifacts = _build(valid_bundle_inputs)

    assert set(artifacts.files) == set(OUTPUT_FILES)
    assert artifacts.manifest["counts"] == {
        "sample_queries": 200,
        "token_audit_pairs": 8060,
        "forward_pairs": 4000,
        "runtime_log_events": 6,
    }
    output_pins = artifacts.manifest["outputs"]
    assert isinstance(output_pins, dict)
    for filename in OUTPUT_FILES[:-1]:
        pin = output_pins[filename]
        assert isinstance(pin, dict)
        assert pin["bytes"] == len(artifacts.files[filename])
        assert pin["sha256"] == hashlib.sha256(artifacts.files[filename]).hexdigest()
    assert output_pins[SAMPLE_FILE]["records"] == 200
    assert output_pins[TOKEN_AUDIT_FILE]["records"] == 8060
    assert output_pins[FORWARD_TRACE_FILE]["records"] == 4000
    assert output_pins[RUNTIME_LOG_FILE]["records"] == 6
    assert REPORT_FILE in output_pins
    assert MANIFEST_FILE not in output_pins

    output_directory = tmp_path / "preflight"
    manifest_path = freeze_resource_preflight_artifacts(output_directory, artifacts)
    assert manifest_path == output_directory / MANIFEST_FILE
    verified = verify_resource_preflight_artifacts(
        output_directory,
        expected_sample=sample,
        expected_input_pins=input_pins,
        expected_git=expected_git,
        expected_model=expected_model,
    )
    assert verified.files == artifacts.files
    assert verified.report == artifacts.report
    assert verified.manifest == artifacts.manifest
    with pytest.raises(ValueError, match="not empty"):
        freeze_resource_preflight_artifacts(output_directory, artifacts)


def test_bundle_rejects_tampered_git_and_model_identity(
    valid_bundle_inputs: BundleInputs,
) -> None:
    _, _, _, report, _, _, _, _ = valid_bundle_inputs

    tampered_git = copy.deepcopy(report)
    git = tampered_git["git"]
    assert isinstance(git, dict)
    git["commit"] = "d" * 40
    with pytest.raises(ValueError, match="Git pin differs"):
        _build(valid_bundle_inputs, report=tampered_git)

    tampered_model = copy.deepcopy(report)
    model = tampered_model["model"]
    assert isinstance(model, dict)
    model["snapshot_identity_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="model identity differs"):
        _build(valid_bundle_inputs, report=tampered_model)


def test_verify_rejects_mismatched_git_and_model_expectations(
    tmp_path: Path, valid_bundle_inputs: BundleInputs
) -> None:
    sample, _, _, _, _, input_pins, expected_git, expected_model = valid_bundle_inputs
    output_directory = tmp_path / "preflight-identity"
    freeze_resource_preflight_artifacts(output_directory, _build(valid_bundle_inputs))

    wrong_git = dict(expected_git)
    wrong_git["commit"] = "d" * 40
    with pytest.raises(ValueError, match="Git pin differs"):
        verify_resource_preflight_artifacts(
            output_directory,
            expected_sample=sample,
            expected_input_pins=input_pins,
            expected_git=wrong_git,
            expected_model=expected_model,
        )

    wrong_model = copy.deepcopy(expected_model)
    wrong_model["snapshot_identity_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="model identity differs"):
        verify_resource_preflight_artifacts(
            output_directory,
            expected_sample=sample,
            expected_input_pins=input_pins,
            expected_git=expected_git,
            expected_model=wrong_model,
        )


def test_verify_rejects_an_extra_output_file(
    tmp_path: Path, valid_bundle_inputs: BundleInputs
) -> None:
    sample, _, _, _, _, input_pins, expected_git, expected_model = valid_bundle_inputs
    output_directory = tmp_path / "preflight-extra-file"
    freeze_resource_preflight_artifacts(output_directory, _build(valid_bundle_inputs))
    (output_directory / "unexpected.txt").write_text("not part of the frozen bundle\n")

    with pytest.raises(ValueError, match="unexpected file set.*unexpected.txt"):
        verify_resource_preflight_artifacts(
            output_directory,
            expected_sample=sample,
            expected_input_pins=input_pins,
            expected_git=expected_git,
            expected_model=expected_model,
        )


def test_bundle_rejects_duplicate_and_missing_rank_window(
    valid_bundle_inputs: BundleInputs,
) -> None:
    _, token_rows, _, _, _, _, _, _ = valid_bundle_inputs
    bad_rows = list(token_rows)
    rank_twenty = bad_rows[19]
    bad_rows[19] = rank_twenty.model_copy(update={"retrieval_rank": 19})

    with pytest.raises(ValueError, match="duplicate candidate keys"):
        _build(valid_bundle_inputs, token_audit=tuple(bad_rows))


def test_bundle_rejects_wrong_sample_digest_and_sample_cardinality(
    valid_bundle_inputs: BundleInputs,
) -> None:
    sample, _, _, _, _, _, _, _ = valid_bundle_inputs
    bad_digest = list(sample)
    bad_digest[0] = bad_digest[0].model_copy(update={"sample_digest": "f" * 64})
    with pytest.raises(ValueError, match="sample digest mismatch"):
        _build(valid_bundle_inputs, sample=tuple(bad_digest))

    with pytest.raises(ValueError, match="exactly 200 queries"):
        _build(valid_bundle_inputs, sample=sample[:-1])


def test_bundle_rejects_missing_and_duplicate_token_rows(
    valid_bundle_inputs: BundleInputs,
) -> None:
    _, token_rows, _, _, _, _, _, _ = valid_bundle_inputs
    with pytest.raises(ValueError, match="exactly 8,060 pairs"):
        _build(valid_bundle_inputs, token_audit=token_rows[:-1])

    duplicate = list(token_rows)
    duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError, match="duplicate candidate keys"):
        _build(valid_bundle_inputs, token_audit=tuple(duplicate))


def test_bundle_rejects_forward_trace_not_equal_to_sample_times_top20(
    valid_bundle_inputs: BundleInputs,
) -> None:
    sample, token_rows, forward_rows, _, _, _, _, _ = valid_bundle_inputs
    sample_keys = {(row.dataset_kind, row.query_id) for row in sample}
    outsider = next(
        row for row in token_rows if (row.dataset_kind, row.query_id) not in sample_keys
    )
    foreign_forward = UntrainedForwardTraceRow(
        dataset_kind=outsider.dataset_kind,
        query_id=outsider.query_id,
        evidence_id=outsider.evidence_id,
        retrieval_rank=outsider.retrieval_rank,
        protect_score=0.5,
        harm_score=0.5,
    )
    bad_forward = (*forward_rows[:-1], foreign_forward)

    with pytest.raises(ValueError, match="ordered sample x Top20"):
        _build(valid_bundle_inputs, forward_trace=bad_forward)


def test_bundle_rejects_forward_head_statistics_not_bound_to_raw_scores(
    valid_bundle_inputs: BundleInputs,
) -> None:
    _, _, forward_rows, _, _, _, _, _ = valid_bundle_inputs
    identical = tuple(
        row.model_copy(update={"protect_score": 0.5, "harm_score": 0.5}) for row in forward_rows
    )
    with pytest.raises(ValueError, match="identical protect/harm"):
        _build(valid_bundle_inputs, forward_trace=identical)

    changed = list(forward_rows)
    changed[0] = changed[0].model_copy(update={"protect_score": 1.0, "harm_score": 0.0})
    with pytest.raises(ValueError, match="head-difference report differs"):
        _build(valid_bundle_inputs, forward_trace=tuple(changed))


def test_bundle_rejects_tampered_training_estimate_derived_fields(
    valid_bundle_inputs: BundleInputs,
) -> None:
    _, _, _, report, _, _, _, _ = valid_bundle_inputs
    tampered = copy.deepcopy(report)
    estimate = tampered["seed13_training_gpu_estimate"]
    assert isinstance(estimate, dict)
    estimate["point_gpu_hours"] = float(estimate["point_gpu_hours"]) * 2

    with pytest.raises(ValueError, match="point GPU hours"):
        _build(valid_bundle_inputs, report=tampered)


def test_bundle_rejects_tampered_forward_throughput_derived_fields(
    valid_bundle_inputs: BundleInputs,
) -> None:
    _, _, _, report, _, _, _, _ = valid_bundle_inputs
    tampered = copy.deepcopy(report)
    forward_probe = tampered["forward_probe"]
    assert isinstance(forward_probe, dict)
    forward_probe["pairs_per_second"] = 999.0

    with pytest.raises(ValueError):
        _build(valid_bundle_inputs, report=tampered)


def test_bundle_rejects_incomplete_report_and_input_pin_categories(
    valid_bundle_inputs: BundleInputs,
) -> None:
    (
        sample,
        token_rows,
        forward_rows,
        report,
        runtime_log,
        input_pins,
        expected_git,
        expected_model,
    ) = valid_bundle_inputs
    bad_report = copy.deepcopy(report)
    forward_probe = bad_report["forward_probe"]
    assert isinstance(forward_probe, dict)
    del forward_probe["peak_memory_allocated_bytes"]
    with pytest.raises(ValueError, match="peak_memory_allocated_bytes"):
        _build(valid_bundle_inputs, report=bad_report)

    bad_pins = dict(input_pins)
    del bad_pins["2wiki/components/component_map.jsonl"]
    with pytest.raises(ValueError, match="miss required categories.*2wiki/components"):
        build_resource_preflight_artifacts(
            sample=sample,
            token_audit=token_rows,
            forward_trace=forward_rows,
            report=report,
            runtime_log=runtime_log,
            input_pins=bad_pins,
            expected_git=expected_git,
            expected_model=expected_model,
        )
