"""Deterministic protocol helpers for the R004 Selector resource preflight.

R004 is deliberately not a model-selection experiment.  These helpers freeze which
``train-modelval`` queries are timed, how token truncation is counted and how a short
ephemeral training probe is converted into a GPU-hour estimate.  They contain no
Torch dependency so the protocol can be tested and audited on a CPU-only machine.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, FiniteFloat, model_validator

DatasetKind: TypeAlias = Literal["niah", "2wiki"]

PREFLIGHT_PROTOCOL_VERSION = "selector-r004-preflight-v1"
PREFLIGHT_SAMPLE_SEED = 20260811
PREFLIGHT_QUESTIONS_PER_DATASET = 100
PREFLIGHT_CANDIDATES_PER_QUERY = 20
PREFLIGHT_FORWARD_PAIRS = 4000
PREFLIGHT_TRAINING_MICROBATCHES = 12
PREFLIGHT_TRAINING_WARMUP_MICROBATCHES = 4
PERCENTILE_METHOD = "nearest-rank"
SAMPLE_FILE = "preflight_sample.jsonl"
TOKEN_AUDIT_FILE = "token_length_audit.jsonl"
FORWARD_TRACE_FILE = "untrained_forward_trace.jsonl"
REPORT_FILE = "resource_preflight_report.json"
RUNTIME_LOG_FILE = "resource_preflight_log.jsonl"
MANIFEST_FILE = "resource_preflight_manifest.json"
OUTPUT_FILES = (
    SAMPLE_FILE,
    TOKEN_AUDIT_FILE,
    FORWARD_TRACE_FILE,
    REPORT_FILE,
    RUNTIME_LOG_FILE,
    MANIFEST_FILE,
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class PreflightSampleQuery(_FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    protocol_version: Literal["selector-r004-preflight-v1"] = "selector-r004-preflight-v1"
    dataset_kind: DatasetKind
    query_id: str = Field(min_length=1)
    sample_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_count: Literal[20] = 20


class TokenLengthAuditRow(_FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset_kind: DatasetKind
    query_id: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    retrieval_rank: int = Field(ge=1, le=PREFLIGHT_CANDIDATES_PER_QUERY)
    question_tokens: int = Field(ge=0)
    candidate_tokens: int = Field(ge=0)
    special_tokens: int = Field(ge=0)
    raw_pair_tokens: int = Field(ge=0)
    encoded_tokens: int = Field(ge=0, le=512)
    truncated_candidate_tokens: int = Field(ge=0)
    truncated: bool

    @model_validator(mode="after")
    def lengths_are_consistent(self) -> TokenLengthAuditRow:
        if self.raw_pair_tokens != (
            self.question_tokens + self.candidate_tokens + self.special_tokens
        ):
            raise ValueError("raw pair length must equal question + candidate + special tokens")
        if self.truncated != (self.truncated_candidate_tokens > 0):
            raise ValueError("truncated flag and truncated token count disagree")
        if self.encoded_tokens > self.raw_pair_tokens:
            raise ValueError("encoded token count cannot exceed the raw pair length")
        if self.raw_pair_tokens - self.encoded_tokens != self.truncated_candidate_tokens:
            raise ValueError("only-second truncation accounting is inconsistent")
        return self


class UntrainedForwardTraceRow(_FrozenModel):
    """Finite output proof only; these random-head scores are never outcome metrics."""

    schema_version: Literal["1.0"] = "1.0"
    interpretation: Literal["UNTRAINED_RESOURCE_PROBE_DO_NOT_INTERPRET"] = (
        "UNTRAINED_RESOURCE_PROBE_DO_NOT_INTERPRET"
    )
    dataset_kind: DatasetKind
    query_id: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    retrieval_rank: int = Field(ge=1, le=PREFLIGHT_CANDIDATES_PER_QUERY)
    protect_score: FiniteFloat = Field(ge=0.0, le=1.0)
    harm_score: FiniteFloat = Field(ge=0.0, le=1.0)


class TrainingGpuEstimate(_FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    formula: Literal[
        "epochs*2*max(ceil(niah_pairs/batch),ceil(2wiki_pairs/batch))*seconds_per_microbatch/3600"
    ] = "epochs*2*max(ceil(niah_pairs/batch),ceil(2wiki_pairs/batch))*seconds_per_microbatch/3600"
    source_ratio: Literal["1:1"] = "1:1"
    niah_supervised_pairs: int = Field(gt=0)
    twowiki_supervised_pairs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    epochs: int = Field(gt=0)
    measured_microbatches: int = Field(gt=0)
    warmup_microbatches: int = Field(ge=0)
    timed_microbatches: int = Field(gt=0)
    microbatches_per_epoch: int = Field(gt=0)
    optimizer_steps_total: int = Field(gt=0)
    timed_microbatch_seconds: tuple[FiniteFloat, ...] = Field(min_length=1)
    mean_seconds_per_microbatch: FiniteFloat = Field(gt=0.0)
    p95_seconds_per_microbatch: FiniteFloat = Field(gt=0.0)
    point_gpu_hours: FiniteFloat = Field(gt=0.0)
    conservative_gpu_hours: FiniteFloat = Field(gt=0.0)

    @model_validator(mode="after")
    def estimate_is_ordered(self) -> TrainingGpuEstimate:
        if self.warmup_microbatches + self.timed_microbatches != self.measured_microbatches:
            raise ValueError("warmup + timed microbatches must equal measured microbatches")
        if len(self.timed_microbatch_seconds) != self.timed_microbatches:
            raise ValueError("timed microbatch values must match timed_microbatches")
        if any(value <= 0.0 for value in self.timed_microbatch_seconds):
            raise ValueError("timed microbatch values must be positive")
        expected_mean = sum(self.timed_microbatch_seconds) / len(self.timed_microbatch_seconds)
        if not math.isclose(self.mean_seconds_per_microbatch, expected_mean, rel_tol=1e-12):
            raise ValueError("mean microbatch time does not match the raw timed values")
        if self.p95_seconds_per_microbatch < self.mean_seconds_per_microbatch:
            raise ValueError("p95 time cannot be smaller than mean time")
        ordered = sorted(self.timed_microbatch_seconds)
        expected_p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
        if not math.isclose(self.p95_seconds_per_microbatch, expected_p95, rel_tol=1e-12):
            raise ValueError("p95 microbatch time does not match the raw timed values")
        expected_microbatches_per_epoch = 2 * max(
            math.ceil(self.niah_supervised_pairs / self.batch_size),
            math.ceil(self.twowiki_supervised_pairs / self.batch_size),
        )
        if self.microbatches_per_epoch != expected_microbatches_per_epoch:
            raise ValueError("microbatches_per_epoch does not match the frozen 1:1 schedule")
        expected_optimizer_steps = self.epochs * math.ceil(
            self.microbatches_per_epoch / self.gradient_accumulation_steps
        )
        if self.optimizer_steps_total != expected_optimizer_steps:
            raise ValueError("optimizer_steps_total does not match per-epoch accumulation")
        total_microbatches = self.epochs * self.microbatches_per_epoch
        expected_point = total_microbatches * self.mean_seconds_per_microbatch / 3600.0
        expected_conservative = total_microbatches * self.p95_seconds_per_microbatch / 3600.0
        if not math.isclose(self.point_gpu_hours, expected_point, rel_tol=1e-12):
            raise ValueError("point GPU hours do not match the raw estimate inputs")
        if not math.isclose(self.conservative_gpu_hours, expected_conservative, rel_tol=1e-12):
            raise ValueError("conservative GPU hours do not match the raw estimate inputs")
        if self.conservative_gpu_hours < self.point_gpu_hours:
            raise ValueError("conservative GPU-hour estimate cannot be smaller than point estimate")
        return self


NonEmpty = Annotated[str, Field(min_length=1)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _InputFilePin(_FrozenModel):
    path: NonEmpty
    bytes: int = Field(ge=0)
    sha256: Sha256


class _GitReport(_FrozenModel):
    branch: NonEmpty
    commit: Annotated[str, Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")]
    dirty: Literal[False]


class _InitializedFingerprint(_FrozenModel):
    schema_version: Literal["selector-dual-head-fingerprint-v1"]
    model_id: NonEmpty
    revision: NonEmpty
    max_length: Literal[512]
    weights_sha256: Sha256


class _ModelReport(_FrozenModel):
    model_id: NonEmpty
    revision: NonEmpty
    snapshot_identity_sha256: Sha256
    snapshot_files: Mapping[str, _InputFilePin]
    random_head_seed: Literal[13]
    initialized_full_state_fingerprint: _InitializedFingerprint
    independent_linear_heads: Literal[True]
    activation: Literal["independent-sigmoid"]
    checkpoint: Literal["NOT_APPLICABLE_EPHEMERAL_MODEL_DISCARDED"]

    @model_validator(mode="after")
    def snapshot_is_complete(self) -> _ModelReport:
        if not self.snapshot_files:
            raise ValueError("model snapshot file pins must not be empty")
        return self


class _SamplingReport(_FrozenModel):
    protocol_version: Literal["selector-r004-preflight-v1"]
    seed: Literal[20260811]
    role: Literal["train-modelval"]
    queries_per_dataset: Literal[100]
    sample_queries: Literal[200]
    candidates_per_query: Literal[20]
    forward_pairs: Literal[4000]
    selection_uses_labels_or_lengths: Literal[False]


class _TokenSummaryReport(_FrozenModel):
    pairs: int = Field(gt=0)
    truncated_pairs: int = Field(ge=0)
    truncated_fraction: FiniteFloat = Field(ge=0.0, le=1.0)
    raw_tokens_p50: int = Field(ge=0)
    raw_tokens_p95: int = Field(ge=0)
    raw_tokens_max: int = Field(ge=0)
    encoded_tokens_p50: int = Field(ge=0, le=512)
    encoded_tokens_p95: int = Field(ge=0, le=512)
    encoded_tokens_max: int = Field(ge=0, le=512)
    truncated_candidate_tokens_total: int = Field(ge=0)

    @model_validator(mode="after")
    def token_summary_is_ordered(self) -> _TokenSummaryReport:
        if self.truncated_pairs > self.pairs:
            raise ValueError("truncated pair count cannot exceed pair count")
        if not self.raw_tokens_p50 <= self.raw_tokens_p95 <= self.raw_tokens_max:
            raise ValueError("raw token percentiles are not ordered")
        if not self.encoded_tokens_p50 <= self.encoded_tokens_p95 <= self.encoded_tokens_max:
            raise ValueError("encoded token percentiles are not ordered")
        expected_fraction = self.truncated_pairs / self.pairs
        if not math.isclose(self.truncated_fraction, expected_fraction, rel_tol=1e-12):
            raise ValueError("truncated fraction does not match pair counts")
        return self


class _TokenAuditReport(_FrozenModel):
    scope: Literal["all-train-modelval-top20"]
    queries: Literal[403]
    pairs: Literal[8060]
    max_length: Literal[512]
    truncation: Literal["only_second"]
    overall: _TokenSummaryReport
    by_dataset: Mapping[DatasetKind, _TokenSummaryReport]

    @model_validator(mode="after")
    def dataset_summaries_are_complete(self) -> _TokenAuditReport:
        if set(self.by_dataset) != {"niah", "2wiki"}:
            raise ValueError("token audit requires exactly NIAH and 2Wiki summaries")
        if self.by_dataset["niah"].pairs != 2060:
            raise ValueError("NIAH token summary must contain 2,060 pairs")
        if self.by_dataset["2wiki"].pairs != 6000:
            raise ValueError("2Wiki token summary must contain 6,000 pairs")
        return self


class _ForwardProbeReport(_FrozenModel):
    interpretation: Literal["UNTRAINED_RESOURCE_PROBE_DO_NOT_INTERPRET"]
    queries: Literal[200]
    pairs: Literal[4000]
    batch_size: int = Field(gt=0)
    warmup_batches: Literal[4]
    timed_batches: int = Field(gt=0)
    batch_seconds: tuple[FiniteFloat, ...] = Field(min_length=1)
    wall_time_seconds: FiniteFloat = Field(gt=0.0)
    pairs_per_second: FiniteFloat = Field(gt=0.0)
    queries_per_second: FiniteFloat = Field(gt=0.0)
    batch_seconds_mean: FiniteFloat = Field(gt=0.0)
    batch_seconds_p50: FiniteFloat = Field(gt=0.0)
    batch_seconds_p95: FiniteFloat = Field(gt=0.0)
    max_head_score_difference: FiniteFloat = Field(gt=0.0, le=1.0)
    mean_head_score_difference: FiniteFloat = Field(gt=0.0, le=1.0)
    peak_memory_allocated_bytes: int = Field(gt=0)
    peak_memory_reserved_bytes: int = Field(gt=0)
    oom_encountered: Literal[False]
    nonfinite_encountered: Literal[False]

    @model_validator(mode="after")
    def forward_derived_values_match_raw_times(self) -> _ForwardProbeReport:
        if len(self.batch_seconds) != self.timed_batches:
            raise ValueError("forward raw batch-time count differs from timed_batches")
        if self.timed_batches != math.ceil(self.pairs / self.batch_size):
            raise ValueError("forward timed_batches differs from pair/batch cardinality")
        if any(value <= 0.0 for value in self.batch_seconds):
            raise ValueError("forward raw batch times must be positive")
        wall = sum(self.batch_seconds)
        derived = {
            "wall_time_seconds": wall,
            "pairs_per_second": self.pairs / wall,
            "queries_per_second": self.queries / wall,
            "batch_seconds_mean": wall / len(self.batch_seconds),
            "batch_seconds_p50": nearest_rank_percentile(self.batch_seconds, 50.0),
            "batch_seconds_p95": nearest_rank_percentile(self.batch_seconds, 95.0),
        }
        for name, expected in derived.items():
            if not math.isclose(getattr(self, name), expected, rel_tol=1e-12):
                raise ValueError(f"forward {name} does not match raw batch times")
        if self.mean_head_score_difference > self.max_head_score_difference:
            raise ValueError("mean head difference cannot exceed maximum head difference")
        if self.peak_memory_reserved_bytes < self.peak_memory_allocated_bytes:
            raise ValueError("forward reserved memory cannot be below allocated memory")
        return self


class _TrainingProbeReport(_FrozenModel):
    interpretation: Literal["EPHEMERAL_RESOURCE_PROBE_NO_CHECKPOINT"]
    source_role: Literal["train-fit"]
    pair_order: Literal[
        "ascending-sha256(dataset-kind,newline,query-id,newline,evidence-id,newline,seed)"
    ]
    loss: Literal["unweighted independent masked BCE for resource measurement only"]
    batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    niah_supervised_pairs: int = Field(gt=0)
    twowiki_supervised_pairs: int = Field(gt=0)
    microbatches: Literal[12]
    warmup_microbatches: Literal[4]
    timed_microbatches: Literal[8]
    microbatch_seconds: tuple[FiniteFloat, ...] = Field(min_length=12, max_length=12)
    losses: tuple[FiniteFloat, ...] = Field(min_length=12, max_length=12)
    protect_effective_labels: tuple[int, ...] = Field(min_length=12, max_length=12)
    harm_effective_labels: tuple[int, ...] = Field(min_length=12, max_length=12)
    optimizer_steps: int = Field(gt=0)
    peak_memory_allocated_bytes: int = Field(gt=0)
    peak_memory_reserved_bytes: int = Field(gt=0)
    checkpoint: Literal["NOT_APPLICABLE_EPHEMERAL_MODEL_DISCARDED"]
    oom_encountered: Literal[False]
    nonfinite_encountered: Literal[False]

    @model_validator(mode="after")
    def training_probe_is_internally_consistent(self) -> _TrainingProbeReport:
        if any(value <= 0.0 for value in self.microbatch_seconds):
            raise ValueError("training microbatch times must be positive")
        if any(value < 0.0 for value in self.losses):
            raise ValueError("training losses must be non-negative")
        counts = (*self.protect_effective_labels, *self.harm_effective_labels)
        if any(value < 0 or value > self.batch_size for value in counts):
            raise ValueError("training effective-label count is outside the batch")
        if any(
            protect + harm <= 0
            for protect, harm in zip(
                self.protect_effective_labels, self.harm_effective_labels, strict=True
            )
        ):
            raise ValueError("every training probe microbatch needs active supervision")
        expected_steps = self.microbatches // self.gradient_accumulation_steps
        if self.microbatches % self.gradient_accumulation_steps:
            expected_steps += 1
        if self.optimizer_steps != expected_steps:
            raise ValueError("training optimizer steps do not match accumulation")
        if self.peak_memory_reserved_bytes < self.peak_memory_allocated_bytes:
            raise ValueError("training reserved memory cannot be below allocated memory")
        return self


class _DeviceReport(_FrozenModel):
    kind: Literal["cuda"]
    requested: NonEmpty
    name: NonEmpty
    total_memory_bytes: int = Field(gt=0)
    compute_capability: Annotated[str, Field(pattern=r"^[0-9]+\.[0-9]+$")]
    torch_version: NonEmpty
    cuda_runtime_version: NonEmpty
    precision: Literal["float32"]
    peak_memory_allocated_bytes: int = Field(gt=0)


class _RuntimeReport(_FrozenModel):
    host: NonEmpty
    python_version: NonEmpty
    started_at_utc: AwareDatetime
    finished_at_utc: AwareDatetime
    wall_time_seconds: FiniteFloat = Field(gt=0.0)
    command: tuple[NonEmpty, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def runtime_is_ordered(self) -> _RuntimeReport:
        if self.finished_at_utc < self.started_at_utc:
            raise ValueError("preflight finish time precedes start time")
        return self


class _BoundariesReport(_FrozenModel):
    model_input_fields: tuple[Literal["question", "candidate_text"], ...]
    provenance_is_model_input: Literal[False]
    unjudged_as_negative: Literal[False]
    selector_policy_constructed: Literal[False]
    checkpoint_written: Literal[False]
    sealed_or_heldout_accessed: Literal[False]
    production_default_changed: Literal[False]

    @model_validator(mode="after")
    def input_fields_are_exact(self) -> _BoundariesReport:
        if self.model_input_fields != ("question", "candidate_text"):
            raise ValueError("model input fields must be exactly question,candidate_text")
        return self


class R004ResourcePreflightReport(_FrozenModel):
    """Typed PASS report whose core claims are recomputable from raw measurements."""

    schema_version: Literal["1.0"]
    run_id: Literal["R004"]
    stage: Literal["label-audit-and-200q-preflight"]
    status: Literal["PASS"]
    interpretation: Literal["RESOURCE_AND_LABEL_FEASIBILITY_ONLY_NOT_SELECTOR_EFFECT"]
    git: _GitReport
    config_sha256: Sha256
    model: _ModelReport
    label_audit: Mapping[DatasetKind, Mapping[str, object]]
    sampling: _SamplingReport
    token_audit: _TokenAuditReport
    forward_probe: _ForwardProbeReport
    training_probe: _TrainingProbeReport
    seed13_training_gpu_estimate: TrainingGpuEstimate
    device: _DeviceReport
    runtime: _RuntimeReport
    boundaries: _BoundariesReport

    @model_validator(mode="after")
    def cross_sections_are_bound(self) -> R004ResourcePreflightReport:
        if set(self.label_audit) != {"niah", "2wiki"}:
            raise ValueError("label audit requires exactly NIAH and 2Wiki")
        for kind, audit in self.label_audit.items():
            if (
                audit.get("protocol_version") != "selector-labels-v2"
                or audit.get("dataset_kind") != kind
                or audit.get("source_split") != "train"
                or audit.get("status") != "LABEL_AUDIT_READY"
            ):
                raise ValueError(f"{kind} label audit is not the frozen source-train protocol")
        training = self.training_probe
        estimate = self.seed13_training_gpu_estimate
        if estimate.niah_supervised_pairs != training.niah_supervised_pairs:
            raise ValueError("NIAH supervised pair count is not bound to the GPU estimate")
        if estimate.twowiki_supervised_pairs != training.twowiki_supervised_pairs:
            raise ValueError("2Wiki supervised pair count is not bound to the GPU estimate")
        if estimate.batch_size != training.batch_size:
            raise ValueError("training batch size is not bound to the GPU estimate")
        if estimate.gradient_accumulation_steps != training.gradient_accumulation_steps:
            raise ValueError("training accumulation is not bound to the GPU estimate")
        if tuple(training.microbatch_seconds[training.warmup_microbatches :]) != tuple(
            estimate.timed_microbatch_seconds
        ):
            raise ValueError("GPU estimate is not bound to raw timed microbatches")
        peak = max(
            self.forward_probe.peak_memory_allocated_bytes,
            self.training_probe.peak_memory_allocated_bytes,
        )
        if self.device.peak_memory_allocated_bytes != peak:
            raise ValueError("device peak memory does not match the two probes")
        return self


@dataclass(frozen=True)
class TokenSummary:
    pairs: int
    truncated_pairs: int
    truncated_fraction: float
    raw_tokens_p50: int
    raw_tokens_p95: int
    raw_tokens_max: int
    encoded_tokens_p50: int
    encoded_tokens_p95: int
    encoded_tokens_max: int
    truncated_candidate_tokens_total: int


@dataclass(frozen=True)
class ResourcePreflightArtifacts:
    files: Mapping[str, bytes]
    report: Mapping[str, object]
    manifest: Mapping[str, object]


def preflight_sample_digest(
    dataset_kind: DatasetKind,
    query_id: str,
    *,
    seed: int = PREFLIGHT_SAMPLE_SEED,
) -> str:
    """Return the exact pre-registered sampling digest for one query."""

    if not query_id:
        raise ValueError("query_id must not be empty")
    payload = f"{PREFLIGHT_PROTOCOL_VERSION}\n{dataset_kind}\n{query_id}\n{seed}".encode()
    return hashlib.sha256(payload).hexdigest()


def select_preflight_queries(
    query_ids_by_dataset: Mapping[str, Collection[str]],
    *,
    count_per_dataset: int = PREFLIGHT_QUESTIONS_PER_DATASET,
    seed: int = PREFLIGHT_SAMPLE_SEED,
) -> tuple[PreflightSampleQuery, ...]:
    """Select equal, label-blind samples by ascending SHA-256 only."""

    if count_per_dataset <= 0:
        raise ValueError("count_per_dataset must be positive")
    if set(query_ids_by_dataset) != {"niah", "2wiki"}:
        raise ValueError("preflight requires exactly NIAH and 2Wiki query collections")
    rows: list[PreflightSampleQuery] = []
    for dataset_kind in ("niah", "2wiki"):
        raw_ids = tuple(query_ids_by_dataset[dataset_kind])
        if any(not query_id for query_id in raw_ids):
            raise ValueError(f"{dataset_kind} contains an empty query ID")
        if len(raw_ids) != len(set(raw_ids)):
            raise ValueError(f"{dataset_kind} contains duplicate query IDs")
        ranked = sorted(
            (
                (preflight_sample_digest(dataset_kind, query_id, seed=seed), query_id)
                for query_id in raw_ids
            ),
            key=lambda item: (item[0], item[1]),
        )
        if len(ranked) < count_per_dataset:
            raise ValueError(
                f"{dataset_kind} has {len(ranked)} eligible queries; "
                f"{count_per_dataset} are required"
            )
        rows.extend(
            PreflightSampleQuery(
                dataset_kind=dataset_kind,
                query_id=query_id,
                sample_digest=digest,
            )
            for digest, query_id in ranked[:count_per_dataset]
        )
    return tuple(rows)


def nearest_rank_percentile(values: Sequence[float | int], percentile: float) -> float:
    """Deterministic nearest-rank percentile, including endpoints 0 and 100."""

    if not values:
        raise ValueError("percentile is undefined for an empty sequence")
    if not math.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be finite and in [0, 100]")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    if percentile == 0.0:
        return ordered[0]
    index = math.ceil(percentile / 100.0 * len(ordered)) - 1
    return ordered[index]


def summarize_token_lengths(rows: Sequence[TokenLengthAuditRow]) -> TokenSummary:
    if not rows:
        raise ValueError("token audit must contain at least one pair")
    raw = [row.raw_pair_tokens for row in rows]
    encoded = [row.encoded_tokens for row in rows]
    truncated = sum(row.truncated for row in rows)
    return TokenSummary(
        pairs=len(rows),
        truncated_pairs=truncated,
        truncated_fraction=truncated / len(rows),
        raw_tokens_p50=int(nearest_rank_percentile(raw, 50.0)),
        raw_tokens_p95=int(nearest_rank_percentile(raw, 95.0)),
        raw_tokens_max=max(raw),
        encoded_tokens_p50=int(nearest_rank_percentile(encoded, 50.0)),
        encoded_tokens_p95=int(nearest_rank_percentile(encoded, 95.0)),
        encoded_tokens_max=max(encoded),
        truncated_candidate_tokens_total=sum(row.truncated_candidate_tokens for row in rows),
    )


def estimate_seed_training_gpu_hours(
    *,
    niah_supervised_pairs: int,
    twowiki_supervised_pairs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
    microbatch_seconds: Sequence[float],
    warmup_microbatches: int,
) -> TrainingGpuEstimate:
    """Estimate one seed under the frozen alternating 1:1 source schedule.

    The larger source determines the number of batches for *both* sources because the
    smaller source is cycled.  Gradient accumulation changes optimizer-step count, not
    the number of forward/backward micro-batches.
    """

    counts = (niah_supervised_pairs, twowiki_supervised_pairs, batch_size, epochs)
    if any(value <= 0 for value in counts):
        raise ValueError("pair counts, batch size and epochs must all be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if not 0 <= warmup_microbatches < len(microbatch_seconds):
        raise ValueError("warmup must leave at least one timed microbatch")
    if any(not math.isfinite(value) or value <= 0.0 for value in microbatch_seconds):
        raise ValueError("microbatch times must be finite and positive")

    timed = tuple(microbatch_seconds[warmup_microbatches:])
    mean = sum(timed) / len(timed)
    p95 = nearest_rank_percentile(timed, 95.0)
    per_source_batches = max(
        math.ceil(niah_supervised_pairs / batch_size),
        math.ceil(twowiki_supervised_pairs / batch_size),
    )
    microbatches_per_epoch = 2 * per_source_batches
    total_microbatches = epochs * microbatches_per_epoch
    # The future trainer restarts its accumulation counter at each epoch and flushes the
    # final partial group, so each epoch contributes its own ceiling.
    optimizer_steps_total = epochs * math.ceil(microbatches_per_epoch / gradient_accumulation_steps)
    return TrainingGpuEstimate(
        niah_supervised_pairs=niah_supervised_pairs,
        twowiki_supervised_pairs=twowiki_supervised_pairs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        epochs=epochs,
        measured_microbatches=len(microbatch_seconds),
        warmup_microbatches=warmup_microbatches,
        timed_microbatches=len(timed),
        microbatches_per_epoch=microbatches_per_epoch,
        optimizer_steps_total=optimizer_steps_total,
        timed_microbatch_seconds=timed,
        mean_seconds_per_microbatch=mean,
        p95_seconds_per_microbatch=p95,
        point_gpu_hours=total_microbatches * mean / 3600.0,
        conservative_gpu_hours=total_microbatches * p95 / 3600.0,
    )


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _canonical_jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return (
        "\n".join(
            json.dumps(
                row,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            for row in rows
        )
        + ("\n" if rows else "")
    ).encode()


def _bytes_pin(payload: bytes, *, records: int | None = None) -> dict[str, object]:
    pin: dict[str, object] = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if records is not None:
        pin["records"] = records
    return pin


def _validate_resource_bundle(
    *,
    sample: Sequence[PreflightSampleQuery],
    token_audit: Sequence[TokenLengthAuditRow],
    forward_trace: Sequence[UntrainedForwardTraceRow],
    report: Mapping[str, object],
    input_pins: Mapping[str, Mapping[str, object]],
    expected_git: Mapping[str, object],
    expected_model: Mapping[str, object],
) -> R004ResourcePreflightReport:
    """Fail closed on every cardinality and resource claim needed for R004 PASS."""

    if len(sample) != 2 * PREFLIGHT_QUESTIONS_PER_DATASET:
        raise ValueError("R004 sample must contain exactly 200 queries")
    expected_sample_counts = {kind: 0 for kind in ("niah", "2wiki")}
    sample_keys: set[tuple[DatasetKind, str]] = set()
    for sample_row in sample:
        expected_sample_counts[sample_row.dataset_kind] += 1
        key = (sample_row.dataset_kind, sample_row.query_id)
        if key in sample_keys:
            raise ValueError(f"R004 sample contains duplicate query {key}")
        sample_keys.add(key)
        if sample_row.sample_digest != preflight_sample_digest(
            sample_row.dataset_kind, sample_row.query_id
        ):
            raise ValueError(f"R004 sample digest mismatch for {key}")
    if expected_sample_counts != {"niah": 100, "2wiki": 100}:
        raise ValueError(f"R004 sample must be 100+100, found {expected_sample_counts}")

    if len(token_audit) != 8060:
        raise ValueError("R004 token audit must contain exactly 8,060 pairs")
    token_evidence_keys = [(row.dataset_kind, row.query_id, row.evidence_id) for row in token_audit]
    token_rank_keys = [(row.dataset_kind, row.query_id, row.retrieval_rank) for row in token_audit]
    if len(token_evidence_keys) != len(set(token_evidence_keys)) or len(token_rank_keys) != len(
        set(token_rank_keys)
    ):
        raise ValueError("R004 token audit contains duplicate candidate keys")
    token_by_query: dict[tuple[DatasetKind, str], list[TokenLengthAuditRow]] = {}
    for token_row in token_audit:
        token_by_query.setdefault((token_row.dataset_kind, token_row.query_id), []).append(
            token_row
        )
    query_counts = {
        kind: sum(dataset_kind == kind for dataset_kind, _ in token_by_query)
        for kind in ("niah", "2wiki")
    }
    if query_counts != {"niah": 103, "2wiki": 300}:
        raise ValueError(f"R004 token query counts must be 103+300, found {query_counts}")
    expected_ranks = set(range(1, PREFLIGHT_CANDIDATES_PER_QUERY + 1))
    for query_key, rows in token_by_query.items():
        ranks = {row.retrieval_rank for row in rows}
        if len(rows) != PREFLIGHT_CANDIDATES_PER_QUERY or ranks != expected_ranks:
            raise ValueError(f"R004 token window is not exact ranks 1..20: {query_key}")
    expected_sample = select_preflight_queries(
        {
            kind: [query_id for dataset_kind, query_id in token_by_query if dataset_kind == kind]
            for kind in ("niah", "2wiki")
        }
    )
    if tuple(sample) != expected_sample:
        raise ValueError("R004 sample is not the pre-registered SHA-256 sample")

    if len(forward_trace) != PREFLIGHT_FORWARD_PAIRS:
        raise ValueError("R004 forward trace must contain exactly 4,000 pairs")
    forward_keys = [
        (row.dataset_kind, row.query_id, row.evidence_id, row.retrieval_rank)
        for row in forward_trace
    ]
    forward_evidence_keys = [key[:3] for key in forward_keys]
    forward_rank_keys = [(kind, query_id, rank) for kind, query_id, _, rank in forward_keys]
    if len(forward_evidence_keys) != len(set(forward_evidence_keys)) or len(
        forward_rank_keys
    ) != len(set(forward_rank_keys)):
        raise ValueError("R004 forward trace contains duplicate candidate keys")
    expected_forward_keys: list[tuple[DatasetKind, str, str, int]] = []
    for sample_row in sample:
        rows = sorted(
            token_by_query[(sample_row.dataset_kind, sample_row.query_id)],
            key=lambda row: (row.retrieval_rank, row.evidence_id),
        )
        expected_forward_keys.extend(
            (row.dataset_kind, row.query_id, row.evidence_id, row.retrieval_rank) for row in rows
        )
    if forward_keys != expected_forward_keys:
        raise ValueError("R004 forward trace is not the ordered sample x Top20 candidate set")

    required_pin_prefixes = (
        "config",
        "model_snapshot/model.safetensors",
        "niah/dataset_manifest",
        "niah/source_parent",
        "niah/labels/",
        "niah/assignment",
        "niah/provenance",
        "niah/candidate_pool",
        "niah/pool_manifest",
        "niah/components/",
        "2wiki/dataset_manifest",
        "2wiki/source_parent",
        "2wiki/labels/",
        "2wiki/candidate_pool",
        "2wiki/pool_manifest",
        "2wiki/components/",
    )
    missing_pin_categories = [
        prefix
        for prefix in required_pin_prefixes
        if not any(name == prefix or name.startswith(prefix) for name in input_pins)
    ]
    if missing_pin_categories:
        raise ValueError(f"R004 input pins miss required categories: {missing_pin_categories}")
    for name, pin in input_pins.items():
        try:
            _InputFilePin.model_validate_json(
                json.dumps(pin, ensure_ascii=True, allow_nan=False, sort_keys=True)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"R004 input pin is invalid: {name}: {error}") from error

    try:
        # Always validate through JSON.  Under strict Pydantic semantics a JSON array is
        # legitimately parsed into a tuple, while a Python list is not; using one path here
        # guarantees that build and verify apply the same contract after serialization.
        typed_report = R004ResourcePreflightReport.model_validate_json(
            json.dumps(report, ensure_ascii=True, allow_nan=False, sort_keys=True)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid typed R004 resource report: {error}") from error

    actual_git = typed_report.git.model_dump(mode="json")
    if actual_git != dict(expected_git):
        raise ValueError("R004 report Git pin differs from the inspected clean worktree")
    expected_model_keys = {
        "model_id",
        "revision",
        "snapshot_identity_sha256",
        "snapshot_files",
    }
    if set(expected_model) != expected_model_keys:
        raise ValueError("expected R004 model identity has the wrong fields")
    actual_model = typed_report.model.model_dump(mode="json")
    if any(actual_model[name] != expected_model[name] for name in expected_model_keys):
        raise ValueError("R004 report model identity differs from the audited snapshot")
    if typed_report.config_sha256 != input_pins["config"]["sha256"]:
        raise ValueError("R004 report config hash differs from the pinned config")
    pinned_snapshot = {
        name.removeprefix("model_snapshot/"): dict(pin)
        for name, pin in input_pins.items()
        if name.startswith("model_snapshot/")
    }
    if pinned_snapshot != expected_model["snapshot_files"]:
        raise ValueError("R004 input pins differ from the audited model snapshot")

    head_differences = [abs(row.protect_score - row.harm_score) for row in forward_trace]
    raw_max_difference = max(head_differences)
    raw_mean_difference = sum(head_differences) / len(head_differences)
    if raw_max_difference <= 0.0:
        raise ValueError("R004 forward trace has identical protect/harm scores for every pair")
    if not math.isclose(
        typed_report.forward_probe.max_head_score_difference,
        raw_max_difference,
        rel_tol=1e-12,
    ) or not math.isclose(
        typed_report.forward_probe.mean_head_score_difference,
        raw_mean_difference,
        rel_tol=1e-12,
    ):
        raise ValueError("R004 forward head-difference report differs from raw scores")

    overall_summary = summarize_token_lengths(token_audit)
    if typed_report.token_audit.overall.model_dump(mode="json") != asdict(overall_summary):
        raise ValueError("R004 overall token report differs from raw token rows")
    dataset_expectations: tuple[tuple[DatasetKind, int], ...] = (
        ("niah", 2060),
        ("2wiki", 6000),
    )
    for kind, expected_pairs in dataset_expectations:
        dataset_rows = [row for row in token_audit if row.dataset_kind == kind]
        if len(dataset_rows) != expected_pairs:
            raise ValueError(f"R004 {kind} token rows differ from the frozen cardinality")
        summary = summarize_token_lengths(dataset_rows)
        if typed_report.token_audit.by_dataset[kind].model_dump(mode="json") != asdict(summary):
            raise ValueError(f"R004 {kind} token report differs from raw token rows")
    return typed_report


def build_resource_preflight_artifacts(
    *,
    sample: Sequence[PreflightSampleQuery],
    token_audit: Sequence[TokenLengthAuditRow],
    forward_trace: Sequence[UntrainedForwardTraceRow],
    report: Mapping[str, object],
    runtime_log: Sequence[Mapping[str, object]],
    input_pins: Mapping[str, Mapping[str, object]],
    expected_git: Mapping[str, object],
    expected_model: Mapping[str, object],
) -> ResourcePreflightArtifacts:
    """Build canonical R004 bytes after the GPU probe has completed successfully."""

    if report.get("status") != "PASS":
        raise ValueError("resource preflight artifacts require a PASS report")
    expected_events = (
        "start",
        "labels-verified",
        "token-audit-complete",
        "forward-complete",
        "ephemeral-training-probe-complete",
        "pass",
    )
    actual_events = tuple(row.get("event") for row in runtime_log)
    if actual_events != expected_events:
        raise ValueError(f"R004 runtime log event sequence differs: {actual_events!r}")
    if any(
        not isinstance(row.get("time_utc"), str) or not row.get("time_utc") for row in runtime_log
    ):
        raise ValueError("R004 runtime log requires a timestamp on every event")
    typed_report = _validate_resource_bundle(
        sample=sample,
        token_audit=token_audit,
        forward_trace=forward_trace,
        report=report,
        input_pins=input_pins,
        expected_git=expected_git,
        expected_model=expected_model,
    )
    canonical_report = typed_report.model_dump(mode="json")

    sample_bytes = _canonical_jsonl_bytes([row.model_dump(mode="json") for row in sample])
    token_bytes = _canonical_jsonl_bytes([row.model_dump(mode="json") for row in token_audit])
    forward_bytes = _canonical_jsonl_bytes([row.model_dump(mode="json") for row in forward_trace])
    report_bytes = _canonical_json_bytes(canonical_report)
    runtime_bytes = _canonical_jsonl_bytes(runtime_log)
    output_payloads = {
        SAMPLE_FILE: sample_bytes,
        TOKEN_AUDIT_FILE: token_bytes,
        FORWARD_TRACE_FILE: forward_bytes,
        REPORT_FILE: report_bytes,
        RUNTIME_LOG_FILE: runtime_bytes,
    }
    output_pins = {
        SAMPLE_FILE: _bytes_pin(sample_bytes, records=len(sample)),
        TOKEN_AUDIT_FILE: _bytes_pin(token_bytes, records=len(token_audit)),
        FORWARD_TRACE_FILE: _bytes_pin(forward_bytes, records=len(forward_trace)),
        REPORT_FILE: _bytes_pin(report_bytes),
        RUNTIME_LOG_FILE: _bytes_pin(runtime_bytes, records=len(runtime_log)),
    }
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "manifest_type": "selector_resource_preflight_manifest",
        "protocol_version": PREFLIGHT_PROTOCOL_VERSION,
        "run_id": "R004",
        "status": "PASS",
        "interpretation": "UNTRAINED_RESOURCE_PROBE_DO_NOT_INTERPRET_AS_SELECTOR_EFFECT",
        "inputs": dict(sorted(input_pins.items())),
        "outputs": output_pins,
        "counts": {
            "sample_queries": len(sample),
            "token_audit_pairs": len(token_audit),
            "forward_pairs": len(forward_trace),
            "runtime_log_events": len(runtime_log),
        },
        "artifact_status": {
            "model_checkpoint": "NOT_APPLICABLE_EPHEMERAL_MODEL_DISCARDED",
            "candidate_scores": "NOT_APPLICABLE_UNTRAINED_RESOURCE_TRACE_ONLY",
            "selector_decision_trace": "NOT_APPLICABLE_NO_SELECTOR_POLICY",
            "selected_sets": "NOT_APPLICABLE_NO_SELECTOR_POLICY",
            "crc": "NOT_APPLICABLE_BEFORE_R008",
            "sealed_or_heldout_effect": "NOT_ACCESSED",
        },
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    return ResourcePreflightArtifacts(
        files={**output_payloads, MANIFEST_FILE: manifest_bytes},
        report=canonical_report,
        manifest=manifest,
    )


def freeze_resource_preflight_artifacts(
    output_directory: Path, artifacts: ResourcePreflightArtifacts
) -> Path:
    """Write the complete preflight bundle once and refuse partial replacement."""

    directory = Path(output_directory)
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"resource preflight output is not a directory: {directory}")
    if directory.is_dir():
        existing = sorted(path.name for path in directory.iterdir())
        if existing:
            raise ValueError(
                f"resource preflight output {directory} is not empty; refusing to overwrite: "
                f"{existing[:5]}"
            )
    directory.mkdir(parents=True, exist_ok=True)
    for filename in OUTPUT_FILES:
        try:
            with (directory / filename).open("xb") as stream:
                stream.write(artifacts.files[filename])
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite resource preflight artifact: {directory / filename}"
            ) from error
    return directory / MANIFEST_FILE


def verify_resource_preflight_artifacts(
    output_directory: Path,
    *,
    expected_sample: Sequence[PreflightSampleQuery],
    expected_input_pins: Mapping[str, Mapping[str, object]],
    expected_git: Mapping[str, object],
    expected_model: Mapping[str, object],
) -> ResourcePreflightArtifacts:
    """Parse and pin-check a completed bundle without repeating nondeterministic timings."""

    directory = Path(output_directory)
    if not directory.is_dir():
        raise ValueError(f"missing resource preflight directory: {directory}")
    actual_names = {path.name for path in directory.iterdir()}
    expected_names = set(OUTPUT_FILES)
    if actual_names != expected_names:
        raise ValueError(
            "resource preflight directory has an unexpected file set: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    try:
        payloads = {filename: (directory / filename).read_bytes() for filename in OUTPUT_FILES}
    except OSError as error:
        raise ValueError(f"missing resource preflight artifact in {directory}: {error}") from error
    try:
        manifest_value = json.loads(payloads[MANIFEST_FILE])
        report_value = json.loads(payloads[REPORT_FILE])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid resource preflight JSON: {error}") from error
    if not isinstance(manifest_value, Mapping) or not isinstance(report_value, Mapping):
        raise ValueError("resource preflight manifest/report must be JSON objects")
    manifest = dict(manifest_value)
    report = dict(report_value)
    if payloads[MANIFEST_FILE] != _canonical_json_bytes(manifest):
        raise ValueError("resource preflight manifest is not canonical")
    if payloads[REPORT_FILE] != _canonical_json_bytes(report):
        raise ValueError("resource preflight report is not canonical")
    if manifest.get("status") != "PASS" or report.get("status") != "PASS":
        raise ValueError("resource preflight bundle is not PASS")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("resource preflight manifest has no output pins")
    for filename in OUTPUT_FILES[:-1]:
        pin = outputs.get(filename)
        if not isinstance(pin, Mapping):
            raise ValueError(f"resource preflight manifest omitted output pin {filename}")
        expected_pin = _bytes_pin(
            payloads[filename],
            records=len(payloads[filename].splitlines())
            if filename in {SAMPLE_FILE, TOKEN_AUDIT_FILE, FORWARD_TRACE_FILE, RUNTIME_LOG_FILE}
            else None,
        )
        if dict(pin) != expected_pin:
            raise ValueError(f"resource preflight output pin mismatch for {filename}")
    if manifest.get("inputs") != dict(sorted(expected_input_pins.items())):
        raise ValueError("resource preflight input pins differ from recomputation")

    def parse_lines(model_type: type[_FrozenModel], filename: str) -> tuple[_FrozenModel, ...]:
        rows: list[_FrozenModel] = []
        for line_number, line in enumerate(payloads[filename].splitlines(), start=1):
            try:
                rows.append(model_type.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"invalid {filename} row {line_number}: {error}") from error
        return tuple(rows)

    sample_rows = tuple(
        row
        for row in parse_lines(PreflightSampleQuery, SAMPLE_FILE)
        if isinstance(row, PreflightSampleQuery)
    )
    token_rows = tuple(
        row
        for row in parse_lines(TokenLengthAuditRow, TOKEN_AUDIT_FILE)
        if isinstance(row, TokenLengthAuditRow)
    )
    forward_rows = tuple(
        row
        for row in parse_lines(UntrainedForwardTraceRow, FORWARD_TRACE_FILE)
        if isinstance(row, UntrainedForwardTraceRow)
    )
    if sample_rows != tuple(expected_sample):
        raise ValueError("resource preflight sample differs from pre-registered recomputation")
    rebuilt = build_resource_preflight_artifacts(
        sample=sample_rows,
        token_audit=token_rows,
        forward_trace=forward_rows,
        report=report,
        runtime_log=tuple(json.loads(line) for line in payloads[RUNTIME_LOG_FILE].splitlines()),
        input_pins=expected_input_pins,
        expected_git=expected_git,
        expected_model=expected_model,
    )
    for filename in OUTPUT_FILES:
        if rebuilt.files[filename] != payloads[filename]:
            raise ValueError(
                f"resource preflight artifact differs from canonical rebuild: {filename}"
            )
    return rebuilt
