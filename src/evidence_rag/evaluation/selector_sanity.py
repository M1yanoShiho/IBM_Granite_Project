"""Pure, auditable protocol and metrics for the R005 dual-head sanity run.

This module deliberately has no Torch dependency and performs no file I/O.  It freezes the
parts of R005 that must not change after looking at model-validation outcomes: label/length
blind sampling, active-label class weights, train-fit score quantiles, the tiny-sample
overfit gate, and the diagnostic safe-corner gate.

Callers may pass the small typed records below directly, or pass ordinary mappings to the
label/prediction helpers.  Outputs are frozen, extra-forbidden Pydantic models so a CLI can
serialize them as canonical evidence without inventing fields at run time.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Annotated, Literal, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictBool, model_validator

from evidence_rag.evaluation.selector_risk import (
    conditional_chain_loss,
    document_recall,
    relative_recall_loss,
)

DatasetKind: TypeAlias = Literal["niah", "2wiki"]
HeadName: TypeAlias = Literal["protect", "harm"]
BinaryLabel: TypeAlias = Literal[0, 1]

SANITY_PROTOCOL_VERSION: Literal["selector-r005-sanity-v1"] = "selector-r005-sanity-v1"
SANITY_SAMPLE_SEED: Literal[20260811] = 20260811
SANITY_QUESTIONS_PER_DATASET: Literal[16] = 16
SANITY_EPOCHS: Literal[30] = 30
SANITY_CLASSIFICATION_THRESHOLD = 0.5
SANITY_MINIMUM_TRAINING_ACCURACY = 0.95
SANITY_MINIMUM_NIAH_PAIRS = 12
SANITY_MINIMUM_PAIR_DIRECTION_ACCURACY = 0.95
SANITY_QUANTILES = (0.99, 0.975, 0.95, 0.90)
SANITY_MAX_RECALL_LOSS = 0.03
SANITY_MAX_CHAIN_LOSS = 0.03
SANITY_COUNT_MATCHED_REPEATS: Literal[100] = 100
EXPECTED_MODELVAL_QUERY_COUNTS: Mapping[DatasetKind, int] = MappingProxyType(
    {"niah": 103, "2wiki": 300}
)
EXPECTED_TRAIN_FIT_QUERY_COUNTS: Mapping[DatasetKind, int] = MappingProxyType(
    {"niah": 920, "2wiki": 2700}
)
TOPK_ACTION_SIZE = 10

NonEmpty = Annotated[str, Field(min_length=1)]
Probability = Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


def _finite_probability(value: float, *, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a real number, not bool")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1]")
    return numeric


def _unique_nonempty(values: Collection[str], *, label: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{label} must be a collection of IDs, not one string")
    result = tuple(values)
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{label} must contain non-empty string IDs")
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must not contain duplicate IDs")
    return result


def canonical_model_bytes(model: BaseModel) -> bytes:
    """Return deterministic JSON bytes suitable for hashing an R005 core record."""

    return (
        json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()


class SanitySampleQuery(_FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    protocol_version: Literal["selector-r005-sanity-v1"] = SANITY_PROTOCOL_VERSION
    dataset_kind: DatasetKind
    query_id: NonEmpty
    role: Literal["train-fit"] = "train-fit"
    selection_index: Annotated[int, Field(ge=1, le=SANITY_QUESTIONS_PER_DATASET)]
    sample_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class SanityTrainingSample(_FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    protocol_version: Literal["selector-r005-sanity-v1"] = SANITY_PROTOCOL_VERSION
    seed: Literal[20260811] = SANITY_SAMPLE_SEED
    role: Literal["train-fit"] = "train-fit"
    questions_per_dataset: Literal[16] = SANITY_QUESTIONS_PER_DATASET
    selection_uses_labels_lengths_or_model_scores: Literal[False] = False
    queries: tuple[SanitySampleQuery, ...] = Field(min_length=32, max_length=32)

    @model_validator(mode="after")
    def sample_is_exact_and_canonical(self) -> SanityTrainingSample:
        expected_order = (("niah", index) for index in range(1, 17))
        expected = tuple(expected_order) + tuple(("2wiki", index) for index in range(1, 17))
        observed = tuple((row.dataset_kind, row.selection_index) for row in self.queries)
        if observed != expected:
            raise ValueError("training sample must contain canonical NIAH 1..16 then 2Wiki 1..16")
        identities = {(row.dataset_kind, row.query_id) for row in self.queries}
        if len(identities) != 32:
            raise ValueError("training sample contains duplicate dataset/query IDs")
        for row in self.queries:
            if row.sample_digest != sanity_sample_digest(row.dataset_kind, row.query_id):
                raise ValueError("training sample digest does not match the frozen protocol")
        return self


def sanity_sample_digest(dataset_kind: DatasetKind, query_id: str) -> str:
    """Hash exactly ``protocol\nkind\nquery_id\nseed`` as pre-registered."""

    if dataset_kind not in ("niah", "2wiki"):
        raise ValueError("dataset_kind must be 'niah' or '2wiki'")
    if not isinstance(query_id, str) or not query_id:
        raise ValueError("query_id must be a non-empty string")
    payload = (
        f"{SANITY_PROTOCOL_VERSION}\n{dataset_kind}\n{query_id}\n{SANITY_SAMPLE_SEED}"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def select_sanity_queries(
    query_ids_by_dataset: Mapping[str, Collection[str]],
) -> SanityTrainingSample:
    """Select exactly 16 train-fit query IDs per source without labels or model scores."""

    if set(query_ids_by_dataset) != {"niah", "2wiki"}:
        raise ValueError("query IDs must contain exactly the 'niah' and '2wiki' datasets")
    selected: list[SanitySampleQuery] = []
    for dataset_kind in cast(tuple[DatasetKind, ...], ("niah", "2wiki")):
        query_ids = _unique_nonempty(
            query_ids_by_dataset[dataset_kind], label=f"{dataset_kind} query IDs"
        )
        if len(query_ids) < SANITY_QUESTIONS_PER_DATASET:
            raise ValueError(f"{dataset_kind} has fewer than 16 train-fit queries")
        ranked = sorted(
            query_ids,
            key=lambda query_id: (sanity_sample_digest(dataset_kind, query_id), query_id),
        )[:SANITY_QUESTIONS_PER_DATASET]
        selected.extend(
            SanitySampleQuery(
                dataset_kind=dataset_kind,
                query_id=query_id,
                selection_index=index,
                sample_digest=sanity_sample_digest(dataset_kind, query_id),
            )
            for index, query_id in enumerate(ranked, start=1)
        )
    return SanityTrainingSample(queries=tuple(selected))


class SanityLabelRecord(_FrozenModel):
    """Minimal projection of one frozen Selector label row used by the CPU core."""

    dataset_kind: DatasetKind
    query_id: NonEmpty
    evidence_id: NonEmpty
    role: Literal["train-fit", "train-modelval"]
    protect_label: BinaryLabel | None
    protect_mask: StrictBool
    harm_label: BinaryLabel | None
    harm_mask: StrictBool

    @model_validator(mode="after")
    def masks_match_labels(self) -> SanityLabelRecord:
        if self.protect_mask != (self.protect_label is not None):
            raise ValueError("protect_mask must be true exactly when protect_label is set")
        if self.harm_mask != (self.harm_label is not None):
            raise ValueError("harm_mask must be true exactly when harm_label is set")
        if self.dataset_kind == "2wiki" and self.harm_mask:
            raise ValueError("2Wiki has no audited R005 harm labels")
        return self


def _project_label_record(value: SanityLabelRecord | Mapping[str, object]) -> SanityLabelRecord:
    if isinstance(value, SanityLabelRecord):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("label records must be SanityLabelRecord instances or mappings")
    fields = (
        "dataset_kind",
        "query_id",
        "evidence_id",
        "role",
        "protect_label",
        "protect_mask",
        "harm_label",
        "harm_mask",
    )
    missing = [field for field in fields if field not in value]
    if missing:
        raise ValueError(f"label record is missing required fields: {missing}")
    return SanityLabelRecord.model_validate({field: value[field] for field in fields})


class SanityClassWeight(_FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    protocol_version: Literal["selector-r005-sanity-v1"] = SANITY_PROTOCOL_VERSION
    dataset_kind: DatasetKind
    head: HeadName
    class_label: BinaryLabel
    active_count: Annotated[int, Field(gt=0)]
    inverse_sqrt_frequency: Annotated[FiniteFloat, Field(gt=0.0)]
    normalized_weight: Annotated[FiniteFloat, Field(gt=0.0)]


def compute_class_weights(
    records: Iterable[SanityLabelRecord | Mapping[str, object]],
) -> tuple[SanityClassWeight, ...]:
    """Compute source × head inverse-sqrt weights with active-example mean equal to one.

    An absent class produces no row and no division.  Normalization is weighted by observed
    examples, i.e. ``sum(n_c * w_c) / sum(n_c) == 1`` within each source/head.
    """

    projected = tuple(_project_label_record(record) for record in records)
    if not projected:
        raise ValueError("class weights require at least one label record")
    identities: set[tuple[str, str, str]] = set()
    counts: Counter[tuple[DatasetKind, HeadName, BinaryLabel]] = Counter()
    for row in projected:
        if row.role != "train-fit":
            raise ValueError("R005 class weights may use train-fit labels only")
        identity = (row.dataset_kind, row.query_id, row.evidence_id)
        if identity in identities:
            raise ValueError(f"duplicate label identity: {identity}")
        identities.add(identity)
        for head in cast(tuple[HeadName, ...], ("protect", "harm")):
            if getattr(row, f"{head}_mask"):
                counts[(row.dataset_kind, head, getattr(row, f"{head}_label"))] += 1

    output: list[SanityClassWeight] = []
    for dataset_kind in cast(tuple[DatasetKind, ...], ("niah", "2wiki")):
        for head in cast(tuple[HeadName, ...], ("protect", "harm")):
            class_counts = {
                class_label: counts[(dataset_kind, head, class_label)]
                for class_label in cast(tuple[BinaryLabel, ...], (0, 1))
                if counts[(dataset_kind, head, class_label)] > 0
            }
            if not class_counts:
                continue
            total = sum(class_counts.values())
            normalizer = total / sum(math.sqrt(count) for count in class_counts.values())
            for class_label in cast(tuple[BinaryLabel, ...], (0, 1)):
                count = class_counts.get(class_label)
                if count is None:
                    continue
                raw = 1.0 / math.sqrt(count)
                output.append(
                    SanityClassWeight(
                        dataset_kind=dataset_kind,
                        head=head,
                        class_label=class_label,
                        active_count=count,
                        inverse_sqrt_frequency=raw,
                        normalized_weight=raw * normalizer,
                    )
                )
    if not output:
        raise ValueError("no active labels were observed")
    return tuple(output)


class SanityCandidateScoreRow(_FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    protocol_version: Literal["selector-r005-sanity-v1"] = SANITY_PROTOCOL_VERSION
    dataset_kind: DatasetKind
    query_id: NonEmpty
    evidence_id: NonEmpty
    document_id: NonEmpty
    role: Literal["train-fit", "train-modelval"]
    retrieval_rank: Annotated[int, Field(ge=1, le=20)]
    score_scope: Literal[
        "train-fit-quantile-and-overfit",
        "train-fit-quantile",
        "sanity-sample-overfit-extra",
        "train-modelval-heldout",
    ]
    text_pair_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    protect_score: Probability
    harm_score: Probability
    safe_score: Probability

    @model_validator(mode="after")
    def derived_fields_are_exact(self) -> SanityCandidateScoreRow:
        expected_safe = min(float(self.harm_score), 1.0 - float(self.protect_score))
        if not math.isclose(float(self.safe_score), expected_safe, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("safe_score must equal min(harm_score, 1-protect_score)")
        if self.score_scope == "train-modelval-heldout":
            if self.role != "train-modelval":
                raise ValueError("train-modelval score scope requires train-modelval role")
        elif self.role != "train-fit":
            raise ValueError("all non-heldout score scopes require train-fit role")
        if self.score_scope == "sanity-sample-overfit-extra":
            if self.retrieval_rank <= TOPK_ACTION_SIZE:
                raise ValueError("overfit-extra scope is reserved for ranks 11--20")
        elif (
            self.score_scope
            in {
                "train-fit-quantile-and-overfit",
                "train-fit-quantile",
            }
            and self.retrieval_rank > TOPK_ACTION_SIZE
        ):
            raise ValueError("train-fit quantile scopes are restricted to ranks 1--10")
        return self


class SanityQuantileThreshold(_FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    protocol_version: Literal["selector-r005-sanity-v1"] = SANITY_PROTOCOL_VERSION
    point_index: Annotated[int, Field(ge=1, le=4)]
    quantile: Probability
    threshold: Probability
    nearest_rank: Annotated[int, Field(gt=0)]
    score_count: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def rank_and_point_match_protocol(self) -> SanityQuantileThreshold:
        if self.quantile != SANITY_QUANTILES[self.point_index - 1]:
            raise ValueError("point index and pre-registered quantile disagree")
        if self.nearest_rank != math.ceil(float(self.quantile) * self.score_count):
            raise ValueError("nearest rank does not match quantile and score count")
        return self


def nearest_rank_quantiles(scores: Sequence[float]) -> tuple[SanityQuantileThreshold, ...]:
    """Map the four frozen quantiles, preserving order and duplicate thresholds."""

    if not scores:
        raise ValueError("nearest-rank quantiles require at least one safe score")
    ordered = sorted(_finite_probability(score, label="safe score") for score in scores)
    output: list[SanityQuantileThreshold] = []
    for point_index, quantile in enumerate(SANITY_QUANTILES, start=1):
        rank = math.ceil(quantile * len(ordered))
        output.append(
            SanityQuantileThreshold(
                point_index=point_index,
                quantile=quantile,
                threshold=ordered[rank - 1],
                nearest_rank=rank,
                score_count=len(ordered),
            )
        )
    return tuple(output)


def derive_train_fit_thresholds(
    rows: Sequence[SanityCandidateScoreRow],
    *,
    expected_query_counts: Mapping[DatasetKind, int] = EXPECTED_TRAIN_FIT_QUERY_COUNTS,
) -> tuple[SanityQuantileThreshold, ...]:
    """Use exactly every train-fit query's ranks 1--10, with no label filtering.

    The runner may additionally score ranks 11--20 for the 32 overfit-sample queries, but those
    rows are not part of this universe and must be projected out before calling this function.
    The production default validates all 920 NIAH plus 2,700 2Wiki train-fit queries; the
    explicit count argument exists for isolated protocol tests.
    """

    if not rows:
        raise ValueError("threshold derivation requires candidate score rows")
    by_query: dict[tuple[DatasetKind, str], list[SanityCandidateScoreRow]] = defaultdict(list)
    identities: set[tuple[str, str, str]] = set()
    for row in rows:
        if row.role != "train-fit":
            raise ValueError("diagnostic thresholds may use train-fit scores only")
        identity = (row.dataset_kind, row.query_id, row.evidence_id)
        if identity in identities:
            raise ValueError(f"duplicate candidate score identity: {identity}")
        identities.add(identity)
        by_query[(row.dataset_kind, row.query_id)].append(row)
    if {key[0] for key in by_query} != {"niah", "2wiki"}:
        raise ValueError("threshold universe must contain both NIAH and 2Wiki train-fit queries")
    if set(expected_query_counts) != {"niah", "2wiki"} or any(
        isinstance(count, bool) or not isinstance(count, int) or count <= 0
        for count in expected_query_counts.values()
    ):
        raise ValueError("expected query counts require positive NIAH and 2Wiki integers")
    observed_query_counts = Counter(key[0] for key in by_query)
    for dataset_kind in cast(tuple[DatasetKind, ...], ("niah", "2wiki")):
        if observed_query_counts[dataset_kind] != expected_query_counts[dataset_kind]:
            raise ValueError(
                f"{dataset_kind} threshold universe has "
                f"{observed_query_counts[dataset_kind]} queries; expected "
                f"{expected_query_counts[dataset_kind]}"
            )
    scores: list[float] = []
    for key, query_rows in by_query.items():
        ranks = [row.retrieval_rank for row in query_rows]
        if len(query_rows) != 10 or sorted(ranks) != list(range(1, 11)):
            raise ValueError(f"train-fit threshold query {key} must contain exactly ranks 1..10")
        if any(
            row.score_scope not in {"train-fit-quantile-and-overfit", "train-fit-quantile"}
            for row in query_rows
        ):
            raise ValueError("threshold universe may contain only train-fit quantile scopes")
        scores.extend(float(row.safe_score) for row in query_rows)
    return nearest_rank_quantiles(scores)


class SanityTrainingPrediction(_FrozenModel):
    """Final-epoch score joined to its sampled train-fit active/masked labels."""

    schema_version: Literal["1.0"] = "1.0"
    protocol_version: Literal["selector-r005-sanity-v1"] = SANITY_PROTOCOL_VERSION
    dataset_kind: DatasetKind
    query_id: NonEmpty
    evidence_id: NonEmpty
    role: Literal["train-fit"] = "train-fit"
    protect_label: BinaryLabel | None
    protect_mask: StrictBool
    harm_label: BinaryLabel | None
    harm_mask: StrictBool
    protect_score: Probability
    harm_score: Probability
    safe_score: Probability

    @model_validator(mode="after")
    def masks_and_safe_score_are_consistent(self) -> SanityTrainingPrediction:
        if self.protect_mask != (self.protect_label is not None):
            raise ValueError("protect mask and label disagree")
        if self.harm_mask != (self.harm_label is not None):
            raise ValueError("harm mask and label disagree")
        if self.dataset_kind == "2wiki" and self.harm_mask:
            raise ValueError("2Wiki harm must remain masked")
        expected = min(float(self.harm_score), 1.0 - float(self.protect_score))
        if not math.isclose(float(self.safe_score), expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("safe_score must equal min(harm_score, 1-protect_score)")
        return self


class SanitySourceHeadLossTrend(_FrozenModel):
    dataset_kind: DatasetKind
    head: HeadName
    initial_loss: Annotated[FiniteFloat, Field(ge=0.0)]
    final_loss: Annotated[FiniteFloat, Field(ge=0.0)]
    decreased: StrictBool

    @model_validator(mode="after")
    def decrease_flag_matches_losses(self) -> SanitySourceHeadLossTrend:
        if self.decreased != (float(self.final_loss) < float(self.initial_loss)):
            raise ValueError("loss decreased flag must mean final_loss < initial_loss")
        return self


class SanityTrainingTrace(_FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    protocol_version: Literal["selector-r005-sanity-v1"] = SANITY_PROTOCOL_VERSION
    epochs_planned: Literal[30] = SANITY_EPOCHS
    termination_reason: Literal["completed", "sample-coverage", "cuda-oom", "nonfinite"]
    epochs_completed: Annotated[int, Field(ge=0, le=30)]
    failure_epoch: Annotated[int, Field(ge=1, le=30)] | None = None
    final_checkpoint_epoch: Annotated[int, Field(ge=0, le=30)] | None
    epoch_mean_losses: tuple[FiniteFloat, ...]
    active_source_head_loss_trends: tuple[SanitySourceHeadLossTrend, ...] = Field(max_length=3)
    protect_head_parameter_l2_change: Annotated[FiniteFloat, Field(ge=0.0)]
    harm_head_parameter_l2_change: Annotated[FiniteFloat, Field(ge=0.0)]
    twowiki_harm_head_gradient_l1: Annotated[FiniteFloat, Field(ge=0.0)] | None
    independent_sigmoid_heads: StrictBool
    oom_encountered: StrictBool
    nonfinite_encountered: StrictBool

    @model_validator(mode="after")
    def trace_counts_match(self) -> SanityTrainingTrace:
        if len(self.epoch_mean_losses) != self.epochs_completed:
            raise ValueError("one finite mean loss is required per completed epoch")
        if self.final_checkpoint_epoch is not None and (
            self.final_checkpoint_epoch != self.epochs_completed
        ):
            raise ValueError("the only R005 checkpoint must be from the final completed epoch")
        if any(float(value) < 0.0 for value in self.epoch_mean_losses):
            raise ValueError("epoch mean losses must be non-negative")

        loss_keys = tuple(
            (row.dataset_kind, row.head) for row in self.active_source_head_loss_trends
        )
        required_loss_keys = {
            ("niah", "protect"),
            ("niah", "harm"),
            ("2wiki", "protect"),
        }
        if len(loss_keys) != len(set(loss_keys)):
            raise ValueError("loss trends must not contain duplicate source/head keys")

        if self.termination_reason == "completed":
            if self.epochs_completed != SANITY_EPOCHS:
                raise ValueError("completed training requires all 30 epochs")
            if self.final_checkpoint_epoch != SANITY_EPOCHS:
                raise ValueError("completed training requires the epoch-30 checkpoint")
            if self.failure_epoch is not None:
                raise ValueError("completed training cannot have a failure epoch")
            if self.oom_encountered or self.nonfinite_encountered:
                raise ValueError("completed training cannot record OOM or non-finite failure")
            if set(loss_keys) != required_loss_keys or len(loss_keys) != 3:
                raise ValueError(
                    "completed training requires exactly NIAH protect/harm and 2Wiki protect "
                    "loss trends"
                )
            if self.twowiki_harm_head_gradient_l1 is None:
                raise ValueError("completed training requires the 2Wiki harm gradient probe")
            return self

        if self.active_source_head_loss_trends:
            raise ValueError("early training failure cannot report final loss trends")
        if self.twowiki_harm_head_gradient_l1 is not None:
            raise ValueError("early training failure cannot report the final gradient probe")

        if self.termination_reason == "sample-coverage":
            if self.epochs_completed != 0 or self.final_checkpoint_epoch != 0:
                raise ValueError("sample-coverage failure requires epoch-0 checkpoint state")
            if self.failure_epoch is not None:
                raise ValueError("sample-coverage failure occurs before any training epoch")
            if self.oom_encountered or self.nonfinite_encountered:
                raise ValueError("sample-coverage failure cannot record OOM or non-finite failure")
            return self

        if self.epochs_completed >= SANITY_EPOCHS:
            raise ValueError("OOM/non-finite failure requires 0..29 completed epochs")
        if self.final_checkpoint_epoch != self.epochs_completed:
            raise ValueError("early runtime failure checkpoint must match the last completed epoch")
        if self.failure_epoch != self.epochs_completed + 1:
            raise ValueError("failure_epoch must be the epoch after the last completed epoch")
        expected_oom = self.termination_reason == "cuda-oom"
        expected_nonfinite = self.termination_reason == "nonfinite"
        if self.oom_encountered != expected_oom or self.nonfinite_encountered != expected_nonfinite:
            raise ValueError("termination reason must match exactly one runtime failure flag")
        return self


class SanityClassAccuracy(_FrozenModel):
    dataset_kind: DatasetKind
    head: HeadName
    class_label: BinaryLabel
    correct: NonNegativeInt
    total: Annotated[int, Field(gt=0)]
    accuracy: Probability

    @model_validator(mode="after")
    def accuracy_matches_counts(self) -> SanityClassAccuracy:
        if self.correct > self.total:
            raise ValueError("correct count cannot exceed total")
        if not math.isclose(float(self.accuracy), self.correct / self.total, abs_tol=1e-12):
            raise ValueError("class accuracy does not match its counts")
        return self


class SanityPairDirection(_FrozenModel):
    paired_queries: NonNegativeInt
    protect_correct: NonNegativeInt
    harm_correct: NonNegativeInt
    safe_correct: NonNegativeInt
    all_three_correct: NonNegativeInt
    protect_accuracy: Probability | None
    harm_accuracy: Probability | None
    safe_accuracy: Probability | None
    all_three_accuracy: Probability | None

    @model_validator(mode="after")
    def pair_rates_match_counts(self) -> SanityPairDirection:
        fields = ("protect", "harm", "safe", "all_three")
        for name in fields:
            count = getattr(self, f"{name}_correct")
            accuracy = getattr(self, f"{name}_accuracy")
            if count > self.paired_queries:
                raise ValueError("pair-direction correct count cannot exceed paired queries")
            if self.paired_queries == 0:
                if count != 0 or accuracy is not None:
                    raise ValueError("zero paired queries require zero counts and N/A accuracies")
            elif accuracy is None or not math.isclose(
                float(accuracy), count / self.paired_queries, abs_tol=1e-12
            ):
                raise ValueError("pair-direction accuracy does not match its count")
        if self.all_three_correct > min(self.protect_correct, self.harm_correct, self.safe_correct):
            raise ValueError("all-three pair successes must be a subset of each direction")
        return self


class SanityTrainingDecision(_FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    protocol_version: Literal["selector-r005-sanity-v1"] = SANITY_PROTOCOL_VERSION
    status: Literal["PASS", "FAIL"]
    class_coverage_pass: StrictBool
    class_accuracy_pass: StrictBool
    pair_direction_pass: StrictBool
    execution_pass: StrictBool
    class_accuracy: tuple[SanityClassAccuracy, ...]
    niah_pair_direction: SanityPairDirection
    failed_checks: tuple[NonEmpty, ...]

    @model_validator(mode="after")
    def status_matches_checks(self) -> SanityTrainingDecision:
        keys = tuple((row.dataset_kind, row.head, row.class_label) for row in self.class_accuracy)
        if len(keys) != len(set(keys)):
            raise ValueError("training decision contains duplicate class-accuracy rows")
        expected_coverage = frozenset(keys) == _REQUIRED_CLASS_KEYS
        if self.class_coverage_pass != expected_coverage:
            raise ValueError("class_coverage_pass does not match class-accuracy rows")
        expected_accuracy = expected_coverage and all(
            float(row.accuracy) >= SANITY_MINIMUM_TRAINING_ACCURACY for row in self.class_accuracy
        )
        if self.class_accuracy_pass != expected_accuracy:
            raise ValueError("class_accuracy_pass does not match per-class accuracies")
        expected_pair = (
            self.niah_pair_direction.paired_queries >= SANITY_MINIMUM_NIAH_PAIRS
            and self.niah_pair_direction.all_three_accuracy is not None
            and float(self.niah_pair_direction.all_three_accuracy)
            >= SANITY_MINIMUM_PAIR_DIRECTION_ACCURACY
        )
        if self.pair_direction_pass != expected_pair:
            raise ValueError("pair_direction_pass does not match pair metrics")
        passed = all(
            (
                self.class_coverage_pass,
                self.class_accuracy_pass,
                self.pair_direction_pass,
                self.execution_pass,
            )
        )
        if (self.status == "PASS") != passed:
            raise ValueError("training decision status does not match its gates")
        if passed != (not self.failed_checks):
            raise ValueError("failed_checks must be empty exactly when all gates pass")
        return self


_REQUIRED_CLASS_KEYS: frozenset[tuple[DatasetKind, HeadName, BinaryLabel]] = frozenset(
    {
        ("niah", "protect", 0),
        ("niah", "protect", 1),
        ("niah", "harm", 0),
        ("niah", "harm", 1),
        ("2wiki", "protect", 1),
    }
)


def evaluate_training_sanity(
    predictions: Sequence[SanityTrainingPrediction],
    trace: SanityTrainingTrace,
) -> SanityTrainingDecision:
    """Evaluate the frozen final-epoch class and NIAH clean/counterfactual gates."""

    if not predictions:
        if trace.termination_reason == "completed":
            raise ValueError("completed training sanity requires final-epoch predictions")
        empty_pair_summary = SanityPairDirection(
            paired_queries=0,
            protect_correct=0,
            harm_correct=0,
            safe_correct=0,
            all_three_correct=0,
            protect_accuracy=None,
            harm_accuracy=None,
            safe_accuracy=None,
            all_three_accuracy=None,
        )
        return SanityTrainingDecision(
            status="FAIL",
            class_coverage_pass=False,
            class_accuracy_pass=False,
            pair_direction_pass=False,
            execution_pass=False,
            class_accuracy=(),
            niah_pair_direction=empty_pair_summary,
            failed_checks=(
                "required-active-class-coverage",
                "per-source-head-class-accuracy",
                "niah-clean-counterfactual-pair-direction",
                "finite-complete-independent-head-training",
            ),
        )
    identities: set[tuple[str, str, str]] = set()
    queries: dict[DatasetKind, set[str]] = {"niah": set(), "2wiki": set()}
    class_counts: Counter[tuple[DatasetKind, HeadName, BinaryLabel]] = Counter()
    correct_counts: Counter[tuple[DatasetKind, HeadName, BinaryLabel]] = Counter()
    by_niah_query: dict[str, list[SanityTrainingPrediction]] = defaultdict(list)
    for row in predictions:
        identity = (row.dataset_kind, row.query_id, row.evidence_id)
        if identity in identities:
            raise ValueError(f"duplicate training prediction identity: {identity}")
        identities.add(identity)
        queries[row.dataset_kind].add(row.query_id)
        if row.dataset_kind == "niah":
            by_niah_query[row.query_id].append(row)
        for head in cast(tuple[HeadName, ...], ("protect", "harm")):
            if not getattr(row, f"{head}_mask"):
                continue
            label = cast(BinaryLabel, getattr(row, f"{head}_label"))
            key = (row.dataset_kind, head, label)
            class_counts[key] += 1
            predicted = int(float(getattr(row, f"{head}_score")) >= SANITY_CLASSIFICATION_THRESHOLD)
            correct_counts[key] += int(predicted == label)
    if trace.termination_reason == "completed" and any(
        len(query_ids) != SANITY_QUESTIONS_PER_DATASET for query_ids in queries.values()
    ):
        raise ValueError("training predictions must cover exactly 16 queries per dataset")

    accuracy_rows = tuple(
        SanityClassAccuracy(
            dataset_kind=dataset_kind,
            head=head,
            class_label=class_label,
            correct=correct_counts[key],
            total=count,
            accuracy=correct_counts[key] / count,
        )
        for key, count in sorted(class_counts.items())
        for dataset_kind, head, class_label in (key,)
    )
    observed_keys = frozenset(class_counts)
    coverage_pass = observed_keys == _REQUIRED_CLASS_KEYS
    class_accuracy_pass = coverage_pass and all(
        float(row.accuracy) >= SANITY_MINIMUM_TRAINING_ACCURACY for row in accuracy_rows
    )

    protect_correct = harm_correct = safe_correct = all_three_correct = 0
    paired_queries = 0
    for query_id, rows in by_niah_query.items():
        clean = [
            row
            for row in rows
            if row.protect_mask and row.harm_mask and row.protect_label == 1 and row.harm_label == 0
        ]
        counterfactual = [
            row
            for row in rows
            if row.protect_mask and row.harm_mask and row.protect_label == 0 and row.harm_label == 1
        ]
        if len(clean) > 1 or len(counterfactual) > 1:
            raise ValueError(f"NIAH query {query_id!r} has ambiguous clean/counterfactual pairing")
        if not clean or not counterfactual:
            continue
        paired_queries += 1
        protect_ok = float(clean[0].protect_score) > float(counterfactual[0].protect_score)
        harm_ok = float(counterfactual[0].harm_score) > float(clean[0].harm_score)
        safe_ok = float(counterfactual[0].safe_score) > float(clean[0].safe_score)
        protect_correct += int(protect_ok)
        harm_correct += int(harm_ok)
        safe_correct += int(safe_ok)
        all_three_correct += int(protect_ok and harm_ok and safe_ok)
    divisor = paired_queries or 1
    pair_summary = SanityPairDirection(
        paired_queries=paired_queries,
        protect_correct=protect_correct,
        harm_correct=harm_correct,
        safe_correct=safe_correct,
        all_three_correct=all_three_correct,
        protect_accuracy=None if not paired_queries else protect_correct / divisor,
        harm_accuracy=None if not paired_queries else harm_correct / divisor,
        safe_accuracy=None if not paired_queries else safe_correct / divisor,
        all_three_accuracy=None if not paired_queries else all_three_correct / divisor,
    )
    pair_pass = (
        paired_queries >= SANITY_MINIMUM_NIAH_PAIRS
        and pair_summary.all_three_accuracy is not None
        and float(pair_summary.all_three_accuracy) >= SANITY_MINIMUM_PAIR_DIRECTION_ACCURACY
    )
    execution_pass = (
        trace.termination_reason == "completed"
        and trace.epochs_completed == SANITY_EPOCHS
        and trace.final_checkpoint_epoch == SANITY_EPOCHS
        and trace.failure_epoch is None
        and all(row.decreased for row in trace.active_source_head_loss_trends)
        and float(trace.protect_head_parameter_l2_change) > 0.0
        and float(trace.harm_head_parameter_l2_change) > 0.0
        and trace.twowiki_harm_head_gradient_l1 is not None
        and float(trace.twowiki_harm_head_gradient_l1) == 0.0
        and trace.independent_sigmoid_heads
        and not trace.oom_encountered
        and not trace.nonfinite_encountered
    )
    checks = []
    if not coverage_pass:
        checks.append("required-active-class-coverage")
    if not class_accuracy_pass:
        checks.append("per-source-head-class-accuracy")
    if not pair_pass:
        checks.append("niah-clean-counterfactual-pair-direction")
    if not execution_pass:
        checks.append("finite-complete-independent-head-training")
    return SanityTrainingDecision(
        status="PASS" if not checks else "FAIL",
        class_coverage_pass=coverage_pass,
        class_accuracy_pass=class_accuracy_pass,
        pair_direction_pass=pair_pass,
        execution_pass=execution_pass,
        class_accuracy=accuracy_rows,
        niah_pair_direction=pair_summary,
        failed_checks=tuple(checks),
    )


class SanityPolicyQueryOutcome(_FrozenModel):
    """One diagnostic point's delete-only outcome for one train-modelval query."""

    schema_version: Literal["1.0"] = "1.0"
    protocol_version: Literal["selector-r005-sanity-v1"] = SANITY_PROTOCOL_VERSION
    dataset_kind: DatasetKind
    query_id: NonEmpty
    role: Literal["train-modelval"] = "train-modelval"
    baseline_document_ids: tuple[NonEmpty, ...] = Field(min_length=1, max_length=10)
    selected_document_ids: tuple[NonEmpty, ...] = Field(min_length=1, max_length=10)
    dropped_evidence_ids: tuple[NonEmpty, ...] = Field(max_length=1)
    dropped_document_ids: tuple[NonEmpty, ...] = Field(max_length=1)
    required_document_ids: tuple[NonEmpty, ...] = Field(min_length=1)
    harmful_document_id: NonEmpty | None = None
    harmful_in_top20_pool: StrictBool | None = None

    @model_validator(mode="after")
    def outcome_is_delete_only_cap1(self) -> SanityPolicyQueryOutcome:
        baseline = _unique_nonempty(self.baseline_document_ids, label="baseline_document_ids")
        selected = _unique_nonempty(self.selected_document_ids, label="selected_document_ids")
        _unique_nonempty(self.required_document_ids, label="required_document_ids")
        if not set(selected).issubset(baseline):
            raise ValueError("selected documents must be a subset of TopK10 documents")
        if len(self.dropped_evidence_ids) != len(self.dropped_document_ids):
            raise ValueError("each dropped candidate needs one evidence and document ID")
        if any(document_id not in set(baseline) for document_id in self.dropped_document_ids):
            raise ValueError("dropped documents must come from TopK10")
        if self.dataset_kind == "niah":
            if self.harmful_document_id is None or self.harmful_in_top20_pool is None:
                raise ValueError("NIAH outcomes require the harmful document and Top20 flag")
        elif self.harmful_document_id is not None or self.harmful_in_top20_pool is not None:
            raise ValueError("2Wiki outcomes must not invent harmful labels")
        return self


class SanityDatasetMetrics(_FrozenModel):
    dataset_kind: DatasetKind
    queries: Annotated[int, Field(gt=0)]
    actual_deletions: NonNegativeInt
    document_recall: Probability
    topk10_relative_recall_loss: Probability
    n_topk10_chain_eligible: NonNegativeInt
    conditional_chain_loss: Probability | None

    @model_validator(mode="after")
    def counts_and_chain_denominator_are_consistent(self) -> SanityDatasetMetrics:
        if self.actual_deletions > self.queries:
            raise ValueError("cap1 deletions cannot exceed the query count")
        if self.n_topk10_chain_eligible > self.queries:
            raise ValueError("chain-eligible count cannot exceed the query count")
        if (self.conditional_chain_loss is None) != (self.n_topk10_chain_eligible == 0):
            raise ValueError("conditional chain loss is N/A exactly when no query is eligible")
        return self


class SanityHarmMetrics(_FrozenModel):
    n_pool_conditional: Annotated[int, Field(gt=0)]
    baseline_pool_conditional_exposure: Probability
    selector_pool_conditional_exposure: Probability
    pool_conditional_harmful_reduction: FiniteFloat = Field(ge=-1.0, le=1.0)
    n_dropped_candidate_actions: NonNegativeInt
    n_harmful_dropped_candidate_actions: NonNegativeInt
    deletion_precision: Probability | None

    @model_validator(mode="after")
    def harm_counts_are_consistent(self) -> SanityHarmMetrics:
        if self.n_harmful_dropped_candidate_actions > self.n_dropped_candidate_actions:
            raise ValueError("harmful drops cannot exceed all drops")
        expected_precision = (
            None
            if self.n_dropped_candidate_actions == 0
            else self.n_harmful_dropped_candidate_actions / self.n_dropped_candidate_actions
        )
        if expected_precision is None:
            if self.deletion_precision is not None:
                raise ValueError("zero drops require N/A deletion precision")
        elif self.deletion_precision is None or not math.isclose(
            float(self.deletion_precision), expected_precision, abs_tol=1e-12
        ):
            raise ValueError("deletion precision does not match drop counts")
        expected_reduction = float(self.baseline_pool_conditional_exposure) - float(
            self.selector_pool_conditional_exposure
        )
        if not math.isclose(
            float(self.pool_conditional_harmful_reduction), expected_reduction, abs_tol=1e-12
        ):
            raise ValueError("harmful reduction must be TopK10 exposure minus Selector exposure")
        return self


class SanityRandomPrecision(_FrozenModel):
    repeats: Literal[100] = SANITY_COUNT_MATCHED_REPEATS
    repeat_precisions: tuple[Probability | None, ...] = Field(min_length=100, max_length=100)
    mean_precision: Probability | None

    @model_validator(mode="after")
    def mean_matches_repeats(self) -> SanityRandomPrecision:
        defined = [float(value) for value in self.repeat_precisions if value is not None]
        if defined and len(defined) != self.repeats:
            raise ValueError("random repeat precisions must be either all finite or all N/A")
        if not defined:
            if self.mean_precision is not None:
                raise ValueError("all-N/A random repeats require an N/A mean precision")
            return self
        expected = sum(defined) / len(defined)
        if self.mean_precision is None or not math.isclose(
            float(self.mean_precision), expected, abs_tol=1e-12
        ):
            raise ValueError("random mean precision does not match its 100 repeats")
        return self


class SanityPolicyMetrics(_FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    protocol_version: Literal["selector-r005-sanity-v1"] = SANITY_PROTOCOL_VERSION
    point_index: Annotated[int, Field(ge=1, le=4)]
    quantile: Probability
    threshold: Probability
    policy_family: Literal["safe-score-0-to-cap1-only"] = "safe-score-0-to-cap1-only"
    niah: SanityDatasetMetrics
    twowiki: SanityDatasetMetrics
    niah_harm: SanityHarmMetrics
    count_matched_random: SanityRandomPrecision

    @model_validator(mode="after")
    def datasets_and_point_are_frozen(self) -> SanityPolicyMetrics:
        if self.quantile != SANITY_QUANTILES[self.point_index - 1]:
            raise ValueError("point index and pre-registered quantile disagree")
        if self.niah.dataset_kind != "niah" or self.twowiki.dataset_kind != "2wiki":
            raise ValueError("policy metrics require explicit NIAH and 2Wiki summaries")
        if self.niah.queries != EXPECTED_MODELVAL_QUERY_COUNTS["niah"]:
            raise ValueError("R005 NIAH metrics must contain exactly 103 modelval queries")
        if self.twowiki.queries != EXPECTED_MODELVAL_QUERY_COUNTS["2wiki"]:
            raise ValueError("R005 2Wiki metrics must contain exactly 300 modelval queries")
        if self.niah_harm.n_pool_conditional > self.niah.queries:
            raise ValueError("pool-conditional denominator cannot exceed NIAH query count")
        if self.niah_harm.n_dropped_candidate_actions != self.niah.actual_deletions:
            raise ValueError("NIAH harm and dataset summaries disagree on actual deletions")
        random_is_na = self.count_matched_random.mean_precision is None
        if random_is_na != (self.niah.actual_deletions == 0):
            raise ValueError(
                "count-matched random precision is N/A exactly when NIAH deletes nothing"
            )
        return self


def _mean(values: Sequence[float], *, label: str) -> float:
    if not values:
        raise ValueError(f"{label} has no eligible observations")
    result = sum(values) / len(values)
    if not math.isfinite(result):
        raise ValueError(f"{label} aggregate is non-finite")
    return result


def _dataset_metrics(
    rows: Sequence[SanityPolicyQueryOutcome], *, dataset_kind: DatasetKind
) -> SanityDatasetMetrics:
    recall = [document_recall(row.selected_document_ids, row.required_document_ids) for row in rows]
    recall_loss = [
        relative_recall_loss(
            baseline_document_ids=row.baseline_document_ids,
            selector_document_ids=row.selected_document_ids,
            gold_document_ids=row.required_document_ids,
        )
        for row in rows
    ]
    chain = [
        conditional_chain_loss(
            baseline_document_ids=row.baseline_document_ids,
            selector_document_ids=row.selected_document_ids,
            gold_document_ids=row.required_document_ids,
        )
        for row in rows
    ]
    eligible_chain = [float(value) for value in chain if value is not None]
    return SanityDatasetMetrics(
        dataset_kind=dataset_kind,
        queries=len(rows),
        actual_deletions=sum(len(row.dropped_evidence_ids) for row in rows),
        document_recall=_mean(recall, label=f"{dataset_kind} recall"),
        topk10_relative_recall_loss=_mean(
            recall_loss, label=f"{dataset_kind} relative recall loss"
        ),
        n_topk10_chain_eligible=len(eligible_chain),
        conditional_chain_loss=(
            None
            if not eligible_chain
            else _mean(eligible_chain, label=f"{dataset_kind} conditional chain loss")
        ),
    )


def evaluate_policy_metrics(
    *,
    point: SanityQuantileThreshold,
    outcomes: Sequence[SanityPolicyQueryOutcome],
    random_repeat_precisions: Sequence[float | None],
) -> SanityPolicyMetrics:
    """Compute one frozen 0--cap1 point from ordinary per-query outcome records."""

    if len(random_repeat_precisions) != SANITY_COUNT_MATCHED_REPEATS:
        raise ValueError("R005 requires exactly 100 count-matched random repeats")
    if not outcomes:
        raise ValueError("policy metrics require query outcomes")
    by_dataset: dict[DatasetKind, list[SanityPolicyQueryOutcome]] = {
        "niah": [],
        "2wiki": [],
    }
    seen: set[tuple[str, str]] = set()
    for row in outcomes:
        identity = (row.dataset_kind, row.query_id)
        if identity in seen:
            raise ValueError(f"duplicate policy query outcome: {identity}")
        seen.add(identity)
        by_dataset[row.dataset_kind].append(row)
    for dataset_kind in cast(tuple[DatasetKind, ...], ("niah", "2wiki")):
        if len(by_dataset[dataset_kind]) != EXPECTED_MODELVAL_QUERY_COUNTS[dataset_kind]:
            raise ValueError(
                f"{dataset_kind} policy outcomes have {len(by_dataset[dataset_kind])} queries; "
                f"expected {EXPECTED_MODELVAL_QUERY_COUNTS[dataset_kind]}"
            )

    niah_metrics = _dataset_metrics(by_dataset["niah"], dataset_kind="niah")
    twowiki_metrics = _dataset_metrics(by_dataset["2wiki"], dataset_kind="2wiki")
    eligible = [row for row in by_dataset["niah"] if row.harmful_in_top20_pool]
    if not eligible:
        raise ValueError("NIAH pool-conditional harmful metric has no eligible queries")
    baseline_exposure = [
        float(cast(str, row.harmful_document_id) in set(row.baseline_document_ids))
        for row in eligible
    ]
    selector_exposure = [
        float(cast(str, row.harmful_document_id) in set(row.selected_document_ids))
        for row in eligible
    ]
    all_drops = [
        (row, document_id) for row in by_dataset["niah"] for document_id in row.dropped_document_ids
    ]
    harmful_drop_count = sum(
        document_id == row.harmful_document_id for row, document_id in all_drops
    )
    base_mean = _mean(baseline_exposure, label="baseline pool-conditional exposure")
    selected_mean = _mean(selector_exposure, label="selector pool-conditional exposure")
    harm_metrics = SanityHarmMetrics(
        n_pool_conditional=len(eligible),
        baseline_pool_conditional_exposure=base_mean,
        selector_pool_conditional_exposure=selected_mean,
        pool_conditional_harmful_reduction=base_mean - selected_mean,
        n_dropped_candidate_actions=len(all_drops),
        n_harmful_dropped_candidate_actions=harmful_drop_count,
        deletion_precision=(None if not all_drops else harmful_drop_count / len(all_drops)),
    )
    random_values = tuple(
        None if value is None else _finite_probability(value, label="random repeat precision")
        for value in random_repeat_precisions
    )
    defined_random_values = tuple(value for value in random_values if value is not None)
    if all_drops:
        if len(defined_random_values) != SANITY_COUNT_MATCHED_REPEATS:
            raise ValueError("nonzero NIAH deletion requires 100 finite random precisions")
    elif defined_random_values:
        raise ValueError("zero NIAH deletion requires 100 N/A random precisions")
    random_summary = SanityRandomPrecision(
        repeat_precisions=random_values,
        mean_precision=(
            None
            if not defined_random_values
            else sum(defined_random_values) / len(defined_random_values)
        ),
    )
    return SanityPolicyMetrics(
        point_index=point.point_index,
        quantile=point.quantile,
        threshold=point.threshold,
        niah=niah_metrics,
        twowiki=twowiki_metrics,
        niah_harm=harm_metrics,
        count_matched_random=random_summary,
    )


class SanityPointDecision(_FrozenModel):
    point_index: Annotated[int, Field(ge=1, le=4)]
    quantile: Probability
    threshold: Probability
    actual_nonzero_drop_pass: StrictBool
    harmful_reduction_pass: StrictBool
    niah_recall_loss_pass: StrictBool
    twowiki_recall_loss_pass: StrictBool
    niah_conditional_chain_loss_pass: StrictBool
    twowiki_conditional_chain_loss_pass: StrictBool
    deletion_precision_pass: StrictBool
    passed: StrictBool

    @model_validator(mode="after")
    def point_pass_is_conjunction(self) -> SanityPointDecision:
        expected = all(
            (
                self.actual_nonzero_drop_pass,
                self.harmful_reduction_pass,
                self.niah_recall_loss_pass,
                self.twowiki_recall_loss_pass,
                self.niah_conditional_chain_loss_pass,
                self.twowiki_conditional_chain_loss_pass,
                self.deletion_precision_pass,
            )
        )
        if self.passed != expected:
            raise ValueError("safe-corner point pass must be the conjunction of every gate")
        return self


class SanitySafeCornerDecision(_FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    protocol_version: Literal["selector-r005-sanity-v1"] = SANITY_PROTOCOL_VERSION
    status: Literal["PASS", "CUT"]
    witness_point_index: Annotated[int, Field(ge=1, le=4)] | None
    points: tuple[SanityPointDecision, ...] = Field(min_length=4, max_length=4)
    thresholds_reused_by_r006_or_r007: Literal[False] = False
    checkpoint_reused_by_r006: Literal[False] = False

    @model_validator(mode="after")
    def decision_is_first_passing_point(self) -> SanitySafeCornerDecision:
        expected_indices = tuple(range(1, 5))
        if tuple(point.point_index for point in self.points) != expected_indices:
            raise ValueError(
                "safe-corner points must retain frozen conservative-to-aggressive order"
            )
        passing = [point.point_index for point in self.points if point.passed]
        expected_witness = passing[0] if passing else None
        if self.witness_point_index != expected_witness:
            raise ValueError("witness must be the first passing point in frozen order")
        if (self.status == "PASS") != bool(passing):
            raise ValueError("safe-corner status does not match point decisions")
        return self


def decide_safe_corner(
    metrics: Sequence[SanityPolicyMetrics],
) -> SanitySafeCornerDecision:
    """Apply the R005 point gates and choose the first conservative passing witness."""

    if len(metrics) != len(SANITY_QUANTILES):
        raise ValueError("safe-corner decision requires all four pre-registered points")
    if tuple(item.point_index for item in metrics) != tuple(range(1, 5)):
        raise ValueError("policy metrics must be in frozen point order 1..4")
    thresholds = tuple(float(item.threshold) for item in metrics)
    if any(left < right for left, right in zip(thresholds[:-1], thresholds[1:], strict=True)):
        raise ValueError("thresholds must be non-increasing from conservative to aggressive")
    points: list[SanityPointDecision] = []
    for item in metrics:
        nonzero = item.niah.actual_deletions + item.twowiki.actual_deletions > 0
        harm = float(item.niah_harm.pool_conditional_harmful_reduction) > 0.0
        niah_recall = float(item.niah.topk10_relative_recall_loss) <= SANITY_MAX_RECALL_LOSS
        twowiki_recall = float(item.twowiki.topk10_relative_recall_loss) <= SANITY_MAX_RECALL_LOSS
        niah_chain = (
            item.niah.n_topk10_chain_eligible > 0
            and item.niah.conditional_chain_loss is not None
            and float(item.niah.conditional_chain_loss) <= SANITY_MAX_CHAIN_LOSS
        )
        twowiki_chain = (
            item.twowiki.n_topk10_chain_eligible > 0
            and item.twowiki.conditional_chain_loss is not None
            and float(item.twowiki.conditional_chain_loss) <= SANITY_MAX_CHAIN_LOSS
        )
        precision = (
            item.niah_harm.deletion_precision is not None
            and item.count_matched_random.mean_precision is not None
            and float(item.niah_harm.deletion_precision)
            > float(item.count_matched_random.mean_precision)
        )
        points.append(
            SanityPointDecision(
                point_index=item.point_index,
                quantile=item.quantile,
                threshold=item.threshold,
                actual_nonzero_drop_pass=nonzero,
                harmful_reduction_pass=harm,
                niah_recall_loss_pass=niah_recall,
                twowiki_recall_loss_pass=twowiki_recall,
                niah_conditional_chain_loss_pass=niah_chain,
                twowiki_conditional_chain_loss_pass=twowiki_chain,
                deletion_precision_pass=precision,
                passed=(
                    nonzero
                    and harm
                    and niah_recall
                    and twowiki_recall
                    and niah_chain
                    and twowiki_chain
                    and precision
                ),
            )
        )
    passing = [point.point_index for point in points if point.passed]
    return SanitySafeCornerDecision(
        status="PASS" if passing else "CUT",
        witness_point_index=passing[0] if passing else None,
        points=tuple(points),
    )
