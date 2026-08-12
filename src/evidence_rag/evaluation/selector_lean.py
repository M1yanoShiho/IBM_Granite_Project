"""Small evaluation helpers for the Lean Selector experiment.

The module has three jobs and deliberately no training or artifact state machine:

* derive the four pre-registered float32 thresholds from train-fit scores;
* choose one development policy with the frozen safety gates and tie-breaks;
* build a read-only evaluation view by joining the already frozen dataset, pool and R002 roles.

Final evaluation accepts only a :class:`FrozenPolicy`; it cannot search a grid or alter a threshold.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import struct
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import ValidationError

from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate, Query
from evidence_rag.evaluation.paired_metric import compare_paired
from evidence_rag.evaluation.selector_components import (
    ROLE_ASSIGNMENTS_FILE,
    DatasetKind,
    NiahSelectorAssignment,
    verify_selector_component_artifacts,
)
from evidence_rag.evaluation.selector_controls import generate_count_matched_drops
from evidence_rag.evaluation.selector_risk import (
    conditional_chain_loss,
    relative_recall_loss,
)
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.materializer.injector import alias_occurrences
from evidence_rag.materializer.provenance import MutationRecord, read_provenance
from evidence_rag.materializer.selector_labels import _verify_counterfactual_pair
from evidence_rag.selector.models import CandidateRiskScore
from evidence_rag.selector.risk_controlled import RiskControlledSelector

LEAN_QUANTILES = (0.995, 0.99, 0.975, 0.95)
LEAN_CAPS = (1, 2)
LEAN_SEEDS = (13, 42)
EvaluationRole = Literal["crc-calibration", "decision-dev"]
Variant = Literal["NLI-base", "NLI-pair"]
_DATASET_KINDS: tuple[DatasetKind, DatasetKind] = ("niah", "2wiki")


def _float32(value: float) -> float:
    return cast(float, struct.unpack("!f", struct.pack("!f", value))[0])


def _probability(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1]")
    return numeric


def _nonnegative(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return numeric


def _nonblank(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-blank string")
    return value


def _sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_digest(value: str, *, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def nearest_rank_float32(values: Sequence[float], quantile: float) -> float:
    """Return ascending item ``ceil(q*n)`` after canonical IEEE float32 conversion."""

    q = _probability(quantile, label="quantile")
    if q == 0.0:
        raise ValueError("quantile must be greater than zero")
    if not values:
        raise ValueError("train score pool must not be empty")
    scores = sorted(
        _float32(_probability(value, label=f"train score {index}"))
        for index, value in enumerate(values)
    )
    return scores[math.ceil(q * len(scores)) - 1]


@dataclass(frozen=True, slots=True)
class TrainThreshold:
    seed: int
    quantile: float
    nearest_rank: int
    score_count: int
    threshold: float


def thresholds_from_train_scores(
    scores_by_seed: Mapping[int, Sequence[float]],
    quantiles: Sequence[float] = LEAN_QUANTILES,
) -> tuple[TrainThreshold, ...]:
    """Derive every seed/quantile threshold without labels or development results."""

    if set(scores_by_seed) != set(LEAN_SEEDS):
        raise ValueError(f"train scores must contain exactly seeds {LEAN_SEEDS}")
    normalized_quantiles = tuple(_probability(value, label="quantile") for value in quantiles)
    if not normalized_quantiles or any(value == 0.0 for value in normalized_quantiles):
        raise ValueError("quantiles must be non-empty and greater than zero")
    if len(set(normalized_quantiles)) != len(normalized_quantiles):
        raise ValueError("quantile grid must not contain duplicate identities")

    output: list[TrainThreshold] = []
    for seed in sorted(scores_by_seed):
        values = tuple(scores_by_seed[seed])
        if not values:
            raise ValueError(f"seed {seed} train score pool must not be empty")
        for quantile in normalized_quantiles:
            output.append(
                TrainThreshold(
                    seed=seed,
                    quantile=quantile,
                    nearest_rank=math.ceil(quantile * len(values)),
                    score_count=len(values),
                    threshold=nearest_rank_float32(values, quantile),
                )
            )
    return tuple(output)


@dataclass(frozen=True, slots=True)
class PolicyCandidate:
    policy_id: str
    quantile: float | None
    cap: int | None
    thresholds_by_seed: tuple[tuple[int, float], ...]
    policy_enabled: bool = True


def build_policy_candidates(
    thresholds: Sequence[TrainThreshold],
    caps: Sequence[int] = LEAN_CAPS,
) -> tuple[PolicyCandidate, ...]:
    """Build P0 plus the frozen 4×2 grid while retaining duplicate threshold values."""

    normalized_caps = tuple(caps)
    if not normalized_caps or any(
        type(cap) is not int or cap not in LEAN_CAPS for cap in normalized_caps
    ):
        raise ValueError(f"caps must contain only {LEAN_CAPS}")
    if len(set(normalized_caps)) != len(normalized_caps):
        raise ValueError("caps must not contain duplicates")

    by_quantile: dict[float, dict[int, TrainThreshold]] = {}
    for row in thresholds:
        if not isinstance(row, TrainThreshold):
            raise TypeError("thresholds must contain TrainThreshold values")
        per_seed = by_quantile.setdefault(row.quantile, {})
        if row.seed in per_seed:
            raise ValueError(f"duplicate threshold for seed={row.seed}, q={row.quantile}")
        per_seed[row.seed] = row
    if tuple(by_quantile) != LEAN_QUANTILES:
        raise ValueError(f"threshold quantiles must follow the frozen order {LEAN_QUANTILES}")

    candidates = [
        PolicyCandidate(
            policy_id="P0_KEEP_TOPK10",
            quantile=None,
            cap=None,
            thresholds_by_seed=(),
            policy_enabled=False,
        )
    ]
    for quantile in LEAN_QUANTILES:
        per_seed = by_quantile[quantile]
        if set(per_seed) != set(LEAN_SEEDS):
            raise ValueError(f"quantile {quantile} must contain exactly seeds {LEAN_SEEDS}")
        seed_thresholds = tuple((seed, per_seed[seed].threshold) for seed in sorted(per_seed))
        for cap in normalized_caps:
            candidates.append(
                PolicyCandidate(
                    policy_id=f"Q{quantile:g}_CAP{cap}",
                    quantile=quantile,
                    cap=cap,
                    thresholds_by_seed=seed_thresholds,
                )
            )
    return tuple(candidates)


@dataclass(frozen=True, slots=True)
class DevelopmentCandidateResult:
    """Already aggregated development result for one common quantile/cap choice."""

    quantile: float
    cap: int
    thresholds_by_seed: tuple[tuple[int, float], ...]
    mean_deletions_per_query: float
    harmful_reduction_seed13: float
    harmful_reduction_seed42: float
    niah_recall_loss_seed13_pp: float
    niah_recall_loss_seed42_pp: float
    twowiki_recall_loss_seed13_pp: float
    twowiki_recall_loss_seed42_pp: float
    niah_chain_loss_seed13_pp: float
    niah_chain_loss_seed42_pp: float
    twowiki_chain_loss_seed13_pp: float
    twowiki_chain_loss_seed42_pp: float
    seed13_harm_beats_random: bool
    seed13_harm_beats_bottom: bool
    seed13_precision_beats_random: bool
    seed13_precision_beats_bottom: bool

    def __post_init__(self) -> None:
        _probability(self.quantile, label="quantile")
        if type(self.cap) is not int or self.cap not in LEAN_CAPS:
            raise ValueError(f"cap must be one of {LEAN_CAPS}")
        if tuple(seed for seed, _ in self.thresholds_by_seed) != LEAN_SEEDS:
            raise ValueError(f"thresholds_by_seed must follow seeds {LEAN_SEEDS}")
        for seed, threshold in self.thresholds_by_seed:
            _probability(threshold, label=f"seed {seed} threshold")
        _nonnegative(self.mean_deletions_per_query, label="mean_deletions_per_query")
        for field in (
            "harmful_reduction_seed13",
            "harmful_reduction_seed42",
            "niah_recall_loss_seed13_pp",
            "niah_recall_loss_seed42_pp",
            "twowiki_recall_loss_seed13_pp",
            "twowiki_recall_loss_seed42_pp",
            "niah_chain_loss_seed13_pp",
            "niah_chain_loss_seed42_pp",
            "twowiki_chain_loss_seed13_pp",
            "twowiki_chain_loss_seed42_pp",
        ):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{field} must be finite")
            if field.startswith("harmful_reduction") and not -1.0 <= float(value) <= 1.0:
                raise ValueError(f"{field} must be in [-1, 1]")
            if field.endswith("_loss_seed13_pp") or field.endswith("_loss_seed42_pp"):
                if not 0.0 <= float(value) <= 100.0:
                    raise ValueError(f"{field} must be in [0, 100] percentage points")
        for field in (
            "seed13_harm_beats_random",
            "seed13_harm_beats_bottom",
            "seed13_precision_beats_random",
            "seed13_precision_beats_bottom",
        ):
            if type(getattr(self, field)) is not bool:
                raise TypeError(f"{field} must be bool")

    @property
    def mean_harmful_reduction(self) -> float:
        return 0.5 * (self.harmful_reduction_seed13 + self.harmful_reduction_seed42)

    @property
    def passes_gate(self) -> bool:
        protection_pairs = (
            (self.niah_recall_loss_seed13_pp, self.niah_recall_loss_seed42_pp),
            (self.twowiki_recall_loss_seed13_pp, self.twowiki_recall_loss_seed42_pp),
            (self.niah_chain_loss_seed13_pp, self.niah_chain_loss_seed42_pp),
            (self.twowiki_chain_loss_seed13_pp, self.twowiki_chain_loss_seed42_pp),
        )
        return (
            self.mean_deletions_per_query > 0.0
            and self.harmful_reduction_seed13 > 0.0
            and self.harmful_reduction_seed42 > 0.0
            and all(seed13 <= 1.0 and seed42 <= 3.0 for seed13, seed42 in protection_pairs)
            and self.seed13_harm_beats_random
            and self.seed13_harm_beats_bottom
            and self.seed13_precision_beats_random
            and self.seed13_precision_beats_bottom
        )


@dataclass(frozen=True, slots=True)
class DevelopmentSelection:
    selected: DevelopmentCandidateResult
    global_max_mean_harmful_reduction: float
    tie_epsilon_pp: float


def select_development_policy(
    results: Sequence[DevelopmentCandidateResult],
    *,
    policy_candidates: Sequence[PolicyCandidate],
    tie_epsilon_pp: float = 0.5,
) -> DevelopmentSelection | None:
    """Apply the pre-registered gate, equivalence band and conservative tie-break."""

    epsilon = _nonnegative(tie_epsilon_pp, label="tie_epsilon_pp") / 100.0
    if not results:
        raise ValueError("development results must not be empty")
    active_candidates = tuple(
        candidate for candidate in policy_candidates if candidate.policy_enabled
    )
    expected = {
        (candidate.quantile, candidate.cap): candidate.thresholds_by_seed
        for candidate in active_candidates
    }
    if len(active_candidates) != len(LEAN_QUANTILES) * len(LEAN_CAPS) or set(expected) != {
        (quantile, cap) for quantile in LEAN_QUANTILES for cap in LEAN_CAPS
    }:
        raise ValueError("policy_candidates must contain the complete frozen 4x2 grid plus P0")
    disabled = tuple(candidate for candidate in policy_candidates if not candidate.policy_enabled)
    if len(policy_candidates) != len(active_candidates) + 1 or disabled != (
        PolicyCandidate(
            policy_id="P0_KEEP_TOPK10",
            quantile=None,
            cap=None,
            thresholds_by_seed=(),
            policy_enabled=False,
        ),
    ):
        raise ValueError("policy_candidates must contain exactly the structural P0")
    expected_ids = {
        (quantile, cap): f"Q{quantile:g}_CAP{cap}"
        for quantile in LEAN_QUANTILES
        for cap in LEAN_CAPS
    }
    if any(
        candidate.quantile is None
        or candidate.cap is None
        or candidate.policy_id != expected_ids[(candidate.quantile, candidate.cap)]
        for candidate in active_candidates
    ):
        raise ValueError("active policy IDs differ from the frozen grid")

    identities: set[tuple[float, int]] = set()
    for result in results:
        if not isinstance(result, DevelopmentCandidateResult):
            raise TypeError("results must contain DevelopmentCandidateResult values")
        identity = (result.quantile, result.cap)
        if identity in identities:
            raise ValueError(f"duplicate development result for {identity}")
        if identity not in expected:
            raise ValueError(f"development result is outside the frozen grid: {identity}")
        if result.thresholds_by_seed != expected[identity]:
            raise ValueError(f"development thresholds differ from train-derived grid: {identity}")
        identities.add(identity)
    if identities != set(expected):
        raise ValueError("development results must cover the complete frozen 4x2 grid")
    eligible = tuple(result for result in results if result.passes_gate)
    if not eligible:
        return None
    maximum = max(result.mean_harmful_reduction for result in eligible)
    equivalent = tuple(
        result for result in eligible if maximum - result.mean_harmful_reduction <= epsilon
    )
    selected = min(
        equivalent,
        key=lambda result: (
            -result.quantile,
            result.cap,
            result.mean_deletions_per_query,
        ),
    )
    return DevelopmentSelection(
        selected=selected,
        global_max_mean_harmful_reduction=maximum,
        tie_epsilon_pp=float(tie_epsilon_pp),
    )


@dataclass(frozen=True, slots=True)
class FrozenPolicy:
    schema_version: str
    variant: Variant
    quantile: float
    cap: int
    thresholds_by_seed: tuple[tuple[int, float], ...]
    checkpoint_sha256_by_seed: tuple[tuple[int, str], ...]
    code_commit: str
    final_input_sha256: str
    development_projection_sha256: str
    generator_sha256: str

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["thresholds_by_seed"] = dict(self.thresholds_by_seed)
        value["checkpoint_sha256_by_seed"] = dict(self.checkpoint_sha256_by_seed)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FrozenPolicy:
        thresholds = value.get("thresholds_by_seed")
        checkpoints = value.get("checkpoint_sha256_by_seed")
        if not isinstance(thresholds, Mapping) or not isinstance(checkpoints, Mapping):
            raise ValueError("frozen policy seed maps are missing")
        try:
            variant = value["variant"]
            cap = value["cap"]
            if not isinstance(variant, str):
                raise TypeError("variant must be a string")
            if type(cap) is not int:
                raise TypeError("cap must be an integer")
            threshold_rows = tuple(
                (int(seed), float(threshold)) for seed, threshold in thresholds.items()
            )
            checkpoint_rows = tuple(
                (int(seed), str(digest)) for seed, digest in checkpoints.items()
            )
            return cls(
                schema_version=str(value["schema_version"]),
                variant=cast(Variant, variant),
                quantile=_probability(value["quantile"], label="quantile"),
                cap=cap,
                thresholds_by_seed=tuple(sorted(threshold_rows)),
                checkpoint_sha256_by_seed=tuple(sorted(checkpoint_rows)),
                code_commit=str(value["code_commit"]),
                final_input_sha256=str(value["final_input_sha256"]),
                development_projection_sha256=str(value["development_projection_sha256"]),
                generator_sha256=str(value["generator_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid frozen policy: {error}") from error

    def __post_init__(self) -> None:
        if self.schema_version != "selector-lean-frozen-policy-v1":
            raise ValueError("unsupported frozen policy schema")
        if self.variant not in ("NLI-base", "NLI-pair"):
            raise ValueError("variant must be NLI-base or NLI-pair")
        _probability(self.quantile, label="quantile")
        if self.quantile not in LEAN_QUANTILES or self.cap not in LEAN_CAPS:
            raise ValueError("frozen policy is outside the pre-registered grid")
        if tuple(seed for seed, _ in self.thresholds_by_seed) != LEAN_SEEDS:
            raise ValueError(f"thresholds must follow seeds {LEAN_SEEDS}")
        if tuple(seed for seed, _ in self.checkpoint_sha256_by_seed) != LEAN_SEEDS:
            raise ValueError(f"checkpoint hashes must follow seeds {LEAN_SEEDS}")
        for seed, threshold in self.thresholds_by_seed:
            _probability(threshold, label=f"seed {seed} threshold")
        for seed, digest in self.checkpoint_sha256_by_seed:
            _require_digest(digest, label=f"seed {seed} checkpoint hash")
        if len(self.code_commit) not in (40, 64) or any(
            character not in "0123456789abcdef" for character in self.code_commit
        ):
            raise ValueError("code_commit must be a lowercase Git object ID")
        _require_digest(self.final_input_sha256, label="final_input_sha256")
        _require_digest(
            self.development_projection_sha256,
            label="development_projection_sha256",
        )
        _require_digest(self.generator_sha256, label="generator_sha256")


def freeze_policy(
    selection: DevelopmentSelection,
    *,
    variant: Variant,
    checkpoint_sha256_by_seed: Mapping[int, str],
    code_commit: str,
    final_input_sha256: str,
    development_projection_sha256: str,
    generator_sha256: str,
) -> FrozenPolicy:
    if not isinstance(selection, DevelopmentSelection):
        raise TypeError("selection must be a DevelopmentSelection")
    result = selection.selected
    return FrozenPolicy(
        schema_version="selector-lean-frozen-policy-v1",
        variant=variant,
        quantile=result.quantile,
        cap=result.cap,
        thresholds_by_seed=result.thresholds_by_seed,
        checkpoint_sha256_by_seed=tuple(sorted(checkpoint_sha256_by_seed.items())),
        code_commit=code_commit,
        final_input_sha256=final_input_sha256,
        development_projection_sha256=development_projection_sha256,
        generator_sha256=generator_sha256,
    )


def require_frozen_final_policy(
    policy: FrozenPolicy,
    *,
    checkpoint_sha256_by_seed: Mapping[int, str],
    code_commit: str,
    final_input_sha256: str,
    development_projection_sha256: str,
    generator_sha256: str,
    requested_quantile: float | None = None,
    requested_cap: int | None = None,
) -> FrozenPolicy:
    """Guard the final path against threshold/cap selection or silent overrides."""

    if not isinstance(policy, FrozenPolicy):
        raise TypeError("final evaluation requires a FrozenPolicy")
    if requested_quantile is not None:
        quantile = _probability(requested_quantile, label="requested_quantile")
        if quantile != policy.quantile:
            raise ValueError("final evaluation cannot change the frozen quantile")
    if requested_cap is not None:
        if type(requested_cap) is not int:
            raise TypeError("requested_cap must be an integer")
        if requested_cap != policy.cap:
            raise ValueError("final evaluation cannot change the frozen cap")
    observed = {
        "checkpoint_sha256_by_seed": tuple(sorted(checkpoint_sha256_by_seed.items())),
        "code_commit": code_commit,
        "final_input_sha256": final_input_sha256,
        "development_projection_sha256": development_projection_sha256,
        "generator_sha256": generator_sha256,
    }
    frozen = {
        "checkpoint_sha256_by_seed": policy.checkpoint_sha256_by_seed,
        "code_commit": policy.code_commit,
        "final_input_sha256": policy.final_input_sha256,
        "development_projection_sha256": policy.development_projection_sha256,
        "generator_sha256": policy.generator_sha256,
    }
    mismatches = sorted(name for name in frozen if observed[name] != frozen[name])
    if mismatches:
        raise ValueError(f"final inputs differ from frozen policy: {mismatches}")
    return policy


@dataclass(frozen=True, slots=True)
class EvaluationQuery:
    dataset_kind: DatasetKind
    dataset_id: str
    dataset_signature: str
    pool_sha256: str
    role: EvaluationRole
    query_id: str
    question: str
    component_id: str
    chain_eligible_topk10: bool
    topk10: tuple[EvidenceCandidate, ...]
    required_document_ids: tuple[str, ...]
    reference_answers: tuple[str, ...]
    harmful_document_id: str | None
    harmful_in_top20_pool: bool

    def identity_row(self) -> dict[str, object]:
        return {
            "dataset_kind": self.dataset_kind,
            "dataset_id": self.dataset_id,
            "dataset_signature": self.dataset_signature,
            "pool_sha256": self.pool_sha256,
            "role": self.role,
            "query_id": self.query_id,
            "question": self.question,
            "component_id": self.component_id,
            "chain_eligible_topk10": self.chain_eligible_topk10,
            "topk10": [candidate.model_dump(mode="json") for candidate in self.topk10],
            "required_document_ids": list(self.required_document_ids),
            "reference_answers": list(self.reference_answers),
            "harmful_document_id": self.harmful_document_id,
            "harmful_in_top20_pool": self.harmful_in_top20_pool,
        }


@dataclass(frozen=True, slots=True)
class EvaluationProjection:
    dataset_kind: DatasetKind
    role: EvaluationRole
    rows: tuple[EvaluationQuery, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class SeedPolicyMetrics:
    """Development point metrics for one seed and one frozen threshold/cap."""

    seed: int
    threshold: float
    cap: int
    mean_deletions_per_query: float
    niah_harmful_reduction: float
    niah_deletion_precision: float
    niah_recall_loss_pp: float
    twowiki_recall_loss_pp: float
    niah_chain_loss_pp: float
    twowiki_chain_loss_pp: float
    random_harmful_reduction: float | None = None
    bottom_harmful_reduction: float | None = None
    random_deletion_precision: float | None = None
    bottom_deletion_precision: float | None = None


@dataclass(frozen=True, slots=True)
class PolicyQueryResult:
    dataset_kind: DatasetKind
    query_id: str
    component_id: str
    selected_evidence_ids: tuple[str, ...]
    dropped_evidence_ids: tuple[str, ...]
    recall_loss: float
    chain_loss: float | None
    harmful_reduction: float | None


@dataclass(frozen=True, slots=True)
class _NiahControlAccumulator:
    harmful_sum: float = 0.0
    harmful_count: int = 0
    harmful_drops: int = 0
    total_drops: int = 0

    def add(
        self,
        *,
        row: EvaluationQuery,
        dropped_evidence_ids: Sequence[str],
    ) -> _NiahControlAccumulator:
        candidate_by_id = {candidate.evidence_id: candidate for candidate in row.topk10}
        dropped_documents = tuple(
            candidate_by_id[evidence_id].document_id for evidence_id in dropped_evidence_ids
        )
        harmful_id = row.harmful_document_id
        harmful_sum = self.harmful_sum
        harmful_count = self.harmful_count
        if row.harmful_in_top20_pool:
            if harmful_id is None:
                raise ValueError("NIAH pool-conditional query is missing harmful_document_id")
            baseline_documents = {candidate.document_id for candidate in row.topk10}
            selected_documents = baseline_documents - set(dropped_documents)
            harmful_sum += float(harmful_id in baseline_documents) - float(
                harmful_id in selected_documents
            )
            harmful_count += 1
        return _NiahControlAccumulator(
            harmful_sum=harmful_sum,
            harmful_count=harmful_count,
            harmful_drops=self.harmful_drops
            + sum(document_id == harmful_id for document_id in dropped_documents),
            total_drops=self.total_drops + len(dropped_documents),
        )

    @property
    def harmful_reduction(self) -> float:
        if self.harmful_count == 0:
            raise ValueError("NIAH projection has no pool-conditional harmful queries")
        return self.harmful_sum / self.harmful_count

    @property
    def deletion_precision(self) -> float:
        return self.harmful_drops / self.total_drops if self.total_drops else 0.0


def _mean(values: Sequence[float], *, label: str) -> float:
    if not values:
        raise ValueError(f"{label} has an empty denominator")
    return sum(values) / len(values)


def apply_seed_policy(
    *,
    projections: Mapping[DatasetKind, EvaluationProjection],
    scores_by_dataset: Mapping[
        DatasetKind, Mapping[str, Mapping[str, CandidateRiskScore]]
    ],
    threshold: float,
    cap: int,
) -> tuple[PolicyQueryResult, ...]:
    """Return one delete-only decision and its paired evidence losses per query."""

    safe_threshold = _probability(threshold, label="threshold")
    if type(cap) is not int or cap not in LEAN_CAPS:
        raise ValueError(f"cap must be one of {LEAN_CAPS}")
    if set(projections) != set(_DATASET_KINDS) or set(scores_by_dataset) != set(
        _DATASET_KINDS
    ):
        raise ValueError("policy evaluation requires NIAH and 2Wiki projections/scores")
    output: list[PolicyQueryResult] = []
    for dataset_kind in _DATASET_KINDS:
        projection = projections[dataset_kind]
        score_table = scores_by_dataset[dataset_kind]
        if {row.query_id for row in projection.rows} != set(score_table):
            raise ValueError(f"{dataset_kind} score queries differ from the projection")
        selector = RiskControlledSelector(
            scores_by_query=score_table,
            safe_threshold=safe_threshold,
            max_delete=cap,
        )
        for row in projection.rows:
            candidate_set = CandidateSet(query_id=row.query_id, candidates=row.topk10)
            result, trace = selector.select_with_trace(
                Query(query_id=row.query_id, text=row.question),
                candidate_set,
                10,
            )
            candidate_by_id = {candidate.evidence_id: candidate for candidate in row.topk10}
            baseline_documents = tuple(candidate.document_id for candidate in row.topk10)
            selected_documents = tuple(
                candidate_by_id[item.evidence_id].document_id for item in result.items
            )
            recall = relative_recall_loss(
                baseline_document_ids=baseline_documents,
                selector_document_ids=selected_documents,
                gold_document_ids=row.required_document_ids,
            )
            chain = conditional_chain_loss(
                baseline_document_ids=baseline_documents,
                selector_document_ids=selected_documents,
                gold_document_ids=row.required_document_ids,
            )
            if (chain is not None) != row.chain_eligible_topk10:
                raise ValueError(f"chain eligibility differs from R002 for {row.query_id}")
            harmful: float | None = None
            if row.harmful_in_top20_pool:
                if row.harmful_document_id is None:
                    raise ValueError("pool-conditional NIAH row has no harmful document")
                harmful = float(row.harmful_document_id in baseline_documents) - float(
                    row.harmful_document_id in selected_documents
                )
            output.append(
                PolicyQueryResult(
                    dataset_kind=dataset_kind,
                    query_id=row.query_id,
                    component_id=row.component_id,
                    selected_evidence_ids=tuple(item.evidence_id for item in result.items),
                    dropped_evidence_ids=trace.dropped_evidence_ids,
                    recall_loss=recall,
                    chain_loss=chain,
                    harmful_reduction=harmful,
                )
            )
    return tuple(output)


def evaluate_seed_policy(
    *,
    seed: int,
    projections: Mapping[DatasetKind, EvaluationProjection],
    scores_by_dataset: Mapping[
        DatasetKind, Mapping[str, Mapping[str, CandidateRiskScore]]
    ],
    threshold: float,
    cap: int,
    include_controls: bool,
    random_repeats: int = 100,
) -> SeedPolicyMetrics:
    """Apply the real delete-only policy and aggregate the pre-registered dev metrics."""

    if seed not in LEAN_SEEDS:
        raise ValueError(f"seed must be one of {LEAN_SEEDS}")
    safe_threshold = _probability(threshold, label="threshold")
    if type(cap) is not int or cap not in LEAN_CAPS:
        raise ValueError(f"cap must be one of {LEAN_CAPS}")
    if type(include_controls) is not bool:
        raise TypeError("include_controls must be bool")
    if include_controls and (type(random_repeats) is not int or random_repeats <= 0):
        raise ValueError("random_repeats must be a positive integer")

    deletion_counts: list[float] = []
    recall_losses: dict[DatasetKind, list[float]] = {"niah": [], "2wiki": []}
    chain_losses: dict[DatasetKind, list[float]] = {"niah": [], "2wiki": []}
    selector_control = _NiahControlAccumulator()
    niah_dropped_by_query: dict[str, tuple[str, ...]] = {}

    query_by_key = {
        (kind, row.query_id): row for kind, projection in projections.items() for row in projection.rows
    }
    decisions = apply_seed_policy(
        projections=projections,
        scores_by_dataset=scores_by_dataset,
        threshold=safe_threshold,
        cap=cap,
    )
    for decision in decisions:
        dataset_kind = decision.dataset_kind
        recall_losses[dataset_kind].append(decision.recall_loss)
        if decision.chain_loss is not None:
            chain_losses[dataset_kind].append(decision.chain_loss)
        deletion_counts.append(float(len(decision.dropped_evidence_ids)))
        if dataset_kind == "niah":
            row = query_by_key[(dataset_kind, decision.query_id)]
            niah_dropped_by_query[row.query_id] = decision.dropped_evidence_ids
            selector_control = selector_control.add(
                row=row,
                dropped_evidence_ids=decision.dropped_evidence_ids,
            )

    random_harm: float | None = None
    random_precision: float | None = None
    bottom_harm: float | None = None
    bottom_precision: float | None = None
    if include_controls:
        random_accumulators = [_NiahControlAccumulator() for _ in range(random_repeats)]
        bottom_accumulator = _NiahControlAccumulator()
        for row in projections["niah"].rows:
            deletion_count = len(niah_dropped_by_query[row.query_id])
            for repeat_index in range(random_repeats):
                drops = generate_count_matched_drops(
                    row.topk10,
                    deletion_count=deletion_count,
                    repeat_index=repeat_index,
                    dataset_id=row.dataset_id,
                    dataset_signature=row.dataset_signature,
                    pool_sha256=row.pool_sha256,
                    query_id=row.query_id,
                )
                random_accumulators[repeat_index] = random_accumulators[repeat_index].add(
                    row=row,
                    dropped_evidence_ids=drops.random_dropped_evidence_ids,
                )
                if repeat_index == 0:
                    bottom_accumulator = bottom_accumulator.add(
                        row=row,
                        dropped_evidence_ids=drops.bottom_rank_dropped_evidence_ids,
                    )
        random_harm = _mean(
            [accumulator.harmful_reduction for accumulator in random_accumulators],
            label="random harmful reduction",
        )
        random_precision = _mean(
            [accumulator.deletion_precision for accumulator in random_accumulators],
            label="random deletion precision",
        )
        bottom_harm = bottom_accumulator.harmful_reduction
        bottom_precision = bottom_accumulator.deletion_precision

    return SeedPolicyMetrics(
        seed=seed,
        threshold=safe_threshold,
        cap=cap,
        mean_deletions_per_query=_mean(deletion_counts, label="mean deletions"),
        niah_harmful_reduction=selector_control.harmful_reduction,
        niah_deletion_precision=selector_control.deletion_precision,
        niah_recall_loss_pp=100.0 * _mean(recall_losses["niah"], label="NIAH recall"),
        twowiki_recall_loss_pp=100.0
        * _mean(recall_losses["2wiki"], label="2Wiki recall"),
        niah_chain_loss_pp=100.0 * _mean(chain_losses["niah"], label="NIAH chain"),
        twowiki_chain_loss_pp=100.0
        * _mean(chain_losses["2wiki"], label="2Wiki chain"),
        random_harmful_reduction=random_harm,
        bottom_harmful_reduction=bottom_harm,
        random_deletion_precision=random_precision,
        bottom_deletion_precision=bottom_precision,
    )


def combine_development_metrics(
    *,
    quantile: float,
    cap: int,
    thresholds_by_seed: tuple[tuple[int, float], ...],
    seed13: SeedPolicyMetrics,
    seed42: SeedPolicyMetrics,
) -> DevelopmentCandidateResult:
    """Convert two separately evaluated seeds into one frozen-grid candidate row."""

    threshold_map = dict(thresholds_by_seed)
    if seed13.seed != 13 or seed42.seed != 42:
        raise ValueError("development metrics must contain seed13 and seed42")
    if threshold_map != {13: seed13.threshold, 42: seed42.threshold}:
        raise ValueError("development metric thresholds differ from the grid")
    controls = (
        seed13.random_harmful_reduction,
        seed13.bottom_harmful_reduction,
        seed13.random_deletion_precision,
        seed13.bottom_deletion_precision,
    )
    if any(value is None for value in controls):
        raise ValueError("seed13 development metrics must include count-matched controls")
    random_harm, bottom_harm, random_precision, bottom_precision = cast(
        tuple[float, float, float, float], controls
    )
    return DevelopmentCandidateResult(
        quantile=quantile,
        cap=cap,
        thresholds_by_seed=thresholds_by_seed,
        mean_deletions_per_query=0.5
        * (seed13.mean_deletions_per_query + seed42.mean_deletions_per_query),
        harmful_reduction_seed13=seed13.niah_harmful_reduction,
        harmful_reduction_seed42=seed42.niah_harmful_reduction,
        niah_recall_loss_seed13_pp=seed13.niah_recall_loss_pp,
        niah_recall_loss_seed42_pp=seed42.niah_recall_loss_pp,
        twowiki_recall_loss_seed13_pp=seed13.twowiki_recall_loss_pp,
        twowiki_recall_loss_seed42_pp=seed42.twowiki_recall_loss_pp,
        niah_chain_loss_seed13_pp=seed13.niah_chain_loss_pp,
        niah_chain_loss_seed42_pp=seed42.niah_chain_loss_pp,
        twowiki_chain_loss_seed13_pp=seed13.twowiki_chain_loss_pp,
        twowiki_chain_loss_seed42_pp=seed42.twowiki_chain_loss_pp,
        seed13_harm_beats_random=seed13.niah_harmful_reduction > random_harm,
        seed13_harm_beats_bottom=seed13.niah_harmful_reduction > bottom_harm,
        seed13_precision_beats_random=seed13.niah_deletion_precision > random_precision,
        seed13_precision_beats_bottom=seed13.niah_deletion_precision > bottom_precision,
    )


def _harm_reduction_for_drops(
    row: EvaluationQuery,
    dropped_evidence_ids: Sequence[str],
) -> float:
    candidate_by_id = {candidate.evidence_id: candidate for candidate in row.topk10}
    baseline_documents = {candidate.document_id for candidate in row.topk10}
    dropped_documents = {
        candidate_by_id[evidence_id].document_id for evidence_id in dropped_evidence_ids
    }
    selected_documents = baseline_documents - dropped_documents
    return float(row.harmful_document_id in baseline_documents) - float(
        row.harmful_document_id in selected_documents
    )


def evidence_inference(
    *,
    decisions: Sequence[PolicyQueryResult],
    projections: Mapping[DatasetKind, EvaluationProjection],
    include_controls: bool,
    random_repeats: int = 100,
) -> Mapping[str, object]:
    """Paired component-bootstrap evidence report for one already-fixed policy."""

    decision_by_key = {(row.dataset_kind, row.query_id): row for row in decisions}
    expected = {
        (kind, row.query_id) for kind, projection in projections.items() for row in projection.rows
    }
    if set(decision_by_key) != expected or len(decision_by_key) != len(decisions):
        raise ValueError("decision rows do not align one-to-one with the projections")

    report: dict[str, object] = {}
    for kind in _DATASET_KINDS:
        projection = projections[kind]
        component_ids = {row.query_id: row.component_id for row in projection.rows}
        zeros = {row.query_id: 0.0 for row in projection.rows}
        recall = {
            row.query_id: decision_by_key[(kind, row.query_id)].recall_loss
            for row in projection.rows
        }
        chain = {
            row.query_id: decision_by_key[(kind, row.query_id)].chain_loss
            for row in projection.rows
        }
        chain_zero = {
            query_id: None if value is None else 0.0 for query_id, value in chain.items()
        }
        report[kind] = {
            "recall_loss": asdict(
                compare_paired(
                    recall,
                    zeros,
                    component_ids=component_ids,
                    seed=13,
                    iterations=10000,
                )
            ),
            "chain_loss": asdict(
                compare_paired(
                    chain,
                    chain_zero,
                    component_ids=component_ids,
                    seed=13,
                    iterations=10000,
                )
            ),
        }

    niah = projections["niah"]
    eligible = tuple(row for row in niah.rows if row.harmful_in_top20_pool)
    components = {row.query_id: row.component_id for row in eligible}
    selector_harm = {
        row.query_id: cast(float, decision_by_key[("niah", row.query_id)].harmful_reduction)
        for row in eligible
    }
    zeros = {row.query_id: 0.0 for row in eligible}
    harm_report: dict[str, object] = {
        "selector_vs_topk10": asdict(
            compare_paired(
                selector_harm,
                zeros,
                component_ids=components,
                seed=13,
                iterations=10000,
            )
        )
    }
    if include_controls:
        random_by_query: dict[str, float] = {}
        bottom_by_query: dict[str, float] = {}
        for row in eligible:
            decision = decision_by_key[("niah", row.query_id)]
            random_values: list[float] = []
            bottom_value: float | None = None
            for repeat_index in range(random_repeats):
                drops = generate_count_matched_drops(
                    row.topk10,
                    deletion_count=len(decision.dropped_evidence_ids),
                    repeat_index=repeat_index,
                    dataset_id=row.dataset_id,
                    dataset_signature=row.dataset_signature,
                    pool_sha256=row.pool_sha256,
                    query_id=row.query_id,
                )

                random_values.append(
                    _harm_reduction_for_drops(row, drops.random_dropped_evidence_ids)
                )
                if repeat_index == 0:
                    bottom_value = _harm_reduction_for_drops(
                        row, drops.bottom_rank_dropped_evidence_ids
                    )
            random_by_query[row.query_id] = _mean(
                random_values, label="random per-query harmful reduction"
            )
            assert bottom_value is not None
            bottom_by_query[row.query_id] = bottom_value
        harm_report["selector_vs_random_mean"] = asdict(
            compare_paired(
                selector_harm,
                random_by_query,
                component_ids=components,
                seed=13,
                iterations=10000,
            )
        )
        harm_report["selector_vs_bottom_rank"] = asdict(
            compare_paired(
                selector_harm,
                bottom_by_query,
                component_ids=components,
                seed=13,
                iterations=10000,
            )
        )
    report["niah_harmful_reduction"] = harm_report
    return report


def macro_answer_inference(
    *,
    selector_by_dataset: Mapping[DatasetKind, Mapping[str, float]],
    topk10_by_dataset: Mapping[DatasetKind, Mapping[str, float]],
    components_by_dataset: Mapping[DatasetKind, Mapping[str, str]],
    iterations: int = 10000,
    seed: int = 13,
) -> Mapping[str, object]:
    """Equal-dataset answer delta with independent component resampling per dataset."""

    if type(iterations) is not int or iterations <= 0:
        raise ValueError("iterations must be positive")
    cluster_values: dict[DatasetKind, dict[str, tuple[float, ...]]] = {}
    dataset_points: dict[DatasetKind, float] = {}
    for kind in _DATASET_KINDS:
        selector = selector_by_dataset[kind]
        baseline = topk10_by_dataset[kind]
        components = components_by_dataset[kind]
        if set(selector) != set(baseline) or set(selector) != set(components) or not selector:
            raise ValueError(f"{kind} answer rows/components are not aligned")
        grouped: dict[str, list[float]] = {}
        diffs: list[float] = []
        for query_id in sorted(selector):
            selector_value = _probability(selector[query_id], label="selector answer")
            baseline_value = _probability(baseline[query_id], label="TopK10 answer")
            component_id = _nonblank(components[query_id], label="answer component")
            diff = selector_value - baseline_value
            diffs.append(diff)
            grouped.setdefault(component_id, []).append(diff)
        cluster_values[kind] = {
            component_id: tuple(values) for component_id, values in grouped.items()
        }
        dataset_points[kind] = _mean(diffs, label=f"{kind} answer delta")

    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(iterations):
        dataset_draws: list[float] = []
        for kind in _DATASET_KINDS:
            sampled_groups = cluster_values[kind]
            clusters = tuple(sorted(sampled_groups))
            sampled = tuple(clusters[rng.randrange(len(clusters))] for _ in clusters)
            values = [
                value for component_id in sampled for value in sampled_groups[component_id]
            ]
            dataset_draws.append(_mean(values, label=f"{kind} bootstrap answer"))
        boot.append(0.5 * sum(dataset_draws))
    boot.sort()
    return {
        "dataset_delta": dict(dataset_points),
        "macro_delta": 0.5 * sum(dataset_points.values()),
        "ci_low": boot[int(0.025 * iterations)],
        "ci_high": boot[min(iterations - 1, int(0.975 * iterations))],
        "iterations": iterations,
        "seed": seed,
        "dataset_weight": "equal-0.5-0.5",
    }


def combine_projection_sha256(
    projections: Mapping[DatasetKind, EvaluationProjection],
) -> str:
    """Bind the exact NIAH and 2Wiki projection identities without reading effects."""

    if set(projections) != {"niah", "2wiki"}:
        raise ValueError("combined projection requires exactly NIAH and 2Wiki")
    if any(projection.dataset_kind != kind for kind, projection in projections.items()):
        raise ValueError("combined projection dataset keys and values disagree")
    roles = {projection.role for projection in projections.values()}
    if len(roles) != 1:
        raise ValueError("combined projections must use one common role")
    return _sha256(
        {
            "schema_version": "selector-lean-combined-projection-v1",
            "role": next(iter(roles)),
            "dataset_sha256": {kind: projections[kind].sha256 for kind in _DATASET_KINDS},
        }
    )


def _read_role_rows(path: Path, role: EvaluationRole) -> dict[str, Mapping[str, object]]:
    output: dict[str, Mapping[str, object]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"unable to read R002 role assignments {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid role assignment at {path}:{line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"role assignment at {path}:{line_number} must be an object")
        if row.get("role") != role:
            continue
        query_id = _nonblank(row.get("query_id"), label="role query_id")
        if query_id in output:
            raise ValueError(f"duplicate role assignment query: {query_id}")
        _nonblank(row.get("component_id"), label=f"component_id for {query_id}")
        if type(row.get("chain_eligible_topk10")) is not bool:
            raise ValueError(f"chain_eligible_topk10 for {query_id} must be bool")
        output[query_id] = row
    if not output:
        raise ValueError(f"R002 role {role!r} has no queries")
    return output


def _read_candidate_sets(path: Path, wanted: set[str]) -> dict[str, CandidateSet]:
    source = path / "candidate_sets.jsonl" if path.is_dir() else path
    output: dict[str, CandidateSet] = {}
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"unable to read candidate pool {source}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            candidate_set = CandidateSet.model_validate_json(line)
        except (ValidationError, ValueError) as error:
            raise ValueError(f"invalid candidate set at {source}:{line_number}: {error}") from error
        if candidate_set.query_id not in wanted:
            continue
        if candidate_set.query_id in output:
            raise ValueError(f"duplicate candidate query: {candidate_set.query_id}")
        ranks = {candidate.retrieval_rank for candidate in candidate_set.candidates}
        if len(candidate_set.candidates) != 20 or ranks != set(range(1, 21)):
            raise ValueError(f"candidate query {candidate_set.query_id} is not exact TopK20")
        output[candidate_set.query_id] = candidate_set
    if set(output) != wanted:
        raise ValueError(f"candidate pool omits role queries: {sorted(wanted - set(output))[:5]}")
    return output


def _read_niah_assignments(path: Path) -> dict[str, NiahSelectorAssignment]:
    output: dict[str, NiahSelectorAssignment] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = NiahSelectorAssignment.model_validate_json(line)
        except (ValidationError, ValueError) as error:
            raise ValueError(f"invalid NIAH assignment at {path}:{line_number}: {error}") from error
        if row.query_id in output:
            raise ValueError(f"duplicate NIAH assignment query: {row.query_id}")
        output[row.query_id] = row
    return output


def build_evaluation_projection(
    *,
    dataset_kind: DatasetKind,
    role: EvaluationRole,
    dataset_manifest_path: Path,
    source_parent_path: Path,
    candidate_pool_path: Path,
    component_directory: Path,
    assignment_path: Path | None = None,
    provenance_path: Path | None = None,
) -> EvaluationProjection:
    """Re-verify R002 inputs, then create the role-filtered read-only metric projection."""

    if dataset_kind not in ("niah", "2wiki"):
        raise ValueError("dataset_kind must be niah or 2wiki")
    if role not in ("crc-calibration", "decision-dev"):
        raise ValueError("role must be crc-calibration or decision-dev")
    if dataset_kind == "niah" and (assignment_path is None or provenance_path is None):
        raise ValueError("NIAH projection requires assignment and provenance paths")
    if dataset_kind == "2wiki" and (assignment_path is not None or provenance_path is not None):
        raise ValueError("2Wiki projection must not receive NIAH assignment/provenance")

    verify_selector_component_artifacts(
        output_directory=component_directory,
        dataset_kind=dataset_kind,
        source_split="dev",
        dataset_manifest_path=dataset_manifest_path,
        source_parent_path=source_parent_path,
        candidate_pool_path=candidate_pool_path,
        assignment_path=assignment_path,
    )
    role_rows = _read_role_rows(component_directory / ROLE_ASSIGNMENTS_FILE, role)
    wanted = set(role_rows)
    bundle = JsonlDatasetAdapter.load(dataset_manifest_path)
    query_by_id = {query.query_id: query for query in bundle.queries}
    gold_by_id = {gold.query_id: gold for gold in bundle.gold_cases}
    missing = (wanted - set(query_by_id)) | (wanted - set(gold_by_id))
    if missing:
        raise ValueError(f"R002 role references missing dataset query/gold: {sorted(missing)[:5]}")
    candidates = _read_candidate_sets(candidate_pool_path, wanted)
    candidate_source = (
        candidate_pool_path / "candidate_sets.jsonl"
        if candidate_pool_path.is_dir()
        else candidate_pool_path
    )
    pool_sha256 = hashlib.sha256(candidate_source.read_bytes()).hexdigest()

    assignments: dict[str, NiahSelectorAssignment] = {}
    provenance_by_id: dict[str, MutationRecord] = {}
    if dataset_kind == "niah":
        assert assignment_path is not None and provenance_path is not None
        assignments = _read_niah_assignments(assignment_path)
        provenance_records = read_provenance(provenance_path)
        provenance_by_id = {record.query_id: record for record in provenance_records}
        if len(provenance_by_id) != len(provenance_records):
            raise ValueError("duplicate NIAH provenance query")
        if wanted - set(assignments) or wanted - set(provenance_by_id):
            raise ValueError("NIAH assignment/provenance omits role queries")

    text_by_document = {document.document_id: document.text for document in bundle.documents}

    rows: list[EvaluationQuery] = []
    for query_id in sorted(wanted):
        role_row = role_rows[query_id]
        gold = gold_by_id[query_id]
        required = tuple(gold.relevant_document_ids or ())
        if not required:
            raise ValueError(f"query {query_id} has no required documents")
        harmful: str | None = None
        harmful_in_top20_pool = False
        if dataset_kind == "niah":
            assignment = assignments[query_id]
            provenance = provenance_by_id[query_id]
            if set(assignment.required_document_ids) != set(required):
                raise ValueError(f"NIAH assignment/official gold mismatch for {query_id}")
            harmful = assignment.harmful_document_id
            harmful_in_top20_pool = any(
                candidate.document_id == harmful
                for candidate in candidates[query_id].candidates
            )
            _verify_counterfactual_pair(
                assignment=assignment,
                record=provenance,
                gold_document_ids=set(required),
                reference_answers=tuple(gold.reference_answers or ()),
                text_by_document=text_by_document,
            )
        ordered = tuple(
            sorted(
                candidates[query_id].candidates,
                key=lambda candidate: (candidate.retrieval_rank, candidate.evidence_id),
            )[:10]
        )
        if tuple(candidate.retrieval_rank for candidate in ordered) != tuple(range(1, 11)):
            raise ValueError(f"query {query_id} does not have exact TopK10 prefix")
        if dataset_kind == "niah":
            record = provenance_by_id[query_id]
            for candidate in candidates[query_id].candidates:
                if candidate.document_id == record.counterfactual_document_id:
                    replacement = str(record.replacement_value)
                    if replacement not in candidate.text:
                        raise ValueError(
                            f"NIAH harmful candidate text omits replacement for {query_id}"
                        )
                    residual = [
                        alias
                        for alias in tuple(gold.reference_answers or ())
                        if alias_occurrences(candidate.text, alias)
                    ]
                    if residual:
                        raise ValueError(
                            f"NIAH harmful candidate retains official answer for {query_id}"
                        )
                if candidate.document_id == record.needle_document_id and not alias_occurrences(
                    candidate.text, str(record.gold_alias_used)
                ):
                    raise ValueError(f"NIAH clean candidate omits gold alias for {query_id}")
        rows.append(
            EvaluationQuery(
                dataset_kind=dataset_kind,
                dataset_id=bundle.manifest.dataset_id,
                dataset_signature=bundle.dataset_signature,
                pool_sha256=pool_sha256,
                role=role,
                query_id=query_id,
                question=query_by_id[query_id].text,
                component_id=str(role_row["component_id"]),
                chain_eligible_topk10=bool(role_row["chain_eligible_topk10"]),
                topk10=ordered,
                required_document_ids=required,
                reference_answers=tuple(gold.reference_answers or ()),
                harmful_document_id=harmful,
                harmful_in_top20_pool=harmful_in_top20_pool,
            )
        )
    identity = [row.identity_row() for row in rows]
    return EvaluationProjection(
        dataset_kind=dataset_kind,
        role=role,
        rows=tuple(rows),
        sha256=_sha256(identity),
    )
