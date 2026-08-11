"""Pure selector-risk metrics and expected-risk CRC policy selection.

This module deliberately has no artifact I/O and no dependency on sealed/held-out data.  It
defines the signs and safety rules used by the adaptive conservative selector:

* recall loss is ``TopK10 - Selector`` over *document IDs*;
* harm reduction is ``TopK10 - Selector`` (positive is beneficial);
* chain loss is scored only when TopK10 contained the complete gold chain;
* P0 is the structural TopK10 fallback, never a policy that is said to have passed CRC.

Policies are ordered P0..P6 from least to most aggressive.  The expected-risk correction uses
the bounded-loss rule ``(sum(losses) + B) / (n + 1)`` with ``B=1`` by default.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

COMPONENT_REPRESENTATIVE_SEED = 20260811
POLICY_COUNT = 7
P0_INDEX = 0
DEFAULT_ALPHA = 0.01
DEFAULT_LOSS_BOUND = 1.0

NIAH_RECALL_RISK = "niah_required_recall"
NIAH_CHAIN_RISK = "niah_conditional_chain"
TWOWIKI_RECALL_RISK = "twowiki_supporting_recall"
TWOWIKI_CHAIN_RISK = "twowiki_conditional_chain"
REQUIRED_RISKS = (
    NIAH_RECALL_RISK,
    NIAH_CHAIN_RISK,
    TWOWIKI_RECALL_RISK,
    TWOWIKI_CHAIN_RISK,
)


def _id_set(values: Collection[str], *, label: str, allow_empty: bool = True) -> frozenset[str]:
    if isinstance(values, str):
        raise TypeError(f"{label} must be a collection of IDs, not one string")
    result = frozenset(values)
    if not allow_empty and not result:
        raise ValueError(f"{label} must not be empty")
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{label} must contain non-empty string IDs")
    return result


def _require_delete_only(
    *, baseline_document_ids: frozenset[str], selector_document_ids: frozenset[str]
) -> None:
    added = selector_document_ids - baseline_document_ids
    if added:
        raise ValueError(
            "selector document IDs must be a subset of the TopK10 baseline for delete-only "
            f"evaluation; added IDs include {sorted(added)[:5]}"
        )


def document_recall(
    selected_document_ids: Collection[str], gold_document_ids: Collection[str]
) -> float:
    """Recall of official required/supporting *documents* in one selected context."""
    selected = _id_set(selected_document_ids, label="selected_document_ids")
    gold = _id_set(gold_document_ids, label="gold_document_ids", allow_empty=False)
    return len(selected & gold) / len(gold)


def relative_recall_loss(
    *,
    baseline_document_ids: Collection[str],
    selector_document_ids: Collection[str],
    gold_document_ids: Collection[str],
) -> float:
    """Return ``Recall(TopK10) - Recall(Selector)``; positive values are harmful."""
    baseline = _id_set(baseline_document_ids, label="baseline_document_ids")
    selector = _id_set(selector_document_ids, label="selector_document_ids")
    _require_delete_only(baseline_document_ids=baseline, selector_document_ids=selector)
    return document_recall(baseline, gold_document_ids) - document_recall(
        selector, gold_document_ids
    )


def conditional_chain_loss(
    *,
    baseline_document_ids: Collection[str],
    selector_document_ids: Collection[str],
    gold_document_ids: Collection[str],
) -> float | None:
    """Whether deletion breaks a chain that TopK10 originally contained completely.

    ``None`` means the query is not chain-eligible because TopK10 itself lacked at least one
    official gold document.  Such a query must not enter the conditional-chain denominator.
    """
    baseline = _id_set(baseline_document_ids, label="baseline_document_ids")
    selector = _id_set(selector_document_ids, label="selector_document_ids")
    gold = _id_set(gold_document_ids, label="gold_document_ids", allow_empty=False)
    _require_delete_only(baseline_document_ids=baseline, selector_document_ids=selector)
    if not gold.issubset(baseline):
        return None
    return float(not gold.issubset(selector))


def _unit_interval(value: float, *, label: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1], got {value!r}")
    return numeric


def harm_reduction(*, baseline_harm: float, selector_harm: float) -> float:
    """Return ``Harm(TopK10) - Harm(Selector)``; positive values are improvements."""
    return _unit_interval(baseline_harm, label="baseline_harm") - _unit_interval(
        selector_harm, label="selector_harm"
    )


def _drop_fraction(
    dropped_document_ids: Sequence[str], labelled_document_ids: Collection[str], *, label: str
) -> float | None:
    if isinstance(dropped_document_ids, str):
        raise TypeError("dropped_document_ids must be one document ID per dropped candidate")
    dropped = tuple(dropped_document_ids)
    if any(not isinstance(document_id, str) or not document_id for document_id in dropped):
        raise ValueError("dropped_document_ids must contain non-empty strings")
    labelled = _id_set(labelled_document_ids, label=label)
    if not dropped:
        return None
    return sum(document_id in labelled for document_id in dropped) / len(dropped)


def deletion_precision(
    *, dropped_document_ids: Sequence[str], harmful_document_ids: Collection[str]
) -> float | None:
    """Fraction of ``DROP_HARM`` candidate actions whose document is truly harmful.

    ``dropped_document_ids`` contains one entry per dropped candidate, so two dropped chunks
    from the same document remain two actions.  ``None`` records an undefined ratio when the
    selector deleted nothing; it must not be silently reported as zero precision.
    """
    return _drop_fraction(
        dropped_document_ids, harmful_document_ids, label="harmful_document_ids"
    )


def required_deletion_rate(
    *, dropped_document_ids: Sequence[str], required_document_ids: Collection[str]
) -> float | None:
    """Fraction of all dropped candidate actions that belong to official required evidence."""
    return _drop_fraction(
        dropped_document_ids, required_document_ids, label="required_document_ids"
    )


@dataclass(frozen=True)
class ComponentCandidate:
    """One eligible query competing to represent a component for one risk."""

    dataset_id: str
    risk_name: str
    component_id: str
    query_id: str

    def __post_init__(self) -> None:
        for field_name in ("dataset_id", "risk_name", "component_id", "query_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-blank string")


def representative_digest(
    candidate: ComponentCandidate, *, seed: int = COMPONENT_REPRESENTATIVE_SEED
) -> str:
    """Stable SHA-256 priority for the frozen representative-selection tuple."""
    payload = json.dumps(
        [
            seed,
            candidate.dataset_id,
            candidate.risk_name,
            candidate.component_id,
            candidate.query_id,
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_component_representatives(
    candidates: Iterable[ComponentCandidate], *, seed: int = COMPONENT_REPRESENTATIVE_SEED
) -> tuple[ComponentCandidate, ...]:
    """Choose the minimum stable hash per ``dataset × risk × component``.

    Input order cannot affect the result.  Duplicate candidate rows are rejected because they
    would conceal an upstream join error even though they happen not to change ``min``.
    """
    groups: dict[tuple[str, str, str], list[ComponentCandidate]] = {}
    seen: set[tuple[str, str, str, str]] = set()
    component_by_query: dict[tuple[str, str, str], str] = {}
    for candidate in candidates:
        identity = (
            candidate.dataset_id,
            candidate.risk_name,
            candidate.component_id,
            candidate.query_id,
        )
        if identity in seen:
            raise ValueError(f"duplicate component representative candidate: {identity}")
        seen.add(identity)
        query_key = (candidate.dataset_id, candidate.risk_name, candidate.query_id)
        previous_component = component_by_query.setdefault(
            query_key, candidate.component_id
        )
        if previous_component != candidate.component_id:
            raise ValueError(
                "one representative candidate query cannot belong to two components: "
                f"{query_key} maps to {previous_component!r} and {candidate.component_id!r}"
            )
        key = identity[:3]
        groups.setdefault(key, []).append(candidate)

    representatives: list[ComponentCandidate] = []
    for key in sorted(groups):
        representative = min(
            groups[key],
            key=lambda item: (representative_digest(item, seed=seed), item.query_id),
        )
        representatives.append(representative)
    return tuple(representatives)


def validate_nested_policy_sets(
    selected_ids_by_policy: Sequence[Collection[str]],
    *,
    baseline_ids: Collection[str],
) -> tuple[frozenset[str], ...]:
    """Validate exact TopK P0 plus ``S0 ⊇ S1 ⊇ ... ⊇ S6``."""
    if len(selected_ids_by_policy) != POLICY_COUNT:
        raise ValueError(
            f"expected exactly P0..P{POLICY_COUNT - 1}, got {len(selected_ids_by_policy)} sets"
        )
    normalized = tuple(
        _id_set(values, label=f"selected_ids_by_policy[{index}]")
        for index, values in enumerate(selected_ids_by_policy)
    )
    baseline = _id_set(baseline_ids, label="baseline_ids")
    if normalized[P0_INDEX] != baseline:
        missing = sorted(baseline - normalized[P0_INDEX])[:5]
        added = sorted(normalized[P0_INDEX] - baseline)[:5]
        raise ValueError(
            "P0 must be exactly the frozen TopK baseline; "
            f"missing={missing}, added={added}"
        )
    for policy_index in range(1, POLICY_COUNT):
        introduced = normalized[policy_index] - normalized[policy_index - 1]
        if introduced:
            raise ValueError(
                f"P{policy_index} is not a subset of P{policy_index - 1}; introduced IDs "
                f"include {sorted(introduced)[:5]}"
            )
    return normalized


def _bounded_losses(losses: Sequence[float], *, loss_bound: float) -> tuple[float, ...]:
    if not math.isfinite(loss_bound) or loss_bound <= 0.0:
        raise ValueError("loss_bound must be finite and positive")
    normalized = tuple(float(loss) for loss in losses)
    for loss in normalized:
        if not math.isfinite(loss) or not 0.0 <= loss <= loss_bound:
            raise ValueError(f"loss must be finite and in [0, {loss_bound}], got {loss!r}")
    return normalized


def validate_monotone_losses(
    losses_by_policy: Sequence[Sequence[float]], *, loss_bound: float = DEFAULT_LOSS_BOUND
) -> tuple[tuple[float, ...], ...]:
    """Validate equal rows and elementwise ``L(P0) ≤ ... ≤ L(P6)``."""
    if len(losses_by_policy) != POLICY_COUNT:
        raise ValueError(
            f"expected exactly P0..P{POLICY_COUNT - 1}, got {len(losses_by_policy)} loss rows"
        )
    normalized = tuple(
        _bounded_losses(losses, loss_bound=loss_bound) for losses in losses_by_policy
    )
    sizes = {len(losses) for losses in normalized}
    if len(sizes) != 1:
        raise ValueError("all policies must be scored on the same representative queries")
    for policy_index in range(1, POLICY_COUNT):
        for row_index, (previous, current) in enumerate(
            zip(normalized[policy_index - 1], normalized[policy_index], strict=True)
        ):
            if current < previous:
                raise ValueError(
                    f"loss is not monotone at representative {row_index}: "
                    f"P{policy_index - 1}={previous}, P{policy_index}={current}"
                )
    return normalized


def corrected_risk(
    losses: Sequence[float], *, loss_bound: float = DEFAULT_LOSS_BOUND
) -> float:
    """Expected-risk CRC correction ``(sum(losses) + B) / (n + 1)``."""
    bounded = _bounded_losses(losses, loss_bound=loss_bound)
    return (sum(bounded) + loss_bound) / (len(bounded) + 1)


def minimum_calibration_size(
    *, alpha: float = DEFAULT_ALPHA, loss_bound: float = DEFAULT_LOSS_BOUND
) -> int:
    """Smallest ``n`` for which a zero-loss non-P0 policy can satisfy CRC."""
    if not math.isfinite(alpha) or not 0.0 < alpha <= loss_bound:
        raise ValueError("alpha must be finite, positive, and no larger than loss_bound")
    if not math.isfinite(loss_bound) or loss_bound <= 0.0:
        raise ValueError("loss_bound must be finite and positive")
    size = max(0, math.ceil(loss_bound / alpha - 1.0))
    while loss_bound / (size + 1) > alpha:
        size += 1
    while size > 0 and loss_bound / size <= alpha:
        size -= 1
    return size


def nonzero_policy_can_qualify(
    n_calibration: int,
    *,
    alpha: float = DEFAULT_ALPHA,
    loss_bound: float = DEFAULT_LOSS_BOUND,
) -> bool:
    """Whether sample size alone permits any non-P0 policy to pass with zero loss."""
    if n_calibration < 0:
        raise ValueError("n_calibration must not be negative")
    return n_calibration >= minimum_calibration_size(alpha=alpha, loss_bound=loss_bound)


@dataclass(frozen=True)
class RiskLossTable:
    """One risk scored on fixed representative IDs for every nested policy."""

    risk_name: str
    representative_ids: tuple[str, ...]
    representative_component_ids: tuple[str, ...]
    losses_by_policy: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.risk_name, str) or not self.risk_name.strip():
            raise ValueError("risk_name must be a non-blank string")
        representative_ids = tuple(self.representative_ids)
        component_ids = tuple(self.representative_component_ids)
        if len(set(representative_ids)) != len(representative_ids):
            raise ValueError("representative_ids must be unique")
        if any(
            not isinstance(representative_id, str) or not representative_id.strip()
            for representative_id in representative_ids
        ):
            raise ValueError("representative_ids must contain non-blank strings")
        if len(component_ids) != len(representative_ids):
            raise ValueError("representative component IDs must align with representative_ids")
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("each component may contribute at most one representative")
        if any(
            not isinstance(component_id, str) or not component_id.strip()
            for component_id in component_ids
        ):
            raise ValueError("representative component IDs must contain non-blank strings")
        normalized = validate_monotone_losses(self.losses_by_policy)
        if any(len(losses) != len(representative_ids) for losses in normalized):
            raise ValueError("every policy loss row must align with representative_ids")
        if any(loss != 0.0 for loss in normalized[P0_INDEX]):
            raise ValueError("P0 is structurally identical to TopK10 and must have zero loss")
        object.__setattr__(self, "representative_ids", representative_ids)
        object.__setattr__(self, "representative_component_ids", component_ids)
        object.__setattr__(self, "losses_by_policy", normalized)

    @property
    def n(self) -> int:
        return len(self.representative_ids)


RiskDecisionStatus: TypeAlias = Literal[
    "QUALIFIED_NONZERO",
    "NO_QUALIFIED_NONZERO",
    "NO_CALIBRATION_EVIDENCE",
    "BLOCKED_BY_SAMPLE_SIZE",
]


@dataclass(frozen=True)
class RiskDecision:
    risk_name: str
    n_calibration: int
    corrected_risks: tuple[float | None, ...]
    strongest_qualified_policy: int
    nonzero_possible_by_size: bool
    status: RiskDecisionStatus


@dataclass(frozen=True)
class CRCSelection:
    alpha: float
    loss_bound: float
    final_policy: int
    structural_fallback: bool
    blocked_by_sample_size: bool
    has_no_calibration_evidence: bool
    per_risk: tuple[RiskDecision, ...]

    def decision_for(self, risk_name: str) -> RiskDecision:
        for decision in self.per_risk:
            if decision.risk_name == risk_name:
                return decision
        raise KeyError(risk_name)


def select_crc_policy(
    risk_tables: Iterable[RiskLossTable],
    *,
    alpha: float = DEFAULT_ALPHA,
    loss_bound: float = DEFAULT_LOSS_BOUND,
) -> CRCSelection:
    """Select per-risk max qualified Pj, then the four-risk conservative ``min(j)``.

    P0 is returned only as the structural TopK10 fallback.  Its zero relative loss is not put
    through the corrected-risk formula and is not described as having passed CRC.
    """
    minimum_calibration_size(alpha=alpha, loss_bound=loss_bound)
    by_name: dict[str, RiskLossTable] = {}
    for table in risk_tables:
        if table.risk_name in by_name:
            raise ValueError(f"duplicate risk table: {table.risk_name}")
        by_name[table.risk_name] = table
    found = set(by_name)
    expected = set(REQUIRED_RISKS)
    if found != expected:
        raise ValueError(
            f"CRC requires exactly the four pre-registered risks; "
            f"missing={sorted(expected - found)}, extra={sorted(found - expected)}"
        )

    decisions: list[RiskDecision] = []
    for risk_name in REQUIRED_RISKS:
        table = by_name[risk_name]
        corrected: list[float | None] = [None]
        qualified: list[int] = []
        for policy_index in range(1, POLICY_COUNT):
            risk = corrected_risk(
                table.losses_by_policy[policy_index], loss_bound=loss_bound
            )
            corrected.append(risk)
            if risk <= alpha:
                qualified.append(policy_index)
        strongest = max(qualified, default=P0_INDEX)
        possible_by_size = nonzero_policy_can_qualify(
            table.n, alpha=alpha, loss_bound=loss_bound
        )
        if table.n == 0:
            status: RiskDecisionStatus = "NO_CALIBRATION_EVIDENCE"
        elif not possible_by_size:
            status = "BLOCKED_BY_SAMPLE_SIZE"
        elif strongest == P0_INDEX:
            status = "NO_QUALIFIED_NONZERO"
        else:
            status = "QUALIFIED_NONZERO"
        decisions.append(
            RiskDecision(
                risk_name=risk_name,
                n_calibration=table.n,
                corrected_risks=tuple(corrected),
                strongest_qualified_policy=strongest,
                nonzero_possible_by_size=possible_by_size,
                status=status,
            )
        )

    final_policy = min(decision.strongest_qualified_policy for decision in decisions)
    return CRCSelection(
        alpha=alpha,
        loss_bound=loss_bound,
        final_policy=final_policy,
        structural_fallback=final_policy == P0_INDEX,
        blocked_by_sample_size=any(
            decision.status == "BLOCKED_BY_SAMPLE_SIZE" for decision in decisions
        ),
        has_no_calibration_evidence=any(
            decision.status == "NO_CALIBRATION_EVIDENCE" for decision in decisions
        ),
        per_risk=tuple(decisions),
    )
