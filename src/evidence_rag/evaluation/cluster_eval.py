"""E1 edge/cluster component eval (spec docs/superpowers/specs/2026-07-23-e1-cluster-eval-design.md).

Offline over the gate's own build_clusters — measures how faithfully the exact-string
answer clusters capture agreement/conflict on the injected NIAH pools. No LLM here; the
CLI feeds extracted answers. Statistic unit = injected query.
"""

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from math import sqrt

from evidence_rag.contracts.models import EvidenceCandidate
from evidence_rag.selector.answer_norm import canonicalize_answer
from evidence_rag.selector.clusters import build_clusters, build_clusters_lenient

MISSED_CONFLICT = "selector.cluster.missed_conflict"
FALSE_CONFLICT = "selector.cluster.false_conflict"
NEEDLE_GOLD_RECOVERY = "selector.cluster.needle_gold_recovery"


def contains_alias(text: str, aliases: Sequence[str]) -> bool:
    """Word-boundary alias presence, mirroring the injector's _occurrences pattern."""
    return any(
        re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text) is not None for alias in aliases
    )


@dataclass(frozen=True)
class ClusterEvalCase:
    query_id: str
    needle_in_window: bool
    cf_in_window: bool
    missed_conflict: bool | None
    false_conflict: bool | None
    needle_gold_recovery: bool | None
    fixed_eligible: bool
    fixed_false_conflict: bool | None


def evaluate_case(
    window: Sequence[EvidenceCandidate],
    answers: Sequence[str],
    *,
    query_id: str,
    needle_document_id: str,
    counterfactual_document_id: str,
    gold_value: str,
    gold_aliases: Sequence[str],
    equivalence: Callable[[str, str], bool] | None = None,
) -> ClusterEvalCase:
    """Score one injected query's window. `equivalence` None = exact-string clustering.

    Pass `lenient_equivalent` to cluster (and credit gold recovery) with containment tolerance —
    required when comparing models or extraction strategies that differ in verbosity, since exact
    matching penalises a correct but wordier answer (see results-summary S5).
    """
    clusters = (
        build_clusters_lenient(window, answers, equivalence)
        if equivalence is not None
        else build_clusters(window, answers)
    )
    cluster_answer_by_member = {
        member_id: cluster.answer
        for cluster in clusters
        for member_id in cluster.member_ids
    }
    needle_candidates = [c for c in window if c.document_id == needle_document_id]
    cf_candidates = [c for c in window if c.document_id == counterfactual_document_id]
    needle_cluster_answers = {
        cluster_answer_by_member[c.evidence_id]
        for c in needle_candidates
        if c.evidence_id in cluster_answer_by_member
    }
    cf_cluster_answers = {
        cluster_answer_by_member[c.evidence_id]
        for c in cf_candidates
        if c.evidence_id in cluster_answer_by_member
    }
    needle_in_window = bool(needle_candidates)
    cf_in_window = bool(cf_candidates)
    needle_clustered = bool(needle_cluster_answers)

    if needle_clustered and cf_cluster_answers:
        missed_conflict: bool | None = bool(needle_cluster_answers & cf_cluster_answers)
    else:
        missed_conflict = None

    if needle_in_window:
        needle_gold_recovery: bool | None = (
            any(equivalence(answer, gold_value) for answer in needle_cluster_answers)
            if equivalence is not None
            else canonicalize_answer(gold_value) in needle_cluster_answers
        )
    else:
        needle_gold_recovery = None

    false_conflict: bool | None = None
    if needle_clustered:
        other_gold_clusters = {
            cluster_answer_by_member[c.evidence_id]
            for c in window
            if c.document_id != needle_document_id
            and c.evidence_id in cluster_answer_by_member
            and contains_alias(c.text, gold_aliases)
        }
        if other_gold_clusters:
            false_conflict = bool(other_gold_clusters - needle_cluster_answers)

    # Fixed-denominator G-FC: eligibility depends ONLY on the pool and the passage text,
    # never on the system, so both arms score the same query set. Abstaining passages count as
    # "did not create a conflict"; an abstaining needle therefore scores False, not None.
    gold_alias_others = [
        candidate
        for candidate in window
        if candidate.document_id != needle_document_id
        and contains_alias(candidate.text, gold_aliases)
    ]
    fixed_eligible = needle_in_window and bool(gold_alias_others)
    if not fixed_eligible:
        fixed_false_conflict: bool | None = None
    elif not needle_cluster_answers:
        # An abstaining needle states nothing, so nothing can falsely conflict with it. It scores
        # False rather than None because eligibility must stay system-independent — dropping the
        # query would make the denominator depend on the extractor, which is the trap this
        # metric exists to avoid.
        fixed_false_conflict = False
    else:
        other_answers = {
            cluster_answer_by_member[candidate.evidence_id]
            for candidate in gold_alias_others
            if candidate.evidence_id in cluster_answer_by_member
        }
        fixed_false_conflict = bool(other_answers - needle_cluster_answers)

    return ClusterEvalCase(
        query_id=query_id,
        needle_in_window=needle_in_window,
        cf_in_window=cf_in_window,
        missed_conflict=missed_conflict,
        false_conflict=false_conflict,
        needle_gold_recovery=needle_gold_recovery,
        fixed_eligible=fixed_eligible,
        fixed_false_conflict=fixed_false_conflict,
    )


@dataclass(frozen=True)
class MetricSummary:
    rate: float | None
    n_scored: int
    successes: int
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class ClusterEvalReport:
    missed_conflict: MetricSummary
    false_conflict: MetricSummary
    fixed_false_conflict: MetricSummary
    needle_gold_recovery: MetricSummary
    n_cases: int
    needle_in_window: int
    cf_in_window: int
    n_fixed_eligible: int


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarize(values: Iterable[bool | None]) -> MetricSummary:
    scored = [value for value in values if value is not None]
    n = len(scored)
    successes = sum(1 for value in scored if value)
    if n == 0:
        return MetricSummary(rate=None, n_scored=0, successes=0, ci_low=0.0, ci_high=0.0)
    low, high = wilson_interval(successes, n)
    return MetricSummary(
        rate=successes / n, n_scored=n, successes=successes, ci_low=low, ci_high=high
    )


def aggregate(cases: Sequence[ClusterEvalCase]) -> ClusterEvalReport:
    return ClusterEvalReport(
        missed_conflict=summarize(case.missed_conflict for case in cases),
        false_conflict=summarize(case.false_conflict for case in cases),
        fixed_false_conflict=summarize(case.fixed_false_conflict for case in cases),
        needle_gold_recovery=summarize(case.needle_gold_recovery for case in cases),
        n_cases=len(cases),
        needle_in_window=sum(1 for case in cases if case.needle_in_window),
        cf_in_window=sum(1 for case in cases if case.cf_in_window),
        n_fixed_eligible=sum(1 for case in cases if case.fixed_eligible),
    )


@dataclass(frozen=True)
class SelectionBias:
    n_gold_cases: int
    n_answerable: int
    n_multi_key: int
    multi_key_rate: float | None
    n_injected: int
    skip_rate: float | None


def selection_bias(
    reference_answers_per_case: Iterable[Sequence[str] | None],
    *,
    n_injected: int,
) -> SelectionBias:
    n_total = 0
    n_answerable = 0
    n_multi = 0
    for answers in reference_answers_per_case:
        n_total += 1
        if not answers:
            continue
        n_answerable += 1
        if len({canonicalize_answer(answer) for answer in answers}) != 1:
            n_multi += 1
    return SelectionBias(
        n_gold_cases=n_total,
        n_answerable=n_answerable,
        n_multi_key=n_multi,
        multi_key_rate=(n_multi / n_answerable) if n_answerable else None,
        n_injected=n_injected,
        skip_rate=((n_total - n_injected) / n_total) if n_total else None,
    )
