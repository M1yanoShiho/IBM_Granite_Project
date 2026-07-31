"""Stratify the gate's reach by how much independent support the gold answer actually has.

Motivation. The frozen gate drops a candidate only when a competing cluster beats it by
`margin` and its own support is at most `support_cap`. The question this module answers is
whether the gate's action concentrates on well-corroborated queries — i.e. whether an aggregate
harm reduction is carried by redundancy rather than by the mechanism the project claims.

CORRECTION (2026-07-31, from the R001 measurement). An earlier version of this docstring
asserted that the gate can act on the poison only when `support(gold) >= 3`. That is FALSE, and
the measurement is what caught it. The competing cluster is the strongest cluster with a
DIFFERENT answer, which need not be the gold cluster at all: a distractor answer backed by
enough sources out-votes the poison just as well. The real condition is
`support(winner) >= support(cf) + margin`, and `support(gold)` is only one candidate for the
winner. Empirically the poison is dropped on 3-11% of queries where the gold cluster has no
support at all, which is impossible under the claim that was made here.

So `support(gold) == 1` does NOT make the gate structurally silent — it makes the gate's action
uncorrelated with defending the gold answer, which is a different and arguably worse problem.
Read the strata as "where does the gate act", not "where can the gate act".

This module answers that from an existing E1 dump, on CPU, under both vote units and both
equivalences, so the question can be settled without spending another GPU pass.

Comparing a single stratum ACROSS vote units compares different query sets, because the vote
unit changes which stratum a query lands in. Only the overall rates share a denominator.

It scores the GATE'S ACTION, not the harm metric. Harm additionally depends on truncation of
the survivor list, so `poison_dropped` is a component of harm, never a substitute for it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evidence_rag.evaluation.cluster_rescore import DumpRow, Equivalence
from evidence_rag.selector.answer_norm import canonicalize_answer
from evidence_rag.selector.clusters import AnswerCluster, build_clusters, build_clusters_lenient
from evidence_rag.selector.gated import GateDecision, gate_decision

STRATA = ("0", "1", "2", ">=3")

DEFAULT_MARGIN = 2
DEFAULT_SUPPORT_CAP = 1


def stratum_label(gold_support: int) -> str:
    return str(gold_support) if gold_support < 3 else ">=3"


@dataclass(frozen=True)
class StratumRow:
    query_id: str
    gold_support: int
    poison_dropped: bool | None
    needle_dropped: bool | None
    # What out-voted the needle. A real competing claim and extraction noise from a passage that
    # does not address the question are indistinguishable in the rates, but not in these strings.
    needle_own_answer: str | None
    needle_winner_answer: str | None
    needle_winner_support: int | None


def _gold_support(
    clusters: Sequence[AnswerCluster], gold_value: str, equivalence: Equivalence
) -> int:
    """Support of the cluster carrying the gold answer, or 0 when no cluster does."""
    target = canonicalize_answer(gold_value)
    for cluster in clusters:
        matched = (
            equivalence(cluster.answer, gold_value)
            if equivalence is not None
            else cluster.answer == target
        )
        if matched:
            return cluster.independent_support
    return 0


def _decisions_for(
    document_id: str,
    row: DumpRow,
    clusters: Sequence[AnswerCluster],
    cluster_by_member: Mapping[str, AnswerCluster],
    *,
    margin: int,
    support_cap: int,
) -> list[GateDecision]:
    """Gate decisions for every window entry belonging to this document (empty when absent)."""
    return [
        gate_decision(
            evidence_id,
            answer,
            clusters,
            cluster_by_member,
            margin=margin,
            support_cap=support_cap,
        )
        for evidence_id, own_document_id, answer in zip(
            row.evidence_ids, row.document_ids, row.answers, strict=True
        )
        if own_document_id == document_id
    ]


def _dropped(decisions: Sequence[GateDecision]) -> bool | None:
    if not decisions:
        return None
    return any(decision.action == "drop" for decision in decisions)


def analyse(
    rows: Sequence[DumpRow],
    *,
    equivalence: Equivalence,
    parent_by_document: Mapping[str, str] | None,
    margin: int = DEFAULT_MARGIN,
    support_cap: int = DEFAULT_SUPPORT_CAP,
) -> tuple[StratumRow, ...]:
    result: list[StratumRow] = []
    for row in rows:
        window = row.window()
        clusters = (
            build_clusters_lenient(
                window, row.answers, equivalence, parent_by_document=parent_by_document
            )
            if equivalence is not None
            else build_clusters(window, row.answers, parent_by_document=parent_by_document)
        )
        cluster_by_member = {
            member_id: cluster for cluster in clusters for member_id in cluster.member_ids
        }
        poison = _decisions_for(
            row.counterfactual_document_id, row, clusters, cluster_by_member,
            margin=margin, support_cap=support_cap,
        )
        needle = _decisions_for(
            row.needle_document_id, row, clusters, cluster_by_member,
            margin=margin, support_cap=support_cap,
        )
        # Report the decision that actually dropped the needle when there is one, so the recorded
        # winner is the answer responsible rather than an arbitrary chunk's view.
        culprit = next(
            (decision for decision in needle if decision.action == "drop"),
            needle[0] if needle else None,
        )
        result.append(
            StratumRow(
                query_id=row.query_id,
                gold_support=_gold_support(clusters, row.gold_value, equivalence),
                poison_dropped=_dropped(poison),
                needle_dropped=_dropped(needle),
                needle_own_answer=culprit.answer if culprit else None,
                needle_winner_answer=culprit.winner_answer if culprit else None,
                needle_winner_support=culprit.winner_support if culprit else None,
            )
        )
    return tuple(result)


def _rate(values: Sequence[bool | None]) -> float | None:
    scored = [value for value in values if value is not None]
    return (sum(1 for value in scored if value) / len(scored)) if scored else None


def summarize_strata(rows: Sequence[StratumRow]) -> dict[str, object]:
    histogram = {label: 0 for label in STRATA}
    for row in rows:
        histogram[stratum_label(row.gold_support)] += 1

    strata: dict[str, object] = {}
    for label in STRATA:
        members = [row for row in rows if stratum_label(row.gold_support) == label]
        strata[label] = {
            "n": len(members),
            "poison_dropped_rate": _rate([row.poison_dropped for row in members]),
            "needle_dropped_rate": _rate([row.needle_dropped for row in members]),
        }

    n = len(rows)
    return {
        "n_cases": n,
        "gold_support_histogram": histogram,
        # the actual needle case: exactly one source backs the gold answer
        "true_needle_rate": (histogram["1"] / n) if n else None,
        # share of queries whose GOLD cluster is strong enough to out-vote the poison on its
        # own. Not an upper bound on gate activity — a distractor cluster can out-vote it too.
        "gate_can_fire_rate": (histogram[">=3"] / n) if n else None,
        "poison_dropped_rate": _rate([row.poison_dropped for row in rows]),
        "needle_dropped_rate": _rate([row.needle_dropped for row in rows]),
        "strata": strata,
    }


__all__ = [
    "DEFAULT_MARGIN",
    "DEFAULT_SUPPORT_CAP",
    "STRATA",
    "StratumRow",
    "analyse",
    "stratum_label",
    "summarize_strata",
]
