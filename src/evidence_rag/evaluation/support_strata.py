"""Stratify the gate's reach by how much independent support the gold answer actually has.

Motivation. The frozen gate drops a candidate only when a competing cluster beats it by
`margin` and its own support is at most `support_cap`. The counterfactual twin is always a
single source, so with the frozen margin 2 and cap 1 the gate can act on the poison only when
`support(gold) >= 3`. Two consequences follow directly, and neither is visible in an aggregate
harm number:

  * `support(gold) == 1` is the actual needle case, and there the gate is STRUCTURALLY silent.
    No parameter choice changes that.
  * A large aggregate harm reduction can therefore be carried entirely by well-corroborated
    queries — i.e. by redundancy — while contributing nothing on the queries the project's own
    framing is about.

This module answers that from an existing E1 dump, on CPU, under both vote units and both
equivalences, so the question can be settled without spending another GPU pass.

It scores the GATE'S ACTION, not the harm metric. Harm additionally depends on truncation of
the survivor list, so `poison_dropped` is a component of harm, never a substitute for it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evidence_rag.evaluation.cluster_rescore import DumpRow, Equivalence
from evidence_rag.selector.answer_norm import canonicalize_answer
from evidence_rag.selector.clusters import AnswerCluster, build_clusters, build_clusters_lenient
from evidence_rag.selector.gated import gate_decision

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


def _dropped(
    document_id: str,
    row: DumpRow,
    clusters: Sequence[AnswerCluster],
    cluster_by_member: Mapping[str, AnswerCluster],
    *,
    margin: int,
    support_cap: int,
) -> bool | None:
    """Would the frozen gate drop this document? None when it is not in the window."""
    decisions = [
        gate_decision(
            evidence_id,
            answer,
            clusters,
            cluster_by_member,
            margin=margin,
            support_cap=support_cap,
        ).action
        for evidence_id, own_document_id, answer in zip(
            row.evidence_ids, row.document_ids, row.answers, strict=True
        )
        if own_document_id == document_id
    ]
    if not decisions:
        return None
    return any(action == "drop" for action in decisions)


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
        result.append(
            StratumRow(
                query_id=row.query_id,
                gold_support=_gold_support(clusters, row.gold_value, equivalence),
                poison_dropped=_dropped(
                    row.counterfactual_document_id, row, clusters, cluster_by_member,
                    margin=margin, support_cap=support_cap,
                ),
                needle_dropped=_dropped(
                    row.needle_document_id, row, clusters, cluster_by_member,
                    margin=margin, support_cap=support_cap,
                ),
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
        # the actual needle case, where the gate is structurally silent
        "true_needle_rate": (histogram["1"] / n) if n else None,
        # share where support(gold) >= 3, the only stratum the gate can act on the poison in
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
