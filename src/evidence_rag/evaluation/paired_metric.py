"""Paired sign-flip test and bootstrap CI for a per-query stage metric.

``on`` and ``off`` must describe exactly the same query keys and scoring mask.  A
joint ``None`` is the explicit representation of a pre-declared ineligible query and
is omitted from the numeric pair; a one-sided ``None`` or missing key is an alignment
error rather than permission to choose a more favourable denominator.

Without ``component_ids`` every scored query is its own resampling unit, preserving
the original API.  With components, one sign is drawn per component and bootstrap
sampling draws whole components with all their queries.  The reported point estimate
remains the query-weighted mean difference; ``n_queries`` and ``n_clusters`` make the
independent-unit count explicit so correlated variants cannot masquerade as a large n.
"""

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class PairedComparison:
    mean_on: float
    mean_off: float
    delta: float
    p_value: float
    ci_low: float
    ci_high: float
    n_paired: int
    n_queries: int
    n_clusters: int
    n_total: int
    n_unscored: int


def _validated_query_keys(values: Mapping[str, object], *, label: str) -> set[str]:
    invalid = [key for key in values if not isinstance(key, str) or not key.strip()]
    if invalid:
        raise ValueError(f"{label} query keys must be non-blank strings")
    return set(values)


def _require_same_keys(
    on: Mapping[str, object], off: Mapping[str, object], *, labels: tuple[str, str]
) -> tuple[str, ...]:
    on_keys = _validated_query_keys(on, label=labels[0])
    off_keys = _validated_query_keys(off, label=labels[1])
    if on_keys != off_keys:
        only_on = sorted(on_keys - off_keys)[:5]
        only_off = sorted(off_keys - on_keys)[:5]
        raise ValueError(
            f"{labels[0]}/{labels[1]} query keys differ: "
            f"missing from {labels[1]}={only_on}, missing from {labels[0]}={only_off}"
        )
    return tuple(sorted(on_keys))


def _finite_metric_value(value: object, *, query_id: str, arm: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{arm} metric for {query_id!r} must be a finite number or None"
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{arm} metric for {query_id!r} must be finite")
    return numeric


def compare_paired(
    on: Mapping[str, float | None],
    off: Mapping[str, float | None],
    *,
    component_ids: Mapping[str, str] | None = None,
    seed: int = 13,
    iterations: int = 10000,
) -> PairedComparison:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    query_ids = _require_same_keys(on, off, labels=("on", "off"))
    if component_ids is not None:
        _require_same_keys(on, component_ids, labels=("metric", "component"))
        invalid_components = sorted(
            query_id
            for query_id, component_id in component_ids.items()
            if not isinstance(component_id, str) or not component_id.strip()
        )
        if invalid_components:
            raise ValueError(
                "component IDs must be non-blank strings; invalid queries="
                f"{invalid_components[:5]}"
            )

    diffs: list[float] = []
    on_values: list[float] = []
    off_values: list[float] = []
    cluster_by_query: list[str] = []
    n_unscored = 0
    for query_id in query_ids:
        on_value = on[query_id]
        off_value = off[query_id]
        if (on_value is None) != (off_value is None):
            raise ValueError(
                "on/off scoring masks differ: exactly one arm is unscored for "
                f"query {query_id!r}"
            )
        if on_value is None:
            n_unscored += 1
            continue
        assert off_value is not None
        on_numeric = _finite_metric_value(on_value, query_id=query_id, arm="on")
        off_numeric = _finite_metric_value(off_value, query_id=query_id, arm="off")
        on_values.append(on_numeric)
        off_values.append(off_numeric)
        diffs.append(on_numeric - off_numeric)
        cluster_by_query.append(
            query_id if component_ids is None else component_ids[query_id]
        )
    if not diffs:
        raise ValueError("no paired scored queries")
    n = len(diffs)
    mean_on = sum(on_values) / n
    mean_off = sum(off_values) / n
    delta = mean_on - mean_off
    observed = abs(delta)
    cluster_sums: dict[str, float] = {}
    cluster_sizes: dict[str, int] = {}
    for component_id, diff in zip(cluster_by_query, diffs, strict=True):
        cluster_sums[component_id] = cluster_sums.get(component_id, 0.0) + diff
        cluster_sizes[component_id] = cluster_sizes.get(component_id, 0) + 1
    clusters = tuple(sorted(cluster_sums))
    n_clusters = len(clusters)

    rng = random.Random(seed)
    extreme = 0
    for _ in range(iterations):
        permuted = (
            sum(
                cluster_sums[component_id]
                if rng.random() < 0.5
                else -cluster_sums[component_id]
                for component_id in clusters
            )
            / n
        )
        if abs(permuted) >= observed - 1e-12:
            extreme += 1
    # Plus-one correction: a finite Monte Carlo run may never draw an equally or
    # more extreme permutation, but that is not evidence for a literal p-value of 0.
    p_value = (extreme + 1) / (iterations + 1)
    boot: list[float] = []
    for _ in range(iterations):
        sampled = tuple(clusters[rng.randrange(n_clusters)] for _ in range(n_clusters))
        numerator = sum(cluster_sums[component_id] for component_id in sampled)
        denominator = sum(cluster_sizes[component_id] for component_id in sampled)
        boot.append(numerator / denominator)
    boot.sort()
    ci_low = boot[int(0.025 * iterations)]
    ci_high = boot[min(iterations - 1, int(0.975 * iterations))]
    return PairedComparison(
        mean_on=mean_on,
        mean_off=mean_off,
        delta=delta,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        n_paired=n,
        n_queries=n,
        n_clusters=n_clusters,
        n_total=len(query_ids),
        n_unscored=n_unscored,
    )
