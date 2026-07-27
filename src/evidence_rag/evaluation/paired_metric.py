"""Paired randomization test + bootstrap CI for a per-query stage metric.

Generalizes the harm metric's significance protocol (harm.compare_harm) to any per-query metric,
so recall / precision / etc. across two selector reports are tested the same way harm is: sign-flip
randomization for the p-value, bootstrap percentile CI on the paired mean difference. Statistic unit
= query; only queries scored in BOTH arms are paired.
"""

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


def compare_paired(
    on: Mapping[str, float | None],
    off: Mapping[str, float | None],
    *,
    seed: int = 13,
    iterations: int = 10000,
) -> PairedComparison:
    diffs: list[float] = []
    on_values: list[float] = []
    off_values: list[float] = []
    for query_id, on_value in on.items():
        off_value = off.get(query_id)
        if on_value is None or off_value is None:
            continue
        on_values.append(on_value)
        off_values.append(off_value)
        diffs.append(on_value - off_value)
    if not diffs:
        raise ValueError("no paired scored queries")
    n = len(diffs)
    mean_on = sum(on_values) / n
    mean_off = sum(off_values) / n
    delta = mean_on - mean_off
    observed = abs(delta)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(iterations):
        permuted = sum(d if rng.random() < 0.5 else -d for d in diffs) / n
        if abs(permuted) >= observed - 1e-12:
            extreme += 1
    p_value = extreme / iterations
    boot: list[float] = []
    for _ in range(iterations):
        boot.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    boot.sort()
    ci_low = boot[int(0.025 * iterations)]
    ci_high = boot[int(0.975 * iterations)]
    return PairedComparison(
        mean_on=mean_on,
        mean_off=mean_off,
        delta=delta,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        n_paired=n,
    )
