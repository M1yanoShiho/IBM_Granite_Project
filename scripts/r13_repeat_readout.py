"""Read the three R13 latency repeats and say whether the superlinearity survives them.

R13 concluded that the inverted index scales *superlinearly* on real NQ queries (fitted
exponent 1.15-1.3, `ms/1k_chunks` 0.27 -> 0.55) and on that basis withdrew R9's asymptotic
claim. That conclusion came from a single 6m46s job. The quantities it rests on -- mean,
p50 and especially p95 milliseconds -- are the highest-variance things the probe reports,
and node contention feeds straight into them.

`compare_scaling_runs.py` already carries the noise-floor idea (R6b's lesson), but it
compares a before arm against an after arm on `ms_per_query_mean` only. R13 has no
before/after: it has one arm measured three times, and its claim lives in the *slope* and
in p95. Hence this script.

The criterion is the one written down before the repeats were submitted:

  * if all three fitted exponents sit above 1.0, superlinearity is resolved clear of
    run-to-run variation and R13 stands;
  * if the three exponents straddle 1.0, the reading does not resolve the question and
    "superlinear" must be downgraded to "did not resolve" rather than left in the report.

    python scripts/r13_repeat_readout.py results/r13-repeat/retriever-scaling-nq-inverted-rep{1,2,3}.json
"""

import argparse
import json
import statistics
from math import log
from pathlib import Path

METRICS = ("ms_per_query_mean", "ms_per_query_p50", "ms_per_query_p95")


def _rows(path: Path) -> dict[int, dict[str, float]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {row["chunks"]: row for row in report["measurements"]}


def _exponent(rows: dict[int, dict[str, float]], metric: str) -> float:
    """Least-squares slope of log(latency) on log(chunks): 1.0 is exactly linear."""
    points = [(log(chunks), log(row[metric])) for chunks, row in sorted(rows.items())]
    mean_x = statistics.fmean(x for x, _ in points)
    mean_y = statistics.fmean(y for _, y in points)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in points)
    variance = sum((x - mean_x) ** 2 for x, _ in points)
    return covariance / variance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path, help="the repeat reports, in order")
    arguments = parser.parse_args(argv)

    replicates = [_rows(path) for path in arguments.reports]
    if len(replicates) < 2:
        raise SystemExit("need at least two repeats to say anything about spread")

    shared = sorted(set.intersection(*(set(rows) for rows in replicates)))
    if not shared:
        raise SystemExit("the repeats share no corpus size")
    for path, rows in zip(arguments.reports, replicates):
        if missing := sorted(set(shared) - set(rows)):
            raise SystemExit(f"{path} is missing corpus sizes {missing}")

    n = len(replicates)
    print(f"{n} repeats, {len(shared)} corpus sizes, spread reported as min-max across repeats")
    print()

    for metric in METRICS:
        label = metric.removeprefix("ms_per_query_")
        print(f"== {label} ms")
        print(f"{'chunks':>8} {'min':>9} {'median':>9} {'max':>9} {'spread':>8}")
        for chunks in shared:
            values = [rows[chunks][metric] for rows in replicates]
            low, high = min(values), max(values)
            # Relative spread is the comparable number across sizes; absolute ms is not,
            # because the values run over two orders of magnitude.
            spread = (high - low) / low if low else float("inf")
            print(
                f"{chunks:>8} {low:>9.3f} {statistics.median(values):>9.3f} "
                f"{high:>9.3f} {100 * spread:>7.1f}%"
            )
        print()

    print("== ms per 1k chunks (the number R13 read the trend off)")
    print(f"{'chunks':>8} " + " ".join(f"{'rep' + str(i + 1):>8}" for i in range(n)))
    for chunks in shared:
        per_1k = [1000 * rows[chunks]["ms_per_query_mean"] / chunks for rows in replicates]
        print(f"{chunks:>8} " + " ".join(f"{value:>8.3f}" for value in per_1k))
    print()

    print("== fitted exponent (1.0 = linear; R13 reported 1.15-1.3 from a single run)")
    verdicts = []
    for metric in METRICS:
        label = metric.removeprefix("ms_per_query_")
        exponents = [_exponent(rows, metric) for rows in replicates]
        low, high = min(exponents), max(exponents)
        resolved = low > 1.0
        verdicts.append((label, low, high, resolved))
        marker = "superlinear on every repeat" if resolved else "STRADDLES 1.0"
        print(f"{label:>6}: " + " ".join(f"{e:.3f}" for e in exponents)
              + f"   range {low:.3f}-{high:.3f}  ({marker})")
    print()

    mean_verdict = next(v for v in verdicts if v[0] == "mean")
    if mean_verdict[3]:
        print("VERDICT: superlinearity survives the repeats on the mean -- R13's withdrawal of "
              "R9's asymptotic claim stands, and now has a variance behind it.")
    else:
        print("VERDICT: the fitted exponent on the mean straddles 1.0 across repeats -- "
              "per the pre-registered criterion, 'superlinear' must be downgraded to "
              "'did not resolve' and R9's asymptotic claim cannot be withdrawn on this evidence.")

    p95_spread = max(
        (max(rows[chunks]["ms_per_query_p95"] for rows in replicates)
         - min(rows[chunks]["ms_per_query_p95"] for rows in replicates))
        / min(rows[chunks]["ms_per_query_p95"] for rows in replicates)
        for chunks in shared
    )
    print(f"p95 worst relative spread: {100 * p95_spread:.1f}% -- R13's tail claim "
          "('p95 degrades faster than the mean') is only readable if this is small "
          "relative to the 30x it reported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
