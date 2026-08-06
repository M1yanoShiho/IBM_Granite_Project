"""Compare two `retriever_scaling.py` reports arm by arm, matched on corpus size.

Exists because R6's before/after came from two jobs on two nodes, and the node
difference showed up as an impossible result: index build got 10-16% *faster* after a
change that makes the build do strictly more work. A speedup measured across that gap
carries the node's variance inside it.

With `--replicate` (a second run of the *before* arm, later in the same job) the
comparison reports its own noise floor. If the two before arms differ by an amount
comparable to the before/after gap, the measurement is not resolving the change and
the script says so rather than leaving the reader to notice.

    python scripts/compare_scaling_runs.py --before B.json --after A.json [--replicate B2.json]
"""

import argparse
import json
from pathlib import Path
from typing import Any


def _rows(path: Path) -> dict[int, dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {row["chunks"]: row for row in report["measurements"]}


def _ratio(before: float, after: float) -> float:
    return before / after if after else float("inf")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument(
        "--replicate",
        type=Path,
        help="a second run of the BEFORE arm, used to estimate the noise floor",
    )
    arguments = parser.parse_args(argv)

    before, after = _rows(arguments.before), _rows(arguments.after)
    replicate = _rows(arguments.replicate) if arguments.replicate else {}

    shared = sorted(set(before) & set(after))
    if not shared:
        raise SystemExit("the two reports share no corpus size")
    for name, rows in (("before", before), ("after", after)):
        if missing := sorted(set(shared) - set(rows)):
            raise SystemExit(f"{name} is missing corpus sizes {missing}")

    print(f"{'chunks':>7} {'before_ms':>10} {'after_ms':>9} {'speedup':>8} "
          f"{'build_b':>8} {'build_a':>8} {'rss_b':>7} {'rss_a':>7} {'d_rss':>7}")
    speedups, build_ratios, rss_deltas = [], [], []
    for chunks in shared:
        b, a = before[chunks], after[chunks]
        speedup = _ratio(b["ms_per_query_mean"], a["ms_per_query_mean"])
        build_ratio = _ratio(b["build_seconds"], a["build_seconds"])
        speedups.append(speedup)
        build_ratios.append(build_ratio)
        rss_b, rss_a = b.get("peak_rss_mb"), a.get("peak_rss_mb")
        delta = None if rss_b is None or rss_a is None else rss_a - rss_b
        if delta is not None:
            rss_deltas.append(delta)
        print(
            f"{chunks:>7} {b['ms_per_query_mean']:>10.2f} {a['ms_per_query_mean']:>9.2f} "
            f"{speedup:>7.2f}x {b['build_seconds']:>8.3f} {a['build_seconds']:>8.3f} "
            f"{'-' if rss_b is None else format(rss_b, '.1f'):>7} "
            f"{'-' if rss_a is None else format(rss_a, '.1f'):>7} "
            f"{'-' if delta is None else format(delta, '+.1f'):>7}"
        )

    print()
    print(f"per-query speedup : {min(speedups):.2f}x - {max(speedups):.2f}x")
    # The hoist moves work *into* the build, so a build that got faster cannot be the
    # change; it is the machine. Reported as a signed check rather than buried.
    verdict = (
        "build got slower, consistent with the extra work"
        if min(build_ratios) < 1.0
        else "BUILD GOT FASTER, impossible from this change; suspect node/load"
    )
    print(f"build before/after: {min(build_ratios):.2f}x - {max(build_ratios):.2f}x ({verdict})")
    if rss_deltas:
        print(f"peak RSS delta    : {min(rss_deltas):+.1f} MB to {max(rss_deltas):+.1f} MB")

    if not replicate:
        print()
        print("no --replicate given, so this comparison has no noise floor: a speedup "
              "close to 1x could not be distinguished from run-to-run variation.")
        return 0

    shared_replicate = sorted(set(shared) & set(replicate))
    drifts = [
        abs(_ratio(replicate[chunks]["ms_per_query_mean"], before[chunks]["ms_per_query_mean"]) - 1.0)
        for chunks in shared_replicate
    ]
    worst_drift = max(drifts)
    effect = min(speedups) - 1.0
    print()
    print(f"before-arm drift  : up to {100 * worst_drift:.1f}% between the two before runs")
    if worst_drift >= effect / 2:
        print("VERDICT: drift is comparable to the effect, so this run does NOT resolve "
              "the change; treat the speedup as unmeasured and rerun.")
    else:
        print(f"VERDICT: drift is {effect / worst_drift:.0f}x smaller than the smallest "
              "speedup, so the effect is resolved well clear of the noise floor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
