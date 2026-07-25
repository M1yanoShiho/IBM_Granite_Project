"""CLI: paired randomization test for a stage metric across two selector reports (CPU, offline).

Reads the per_case metric values from two stage_evaluation reports (e.g. selector_report.json),
pairs them by query_id, and reports the paired mean difference with a randomization p-value and a
bootstrap CI — the same protocol harm_cli uses for harm. No GPU; runs on the login node in seconds.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.evaluation.paired_metric import compare_paired


def _read_metric(path: Path, metric: str) -> dict[str, float | None]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    values: dict[str, float | None] = {}
    for case in report["per_case"]:
        entry = case["metrics"].get(metric)
        values[case["query_id"]] = entry["value"] if entry is not None else None
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paired randomization test for a stage metric")
    parser.add_argument("--on-report", required=True, type=Path)
    parser.add_argument("--off-report", required=True, type=Path)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--iterations", type=int, default=10000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    on = _read_metric(arguments.on_report, arguments.metric)
    off = _read_metric(arguments.off_report, arguments.metric)
    comparison = compare_paired(on, off, seed=arguments.seed, iterations=arguments.iterations)
    print(
        json.dumps(
            {
                "metric": arguments.metric,
                "mean_on": comparison.mean_on,
                "mean_off": comparison.mean_off,
                "delta": comparison.delta,
                "p_value": comparison.p_value,
                "ci_low": comparison.ci_low,
                "ci_high": comparison.ci_high,
                "n_paired": comparison.n_paired,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
