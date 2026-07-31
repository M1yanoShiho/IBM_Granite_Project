"""CLI: rescore an E1 dump on CPU and emit per-query maps for paired testing.

Feeds `paired_metric_cli` — the per-query maps it writes are exactly the mapping shape
`paired_metric.compare_paired` consumes, so significance testing on a finished run costs no GPU.
"""

import argparse
import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.evaluation.cluster_rescore import (
    METRICS,
    Equivalence,
    per_query_metric,
    read_dump,
    rescore,
)
from evidence_rag.selector.answer_equivalence import lenient_equivalent

SCORINGS: dict[str, Equivalence] = {"exact": None, "lenient": lenient_equivalent}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU rescoring of an E1 per-case dump")
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--per-query",
        type=Path,
        help="write per-query metric maps here, for paired randomization testing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    rows = read_dump(arguments.dump)

    payload: dict[str, object] = {"n_rows": len(rows)}
    for name, equivalence in SCORINGS.items():
        report = rescore(rows, equivalence=equivalence)
        payload[name] = {
            "missed_conflict": dataclasses.asdict(report.missed_conflict),
            "false_conflict": dataclasses.asdict(report.false_conflict),
            "fixed_false_conflict": dataclasses.asdict(report.fixed_false_conflict),
            "needle_gold_recovery": dataclasses.asdict(report.needle_gold_recovery),
            "n_cases": report.n_cases,
            "n_fixed_eligible": report.n_fixed_eligible,
        }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if arguments.per_query is not None:
        mapping = {
            name: {
                metric: per_query_metric(rows, metric=metric, equivalence=equivalence)
                for metric in METRICS
            }
            for name, equivalence in SCORINGS.items()
        }
        arguments.per_query.parent.mkdir(parents=True, exist_ok=True)
        arguments.per_query.write_text(
            json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8"
        )

    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
