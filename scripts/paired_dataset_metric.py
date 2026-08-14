"""Paired randomization test for one metric across two `DatasetEvaluationReport` files.

Exists because `evidence_rag.evaluation.paired_metric_cli` cannot read these reports.
That CLI validates against `StageEvaluationReport`, whose per-case entries carry a flat
`metrics` mapping; the pipeline stage writes `evaluation_report.json`, a
`DatasetEvaluationReport` whose per-case entries are split by module into
`system` / `retriever` / `selector` / `generator`. Ledger R14 lost a submission to this:
its pre-registered read-out named `generator_report.json`, which the pipeline stage never
writes, and swapping in `evaluation_report.json` then failed schema validation instead.

Verified against ledger R9's published 2Wiki numbers before first use: 60x10 mean 0.5405,
120x5 mean 0.5185, delta +0.0220; 200x3 delta -0.0485 -- all bit-identical to the ledger.
The p-values differ in the fourth decimal because the randomization seed was not recorded
with the original figures.

    python scripts/paired_dataset_metric.py --on-report ON.json --off-report OFF.json \
        --metric system.core.answer_match
"""

import argparse
import json
from pathlib import Path

from evidence_rag.evaluation.paired_metric import compare_paired

# The per-case groups a DatasetEvaluationReport splits its metrics into. A metric name is
# prefixed with its group, so the prefix alone locates it.
GROUPS = ("system", "retriever", "selector", "generator")


def read_metric(path: Path, metric: str) -> dict[str, float | None]:
    """Return {query_id: value} for `metric`, or raise if the report cannot supply it."""
    group = metric.split(".", 1)[0]
    if group not in GROUPS:
        raise ValueError(f"metric {metric!r} names no known group; expected one of {GROUPS}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    values: dict[str, float | None] = {}
    for case in payload["per_case"]:
        entry = case[group][metric]
        # Reports written before the metric objects gained provenance store a bare number.
        values[case["query_id"]] = entry["value"] if isinstance(entry, dict) else entry
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--on-report", required=True, type=Path)
    parser.add_argument("--off-report", required=True, type=Path)
    parser.add_argument("--metric", required=True)
    arguments = parser.parse_args(argv)

    on = read_metric(arguments.on_report, arguments.metric)
    off = read_metric(arguments.off_report, arguments.metric)
    result = compare_paired(on, off)

    print(f"{arguments.on_report} vs {arguments.off_report}  [{arguments.metric}]")
    print(f"  mean_on {result.mean_on:.4f}  mean_off {result.mean_off:.4f}  delta {result.delta:+.4f}")
    print(
        f"  p {result.p_value:.4f}  95% CI [{result.ci_low:+.4f}, {result.ci_high:+.4f}]"
        f"  n {result.n_paired}  (unscored {result.n_unscored} / total {result.n_total})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
