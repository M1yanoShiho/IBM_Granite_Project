"""Strip a pipeline report's per-case traces so the raw result can be pulled back into git.

The ledger's own rule is that a result living only in a `.out` log or a loose scp'd file is not
recorded: raw results are `git add -f`'d on bp1 and travel back through git. Complete
`evaluation_report.json` files are far too large for that -- R7 and R9 hit 660 MB across seven
arms -- and it is the per-case `trace` that carries the weight, holding every candidate's text.
Dropping it took those seven to 14 MB, and every reading since has been done without it:
`conditional_miss_rate.py` and `paired_dataset_metric.py` read the metric groups only.

So this drops `trace` and nothing else. Aggregates, case IDs, the four metric groups and all
four signatures survive, which is what makes the pulled-back copy checkable against the run it
came from. The full trace stays on bp1.

Verified in passing: the local trace-stripped copies of ledger R9 reproduce that entry's
published figures, confidence intervals included, from the same scripts that read the full
reports on bp1.

    PYTHONPATH=src python scripts/export_per_case.py \
        --report runs/chunk-nq-b-c60o10/evaluation_report.json \
        --output results/r14-nq-b-c60o10-per-case.json
"""

import argparse
import json
from pathlib import Path

DROPPED = "trace"


def strip_traces(payload: dict[str, object]) -> tuple[dict[str, object], int]:
    """Return the report without per-case traces, and how many were dropped."""
    cases = payload.get("per_case")
    if not isinstance(cases, list):
        raise ValueError("report has no per_case list; is this an evaluation report?")
    dropped = 0
    stripped = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("per_case entries must be objects")
        if DROPPED in case:
            dropped += 1
        stripped.append({key: value for key, value in case.items() if key != DROPPED})
    return {**payload, "per_case": stripped}, dropped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)

    payload = json.loads(arguments.report.read_text(encoding="utf-8"))
    stripped, dropped = strip_traces(payload)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(stripped, ensure_ascii=False), encoding="utf-8", newline="\n"
    )

    before = arguments.report.stat().st_size / 1e6
    after = arguments.output.stat().st_size / 1e6
    print(
        f"{arguments.report} -> {arguments.output}\n"
        f"  {len(stripped['per_case'])} cases, {dropped} traces dropped, "  # type: ignore[arg-type]
        f"{before:.1f} MB -> {after:.1f} MB"
    )
    if dropped == 0:
        print("  note: no traces were present, so the copy is the report verbatim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
