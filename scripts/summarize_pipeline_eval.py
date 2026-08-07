#!/usr/bin/env python3
"""Summarise evaluation_report.json across pipeline configs, upstream beside downstream.

Usage:
    python scripts/summarize_pipeline_eval.py CONFIG.toml [CONFIG.toml ...]

R7 asks whether a retrieval gain reaches the generator, so the table has to show both
ends of the chain on one line; reading two separate tables is how a non-transfer gets
missed. Columns run left to right in pipeline order: retriever, then selector, then
system.

The last column is the one the question turns on. `transfer` divides the spread in
`system.core.answer_match` by the spread in `retriever.core.document_mrr` across the
arms shown — how much downstream movement each point of upstream movement bought. It
is printed once, under the table, because it is a property of the comparison rather
than of any single arm.

A saturation warning fires when every arm scores above 0.9 on answer_match. With the
extractive generator that metric is substring containment over the selected chunks, so
it can top out and stop discriminating; R7 pre-registers a rerun at max_selected=1 for
that case rather than reading saturation as successful transfer.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tomllib
from collections.abc import Sequence

METRICS = (
    ("MRR", "retriever.core.document_mrr"),
    ("Recall", "retriever.core.document_recall"),
    ("selRecall", "selector.core.conditional_document_recall"),
    ("selPrec", "selector.core.document_precision"),
    ("sysRecall", "system.core.final_document_recall"),
    ("answer", "system.core.answer_match"),
    ("citePrec", "system.core.cited_document_precision"),
)
MRR_KEY = "retriever.core.document_mrr"
ANSWER_KEY = "system.core.answer_match"
SATURATION = 0.9


def _resolve_output(config_path: pathlib.Path) -> pathlib.Path:
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return (config_path.parent / raw["output"]["directory"]).resolve()


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _row(config_path: pathlib.Path) -> dict[str, object]:
    output_dir = _resolve_output(config_path)
    report_path = output_dir / "evaluation_report.json"
    row: dict[str, object] = {
        "config": config_path.name,
        "output": str(output_dir),
        "found": report_path.is_file(),
        "metrics": {},
        "n_scored": None,
        "n_total": None,
    }
    if not report_path.is_file():
        return row
    aggregate = json.loads(report_path.read_text(encoding="utf-8")).get("aggregate", {})
    metrics: dict[str, float | None] = {}
    for _, key in METRICS:
        entry = aggregate.get(key) or {}
        metrics[key] = entry.get("mean")
        # n_scored differs per metric once a stage abstains, so report the retriever's,
        # which is the only one every arm always scores.
        if key == MRR_KEY and entry:
            row["n_scored"] = entry.get("n_scored")
            row["n_total"] = entry.get("n_total")
    row["metrics"] = metrics
    return row


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=pathlib.Path)
    parser.add_argument("--json", action="store_true", help="also print a JSON blob")
    args = parser.parse_args(argv)

    rows = [_row(config) for config in args.configs]

    name_width = max((len(str(r["config"])) for r in rows), default=6)
    header = f"{'config':<{name_width}}  " + "  ".join(f"{label:>9}" for label, _ in METRICS)
    header += "   n_scored/n_total"
    print(header)
    print("-" * len(header))
    for r in rows:
        if not r["found"]:
            print(f"{str(r['config']):<{name_width}}  (no evaluation_report.json at {r['output']})")
            continue
        metrics = r["metrics"]
        cells = "  ".join(f"{_fmt(metrics[key]):>9}" for _, key in METRICS)
        print(f"{str(r['config']):<{name_width}}  {cells}   {r['n_scored']}/{r['n_total']}")

    scored = [r["metrics"] for r in rows if r["found"]]
    mrrs = [m[MRR_KEY] for m in scored if m.get(MRR_KEY) is not None]
    answers = [m[ANSWER_KEY] for m in scored if m.get(ANSWER_KEY) is not None]

    print()
    if len(mrrs) >= 2 and len(answers) >= 2:
        mrr_span = max(mrrs) - min(mrrs)
        answer_span = max(answers) - min(answers)
        print(f"MRR span    : {min(mrrs):.4f} -> {max(mrrs):.4f}  (spread {mrr_span:+.4f})")
        print(f"answer span : {min(answers):.4f} -> {max(answers):.4f}  (spread {answer_span:+.4f})")
        if mrr_span > 0:
            print(f"transfer    : {answer_span / mrr_span:.3f} downstream points per upstream point")
    else:
        print("transfer    : needs at least two scored arms")

    if answers and min(answers) > SATURATION:
        print()
        print(f"WARNING: every arm scores above {SATURATION} on {ANSWER_KEY}. The metric has "
              "saturated and cannot discriminate; per R7 this is 'did not resolve', not "
              "'transfer is good'. Rerun at max_selected=1.")

    if args.json:
        print()
        print(json.dumps(rows, indent=2, sort_keys=True))

    missing = [r["config"] for r in rows if not r["found"]]
    if missing:
        print(f"\nWARNING: {len(missing)} report(s) missing: {', '.join(map(str, missing))}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
