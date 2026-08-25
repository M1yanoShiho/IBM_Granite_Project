#!/usr/bin/env python3
"""Summarise evaluation_report.json across pipeline configs, upstream beside downstream.

Usage:
    python scripts/summarize_pipeline_eval.py CONFIG.toml [CONFIG.toml ...]

R7 asks whether a retrieval gain reaches the generator, so the table has to show both
ends of the chain on one line; reading two separate tables is how a non-transfer gets
missed. Columns run left to right in pipeline order: retriever, then selector, then
system.

Under the table it reports how much downstream movement each point of upstream movement
bought — but *segment by segment*, between adjacent arms sorted by MRR, not as one ratio
over the whole span. R7 measured slopes of 0.416, 0.719 and 1.728 across its three
segments, so a single span-endpoint ratio (0.506 there) averages away a fourfold
difference and reads as a constant that does not exist. The earlier version of this
script printed exactly that, and the R7 entry records it as a defect of this tool.

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


def _varied_sections(config_paths: list[pathlib.Path]) -> set[str]:
    """Which config sections actually differ across the arms being compared.

    The transfer slopes below only mean anything when the retriever is what varies.
    R9 swept chunk size instead, and the slopes printed −26.5 and −6.5 — arithmetically
    fine, causally empty, and looking exactly like findings. Worse, changing the chunker
    changes how much text reaches the generator, so the downstream metric is not
    comparable across those arms at all. Reporting which sections vary lets the caller
    be told that rather than left to remember it.
    """

    seen: dict[str, set[str]] = {}
    for path in config_paths:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        for section in ("retriever", "selector", "generator", "chunker", "run"):
            rendered = json.dumps(raw.get(section, {}), sort_keys=True)
            seen.setdefault(section, set()).add(rendered)
    return {section for section, values in seen.items() if len(values) > 1}


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

    paired = [
        (r["config"], r["metrics"][MRR_KEY], r["metrics"][ANSWER_KEY])
        for r in rows
        if r["found"]
        and r["metrics"].get(MRR_KEY) is not None
        and r["metrics"].get(ANSWER_KEY) is not None
    ]
    paired.sort(key=lambda item: item[1])

    varied = _varied_sections(list(args.configs))
    # Bound outside the branch below: the saturation check reads them, and a run where
    # only one arm has finished must still print its table rather than crash.
    mrrs = [m for _, m, _ in paired]
    answers = [a for _, _, a in paired]

    print()
    if len(paired) >= 2:
        print(f"MRR span    : {min(mrrs):.4f} -> {max(mrrs):.4f}  (spread {max(mrrs) - min(mrrs):+.4f})")
        print(f"answer span : {min(answers):.4f} -> {max(answers):.4f}  (spread {max(answers) - min(answers):+.4f})")
    if len(paired) < 2:
        print("transfer    : needs at least two scored arms")
    elif "chunker" in varied or "run" in varied:
        # Both change how much text reaches the generator, so answer_match is not
        # comparable across these arms and a slope against MRR would be meaningless.
        moving = " and ".join(sorted(varied & {"chunker", "run"}))
        print(f"transfer    : not reported. These arms vary [{moving}], which changes how")
        print("              much evidence reaches the generator, so answer_match is not")
        print("              comparable across them. Compare only at a fixed evidence budget.")
    elif "retriever" not in varied:
        print("transfer    : not reported. The retriever is identical across these arms,")
        print("              so there is no upstream retrieval change to attribute anything to.")
    else:
        print("transfer, by segment (downstream points per upstream point):")
        # strict=False is deliberate: paired[1:] is one shorter by construction.
        for (low, mrr_low, ans_low), (high, mrr_high, ans_high) in zip(
            paired, paired[1:], strict=False
        ):
            gap = mrr_high - mrr_low
            segment = f"{low} -> {high}"
            if gap <= 0:
                print(f"  {segment:<58} (arms tie on MRR; slope undefined)")
                continue
            print(f"  {segment:<58} {(ans_high - ans_low) / gap:>7.3f}")
        print("  a single whole-span ratio is not reported: R7 found these slopes to differ")
        print("  fourfold, so one number would read as a constant that does not exist.")

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
