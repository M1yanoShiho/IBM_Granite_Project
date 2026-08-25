#!/usr/bin/env python3
"""Summarise retriever_report.json across several experiment configs into one table.

Usage:
    python scripts/summarize_retriever_eval.py CONFIG.toml [CONFIG.toml ...]

For each config it resolves ``[output] directory`` (relative to the config file, the
same rule the experiment loader uses), reads ``retriever_report.json`` from there, and
prints MRR / Recall@{5,10,20} / Recall from the ``aggregate`` block. Missing reports are
reported as ``-`` so a partial run still yields a table. Output is plain text for the
Slurm ``.out`` log; add ``--json`` to also emit a machine-readable blob.
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
    ("R@5", "retriever.core.document_recall_at_5"),
    ("R@10", "retriever.core.document_recall_at_10"),
    ("R@20", "retriever.core.document_recall_at_20"),
    ("Recall", "retriever.core.document_recall"),
)


def _resolve_output(config_path: pathlib.Path) -> pathlib.Path:
    """Resolve [output] directory relative to the config file (loader semantics)."""

    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return (config_path.parent / raw["output"]["directory"]).resolve()


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _row(config_path: pathlib.Path) -> dict[str, object]:
    output_dir = _resolve_output(config_path)
    report_path = output_dir / "retriever_report.json"
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
        if row["n_scored"] is None and entry:
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
    header = f"{'config':<{name_width}}  " + "  ".join(f"{label:>7}" for label, _ in METRICS)
    header += "   n_scored/n_total"
    print(header)
    print("-" * len(header))
    for r in rows:
        if not r["found"]:
            print(f"{str(r['config']):<{name_width}}  (no retriever_report.json at {r['output']})")
            continue
        metrics = r["metrics"]
        cells = "  ".join(f"{_fmt(metrics[key]):>7}" for _, key in METRICS)
        counts = f"{r['n_scored']}/{r['n_total']}"
        print(f"{str(r['config']):<{name_width}}  {cells}   {counts}")

    if args.json:
        print()
        print(json.dumps(rows, indent=2, sort_keys=True))

    missing = [r["config"] for r in rows if not r["found"]]
    if missing:
        print(f"\nWARNING: {len(missing)} report(s) missing: {', '.join(map(str, missing))}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
