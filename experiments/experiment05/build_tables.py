"""Rebuild Experiment 05 public tables from audited aggregate results."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.evaluation.public_tables import rebuild_experiment05_tables

ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results/experiment05",
        help="directory containing table1.json, table2.json, and claim_labels.json",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    paths = rebuild_experiment05_tables(arguments.results_dir, arguments.output_dir)
    print(json.dumps([str(path) for path in paths], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
