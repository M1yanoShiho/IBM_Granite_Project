"""Rebuild Experiment 04 public tables from the frozen aggregate result."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.evaluation.public_tables import rebuild_experiment04_tables

ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-results",
        type=Path,
        default=ROOT / "results/experiment04/final_results.json",
        help="frozen experiment04.final_results.v1 aggregate",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    paths = rebuild_experiment04_tables(arguments.final_results, arguments.output_dir)
    print(json.dumps([str(path) for path in paths], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
