"""CLI: re-classify the needle-probe `wrong` rows into matching-artifact tiers (CPU, instant).

Reads the probe dump jsonl, keeps the visible WRONG rows by default, and reports how many are
recoverable by a better answer-matcher (equal-after-norm / containment) versus genuinely
DIFFERENT (real QA error or synonym). Prints a sample of the DIFFERENT residual so the true
floor can be eyeballed. No GPU -- runs on the login node.
"""

import argparse
import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.evaluation.wrong_reclassify import DIFFERENT, reclassify, summarize_reclass


def _read_dump(path: Path) -> list[dict[str, object]]:
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-classify needle-probe WRONG rows")
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--all", action="store_true", help="include non-visible WRONG rows too")
    parser.add_argument("--examples", type=int, default=15)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    rows = _read_dump(arguments.dump)
    wrong = [
        row
        for row in rows
        if row.get("outcome") == "wrong" and (arguments.all or row.get("visible") is True)
    ]
    summary = summarize_reclass(wrong)
    payload = {"only_visible": not arguments.all, **dataclasses.asdict(summary)}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))

    different = [
        row for row in wrong if reclassify(str(row["extracted"]), str(row["gold_value"])) == DIFFERENT
    ]
    print(f"=== DIFFERENT residual: {len(different)} rows (first {arguments.examples}) ===")
    for row in different[: arguments.examples]:
        print(
            json.dumps(
                {
                    "question": row.get("question"),
                    "gold_value": row.get("gold_value"),
                    "extracted": row.get("extracted"),
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
