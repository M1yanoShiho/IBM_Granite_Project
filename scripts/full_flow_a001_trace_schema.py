"""Materialise the frozen A001 per-query Generator trace JSON schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidence_rag.generator.trace import GeneratorTrace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(GeneratorTrace.model_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
