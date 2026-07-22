"""CLI: materialize a counterfactual-injected dataset from a base manifest (spec §9)."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.materializer.answer_bank import build_answer_bank
from evidence_rag.materializer.injector import (
    materialize_counterfactuals,
    write_injected_dataset,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inject counterfactual twins into a base dataset")
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    bundle = JsonlDatasetAdapter.load(arguments.base_manifest)
    answers = [
        answer for gold_case in bundle.gold_cases for answer in (gold_case.reference_answers or ())
    ]
    bank = build_answer_bank(answers, seed=arguments.seed)
    result = materialize_counterfactuals(bundle, bank, seed=arguments.seed)
    write_injected_dataset(bundle, result, arguments.output, seed=arguments.seed)
    print(
        json.dumps(
            {
                "manifest": str(arguments.output / "manifest.json"),
                "injected": len(result.records),
                "skipped": len(result.skipped_query_ids),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
