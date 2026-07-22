"""CLI: materialize the dpr-w100 NQ base dataset (spec §6).

Production loads the dataset through ir_datasets; a provider can be injected for tests.
"""

import argparse
import importlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from evidence_rag.materializer.base_loader import BaseDataset, materialize_niah_base


class BaseProvider(Protocol):
    version: str

    def load(self, dataset_id: str) -> BaseDataset: ...


class _IrDatasetsProvider:
    def __init__(self) -> None:
        self._module = importlib.import_module("ir_datasets")
        self.version = "ir-datasets"

    def load(self, dataset_id: str) -> BaseDataset:
        dataset: BaseDataset = self._module.load(dataset_id)
        return dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize the dpr-w100 NQ base dataset")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--corpus-size", type=int, default=100000)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None, *, provider: BaseProvider | None = None) -> int:
    arguments = _parser().parse_args(argv)
    active = provider if provider is not None else _IrDatasetsProvider()
    dataset = active.load(f"dpr-w100/natural-questions/{arguments.split}")
    result = materialize_niah_base(
        dataset,
        arguments.output,
        corpus_size=arguments.corpus_size,
        query_limit=arguments.query_limit,
        seed=arguments.seed,
    )
    print(
        json.dumps(
            {
                "manifest": str(result.manifest_path),
                "documents": result.document_count,
                "queries": result.query_count,
                "gold_cases": result.gold_case_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
