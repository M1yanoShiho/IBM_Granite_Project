import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.infrastructure.benchmarks import BenchmarkProvider, materialize_beir_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize a BEIR benchmark into evidence-rag JSONL files"
    )
    parser.add_argument("name", help="BEIR dataset name, for example scifact")
    parser.add_argument("--split", default="test", help="dataset split (default: test)")
    parser.add_argument("--output", required=True, type=Path, help="output dataset directory")
    parser.add_argument("--query-limit", type=int, help="keep the first N sorted query IDs")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    provider: BenchmarkProvider | None = None,
) -> int:
    arguments = _parser().parse_args(argv)
    summary = materialize_beir_dataset(
        name=arguments.name,
        split=arguments.split,
        output_directory=arguments.output,
        query_limit=arguments.query_limit,
        provider=provider,
    )
    print(
        json.dumps(
            {
                "dataset_id": summary.dataset_id,
                "document_count": summary.document_count,
                "gold_case_count": summary.gold_case_count,
                "manifest": str(summary.manifest_path),
                "query_count": summary.query_count,
                "split": summary.split,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
