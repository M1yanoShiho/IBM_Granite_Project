"""CLI: materialise the Gate 0B-2 task probe from a NIAH manifest + mutation log.

Point this at the TRAIN split. Building the probe from dev or the sealed test would put the
acceptance set and the evaluation set on the same queries, which is the thing the whole
pre-registration protocol exists to prevent.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.materializer.provenance import read_provenance
from evidence_rag.relations.task_probe import build_probe_pairs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the Gate 0B-2 task probe")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    bundle = JsonlDatasetAdapter.load(arguments.manifest)
    records = read_provenance(arguments.provenance)
    pairs = build_probe_pairs(
        records=records,
        question_by_query={query.query_id: query.text for query in bundle.queries},
        text_by_document={document.document_id: document.text for document in bundle.documents},
    )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        "".join(
            json.dumps(
                {
                    "premise": pair.premise,
                    "hypothesis": pair.hypothesis,
                    "label": pair.label.value,
                    "group": pair.group,
                    "kind": pair.kind,
                    "query_id": pair.query_id,
                },
                sort_keys=True,
            )
            + "\n"
            for pair in pairs
        ),
        encoding="utf-8",
    )
    report = {
        "n_records": len(records),
        "n_pairs": len(pairs),
        "n_skipped_records": len(records) - len(pairs) // 4,
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
