"""CLI: build the document -> source_parent_id sidecar from a materialized documents.jsonl.

Prerequisite for two separate things (spec §3.4): counting independent support by source parent
rather than by chunk, and the parent-page axis of the sealed-600 leakage audit. The second holds
even if the first is not adopted, so this step is never optional.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.materializer.source_parent import parse_parent, write_parent_index


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the source_parent_id sidecar")
    parser.add_argument("--documents", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    parent_by_document: dict[str, str] = {}
    n_documents = 0
    for line in arguments.documents.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n_documents += 1
        row = json.loads(line)
        parent = parse_parent(row["text"])
        if parent is not None:
            parent_by_document[row["document_id"]] = parent

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_parent_index(arguments.output, parent_by_document)
    report = {
        "n_documents": n_documents,
        "n_resolved": len(parent_by_document),
        "n_unresolved": n_documents - len(parent_by_document),
        "n_parents": len(set(parent_by_document.values())),
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
