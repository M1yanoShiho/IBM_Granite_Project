"""CLI: how often do two passages of the same Wikipedia article share a retrieval window?

Background quantity for spec §3.4. It reports how much the document -> parent support fix
matters; it is deliberately NOT a decision gate. Counting document_ids contradicts the stated
definition of independent support regardless of the rate, so the fix is adopted unconditionally
and this number only says how large its effect is.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.contracts.models import CandidateSet
from evidence_rag.materializer.source_parent import read_parent_index


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Same-parent collision rate in retrieval windows")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--parent-index", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.top_n <= 0:
        raise ValueError("--top-n must be positive")
    index = read_parent_index(arguments.parent_index)

    n_queries = 0
    n_with_collision = 0
    total_documents = 0
    total_parents = 0
    n_unresolved = 0
    for line in arguments.candidates.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        candidate_set = CandidateSet.model_validate_json(line)
        window = sorted(candidate_set.candidates, key=lambda item: item.retrieval_rank)[
            : arguments.top_n
        ]
        document_ids = {candidate.document_id for candidate in window}
        parents = {index.parent_of(document_id) for document_id in document_ids}
        n_queries += 1
        total_documents += len(document_ids)
        total_parents += len(parents)
        n_unresolved += index.n_unresolved(document_ids)
        if len(parents) < len(document_ids):
            n_with_collision += 1

    report = {
        "n_queries": n_queries,
        "n_queries_with_collision": n_with_collision,
        "collision_rate": (n_with_collision / n_queries) if n_queries else None,
        "mean_documents_per_window": (total_documents / n_queries) if n_queries else None,
        "mean_parents_per_window": (total_parents / n_queries) if n_queries else None,
        "n_unresolved_documents": n_unresolved,
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
