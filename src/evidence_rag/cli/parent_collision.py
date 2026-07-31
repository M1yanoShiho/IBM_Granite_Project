"""CLI: how often do two passages of the same Wikipedia article share a retrieval window?

Background quantity for spec §3.4. It reports how much the document -> parent support fix
matters; it is deliberately NOT a decision gate. Counting document_ids contradicts the stated
definition of independent support regardless of the rate, so the fix is adopted unconditionally
and this number only says how large its effect is.

The window-wide collision rate is an average and does not say whether the GOLD cluster in
particular was inflated — which is what actually determines whether the gate's margin was
propped up by one article counted several times. With `--provenance`, the needle-parent
diagnostics answer that directly: `needle_parent_inflation_rate` is the share of injected
queries whose window contains a second passage of the needle's own article. The counterfactual
twin is excluded, since it is a copy of the needle's passage and shares its parent by
construction; `cf_shares_needle_parent` is reported separately as a sanity check on that
construction rather than as evidence about the corpus.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.contracts.models import CandidateSet
from evidence_rag.materializer.provenance import read_provenance
from evidence_rag.materializer.source_parent import ParentIndex, read_parent_index


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Same-parent collision rate in retrieval windows")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--parent-index", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--provenance",
        type=Path,
        help="mutation log; enables the needle-parent diagnostics (see module docstring)",
    )
    return parser


def _needle_parent_inflated(
    window: Sequence[str],
    index: ParentIndex,
    *,
    needle_document_id: str,
    counterfactual_document_id: str,
) -> bool | None:
    """Did the needle's own article contribute a SECOND passage to this window?

    That is what inflates the gold cluster's support, and the window-wide collision rate cannot
    show it. The counterfactual twin is excluded because it is a copy of the needle's passage and
    therefore shares its parent by construction — counting it would report injector bookkeeping
    as corpus structure. None when the needle is not in the window.
    """
    if needle_document_id not in window:
        return None
    needle_parent = index.parent_of(needle_document_id)
    return any(
        document_id not in (needle_document_id, counterfactual_document_id)
        and index.parent_of(document_id) == needle_parent
        for document_id in window
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.top_n <= 0:
        raise ValueError("--top-n must be positive")
    index = read_parent_index(arguments.parent_index)
    records = (
        {record.query_id: record for record in read_provenance(arguments.provenance)}
        if arguments.provenance is not None
        else {}
    )

    n_queries = 0
    n_with_collision = 0
    total_documents = 0
    total_parents = 0
    n_unresolved = 0
    n_injected_scored = 0
    n_needle_inflated = 0
    n_cf_shares_parent = 0
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

        record = records.get(candidate_set.query_id)
        if record is not None:
            inflated = _needle_parent_inflated(
                sorted(document_ids),
                index,
                needle_document_id=record.needle_document_id,
                counterfactual_document_id=record.counterfactual_document_id,
            )
            if inflated is not None:
                n_injected_scored += 1
                n_needle_inflated += int(inflated)
                if index.parent_of(record.counterfactual_document_id) == index.parent_of(
                    record.needle_document_id
                ):
                    n_cf_shares_parent += 1

    report: dict[str, object] = {
        "n_queries": n_queries,
        "n_queries_with_collision": n_with_collision,
        "collision_rate": (n_with_collision / n_queries) if n_queries else None,
        "mean_documents_per_window": (total_documents / n_queries) if n_queries else None,
        "mean_parents_per_window": (total_parents / n_queries) if n_queries else None,
        "n_unresolved_documents": n_unresolved,
    }
    if arguments.provenance is not None:
        report.update(
            {
                "n_injected_scored": n_injected_scored,
                "needle_parent_inflated": n_needle_inflated,
                "needle_parent_inflation_rate": (
                    (n_needle_inflated / n_injected_scored) if n_injected_scored else None
                ),
                "cf_shares_needle_parent": n_cf_shares_parent,
            }
        )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
