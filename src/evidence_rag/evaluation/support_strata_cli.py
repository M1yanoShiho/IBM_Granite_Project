"""CLI: stratify gate reach by gold support, from an existing E1 dump (CPU only).

Reports all four combinations of vote unit (document / parent) and answer equivalence
(exact / lenient), because the two questions interact: the parent unit is what makes the
support counts honest, and the lenient equivalence is what stops a correct-but-reworded answer
from being counted as a different claim.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.evaluation.cluster_rescore import Equivalence, read_dump
from evidence_rag.evaluation.support_strata import analyse, summarize_strata
from evidence_rag.materializer.source_parent import read_parent_index
from evidence_rag.selector.answer_equivalence import lenient_equivalent

SCORINGS: dict[str, Equivalence] = {"exact": None, "lenient": lenient_equivalent}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate reach stratified by gold support")
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--parent-index",
        type=Path,
        help="SAME_SOURCE sidecar; without it the parent columns repeat the document columns",
    )
    parser.add_argument("--margin", type=int, default=2)
    parser.add_argument("--support-cap", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    rows = read_dump(arguments.dump)
    parents = (
        read_parent_index(arguments.parent_index).parent_by_document
        if arguments.parent_index is not None
        else None
    )

    payload: dict[str, object] = {
        "n_rows": len(rows),
        "margin": arguments.margin,
        "support_cap": arguments.support_cap,
        "parent_index": str(arguments.parent_index) if arguments.parent_index else None,
    }
    for unit, mapping in (("document", None), ("parent", parents)):
        payload[unit] = {
            name: summarize_strata(
                analyse(
                    rows,
                    equivalence=equivalence,
                    parent_by_document=mapping,
                    margin=arguments.margin,
                    support_cap=arguments.support_cap,
                )
            )
            for name, equivalence in SCORINGS.items()
        }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
