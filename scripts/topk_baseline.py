"""Synthesise the TopK arm from an existing pool, so this line's numbers can be read against
the baseline everyone else quotes.

    export PYTHONPATH=src
    python scripts/topk_baseline.py --candidates runs/e2-gate-off/candidate_sets.jsonl \
      --max-selected 10 --output runs/e2-topk/selected_evidence_sets.jsonl

E2's "gate-off" arm is NOT TopK. Its config runs the corroboration selector with the drop gate
disabled (`name = "corroboration"`, alpha 0.6), so the pool has already been reranked before the
gate acts. S1's -11.2pp therefore credits the gate alone, measured against a baseline that had
already done work -- which is the honest way to isolate the gate, and the wrong number to put
beside anyone else's TopK figure. The Reliability-MIS table quotes TopK harm at 91.77% where
this line's gate-off reads 68.0%; those two cannot both be the same baseline, and comparing them
supports a false conclusion in whichever direction the reader is already leaning.

So this writes the arm that was never run: the first N candidates by retrieval rank, no
rerank, no gate. Same pool, same provenance, same metric, nothing else moving, which is what
makes `harm_cli` and `paired_metric_cli` able to read it beside the other arms.

Deliberately NOT an ArtifactStore run directory. It carries no manifest and no metadata
sidecar, because it is derived from a pool that already has both, and minting a second
provenance record for a view of existing data is exactly the confusion this exists to remove.
It is a diagnostic, and a diagnostic that claimed to be a run would be worse than none.
"""

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evidence_rag.contracts.models import CandidateSet, SelectedEvidenceSet  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument(
        "--max-selected",
        type=int,
        default=10,
        help="must match the arms being compared against; E2's configs freeze it at 10",
    )
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)

    try:
        text = arguments.candidates.read_text(encoding="utf-8")
    except OSError as error:
        raise SystemExit(f"unable to read {arguments.candidates}: {error}") from error
    pools = [
        CandidateSet.model_validate_json(line) for line in text.splitlines() if line.strip()
    ]
    if not pools:
        raise SystemExit(f"{arguments.candidates} holds no candidate sets")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    sizes: list[int] = []
    # newline="\n" so the bytes do not depend on the OS. The same omission in write_index made
    # an index manifest hash differently on Windows and Linux, which cost a day to trace.
    with arguments.output.open("w", encoding="utf-8", newline="\n") as handle:
        for pool in pools:
            # Retrieval rank IS the definition of this arm. Nothing asserts that the file's line
            # order matches it, so sort instead of trusting it -- reading a pool in stored order
            # and calling the result "TopK" is the kind of quiet mismatch that produces a
            # plausible number rather than an error.
            chosen = tuple(sorted(pool.candidates, key=lambda item: item.retrieval_rank))[
                : arguments.max_selected
            ]
            sizes.append(len(chosen))
            handle.write(
                SelectedEvidenceSet(query_id=pool.query_id, evidence=chosen).model_dump_json()
                + "\n"
            )

    print(
        json.dumps(
            {
                "n_queries": len(pools),
                "max_selected": arguments.max_selected,
                "n_selected_min": min(sizes),
                "n_selected_max": max(sizes),
                "short_pools": sum(1 for size in sizes if size < arguments.max_selected),
                "output": str(arguments.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
