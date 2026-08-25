"""CLI: pin the sealed-600 candidate pool into the freeze record (M0 §4).

    export PYTHONPATH=src
    python -m evidence_rag.cli.pin_candidates \
      --sealed runs/niah-sealed600 \
      --candidates runs/sealed600-bm25/candidate_sets.jsonl \
      --top-n 20

Run this ONCE, immediately after the frozen retriever has produced the pool and before any
experiment reads it. It writes `candidate_freeze.json` into the sealed directory: the retriever
that built the pool, the pool's sha256, its window count and its top-n. Gate 0A's
`candidate_windows` check then has something to verify against, and cannot reach PASS without it.

WHY THIS IS A SEPARATE STEP AND NOT PART OF THE BUILD. The sealed manifest is frozen before any
retrieval runs — that is the point of it — and a manifest that could be rewritten afterwards is
not a freeze. The pool does not exist at that moment, so its hash cannot be in there. This is the
half of the freeze that can only be written later, and it is write-once for the same reason the
manifest is.

WHAT IT REFUSES. A pool that names no retriever, a pool that names more than one, a pool from any
retriever other than the frozen one, and a pool that is not over exactly the sealed queries. None
of those can be fixed by writing the pin: the id has to come from the run that produced the pool,
not from the operator pinning it afterwards, or the record is an assumption wearing a
measurement's clothes.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.contracts.models import CandidateSet
from evidence_rag.materializer.sealed600 import (
    FROZEN_RETRIEVER,
    PROTOCOL_VERSION,
    SEALED_MANIFEST_FILE,
    CandidateFreeze,
    freeze_candidate_pin,
    read_sealed_manifest,
    sha256_file,
    sole_retriever,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the sealed-600 candidate pool's identity (M0 §4)"
    )
    parser.add_argument("--sealed", required=True, type=Path)
    parser.add_argument(
        "--candidates",
        required=True,
        type=Path,
        help="candidate_sets.jsonl produced by the frozen retriever",
    )
    parser.add_argument("--top-n", type=int, default=20)
    return parser


def read_candidate_sets(path: Path) -> tuple[CandidateSet, ...]:
    return tuple(
        CandidateSet.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    manifest = read_sealed_manifest(arguments.sealed)
    candidate_sets = read_candidate_sets(arguments.candidates)

    # Raises on an empty pool, on a pool that names no producer, and on one that names several.
    # Nothing may pin a pool it cannot identify, so every ambiguity is fatal here even though the
    # audit only reports on it.
    retriever = sole_retriever(candidate_sets)
    if retriever.name != FROZEN_RETRIEVER:
        raise ValueError(
            f"this pool was built by {retriever.name!r}, and M0 §4 freezes the retriever to "
            f"{FROZEN_RETRIEVER!r}: Graph 2.0's claim is conditional on a fixed candidate pool, "
            "and §3.5's G-FC baseline and §5.2's recall reference were both measured on the "
            f"{FROZEN_RETRIEVER} pool. A {retriever.name} arm is a separate generalisation table "
            "(§4) and does not enter the C1 judgement, so it is not pinned here."
        )

    sealed_queries = set(manifest.query_ids)
    found = {candidate_set.query_id for candidate_set in candidate_sets}
    if found != sealed_queries:
        missing = sorted(sealed_queries - found)[:5]
        extra = sorted(found - sealed_queries)[:5]
        raise ValueError(
            f"this pool is not over the sealed set: {len(sealed_queries - found)} sealed queries "
            f"have no window (e.g. {missing}) and {len(found - sealed_queries)} windows belong to "
            f"queries outside it (e.g. {extra})."
        )
    wrong = sorted(
        candidate_set.query_id
        for candidate_set in candidate_sets
        if len(candidate_set.candidates) != arguments.top_n
    )
    if wrong:
        raise ValueError(
            f"{len(wrong)} queries do not have exactly {arguments.top_n} candidates (e.g. "
            f"{wrong[:5]}). A short window changes that query's own recall and harmful "
            "denominators and reads as an ordinary data point downstream (M0 §6 item 3)."
        )

    pin = CandidateFreeze(
        protocol_version=PROTOCOL_VERSION,
        sealed_manifest_sha256=sha256_file(arguments.sealed / SEALED_MANIFEST_FILE),
        candidate_file=str(arguments.candidates),
        candidate_sha256=sha256_file(arguments.candidates),
        n_windows=len(candidate_sets),
        top_n=arguments.top_n,
        retriever=retriever,
    )
    path = freeze_candidate_pin(arguments.sealed, pin)
    print(
        json.dumps(
            {
                "pin": str(path),
                "retriever": retriever.name,
                "implementation_version": retriever.implementation_version,
                "candidate_sha256": pin.candidate_sha256,
                "n_windows": pin.n_windows,
                "top_n": pin.top_n,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
