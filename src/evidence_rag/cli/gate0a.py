"""CLI: run the Gate 0A audit over a frozen sealed set (M0 §6).

    export PYTHONPATH=src
    python -m evidence_rag.cli.gate0a \
      --sealed runs/niah-sealed600 \
      --existing-split runs/niah-injected \
      --existing-split runs/niah-train-injected \
      --protocol-doc docs/selector/M0_PROTOCOL_FREEZE.md \
      --no-candidates --utility-labels-absent \
      --output results/sealed600/gate0a.json

Exit code is 0 only for PASS. A freshly built set audits as INCOMPLETE and exits 1, because §6
item 3 needs the Top-20 windows and retrieval has not run yet — that is the honest reading, and
`--no-candidates` has to be typed so the gap is a stated fact rather than an omission. Re-run
with `--candidates runs/sealed600-bm25/candidate_sets.jsonl` once the frozen retriever has run.

`--utility-labels-absent` is likewise mandatory rather than implied. §6 item 3 also names
`utility range` and `derived flags`, which came from the v1 LightGBM selector; under D1=A the
selector has no learned parameters and NOTHING in this repository produces a `utility_grade`.
The audit records that as `not_applicable` with the reason attached — it does not quietly drop
the item, and it does not invent a label schema to have something to check.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.contracts.models import CandidateSet
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.materializer.gate0a import (
    audit_axes,
    audit_candidates,
    audit_counterfactuals,
    audit_label_provenance,
    audit_manifest_freeze,
    audit_size_derivation,
    gate0a_report,
    not_applicable,
    render,
    unevaluated,
)
from evidence_rag.materializer.provenance import read_provenance
from evidence_rag.materializer.sealed600 import (
    fingerprint_bundle,
    read_sealed_manifest,
    read_split_fingerprint,
    sha256_file,
)

UTILITY_REASON = (
    "no producer under D1=A: the selector has no learned parameters, so nothing in this "
    "repository emits utility_grade or its derived flags (M0 §1 D1; the item is inherited from "
    "TRAINING_PLAN §5's LightGBM-era Gate 0A)"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate 0A audit of a frozen sealed set")
    parser.add_argument("--sealed", required=True, type=Path)
    parser.add_argument(
        "--existing-split", required=True, action="append", type=Path, dest="existing_splits"
    )
    parser.add_argument("--protocol-doc", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--candidates", type=Path, help="candidate_sets.jsonl from the frozen retriever")
    group.add_argument(
        "--no-candidates",
        action="store_true",
        help="state that retrieval has not run; the verdict becomes INCOMPLETE",
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--utility-labels-absent",
        action="store_true",
        help="acknowledge that utility_grade has no producer under D1=A (required)",
    )
    parser.add_argument("--output", type=Path)
    return parser


def _read_candidates(path: Path) -> tuple[CandidateSet, ...]:
    return tuple(
        CandidateSet.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if not arguments.utility_labels_absent:
        # Not a rhetorical flag. Leaving the item silent is how a checklist rots, and the
        # operator stating it is the only thing separating "no producer exists" from "nobody
        # checked".
        parser.error(f"--utility-labels-absent is required: {UTILITY_REASON}")

    manifest = read_sealed_manifest(arguments.sealed)
    bundle = JsonlDatasetAdapter.load(arguments.sealed / "manifest.json")
    records = read_provenance(arguments.sealed / "provenance.jsonl")
    existing = tuple(read_split_fingerprint(path) for path in arguments.existing_splits)

    checks = [
        audit_size_derivation(manifest),
        audit_manifest_freeze(arguments.sealed, manifest, sha256_file(arguments.protocol_doc)),
        audit_label_provenance(manifest, bundle, records),
        audit_counterfactuals(bundle, records),
        not_applicable("utility_range_and_derived_flags", UTILITY_REASON),
    ]
    if arguments.candidates is not None:
        checks.append(
            audit_candidates(
                bundle, records, _read_candidates(arguments.candidates), top_n=arguments.top_n
            )
        )
    else:
        checks.append(
            unevaluated(
                "candidate_windows",
                "retrieval has not run: M0 §6 item 3 cannot be evaluated without the frozen "
                "Top-20 windows, so this run is INCOMPLETE rather than PASS",
            )
        )

    report = gate0a_report(
        axes=audit_axes(fingerprint_bundle("sealed", bundle, records), existing), checks=checks
    )
    rendered = render(report)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print(json.dumps({"verdict": report.verdict}, sort_keys=True))
    return 0 if report.passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
