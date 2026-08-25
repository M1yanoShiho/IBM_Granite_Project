"""CLI: freeze one exact-recovery Hybrid-RRF Top-20 pool for Selector-v2.

Example::

    python -m evidence_rag.cli.pin_selector_pool \
      --pool-dir runs/selector-beam-v1/pools/niah-train \
      --dataset-manifest runs/niah-train-injected/manifest.json \
      --data-role niah-train \
      --recovery-status exact-recovery

This command does not touch the sealed600 BM25 ``candidate_freeze.json``.  The new
write-once manifest is placed beside the recovered Hybrid ``candidate_sets.jsonl``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from evidence_rag.materializer.selector_pool import (
    DataRole,
    RecoveryStatus,
    build_selector_candidate_pool_manifest_v2,
    freeze_selector_candidate_pool_manifest_v2,
    verify_selector_candidate_pool_v2,
)

DATA_ROLES: tuple[DataRole, ...] = (
    "niah-train",
    "niah-dev",
    "niah-sealed600",
    "2wiki-train",
    "2wiki-dev",
    "2wiki-heldout",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze an exact-recovery Selector-v2 Hybrid-RRF Top-20 candidate pool"
    )
    parser.add_argument("--pool-dir", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--data-role", choices=DATA_ROLES)
    parser.add_argument(
        "--recovery-status",
        choices=("exact-recovery", "new-v2-run"),
        help="state whether bytes are one of the six audited recoveries or a newly frozen v2 run",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="re-verify an existing manifest and every dependency without writing anything",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.verify_only:
        if arguments.data_role is not None or arguments.recovery_status is not None:
            _parser().error("--verify-only reads role/status from the frozen manifest; do not pass them")
        manifest = verify_selector_candidate_pool_v2(
            arguments.pool_dir, arguments.dataset_manifest
        )
        print(
            json.dumps(
                {
                    "verified": str(arguments.pool_dir),
                    "data_role": manifest.data_role,
                    "recovery_status": manifest.recovery_status,
                    "n_queries": manifest.n_queries,
                    "candidate_sha256": manifest.candidate_sha256,
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.data_role is None or arguments.recovery_status is None:
        _parser().error("--data-role and --recovery-status are required unless --verify-only is used")
    manifest = build_selector_candidate_pool_manifest_v2(
        arguments.pool_dir,
        arguments.dataset_manifest,
        data_role=cast(DataRole, arguments.data_role),
        recovery_status=cast(RecoveryStatus, arguments.recovery_status),
    )
    path = freeze_selector_candidate_pool_manifest_v2(arguments.pool_dir, manifest)
    print(
        json.dumps(
            {
                "manifest": str(path),
                "data_role": manifest.data_role,
                "recovery_status": manifest.recovery_status,
                "n_queries": manifest.n_queries,
                "top_n": manifest.top_n,
                "retriever": manifest.retriever.model_dump(mode="json"),
                "candidate_sha256": manifest.candidate_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
