"""CLI: freeze or verify leakage-safe Selector component and role artifacts.

Example::

    python -m evidence_rag.cli.build_selector_components \
      --dataset-kind niah --source-split dev \
      --dataset-manifest runs/niah-injected/manifest.json \
      --source-parent runs/niah-injected/source_parent.jsonl \
      --candidate-pool runs/selector-beam-v1/pools/niah-dev \
      --niah-assignment runs/selector-beam-v1/data/niah/niah_dev_assignments.jsonl \
      --output-dir runs/selector-adaptive-risk-v1/components/niah-dev
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from evidence_rag.evaluation.selector_components import (
    MANIFEST_FILE,
    REPORT_FILE,
    DatasetKind,
    SourceSplit,
    build_selector_component_artifacts,
    freeze_selector_component_artifacts,
    verify_selector_component_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze deterministic Selector components, derived roles and representatives"
    )
    parser.add_argument("--dataset-kind", required=True, choices=("niah", "2wiki"))
    parser.add_argument("--source-split", required=True, choices=("train", "dev"))
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--source-parent", required=True, type=Path)
    parser.add_argument(
        "--candidate-pool",
        required=True,
        type=Path,
        help="frozen pool directory containing candidate_sets.jsonl and its v2 manifest",
    )
    parser.add_argument(
        "--niah-assignment",
        type=Path,
        help="required only for NIAH; the frozen leakage-filtered assignment JSONL",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="recompute from every input and require byte-identical existing outputs",
    )
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    dataset_kind = cast(DatasetKind, arguments.dataset_kind)
    source_split = cast(SourceSplit, arguments.source_split)
    if dataset_kind == "niah" and arguments.niah_assignment is None:
        parser.error("--niah-assignment is required for --dataset-kind niah")
    if dataset_kind == "2wiki" and arguments.niah_assignment is not None:
        parser.error("--niah-assignment is not allowed for --dataset-kind 2wiki")
    if not arguments.candidate_pool.is_dir():
        parser.error("--candidate-pool must be a frozen Selector-v2 pool directory")
    pool_manifest = arguments.candidate_pool / "selector_candidate_pool_manifest_v2.json"
    if not pool_manifest.is_file():
        parser.error(f"--candidate-pool is missing its v2 manifest: {pool_manifest}")

    if arguments.verify_only:
        artifacts = verify_selector_component_artifacts(
            output_directory=arguments.output_dir,
            dataset_kind=dataset_kind,
            source_split=source_split,
            dataset_manifest_path=arguments.dataset_manifest,
            source_parent_path=arguments.source_parent,
            candidate_pool_path=arguments.candidate_pool,
            assignment_path=arguments.niah_assignment,
        )
        action = "verified"
    else:
        artifacts = build_selector_component_artifacts(
            dataset_kind=dataset_kind,
            source_split=source_split,
            dataset_manifest_path=arguments.dataset_manifest,
            source_parent_path=arguments.source_parent,
            candidate_pool_path=arguments.candidate_pool,
            assignment_path=arguments.niah_assignment,
        )
        freeze_selector_component_artifacts(arguments.output_dir, artifacts)
        action = "frozen"

    manifest_path = arguments.output_dir / MANIFEST_FILE
    report_path = arguments.output_dir / REPORT_FILE
    print(
        json.dumps(
            {
                "action": action,
                "output_dir": str(arguments.output_dir),
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "report": str(report_path),
                "report_sha256": _sha256(report_path),
                "dataset_signature": artifacts.manifest["dataset_signature"],
                "counts": artifacts.report["counts"],
                "crossing": artifacts.report["crossing"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
