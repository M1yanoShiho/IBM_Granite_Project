"""CLI for R003 TopK baselines and the frozen count-matched protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from evidence_rag.evaluation.selector_components import (
    MANIFEST_FILE as COMPONENT_MANIFEST_FILE,
)
from evidence_rag.evaluation.selector_components import DatasetKind, SourceSplit
from evidence_rag.evaluation.selector_controls import (
    COUNT_MATCHED_MANIFEST_FILE,
    COUNT_MATCHED_PROTOCOL_FILE,
    TOPK_MANIFEST_FILE,
    TOPK_REPORT_FILE,
    build_count_matched_protocol_artifacts,
    build_topk_control_artifacts,
    freeze_count_matched_protocol_artifacts,
    freeze_topk_control_artifacts,
    verify_count_matched_protocol_artifacts,
    verify_topk_control_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze or verify R003 TopK and count-matched Selector controls"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    topk = subparsers.add_parser("topk", help="run TopK10/9/8/7 on one verified frozen pool")
    topk.add_argument("--dataset-kind", required=True, choices=("niah", "2wiki"))
    topk.add_argument("--source-split", required=True, choices=("train", "dev"))
    topk.add_argument("--dataset-manifest", required=True, type=Path)
    topk.add_argument("--source-parent", required=True, type=Path)
    topk.add_argument(
        "--candidate-pool",
        required=True,
        type=Path,
        help="frozen pool directory containing candidate_sets.jsonl and its v2 manifest",
    )
    topk.add_argument(
        "--components-dir",
        required=True,
        type=Path,
        help="frozen, verified R002 component bundle for this exact pool and dataset",
    )
    topk.add_argument(
        "--niah-assignment",
        type=Path,
        help="required only for NIAH; frozen leakage-filtered assignment JSONL",
    )
    topk.add_argument("--output-dir", required=True, type=Path)
    topk.add_argument(
        "--verify-only",
        action="store_true",
        help="recompute from all inputs and require byte-identical existing outputs",
    )

    protocol = subparsers.add_parser(
        "freeze-protocol",
        help="freeze the input-free 100-repeat count-matched generator protocol",
    )
    protocol.add_argument("--output-dir", required=True, type=Path)
    protocol.add_argument(
        "--verify-only",
        action="store_true",
        help="recompute and require byte-identical existing protocol outputs",
    )
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_verified_pool(parser: argparse.ArgumentParser, path: Path) -> None:
    if not path.is_dir():
        parser.error("--candidate-pool must be a frozen Selector-v2 pool directory")
    manifest = path / "selector_candidate_pool_manifest_v2.json"
    if not manifest.is_file():
        parser.error(f"--candidate-pool is missing its v2 manifest: {manifest}")


def _require_components(parser: argparse.ArgumentParser, path: Path) -> None:
    if not path.is_dir():
        parser.error("--components-dir must be a frozen R002 component directory")
    manifest = path / COMPONENT_MANIFEST_FILE
    if not manifest.is_file():
        parser.error(f"--components-dir is missing its R002 manifest: {manifest}")


def _run_topk(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> int:
    dataset_kind = cast(DatasetKind, arguments.dataset_kind)
    source_split = cast(SourceSplit, arguments.source_split)
    if dataset_kind == "niah" and arguments.niah_assignment is None:
        parser.error("--niah-assignment is required for --dataset-kind niah")
    if dataset_kind == "2wiki" and arguments.niah_assignment is not None:
        parser.error("--niah-assignment is not allowed for --dataset-kind 2wiki")
    _require_verified_pool(parser, arguments.candidate_pool)
    _require_components(parser, arguments.components_dir)

    common = {
        "dataset_kind": dataset_kind,
        "source_split": source_split,
        "dataset_manifest_path": arguments.dataset_manifest,
        "source_parent_path": arguments.source_parent,
        "candidate_pool_path": arguments.candidate_pool,
        "component_directory": arguments.components_dir,
        "assignment_path": arguments.niah_assignment,
    }
    if arguments.verify_only:
        artifacts = verify_topk_control_artifacts(output_directory=arguments.output_dir, **common)
        action = "verified"
    else:
        artifacts = build_topk_control_artifacts(**common)
        freeze_topk_control_artifacts(arguments.output_dir, artifacts)
        action = "frozen"

    manifest_path = arguments.output_dir / TOPK_MANIFEST_FILE
    report_path = arguments.output_dir / TOPK_REPORT_FILE
    print(
        json.dumps(
            {
                "action": action,
                "kind": "topk-controls",
                "output_dir": str(arguments.output_dir),
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "report": str(report_path),
                "report_sha256": _sha256(report_path),
                "dataset_signature": artifacts.manifest["dataset_signature"],
                "counts": artifacts.report["counts"],
            },
            sort_keys=True,
        )
    )
    return 0


def _run_protocol(arguments: argparse.Namespace) -> int:
    if arguments.verify_only:
        artifacts = verify_count_matched_protocol_artifacts(arguments.output_dir)
        action = "verified"
    else:
        artifacts = build_count_matched_protocol_artifacts()
        freeze_count_matched_protocol_artifacts(arguments.output_dir, artifacts)
        action = "frozen"
    manifest_path = arguments.output_dir / COUNT_MATCHED_MANIFEST_FILE
    protocol_path = arguments.output_dir / COUNT_MATCHED_PROTOCOL_FILE
    print(
        json.dumps(
            {
                "action": action,
                "kind": "count-matched-protocol",
                "output_dir": str(arguments.output_dir),
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "protocol": str(protocol_path),
                "protocol_sha256": _sha256(protocol_path),
                "status": artifacts.protocol["status"],
                "repeat_count": artifacts.protocol["repeat_count"],
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "topk":
        return _run_topk(parser, arguments)
    if arguments.command == "freeze-protocol":
        return _run_protocol(arguments)
    raise AssertionError(f"unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
