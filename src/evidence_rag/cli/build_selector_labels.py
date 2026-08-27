"""Freeze or independently verify one R004 conservative label bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict, cast

from evidence_rag.evaluation.selector_components import (
    MANIFEST_FILE as COMPONENT_MANIFEST_FILE,
)
from evidence_rag.evaluation.selector_components import DatasetKind
from evidence_rag.materializer.selector_labels import (
    MANIFEST_FILE,
    REPORT_FILE,
    SelectorLabelArtifacts,
    build_selector_label_artifacts,
    freeze_selector_label_artifacts,
    verify_selector_label_artifacts,
)
from evidence_rag.materializer.selector_pool import SELECTOR_POOL_MANIFEST_FILE


class _ArtifactArguments(TypedDict):
    dataset_kind: DatasetKind
    dataset_manifest_path: Path
    source_parent_path: Path
    candidate_pool_path: Path
    component_directory: Path
    assignment_path: Path | None
    provenance_path: Path | None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze or verify strict masked labels for the R004 dual-head Selector"
    )
    parser.add_argument("--dataset-kind", required=True, choices=("niah", "2wiki"))
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--source-parent", required=True, type=Path)
    parser.add_argument(
        "--candidate-pool",
        required=True,
        type=Path,
        help="source-train pool directory with a verified Selector-v2 manifest",
    )
    parser.add_argument("--components-dir", required=True, type=Path)
    parser.add_argument("--niah-assignment", type=Path)
    parser.add_argument("--niah-provenance", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_inputs(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> None:
    if not arguments.candidate_pool.is_dir():
        parser.error("--candidate-pool must be a Selector-v2 pool directory")
    if not (arguments.candidate_pool / SELECTOR_POOL_MANIFEST_FILE).is_file():
        parser.error(f"--candidate-pool is missing {SELECTOR_POOL_MANIFEST_FILE}")
    if not arguments.components_dir.is_dir():
        parser.error("--components-dir must be an R002 component directory")
    if not (arguments.components_dir / COMPONENT_MANIFEST_FILE).is_file():
        parser.error(f"--components-dir is missing {COMPONENT_MANIFEST_FILE}")
    if arguments.dataset_kind == "niah":
        if arguments.niah_assignment is None or arguments.niah_provenance is None:
            parser.error("NIAH requires both --niah-assignment and --niah-provenance")
    elif arguments.niah_assignment is not None or arguments.niah_provenance is not None:
        parser.error("2Wiki must not receive NIAH assignment or provenance")


def _artifact_arguments(arguments: argparse.Namespace) -> _ArtifactArguments:
    return {
        "dataset_kind": cast(DatasetKind, arguments.dataset_kind),
        "dataset_manifest_path": arguments.dataset_manifest,
        "source_parent_path": arguments.source_parent,
        "candidate_pool_path": arguments.candidate_pool,
        "component_directory": arguments.components_dir,
        "assignment_path": arguments.niah_assignment,
        "provenance_path": arguments.niah_provenance,
    }


def _result(
    *,
    action: str,
    output: Path,
    artifacts: SelectorLabelArtifacts,
) -> dict[str, object]:
    report_path = output / REPORT_FILE
    manifest_path = output / MANIFEST_FILE
    return {
        "action": action,
        "kind": "selector-labels",
        "output_dir": str(output),
        "report": str(report_path),
        "report_sha256": _sha256(report_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "dataset_kind": artifacts.report["dataset_kind"],
        "status": artifacts.report["status"],
        "counts": artifacts.report["counts"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    _require_inputs(parser, arguments)
    common = _artifact_arguments(arguments)
    if arguments.verify_only:
        artifacts = verify_selector_label_artifacts(
            output_directory=arguments.output_dir,
            **common,
        )
        action = "verified"
    else:
        artifacts = build_selector_label_artifacts(**common)
        freeze_selector_label_artifacts(arguments.output_dir, artifacts)
        action = "frozen"
    print(
        json.dumps(
            _result(action=action, output=arguments.output_dir, artifacts=artifacts), sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
