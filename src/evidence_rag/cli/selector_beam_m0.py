"""Validate and summarize the frozen inputs for Beam Selector M0."""

from __future__ import annotations

import argparse
import json
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.materializer.selector_beam_split import sha256_file
from evidence_rag.materializer.source_parent import read_parent_index

FROZEN_POOL_NAMES = {
    "niah-train",
    "niah-dev",
    "sealed600",
    "2wiki-train",
    "2wiki-dev",
    "2wiki-heldout",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Beam Selector M0 inputs")
    parser.add_argument("--niah-audit", required=True, type=Path)
    parser.add_argument("--niah-train-assignments", required=True, type=Path)
    parser.add_argument("--niah-dev-assignments", required=True, type=Path)
    parser.add_argument("--niah-train-manifest", required=True, type=Path)
    parser.add_argument("--niah-dev-manifest", required=True, type=Path)
    parser.add_argument("--niah-sealed-manifest", required=True, type=Path)
    parser.add_argument("--twowiki-train", required=True, type=Path)
    parser.add_argument("--twowiki-dev", required=True, type=Path)
    parser.add_argument("--twowiki-heldout", required=True, type=Path)
    parser.add_argument(
        "--pool-manifest",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="candidate-pool audit manifest; repeat for every frozen pool",
    )
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--model-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _read_json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _twowiki_summary(manifest_path: Path) -> tuple[dict[str, object], set[str]]:
    bundle = JsonlDatasetAdapter.load(manifest_path)
    query_ids = {query.query_id for query in bundle.queries}
    parent_path = manifest_path.parent / "source_parent.jsonl"
    parents = read_parent_index(parent_path)
    unresolved = parents.n_unresolved(document.document_id for document in bundle.documents)
    if unresolved:
        raise ValueError(f"{manifest_path} has {unresolved} unresolved source parents")
    return (
        {
            "split": bundle.manifest.split,
            "queries": len(bundle.queries),
            "documents": len(bundle.documents),
            "dataset_signature": bundle.dataset_signature,
            "manifest_sha256": sha256_file(manifest_path),
            "source_parent_sha256": sha256_file(parent_path),
        },
        query_ids,
    )


def _niah_dataset_summary(
    manifest_path: Path, expected_hashes: Mapping[str, object]
) -> dict[str, object]:
    bundle = JsonlDatasetAdapter.load(manifest_path)
    root = manifest_path.parent
    names = (
        "manifest.json",
        "documents.jsonl",
        "queries.jsonl",
        "gold_cases.jsonl",
        "provenance.jsonl",
    )
    actual = {name: sha256_file(root / name) for name in names}
    if any(expected_hashes.get(name) != digest for name, digest in actual.items()):
        raise ValueError(f"NIAH split audit no longer matches {manifest_path}")
    return {
        "split": bundle.manifest.split,
        "queries": len(bundle.queries),
        "documents": len(bundle.documents),
        "dataset_signature": bundle.dataset_signature,
        "artifact_sha256": actual,
    }


def _parse_pool_arguments(values: Sequence[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError("--pool-manifest must use NAME=PATH")
        if name in parsed:
            raise ValueError(f"duplicate pool name: {name}")
        parsed[name] = Path(raw_path)
    if set(parsed) != FROZEN_POOL_NAMES:
        missing = sorted(FROZEN_POOL_NAMES - set(parsed))
        extra = sorted(set(parsed) - FROZEN_POOL_NAMES)
        raise ValueError(f"frozen pool names differ: missing={missing}, extra={extra}")
    return parsed


def _pool_summary(path: Path) -> dict[str, object]:
    manifest = _read_json(path)
    dataset = _mapping(manifest.get("dataset"), "dataset")
    pool = _mapping(manifest.get("candidate_pool"), "candidate_pool")
    audit = _mapping(manifest.get("audit"), "audit")
    retriever = _mapping(manifest.get("retriever"), "retriever")
    if pool.get("top_n") != 20 or pool.get("exact_top_n_rate") != 1.0:
        raise ValueError(f"{path} is not an exact Top-20 pool")
    if audit.get("unresolved_parent_count") != 0:
        raise ValueError(f"{path} contains unresolved source parents")
    if retriever.get("name") != "hybrid":
        raise ValueError(f"{path} was not produced by Hybrid RRF")
    candidate_path = Path(str(pool.get("path")))
    if not candidate_path.is_file() or sha256_file(candidate_path) != pool.get("sha256"):
        raise ValueError(f"{path} does not match its actual candidate-pool file")
    source_parent = _mapping(manifest.get("source_parent"), "source_parent")
    source_parent_path = Path(str(source_parent.get("path")))
    if not source_parent_path.is_file() or sha256_file(source_parent_path) != source_parent.get(
        "sha256"
    ):
        raise ValueError(f"{path} does not match its actual source-parent file")
    return {
        "dataset_signature": dataset.get("signature"),
        "query_count": pool.get("query_count"),
        "sha256": pool.get("sha256"),
        "path": str(candidate_path.resolve()),
        "required_recall_at_top20": audit.get("required_recall_at_top_n"),
        "harmful_pool_hit_rate": audit.get("harmful_pool_hit_rate"),
        "manifest_sha256": sha256_file(path),
        "manifest_path": str(path.resolve()),
        "source_parent_sha256": source_parent.get("sha256"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    niah = _read_json(arguments.niah_audit)
    overlaps = _mapping(niah.get("overlap_after_filtering"), "NIAH final overlaps")
    if any(
        value
        for pair in overlaps.values()
        for value in _mapping(pair, "NIAH overlap pair").values()
    ):
        raise ValueError("NIAH assignments still leak across splits")
    niah_outputs = _mapping(niah.get("outputs"), "NIAH assignment outputs")
    niah_inputs = _mapping(niah.get("inputs"), "NIAH source inputs")
    assignment_paths = {
        "train": arguments.niah_train_assignments,
        "dev": arguments.niah_dev_assignments,
    }
    assignments: dict[str, dict[str, object]] = {}
    for name, assignment_path in assignment_paths.items():
        expected = niah_outputs.get(f"niah_{name}_assignments.jsonl")
        actual = sha256_file(assignment_path)
        if expected != actual:
            raise ValueError(f"NIAH {name} assignment SHA differs from the split audit")
        assignments[name] = {
            "path": str(assignment_path.resolve()),
            "sha256": actual,
        }
    niah_datasets = {
        name: _niah_dataset_summary(
            manifest_path,
            _mapping(niah_inputs.get(name), f"NIAH {name} source hashes"),
        )
        for name, manifest_path in (
            ("train", arguments.niah_train_manifest),
            ("dev", arguments.niah_dev_manifest),
            ("sealed", arguments.niah_sealed_manifest),
        )
    }

    twowiki: dict[str, dict[str, object]] = {}
    query_sets: dict[str, set[str]] = {}
    for name, path in (
        ("train", arguments.twowiki_train),
        ("dev", arguments.twowiki_dev),
        ("heldout", arguments.twowiki_heldout),
    ):
        twowiki[name], query_sets[name] = _twowiki_summary(path)
    query_overlap = {
        "train_dev": len(query_sets["train"] & query_sets["dev"]),
        "train_heldout": len(query_sets["train"] & query_sets["heldout"]),
        "dev_heldout": len(query_sets["dev"] & query_sets["heldout"]),
    }
    if any(query_overlap.values()):
        raise ValueError(f"2Wiki query leakage: {query_overlap}")

    model_config = tomllib.loads(arguments.model_config.read_text(encoding="utf-8"))
    model = _mapping(model_config.get("model"), "model config")
    actual_model_sha = sha256_file(arguments.model_file)
    if model.get("model_sha256") != actual_model_sha:
        raise ValueError("model SHA-256 differs from the frozen config")

    pools = {
        name: _pool_summary(path)
        for name, path in sorted(_parse_pool_arguments(arguments.pool_manifest).items())
    }
    expected_pool_signatures = {
        "niah-train": niah_datasets["train"]["dataset_signature"],
        "niah-dev": niah_datasets["dev"]["dataset_signature"],
        "sealed600": niah_datasets["sealed"]["dataset_signature"],
        "2wiki-train": twowiki["train"]["dataset_signature"],
        "2wiki-dev": twowiki["dev"]["dataset_signature"],
        "2wiki-heldout": twowiki["heldout"]["dataset_signature"],
    }
    for name, expected_signature in expected_pool_signatures.items():
        if pools[name]["dataset_signature"] != expected_signature:
            raise ValueError(f"{name} candidate pool belongs to a different dataset")
    report = {
        "schema_version": "1.0",
        "status": "PASS",
        "niah": {
            "counts": niah.get("counts"),
            "overlap_after_filtering": overlaps,
            "split_audit_sha256": sha256_file(arguments.niah_audit),
            "assignments": assignments,
            "datasets": niah_datasets,
        },
        "twowiki": twowiki,
        "twowiki_query_overlap": query_overlap,
        "model": {
            "model_id": model.get("model_id"),
            "revision": model.get("revision"),
            "model_sha256": actual_model_sha,
            "config_sha256": sha256_file(arguments.model_config),
        },
        "candidate_pools": pools,
    }
    arguments.output.mkdir(parents=True, exist_ok=True)
    json_path = arguments.output / "M0_REPORT.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = [
        "# Beam Selector M0 Report",
        "",
        "**Status:** PASS",
        "",
        "- NIAH train/dev/sealed leakage: 0 on query, parent and synthetic-family axes.",
        "- 2Wiki train/dev/heldout query overlap: 0.",
        "- Every frozen candidate pool has exactly Hybrid RRF Top-20 and no unresolved parent.",
        f"- Model: `{model.get('model_id')}@{model.get('revision')}`.",
        "",
        "The machine-readable counts, hashes and pool checks are in `M0_REPORT.json`.",
    ]
    (arguments.output / "M0_REPORT.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(json_path), "status": "PASS"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
