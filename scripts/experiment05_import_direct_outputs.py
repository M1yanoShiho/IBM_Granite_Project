#!/usr/bin/env python3
"""Audit and import externally generated Experiment 05 direct-ablation outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evidence_rag.evaluation.experiment05_data import read_runtime_bundle  # noqa: E402
from evidence_rag.evaluation.experiment05_generation import SystemOutput  # noqa: E402
from evidence_rag.evaluation.experiment05_io import read_jsonl  # noqa: E402

DATASETS = ("kilt-nq", "kilt-tqa", "alce-asqa")
ARM = "ablation_direct_generator"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _generation(root: Path, dataset: str) -> Path:
    return root / "goal4/outputs" / dataset / "generations" / f"{ARM}.jsonl"


def _evidence(root: Path, dataset: str, digest: str) -> Path:
    return root / "sealed_evidence/formal" / dataset / ARM / digest[:2] / f"{digest}.txt"


def _copy_verified(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    source_hash = _sha256(source)
    if target.exists():
        if _sha256(target) != source_hash:
            raise ValueError(f"existing target differs: {target}")
        return
    temporary = target.with_name(f".{target.name}.importing")
    shutil.copy2(source, temporary)
    if _sha256(temporary) != source_hash:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"copied file hash differs: {target}")
    temporary.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--runroot", type=Path, required=True)
    parser.add_argument("--goal1-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "schema_version": "experiment05.goal4_direct_import.v1",
        "status": "PASS",
        "arm": ARM,
        "datasets": {},
    }
    for dataset in DATASETS:
        runtime = list(
            read_runtime_bundle(
                args.goal1_root / "bundles/runtime/formal" / f"{dataset}.jsonl",
                dataset=dataset,
            )
        )
        expected_ids = [str(row["query_id"]) for row in runtime]
        source = _generation(args.staging_root, dataset)
        rows = [SystemOutput.model_validate(item) for item in read_jsonl(source)]
        if (
            len(rows) != 400
            or [row.query_id for row in rows] != expected_ids
            or any(row.dataset != dataset or row.arm_id != ARM for row in rows)
        ):
            raise ValueError(f"{dataset}/{ARM} does not cover the exact frozen IDs")
        errors = sum(row.runtime_error is not None for row in rows)
        if errors:
            raise ValueError(f"{dataset}/{ARM} contains runtime errors")

        evidence_hashes = {
            record.text_sha256
            for row in rows
            for record in (*row.selected_evidence_records, *row.presented_evidence_records)
        }
        for digest in sorted(evidence_hashes):
            artifact = _evidence(args.staging_root, dataset, digest)
            if not artifact.is_file() or _sha256(artifact) != digest:
                raise ValueError(f"sealed evidence differs: {dataset}/{digest}")
            _copy_verified(artifact, _evidence(args.runroot, dataset, digest))

        _copy_verified(source, _generation(args.runroot, dataset))
        for manifest in source.parent.parent.glob("run_manifest.direct.*.json"):
            relative = manifest.relative_to(args.staging_root)
            _copy_verified(manifest, args.runroot / relative)
        if _sha256(_generation(args.runroot, dataset)) != _sha256(source):
            raise ValueError(f"final generation hash differs: {dataset}")

        report["datasets"][dataset] = {
            "count": len(rows),
            "unique_ids": len({row.query_id for row in rows}),
            "runtime_errors": errors,
            "generation_sha256": _sha256(source),
            "sealed_evidence_files": len(evidence_hashes),
        }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "datasets": 3, "outputs": 1200}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
