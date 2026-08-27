#!/usr/bin/env python3
"""Validate and freeze Experiment 05 Goal 3 or Goal 4 generation outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evidence_rag.evaluation.experiment05_data import read_runtime_bundle  # noqa: E402
from evidence_rag.evaluation.experiment05_generation import SystemOutput  # noqa: E402
from evidence_rag.evaluation.experiment05_io import read_jsonl  # noqa: E402
from evidence_rag.evaluation.experiment05_runtime import (  # noqa: E402
    ABLATION_ARMS,
    MAIN_ARMS,
    PreparedQuery,
)

DATASETS = ("kilt-nq", "kilt-tqa", "alce-asqa")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _load_arm(path: Path, *, dataset: str, arm: str, expected_ids: list[str]) -> list[SystemOutput]:
    rows = [SystemOutput.model_validate(item) for item in read_jsonl(path)]
    if (
        len(rows) != 400
        or [row.query_id for row in rows] != expected_ids
        or any(row.dataset != dataset or row.arm_id != arm for row in rows)
    ):
        raise ValueError(f"{dataset}/{arm} does not cover the exact 400 frozen IDs")
    return rows


def _artifact_path(runroot: Path, dataset: str, arm: str, digest: str) -> Path:
    return runroot / "sealed_evidence/formal" / dataset / arm / digest[:2] / f"{digest}.txt"


def _arm_manifest(
    *,
    runroot: Path,
    path: Path,
    dataset: str,
    arm: str,
    rows: list[SystemOutput],
) -> dict[str, Any]:
    errors = sum(row.runtime_error is not None for row in rows)
    for row in rows:
        for record in (*row.selected_evidence_records, *row.presented_evidence_records):
            artifact = _artifact_path(runroot, dataset, arm, record.text_sha256)
            if not artifact.is_file() or _sha256(artifact) != record.text_sha256:
                raise ValueError(f"sealed evidence artifact differs: {dataset}/{arm}")
    if errors / len(rows) > 0.01:
        raise ValueError(f"runtime error guard failed: {dataset}/{arm}")
    return {
        "count": len(rows),
        "file_sha256": _sha256(path),
        "runtime_errors": errors,
        "runtime_error_rate": errors / len(rows),
        "abstentions": sum(row.abstained for row in rows),
        "presented_records": sum(len(row.presented_evidence_records) for row in rows),
    }


def _generation_path(runroot: Path, phase: str, dataset: str, arm: str) -> Path:
    return runroot / phase / "outputs" / dataset / "generations" / f"{arm}.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("goal3", "goal4"), required=True)
    parser.add_argument("--runroot", type=Path, required=True)
    parser.add_argument("--goal1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.phase == "goal4":
        goal3_manifest_path = args.runroot / "goal3/freeze_manifest.json"
        goal3_manifest = json.loads(goal3_manifest_path.read_text(encoding="utf-8"))
        if goal3_manifest.get("status") != "PASS":
            raise ValueError("Goal 3 outputs are not frozen PASS")

    report: dict[str, Any] = {
        "schema_version": f"experiment05.{args.phase}_freeze.v1",
        "status": "PASS",
        "phase": args.phase,
        "outputs_frozen_before_scoring": True,
        "datasets": {},
    }
    total = 0
    for dataset in DATASETS:
        runtime = list(
            read_runtime_bundle(
                args.goal1_root / "bundles/runtime/formal" / f"{dataset}.jsonl",
                dataset=dataset,
            )
        )
        expected_ids = [str(row["query_id"]) for row in runtime]
        prepared_path = args.runroot / "prepared/formal" / f"{dataset}.jsonl"
        prepared = [PreparedQuery.model_validate(row) for row in read_jsonl(prepared_path)]
        if len(prepared) != 400 or [row.query_id for row in prepared] != expected_ids:
            raise ValueError(f"{dataset} formal preparation differs from frozen IDs")
        arms = MAIN_ARMS if args.phase == "goal3" else ABLATION_ARMS
        arm_rows: dict[str, list[SystemOutput]] = {}
        arm_reports: dict[str, Any] = {}
        for arm in arms:
            path = _generation_path(args.runroot, args.phase, dataset, arm)
            rows = _load_arm(path, dataset=dataset, arm=arm, expected_ids=expected_ids)
            arm_rows[arm] = rows
            arm_reports[arm] = _arm_manifest(
                runroot=args.runroot,
                path=path,
                dataset=dataset,
                arm=arm,
                rows=rows,
            )
            total += len(rows)

        if args.phase == "goal3":
            goal4_dir = args.runroot / "goal4/outputs" / dataset / "generations"
            if goal4_dir.exists() and any(goal4_dir.iterdir()):
                raise ValueError("Goal 4 outputs exist before Goal 3 PASS")
        else:
            full = _load_arm(
                _generation_path(args.runroot, "goal3", dataset, "ours_seed13"),
                dataset=dataset,
                arm="ours_seed13",
                expected_ids=expected_ids,
            )
            direct = arm_rows["ablation_direct_generator"]
            no_selector = arm_rows["ablation_no_selector"]
            bm25 = arm_rows["ablation_bm25_retriever"]
            for full_row, direct_row, no_selector_row, bm25_row in zip(
                full, direct, no_selector, bm25, strict=True
            ):
                full_selected = tuple(
                    item.text_sha256 for item in full_row.selected_evidence_records
                )
                if tuple(item.text_sha256 for item in direct_row.selected_evidence_records) != full_selected:
                    raise ValueError("Direct Generator ablation changed upstream evidence")
                if no_selector_row.retrieved_evidence_ids != full_row.retrieved_evidence_ids:
                    raise ValueError("no-Selector ablation changed the Hybrid Retriever")
                if bm25_row.config_fingerprint != full_row.config_fingerprint:
                    raise ValueError("BM25 Retriever ablation changed the Generator config")
                if bm25_row.prompt_fingerprint != full_row.prompt_fingerprint:
                    raise ValueError("BM25 Retriever ablation changed the shared prompt")

        scorer_dir = args.runroot / "goal5/scoring"
        if scorer_dir.exists() and any(path.is_file() for path in scorer_dir.rglob("*")):
            raise ValueError("formal scoring exists before all generation outputs are frozen")
        sidecar_path = (
            args.goal1_root / "bundles/scorer_only/formal" / f"{dataset}.jsonl"
        )
        if stat.S_IMODE(sidecar_path.stat().st_mode) != 0o600:
            raise ValueError("formal scorer-only sidecar is not mode 0600")
        report["datasets"][dataset] = {
            "runtime_count": len(runtime),
            "prepared_sha256": _sha256(prepared_path),
            "arms": arm_reports,
            "scorer_sidecar_mode": "0600_unread",
        }

    expected_total = 8_400 if args.phase == "goal3" else 3_600
    if total != expected_total:
        raise ValueError(f"{args.phase} output count differs: {total} != {expected_total}")
    report["output_count"] = total
    report["formal_scores_created"] = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for dataset in DATASETS:
        for arm in MAIN_ARMS if args.phase == "goal3" else ABLATION_ARMS:
            _generation_path(args.runroot, args.phase, dataset, arm).chmod(0o444)
    print(json.dumps({"phase": args.phase, "status": "PASS", "outputs": total}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
