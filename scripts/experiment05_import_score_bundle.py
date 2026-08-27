#!/usr/bin/env python3
"""Audit and atomically import one externally scored Experiment 05 arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evidence_rag.evaluation.experiment05_data import read_runtime_bundle  # noqa: E402
from evidence_rag.evaluation.experiment05_io import read_jsonl  # noqa: E402

METRICS = {"rfc", "vrfc", "ucr", "cp", "cr", "rr"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _replace_verified(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    source_hash = _sha256(source)
    if target.exists() and _sha256(target) == source_hash:
        return
    temporary = target.with_name(f".{target.name}.importing")
    shutil.copy2(source, temporary)
    if _sha256(temporary) != source_hash:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"copied file hash differs: {target}")
    temporary.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--runroot", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    expected_ids = [
        str(row["query_id"])
        for row in read_runtime_bundle(args.runtime, dataset=args.dataset)
    ]
    if len(expected_ids) != 400:
        raise ValueError("runtime must contain exactly 400 frozen IDs")

    scores_path = args.source_dir / "query_scores.jsonl"
    traces_path = args.source_dir / "claim_traces.jsonl"
    aggregate_path = args.source_dir / "aggregate.json"
    scores = read_jsonl(scores_path)
    traces = read_jsonl(traces_path)
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))

    if [str(row.get("query_id")) for row in scores] != expected_ids:
        raise ValueError("score rows do not cover the exact ordered frozen IDs")
    if [str(row.get("query_id")) for row in traces] != expected_ids:
        raise ValueError("claim traces do not cover the exact ordered frozen IDs")
    if any(
        row.get("schema_version") != "experiment05.query_score.v1"
        or row.get("dataset") != args.dataset
        or row.get("arm_id") != args.arm
        or row.get("scorer_error") is not False
        or set(row.get("metrics", {})) != METRICS
        for row in scores
    ):
        raise ValueError("score rows fail schema, identity, metric, or error audit")
    if any(
        row.get("schema_version") != "experiment05.scorer_claim_trace.v1"
        or row.get("dataset") != args.dataset
        or row.get("arm_id") != args.arm
        for row in traces
    ):
        raise ValueError("claim traces fail schema or identity audit")
    if (
        aggregate.get("schema_version") != "experiment05.arm_aggregate.v1"
        or aggregate.get("dataset") != args.dataset
        or aggregate.get("arm_id") != args.arm
        or aggregate.get("n_total") != 400
        or aggregate.get("n_scorer_error") != 0
        or set(aggregate.get("metrics", {})) != METRICS
    ):
        raise ValueError("aggregate fails completeness or schema audit")

    destination = args.runroot / "goal5/scoring" / args.dataset / args.arm
    for source in (scores_path, traces_path, aggregate_path):
        _replace_verified(source, destination / source.name)

    report = {
        "schema_version": "experiment05.goal5_score_import.v1",
        "status": "PASS",
        "dataset": args.dataset,
        "arm": args.arm,
        "count": 400,
        "source_hashes": {
            source.name: _sha256(source)
            for source in (scores_path, traces_path, aggregate_path)
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "dataset": args.dataset, "arm": args.arm}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
