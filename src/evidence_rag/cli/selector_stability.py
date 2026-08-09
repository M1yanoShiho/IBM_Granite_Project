"""Compare the fixed 60-query repeat with the formal Selector outputs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from evidence_rag.contracts.models import SelectedEvidenceSet
from evidence_rag.evaluation.selector_experiment import read_jsonl


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check exact Selector output stability")
    parser.add_argument("--formal-arms", required=True, type=Path)
    parser.add_argument("--repeat-arms", required=True, type=Path)
    parser.add_argument("--query-ids", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _manifest(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"invalid run manifest: {path}")
    if value.get("status") != "complete":
        raise ValueError(f"run is not complete: {path}")
    return value


def _selected(path: Path) -> dict[str, tuple[str, ...]]:
    records = read_jsonl(path, SelectedEvidenceSet)
    return {
        record.query_id: tuple(item.evidence_id for item in record.evidence)
        for record in records
    }


def _arm_report(
    method: str,
    formal_root: Path,
    repeat_root: Path,
    query_ids: Sequence[str],
) -> dict[str, object]:
    formal_manifest = _manifest(formal_root / method / "run_manifest.json")
    repeat_manifest = _manifest(repeat_root / method / "run_manifest.json")
    if formal_manifest.get("candidate_pool_sha256") != repeat_manifest.get(
        "candidate_pool_sha256"
    ):
        raise ValueError(f"{method} formal and repeat runs used different candidate pools")
    formal = _selected(formal_root / method / "selected_evidence_sets.jsonl")
    repeat = _selected(repeat_root / method / "selected_evidence_sets.jsonl")
    unknown = set(query_ids) - set(formal)
    if unknown:
        raise ValueError(f"formal {method} output is missing query {sorted(unknown)[0]}")
    if set(repeat) != set(query_ids):
        raise ValueError(f"repeat {method} output does not match the frozen query-ID set")
    mismatches = [query_id for query_id in query_ids if formal[query_id] != repeat[query_id]]
    return {
        "candidate_pool_sha256": formal_manifest["candidate_pool_sha256"],
        "n_queries": len(query_ids),
        "exact_agreement_count": len(query_ids) - len(mismatches),
        "exact_agreement_rate": (len(query_ids) - len(mismatches)) / len(query_ids),
        "mismatched_query_ids": mismatches,
        "pass": not mismatches,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    query_ids = tuple(
        line.strip()
        for line in arguments.query_ids.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(query_ids) != 60 or len(query_ids) != len(set(query_ids)):
        raise ValueError("stability query-ID file must contain exactly 60 unique IDs")
    arms = {
        method: _arm_report(
            method,
            arguments.formal_arms,
            arguments.repeat_arms,
            query_ids,
        )
        for method in ("top-k", "reliability-mis")
    }
    report = {
        "schema_version": "1.0",
        "query_ids_path": str(arguments.query_ids.resolve()),
        "query_count": len(query_ids),
        "arms": arms,
        "pass": all(bool(arm["pass"]) for arm in arms.values()),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
