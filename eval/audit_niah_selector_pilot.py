"""Audit the frozen controlled-NIAH candidate and feature artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from src.retrieval.corroboration import is_valid_answer


def _split_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    candidate_counts = [len(row["candidates"]) for row in rows]
    required_hits = 0
    harmful_hits = 0
    generated_counterfactual = selected_counterfactual = 0
    generated_non_answer = selected_non_answer = 0
    source_counts: Counter[str] = Counter()
    grade_counts: Counter[int] = Counter()
    candidate_total = 0
    for row in rows:
        candidates = list(row["candidates"])
        candidate_total += len(candidates)
        grades = [int(candidate["utility_grade"]) for candidate in candidates]
        required_hits += int(any(grade >= 3 for grade in grades))
        harmful_hits += int(any(grade == 0 for grade in grades))
        sources = [str(candidate["source"]) for candidate in candidates]
        source_counts.update(sources)
        grade_counts.update(grades)
        if bool(row.get("counterfactual_valid", False)):
            generated_counterfactual += 1
            selected_counterfactual += int("counterfactual" in sources)
        if bool(row.get("non_answer_valid", False)):
            generated_non_answer += 1
            selected_non_answer += int("generative_non_answer" in sources)
    query_count = len(rows)
    return {
        "query_count": query_count,
        "candidate_count": candidate_total,
        "candidate_count_min": min(candidate_counts) if candidate_counts else 0,
        "candidate_count_max": max(candidate_counts) if candidate_counts else 0,
        "required_evidence_recall_at_pool": required_hits / query_count if query_count else 0.0,
        "harmful_query_exposure_rate": harmful_hits / query_count if query_count else 0.0,
        "counterfactual_entry_rate_when_generated": (
            selected_counterfactual / generated_counterfactual
            if generated_counterfactual
            else 0.0
        ),
        "non_answer_entry_rate_when_generated": (
            selected_non_answer / generated_non_answer if generated_non_answer else 0.0
        ),
        "source_counts": dict(sorted(source_counts.items())),
        "utility_grade_counts": {
            str(grade): grade_counts.get(grade, 0) for grade in range(5)
        },
    }


def audit_ranked_rows(
    rows: Sequence[Mapping[str, object]], *, expected_candidates: int | None = 20
) -> dict[str, object]:
    """Validate frozen top-N groups and summarize selection headroom."""

    seen_queries: set[str] = set()
    split_rows: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        query_id = str(row["query_id"])
        if query_id in seen_queries:
            raise ValueError(f"duplicate query_id {query_id!r}")
        seen_queries.add(query_id)
        candidates = list(row["candidates"])
        if expected_candidates is not None and len(candidates) != expected_candidates:
            raise ValueError(
                f"query {query_id!r} has {len(candidates)} candidates; "
                f"expected {expected_candidates} candidates"
            )
        candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"query {query_id!r} has duplicate candidate_id values")
        split_rows.setdefault(str(row["split"]), []).append(row)
    result = _split_summary(rows)
    result["splits"] = {
        split: _split_summary(group) for split, group in sorted(split_rows.items())
    }
    return result


def audit_feature_rows(
    rows: Sequence[Mapping[str, object]], *, expected_candidates: int | None = 20
) -> dict[str, object]:
    """Validate the feature cache and summarize extraction reliability."""

    result = audit_ranked_rows(rows, expected_candidates=expected_candidates)
    candidates = [candidate for row in rows for candidate in row["candidates"]]
    candidate_count = len(candidates)
    query_count = len(rows)
    extraction_failures = sum(float(row["extraction_failure"]) >= 0.5 for row in candidates)
    parametric_failures = sum(
        not is_valid_answer(str(row.get("parametric_answer", ""))) for row in rows
    )
    vote_counts = [float(row["exact_vote_count"]) for row in candidates]
    result.update(
        {
            "candidate_extraction_failure_rate": (
                extraction_failures / candidate_count if candidate_count else 0.0
            ),
            "parametric_extraction_failure_rate": (
                parametric_failures / query_count if query_count else 0.0
            ),
            "candidate_with_vote_rate": (
                sum(value > 0 for value in vote_counts) / candidate_count
                if candidate_count
                else 0.0
            ),
            "mean_exact_vote_count": (
                sum(vote_counts) / candidate_count if candidate_count else 0.0
            ),
        }
    )
    return result


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("ranked", "features"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-candidates", type=int, default=20)
    args = parser.parse_args(argv)
    rows = _read_jsonl(args.input)
    if args.kind == "ranked":
        audit = audit_ranked_rows(rows, expected_candidates=args.expected_candidates)
    else:
        audit = audit_feature_rows(rows, expected_candidates=args.expected_candidates)
    payload = {
        "artifact": str(args.input),
        "artifact_sha256": _sha256(args.input),
        "audit": audit,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
