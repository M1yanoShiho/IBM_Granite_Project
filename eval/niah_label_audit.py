"""Create a deterministic, model-assisted audit packet for controlled NIAH labels."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import re
from typing import Mapping, Sequence

from eval.niah_selector_pilot import GraniteBatchGenerator
from src.prompts.judge import LABEL_AUDIT_PROMPT_A, LABEL_AUDIT_PROMPT_B


# Backwards-compat aliases for any external caller that imported the old uppercase
# names. The canonical names live in ``src.prompts.judge``.
PROMPT_A = LABEL_AUDIT_PROMPT_A
PROMPT_B = LABEL_AUDIT_PROMPT_B


def parse_grade(value: str) -> int | None:
    """Parse one unambiguous grade from a model response."""

    values = {int(match) for match in re.findall(r"(?<!\d)([0-4])(?!\d)", value)}
    return next(iter(values)) if len(values) == 1 else None


def quadratic_weighted_kappa(first: Sequence[int], second: Sequence[int]) -> float:
    """Compute quadratic weighted kappa for the canonical grades 0 through 4."""

    if len(first) != len(second) or not first:
        raise ValueError("kappa inputs must be non-empty and have equal length")
    if any(label not in range(5) for label in (*first, *second)):
        raise ValueError("kappa labels must be integers from 0 to 4")
    total = len(first)
    first_counts = Counter(first)
    second_counts = Counter(second)
    observed = sum(((a - b) ** 2 / 16.0) for a, b in zip(first, second)) / total
    expected = sum(
        ((a - b) ** 2 / 16.0)
        * first_counts[a]
        * second_counts[b]
        / (total * total)
        for a in range(5)
        for b in range(5)
    )
    if expected == 0.0:
        return 1.0 if observed == 0.0 else 0.0
    return 1.0 - observed / expected


def select_query_groups(
    rows: Sequence[Mapping[str, object]], *, quotas: Mapping[str, int], seed: int
) -> list[Mapping[str, object]]:
    """Select deterministic split-stratified complete query groups."""

    rng = random.Random(seed)
    selected: list[Mapping[str, object]] = []
    for split, quota in quotas.items():
        candidates = [row for row in rows if str(row["split"]) == split]
        if len(candidates) < quota:
            raise ValueError(f"split {split!r} has {len(candidates)} rows, needs {quota}")
        selected.extend(rng.sample(candidates, quota))
    return sorted(selected, key=lambda row: (str(row["split"]), str(row["query_id"])))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def create_label_audit(
    *, pool: Path, out_csv: Path, out_json: Path, model_id: str, device: str, batch_size: int
) -> None:
    """Audit 80 query groups with two frozen Granite prompts."""

    rows = _read_jsonl(pool)
    eligible = [
        row
        for row in rows
        if {"counterfactual", "generative_non_answer"}
        <= {str(candidate["source"]) for candidate in row["candidates"]}
    ]
    selected = select_query_groups(
        eligible, quotas={"train": 32, "dev": 24, "test": 24}, seed=42
    )
    audit_rows: list[dict[str, object]] = []
    for row in selected:
        by_id = {str(candidate["candidate_id"]): candidate for candidate in row["candidates"]}
        needle = by_id[str(row["needle_doc_id"])]
        counterfactual = next(
            candidate for candidate in row["candidates"] if candidate["source"] == "counterfactual"
        )
        non_answer = next(
            candidate
            for candidate in row["candidates"]
            if candidate["source"] == "generative_non_answer"
        )
        for candidate in (needle, counterfactual, non_answer):
            audit_rows.append(
                {
                    "query_id": row["query_id"],
                    "split": row["split"],
                    "question": row["question"],
                    "reference_answers": json.dumps(row["answers"], ensure_ascii=False),
                    "candidate_id": candidate["candidate_id"],
                    "source": candidate["source"],
                    "canonical_grade": int(candidate["utility_grade"]),
                    "passage": candidate["text"],
                }
            )
    prompts_a = [
        PROMPT_A.format(
            question=row["question"],
            answers=row["reference_answers"],
            passage=str(row["passage"])[:900],
        )
        for row in audit_rows
    ]
    prompts_b = [
        PROMPT_B.format(
            question=row["question"],
            answers=row["reference_answers"],
            passage=str(row["passage"])[:900],
        )
        for row in audit_rows
    ]
    generator = GraniteBatchGenerator(model_id, device=device)
    raw_a = generator.generate(prompts_a, batch_size=batch_size, max_new_tokens=12)
    raw_b = generator.generate(prompts_b, batch_size=batch_size, max_new_tokens=12)
    for row, value_a, value_b in zip(audit_rows, raw_a, raw_b):
        row["model_a_raw"] = value_a
        row["model_a_grade"] = parse_grade(value_a)
        row["model_b_raw"] = value_b
        row["model_b_grade"] = parse_grade(value_b)
        row["human_annotator_1"] = ""
        row["human_annotator_2"] = ""
        row["human_adjudication"] = ""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    valid = [
        row
        for row in audit_rows
        if row["model_a_grade"] is not None and row["model_b_grade"] is not None
    ]
    canonical = [int(row["canonical_grade"]) for row in valid]
    labels_a = [int(row["model_a_grade"]) for row in valid]
    labels_b = [int(row["model_b_grade"]) for row in valid]
    summary = {
        "audit_type": "model_assisted_not_human_inter_annotator_audit",
        "query_groups": len(selected),
        "candidate_rows": len(audit_rows),
        "valid_dual_model_rows": len(valid),
        "parse_rate": len(valid) / len(audit_rows),
        "quadratic_weighted_kappa": {
            "canonical_vs_prompt_a": quadratic_weighted_kappa(canonical, labels_a),
            "canonical_vs_prompt_b": quadratic_weighted_kappa(canonical, labels_b),
            "prompt_a_vs_prompt_b": quadratic_weighted_kappa(labels_a, labels_b),
        },
        "exact_agreement": {
            "canonical_vs_prompt_a": sum(a == b for a, b in zip(canonical, labels_a))
            / len(valid),
            "canonical_vs_prompt_b": sum(a == b for a, b in zip(canonical, labels_b))
            / len(valid),
            "prompt_a_vs_prompt_b": sum(a == b for a, b in zip(labels_a, labels_b))
            / len(valid),
        },
        "human_review_status": "pending_team_annotation_in_csv",
    }
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    create_label_audit(
        pool=args.pool,
        out_csv=args.out_csv,
        out_json=args.out_json,
        model_id=args.model_id,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
