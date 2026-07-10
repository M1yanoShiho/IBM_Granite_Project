"""Materialize a transparent ContractNLI cross-domain selector diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from eval.niah_selector_pilot import minmax


def materialize_contract_features(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Keep eligible official-test groups and mark unavailable QA signals missing."""

    output: list[dict[str, object]] = []
    for row in rows:
        if str(row["split"]) != "test" or not bool(row["eligible_for_training"]):
            continue
        candidates = [dict(candidate) for candidate in row["candidates"]]
        normalized = minmax(
            [float(candidate["relevance_score"]) for candidate in candidates]
        )
        for candidate, relevance_normalized in zip(candidates, normalized):
            rank = int(candidate["original_rank"])
            candidate.update(
                {
                    "relevance_normalized": relevance_normalized,
                    "reciprocal_rank": 1.0 / rank,
                    "exact_vote_count": 0.0,
                    "vote_ratio": 0.0,
                    "parametric_agreement": 0.0,
                    "extraction_failure": 1.0,
                    "source_dedup_vote_count": 0.0,
                    "dominant_answer_agreement": 0.0,
                    "answer_length": 0.0,
                    "judge_direct_support": 0.0,
                    "judge_condition_coverage": 0.0,
                    "judge_evidence_sufficiency": 0.0,
                    "judge_parse_failure": 1.0,
                }
            )
        materialized = dict(row)
        materialized["candidates"] = candidates
        output.append(materialized)
    return output


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranked-pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    _write_jsonl(args.out, materialize_contract_features(_read_jsonl(args.ranked_pool)))


if __name__ == "__main__":
    main()
