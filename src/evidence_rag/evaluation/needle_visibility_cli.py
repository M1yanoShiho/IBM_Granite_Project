"""CLI: model-free needle-visibility triage over the injected NIAH pools.

Reads the E2 candidate_sets.jsonl + injected manifest + provenance, and for every
needle-in-window case reports whether the injected gold alias was inside what the extractor
actually reads (text[:passage_chars]). No LLM, no GPU — runs on the login node in seconds.
Compare visible_rate to the report's needle_gold_recovery to decide whether a bigger extractor
could help (visible_rate >> recovery) or the bottleneck is truncation/chunking (visible_rate ~= recovery).
"""

import argparse
import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.contracts.models import CandidateSet
from evidence_rag.evaluation.needle_visibility import classify_needle, summarize_visibility
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.materializer.provenance import read_provenance


def _read_candidates(path: Path) -> dict[str, CandidateSet]:
    text = Path(path).read_text(encoding="utf-8")
    sets = (
        CandidateSet.model_validate_json(line) for line in text.splitlines() if line.strip()
    )
    return {candidate_set.query_id: candidate_set for candidate_set in sets}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model-free needle-visibility triage")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--passage-chars", type=int, default=600)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    bundle = JsonlDatasetAdapter.load(arguments.manifest)
    gold_by_id = {gold_case.query_id: gold_case for gold_case in bundle.gold_cases}
    candidates_by_id = _read_candidates(arguments.candidates)
    records = read_provenance(arguments.provenance)

    classes: list[str | None] = []
    for record in records:
        candidate_set = candidates_by_id[record.query_id]
        gold_case = gold_by_id.get(record.query_id)
        window = tuple(
            sorted(candidate_set.candidates, key=lambda item: item.retrieval_rank)
        )[: arguments.top_n]
        classes.append(
            classify_needle(
                window,
                record.needle_document_id,
                (gold_case.reference_answers or ()) if gold_case else (),
                passage_chars=arguments.passage_chars,
            )
        )

    summary = summarize_visibility(classes)
    payload = {"passage_chars": arguments.passage_chars, **dataclasses.asdict(summary)}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
