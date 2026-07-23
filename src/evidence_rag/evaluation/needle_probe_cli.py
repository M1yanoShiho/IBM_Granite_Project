"""CLI: needle-extraction failure-mode probe (systematic-debugging Phase 3).

For every needle-in-window injected query, extract from the needle chunk ONLY (1 call/chunk,
not the full 20-window), classify recovered/wrong/none, and record whether the gold alias was
visible in text[:passage_chars]. Cheap (~one call per injected query, ~15-20 min on 3B). Writes
a summary json + a per-case jsonl so the actual failures can be eyeballed.
"""

import argparse
import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.contracts.models import CandidateSet
from evidence_rag.evaluation.needle_probe import (
    ProbeOutcome,
    classify_extraction,
    summarize_probe,
)
from evidence_rag.evaluation.needle_visibility import VISIBLE, classify_needle
from evidence_rag.generator.granite import GraniteLLMClient, TextGenerator
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.materializer.provenance import read_provenance
from evidence_rag.selector.extraction import AnswerExtractionEngine


def _read_candidates(path: Path) -> dict[str, CandidateSet]:
    text = Path(path).read_text(encoding="utf-8")
    sets = (
        CandidateSet.model_validate_json(line) for line in text.splitlines() if line.strip()
    )
    return {candidate_set.query_id: candidate_set for candidate_set in sets}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Needle-extraction failure-mode probe")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--passage-chars", type=int, default=600)
    return parser


def main(argv: Sequence[str] | None = None, *, llm: TextGenerator | None = None) -> int:
    arguments = _parser().parse_args(argv)
    bundle = JsonlDatasetAdapter.load(arguments.manifest)
    query_by_id = {query.query_id: query for query in bundle.queries}
    gold_by_id = {gold_case.query_id: gold_case for gold_case in bundle.gold_cases}
    candidates_by_id = _read_candidates(arguments.candidates)
    records = read_provenance(arguments.provenance)

    engine = AnswerExtractionEngine(
        llm if llm is not None else GraniteLLMClient(),
        passage_chars=arguments.passage_chars,
        use_parametric=False,
    )

    outcomes: list[ProbeOutcome] = []
    dump_rows: list[dict[str, object]] = []
    for record in records:
        candidate_set = candidates_by_id[record.query_id]
        query = query_by_id[record.query_id]
        gold_case = gold_by_id.get(record.query_id)
        gold_aliases = (gold_case.reference_answers or ()) if gold_case else ()
        window = tuple(
            sorted(candidate_set.candidates, key=lambda item: item.retrieval_rank)
        )[: arguments.top_n]
        needle_chunks = tuple(c for c in window if c.document_id == record.needle_document_id)
        if not needle_chunks:
            continue  # needle not retrieved into the window; matches the recovery denominator
        answers = engine.extract(query, needle_chunks).answers
        outcome = classify_extraction(answers, record.gold_value)
        visible = classify_needle(
            window, record.needle_document_id, gold_aliases, passage_chars=arguments.passage_chars
        ) == VISIBLE
        outcomes.append(
            ProbeOutcome(
                query_id=record.query_id,
                visible=visible,
                outcome=outcome,
                extracted=answers[0],
                gold_value=record.gold_value,
            )
        )
        dump_rows.append(
            {
                "query_id": record.query_id,
                "question": query.text,
                "gold_value": record.gold_value,
                "extracted": answers[0],
                "visible": visible,
                "outcome": outcome,
            }
        )

    summary = summarize_probe(outcomes)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(dataclasses.asdict(summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    arguments.dump.parent.mkdir(parents=True, exist_ok=True)
    arguments.dump.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in dump_rows), encoding="utf-8"
    )
    print(json.dumps(dataclasses.asdict(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
