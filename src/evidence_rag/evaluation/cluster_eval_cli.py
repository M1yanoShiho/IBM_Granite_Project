"""CLI: offline E1 edge/cluster component eval over the injected NIAH pools (spec §12).

Reuses the E2 candidate_sets.jsonl (pools carry passage text), runs one Granite extraction
pass over each injected query's top_n window, and reports missed/false-conflict, needle-gold
recovery (Wilson CIs), plus the deterministic injection selection-bias framing line.
"""

import argparse
import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.contracts.models import CandidateSet
from evidence_rag.evaluation.cluster_eval import aggregate, evaluate_case, selection_bias
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
    parser = argparse.ArgumentParser(description="Offline E1 edge/cluster component eval")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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

    cases = []
    for record in records:
        query = query_by_id[record.query_id]
        candidate_set = candidates_by_id[record.query_id]
        gold_case = gold_by_id.get(record.query_id)
        window = tuple(
            sorted(candidate_set.candidates, key=lambda item: item.retrieval_rank)
        )[: arguments.top_n]
        extracted = engine.extract(query, window)
        cases.append(
            evaluate_case(
                window,
                extracted.answers,
                query_id=record.query_id,
                needle_document_id=record.needle_document_id,
                counterfactual_document_id=record.counterfactual_document_id,
                gold_value=record.gold_value,
                gold_aliases=(gold_case.reference_answers or ()) if gold_case else (),
            )
        )

    report = aggregate(cases)
    bias = selection_bias(
        (gold_case.reference_answers for gold_case in bundle.gold_cases),
        n_injected=len(records),
    )
    payload = {
        "missed_conflict": dataclasses.asdict(report.missed_conflict),
        "false_conflict": dataclasses.asdict(report.false_conflict),
        "needle_gold_recovery": dataclasses.asdict(report.needle_gold_recovery),
        "n_cases": report.n_cases,
        "needle_in_window": report.needle_in_window,
        "cf_in_window": report.cf_in_window,
        "selection_bias": dataclasses.asdict(bias),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
