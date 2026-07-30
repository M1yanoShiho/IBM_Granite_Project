"""CLI: offline E1 edge/cluster component eval over the injected NIAH pools (spec §12).

Reuses the E2 candidate_sets.jsonl (pools carry passage text), runs one Granite extraction
pass over each injected query's top_n window, and reports missed/false-conflict, needle-gold
recovery (Wilson CIs), plus the deterministic injection selection-bias framing line.
"""

import argparse
import dataclasses
import json
import os
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate
from evidence_rag.evaluation.cluster_eval import (
    ClusterEvalCase,
    aggregate,
    contains_alias,
    evaluate_case,
    selection_bias,
)
from evidence_rag.evaluation.missed_conflict_probe import STAGE_A_PROMPT, STAGE_B_PROMPT
from evidence_rag.generator.granite import GraniteLLMClient, TextGenerator
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.materializer.provenance import read_provenance
from evidence_rag.selector.answer_equivalence import lenient_equivalent
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
    parser.add_argument(
        "--extraction",
        choices=("single", "decoupled"),
        default="single",
        help="single = the shared EXTRACT_PROMPT; decoupled = Stage A target + Stage B copy (S5)",
    )
    parser.add_argument(
        "--dump",
        type=Path,
        help="write per-case rows (raw answers + alias flags) to this jsonl; required by M0 G-PQ",
    )
    parser.add_argument("--limit", type=int, default=0, help="subsample the first N injected queries")
    return parser


def _decoupled_answers(
    llm: TextGenerator,
    question: str,
    window: Sequence[EvidenceCandidate],
    passage_chars: int,
) -> tuple[str, ...]:
    """Two-stage extraction (S5): name the queried target once, then copy it from each passage."""
    target = llm.generate(STAGE_A_PROMPT.format(question=question)).strip()
    return tuple(
        llm.generate(
            STAGE_B_PROMPT.format(
                target=target, question=question, passage=candidate.text[:passage_chars]
            )
        ).strip()
        for candidate in window
    )


def main(argv: Sequence[str] | None = None, *, llm: TextGenerator | None = None) -> int:
    arguments = _parser().parse_args(argv)
    bundle = JsonlDatasetAdapter.load(arguments.manifest)
    query_by_id = {query.query_id: query for query in bundle.queries}
    gold_by_id = {gold_case.query_id: gold_case for gold_case in bundle.gold_cases}
    candidates_by_id = _read_candidates(arguments.candidates)
    records = read_provenance(arguments.provenance)
    # selection-bias is a property of the dataset, so it is computed from the FULL injected set even
    # when --limit subsamples the evaluated queries (otherwise skip_rate reports the subsample size).
    n_injected_total = len(records)
    if arguments.limit > 0:
        records = records[: arguments.limit]

    client = llm if llm is not None else GraniteLLMClient()
    engine = AnswerExtractionEngine(
        client,
        passage_chars=arguments.passage_chars,
        use_parametric=False,
    )

    exact_cases: list[ClusterEvalCase] = []
    lenient_cases: list[ClusterEvalCase] = []
    dump_rows: list[dict[str, object]] = []
    for record in records:
        query = query_by_id[record.query_id]
        candidate_set = candidates_by_id[record.query_id]
        gold_case = gold_by_id.get(record.query_id)
        window = tuple(
            sorted(candidate_set.candidates, key=lambda item: item.retrieval_rank)
        )[: arguments.top_n]
        if arguments.extraction == "decoupled":
            answers = _decoupled_answers(client, query.text, window, arguments.passage_chars)
        else:
            answers = engine.extract(query, window).answers
        gold_aliases = (gold_case.reference_answers or ()) if gold_case else ()
        exact_cases.append(
            evaluate_case(
                window,
                answers,
                query_id=record.query_id,
                needle_document_id=record.needle_document_id,
                counterfactual_document_id=record.counterfactual_document_id,
                gold_value=record.gold_value,
                gold_aliases=gold_aliases,
            )
        )
        lenient_cases.append(
            evaluate_case(
                window,
                answers,
                query_id=record.query_id,
                needle_document_id=record.needle_document_id,
                counterfactual_document_id=record.counterfactual_document_id,
                gold_value=record.gold_value,
                gold_aliases=gold_aliases,
                equivalence=lenient_equivalent,
            )
        )
        if arguments.dump is not None:
            dump_rows.append(
                {
                    "query_id": record.query_id,
                    "needle_document_id": record.needle_document_id,
                    "counterfactual_document_id": record.counterfactual_document_id,
                    "gold_value": record.gold_value,
                    "gold_aliases": list(gold_aliases),
                    "window": [
                        {
                            "evidence_id": candidate.evidence_id,
                            "document_id": candidate.document_id,
                            "retrieval_rank": candidate.retrieval_rank,
                            "answer": answer,
                            "contains_gold_alias": contains_alias(candidate.text, gold_aliases),
                        }
                        for candidate, answer in zip(window, answers, strict=True)
                    ],
                    "exact": dataclasses.asdict(exact_cases[-1]),
                    "lenient": dataclasses.asdict(lenient_cases[-1]),
                }
            )

    bias = selection_bias(
        (gold_case.reference_answers for gold_case in bundle.gold_cases),
        n_injected=n_injected_total,
    )

    def _scored(cases: list[ClusterEvalCase]) -> dict[str, object]:
        report = aggregate(cases)
        return {
            "missed_conflict": dataclasses.asdict(report.missed_conflict),
            "false_conflict": dataclasses.asdict(report.false_conflict),
            "needle_gold_recovery": dataclasses.asdict(report.needle_gold_recovery),
            "n_cases": report.n_cases,
            "needle_in_window": report.needle_in_window,
            "cf_in_window": report.cf_in_window,
        }

    payload = {
        "model": os.environ.get("GRANITE_MODEL_ID", "unknown"),
        "extraction": arguments.extraction,
        "exact": _scored(exact_cases),
        "lenient": _scored(lenient_cases),
        "selection_bias": dataclasses.asdict(bias),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if arguments.dump is not None:
        arguments.dump.parent.mkdir(parents=True, exist_ok=True)
        arguments.dump.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in dump_rows), encoding="utf-8"
        )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
