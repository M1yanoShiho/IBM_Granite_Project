"""CLI: 2x2 prompt/model probe for the twin missed-conflict (systematic-debugging, GPU).

For each injected query it extracts the needle and counterfactual passages under three strategies and
reports missed-conflict / needle-gold / cf-replacement per strategy:
  - baseline   : the current single-stage EXTRACT_PROMPT;
  - attribute  : the best single-stage targeted prompt from Phase 3;
  - decoupled  : two-stage — Stage A names the queried target once per query, Stage B copies that
                 target's value from each passage (separates reasoning from span-extraction).
The model is whatever GRANITE_MODEL_ID selects (3B vs 8B), recorded in the report — run it under each
model for the full 2x2. --limit subsamples for a fast pilot. No retrieval: extracts from source docs.
"""

import argparse
import dataclasses
import json
import os
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.evaluation.missed_conflict_probe import (
    PROMPTS,
    STAGE_A_PROMPT,
    STAGE_B_PROMPT,
    PairOutcome,
    classify_pair,
    summarize_prompt,
)
from evidence_rag.generator.granite import GraniteLLMClient, TextGenerator
from evidence_rag.materializer.provenance import read_provenance

SINGLE_STAGE = ("baseline", "attribute")
STRATEGIES = (*SINGLE_STAGE, "decoupled")


def _read_jsonl_field(
    path: Path, key_field: str, value_field: str, keep: set[str] | None = None
) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row[key_field]
        if keep is None or key in keep:
            result[key] = row[value_field]
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="2x2 prompt/model probe for the twin missed-conflict")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--passage-chars", type=int, default=600)
    parser.add_argument("--limit", type=int, default=0, help="subsample the first N queries (0 = all)")
    return parser


def _pair_answers(
    llm: TextGenerator, strategy: str, question: str, needle_text: str, cf_text: str
) -> tuple[str, str]:
    if strategy == "decoupled":
        target = llm.generate(STAGE_A_PROMPT.format(question=question)).strip()
        needle = llm.generate(STAGE_B_PROMPT.format(target=target, passage=needle_text)).strip()
        cf = llm.generate(STAGE_B_PROMPT.format(target=target, passage=cf_text)).strip()
        return needle, cf
    template = PROMPTS[strategy]
    needle = llm.generate(template.format(question=question, passage=needle_text)).strip()
    cf = llm.generate(template.format(question=question, passage=cf_text)).strip()
    return needle, cf


def main(argv: Sequence[str] | None = None, *, llm: TextGenerator | None = None) -> int:
    arguments = _parser().parse_args(argv)
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    root = arguments.manifest.parent
    records = read_provenance(arguments.provenance)
    if arguments.limit > 0:
        records = records[: arguments.limit]
    needed = {record.needle_document_id for record in records} | {
        record.counterfactual_document_id for record in records
    }
    docs = _read_jsonl_field(root / manifest["documents_file"], "document_id", "text", keep=needed)
    queries = _read_jsonl_field(root / manifest["queries_file"], "query_id", "text")
    client = llm if llm is not None else GraniteLLMClient()

    outcomes: dict[str, list[PairOutcome]] = {name: [] for name in STRATEGIES}
    for record in records:
        question = queries[record.query_id]
        needle_text = docs[record.needle_document_id][: arguments.passage_chars]
        cf_text = docs[record.counterfactual_document_id][: arguments.passage_chars]
        for strategy in STRATEGIES:
            needle_answer, cf_answer = _pair_answers(
                client, strategy, question, needle_text, cf_text
            )
            outcomes[strategy].append(
                classify_pair(
                    needle_answer,
                    cf_answer,
                    gold_value=record.gold_value,
                    replacement_value=record.replacement_value,
                )
            )

    payload = {
        "model": os.environ.get("GRANITE_MODEL_ID", "unknown"),
        "n_records": len(records),
        "strategies": {
            name: dataclasses.asdict(summarize_prompt(name, outcomes[name])) for name in STRATEGIES
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
