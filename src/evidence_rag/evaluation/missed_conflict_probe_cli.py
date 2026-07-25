"""CLI: prompt probe for the twin missed-conflict (systematic-debugging Phase 3, GPU).

Extracts from the needle and counterfactual passages under each prompt in PROMPTS, classifies whether
the twins collapsed to the same answer, and reports missed-conflict / needle-gold / cf-replacement per
prompt — so baseline can be compared against the targeted prompts. ~2 x len(PROMPTS) LLM calls per
injected query (~9k calls, short GPU job). No retrieval: extracts from the source needle/cf documents,
isolating the prompt's effect on twin separation.
"""

import argparse
import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.evaluation.missed_conflict_probe import (
    PROMPTS,
    PairOutcome,
    classify_pair,
    summarize_prompt,
)
from evidence_rag.generator.granite import GraniteLLMClient, TextGenerator
from evidence_rag.materializer.provenance import read_provenance


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
    parser = argparse.ArgumentParser(description="Prompt probe for the twin missed-conflict")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--passage-chars", type=int, default=600)
    return parser


def main(argv: Sequence[str] | None = None, *, llm: TextGenerator | None = None) -> int:
    arguments = _parser().parse_args(argv)
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    root = arguments.manifest.parent
    records = read_provenance(arguments.provenance)
    needed = {record.needle_document_id for record in records} | {
        record.counterfactual_document_id for record in records
    }
    docs = _read_jsonl_field(root / manifest["documents_file"], "document_id", "text", keep=needed)
    queries = _read_jsonl_field(root / manifest["queries_file"], "query_id", "text")
    client = llm if llm is not None else GraniteLLMClient()

    outcomes: dict[str, list[PairOutcome]] = {name: [] for name in PROMPTS}
    for record in records:
        question = queries[record.query_id]
        needle_text = docs[record.needle_document_id][: arguments.passage_chars]
        cf_text = docs[record.counterfactual_document_id][: arguments.passage_chars]
        for name, template in PROMPTS.items():
            needle_answer = client.generate(
                template.format(question=question, passage=needle_text)
            ).strip()
            cf_answer = client.generate(
                template.format(question=question, passage=cf_text)
            ).strip()
            outcomes[name].append(
                classify_pair(
                    needle_answer,
                    cf_answer,
                    gold_value=record.gold_value,
                    replacement_value=record.replacement_value,
                )
            )

    payload = {
        "n_records": len(records),
        "prompts": {
            name: dataclasses.asdict(summarize_prompt(name, outcomes[name])) for name in PROMPTS
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
