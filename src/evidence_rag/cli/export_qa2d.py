"""CLI: pre-generate the QA2D hypothesis cache (design §2.4, ablation rung 3).

RUN THIS ON THE LOGIN NODE. It is the only step in the rung-3 arm that needs a network and a
seq2seq model; the compute nodes run HF-offline, and the probe itself
(`evidence_rag.cli.export_task_probe --hypothesis-form qa2d`) only reads the cache this writes.
Splitting it that way is what keeps the arm deterministic and auditable: the generated sentences
are frozen artefacts on disk that a reviewer can diff, not a model invocation buried inside the
export.

Point --manifest and --provenance at the SAME train-split pair the probe will use. The cache is
keyed on (question, answer), so a cache built from a different split silently under-covers and
the probe then dies on a Qa2dCacheMiss.

--model is REQUIRED and has no default, deliberately. This repo does not ship unverified
checkpoint ids (see LABEL_ORDER in evidence_rag/cli/gate0b.py, where a guessed entry would swap
two classes while every number stayed plausible); the same rule applies here, where a guessed
converter would produce fluent sentences that are simply not the QA2D transform.

Candidate checkpoints — UNVERIFIED, NOT endorsed by this repo, listed only so a human has
somewhere to start. Confirm the id resolves on the Hub, that it is a question+answer -> statement
converter, and what input format its card specifies, BEFORE any generated cache feeds a reported
number:
  - "domenicrosati/question_converter-3b"
  - "MarkS/bart-base-qa2d"
Neither was checked by the agent that wrote this file. Treat both as leads, not as defaults.
"""

import argparse
import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.materializer.provenance import MutationRecord, read_provenance
from evidence_rag.relations.qa2d import (
    ANSWER_FIELD,
    DECLARATIVE_FIELD,
    QUESTION_FIELD,
    CacheKey,
    cache_key,
)

Qa2dGenerator = Callable[[Sequence[CacheKey]], Sequence[str]]

BATCH_SIZE = 16
MAX_NEW_TOKENS = 64
# Each MutationRecord contributes the gold claim and the replacement claim, and the probe builds
# both from the record's own question.
ANSWERS_PER_RECORD = 2


def qa2d_input(question: str, answer: str) -> str:
    """The string handed to the converter.

    UNVERIFIED in the same way --model is: released QA2D converters differ in how they join the
    question and the answer (a separator token, a period, a task prefix). Confirm this against
    the model card of whatever checkpoint is supplied. A mismatched input format does not crash,
    it degrades into fluent sentences that are not the QA2D transform.
    """
    return f"{question.strip()} {answer.strip()}"


def load_generator(model_id: str) -> Qa2dGenerator:
    """Load the seq2seq converter. Seam for testing: the only place torch/transformers are
    imported, so the unit tests substitute this and never touch a model."""
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[export-qa2d] {model_id} on {device}", flush=True)

    def generate(pairs: Sequence[CacheKey]) -> Sequence[str]:
        sentences: list[str] = []
        for start in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[start : start + BATCH_SIZE]
            encoded = tokenizer(
                [qa2d_input(question, answer) for question, answer in batch],
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                # Greedy decoding: do_sample=False with num_beams=1 makes a re-run of this
                # script reproduce the cache, which is the only reason a seq2seq model is
                # allowed anywhere near a pre-registered arm. Do not add sampling, temperature,
                # or beams without re-freezing the cache and saying so in the protocol record.
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
            sentences.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
            print(f"[export-qa2d] {start + len(batch)}/{len(pairs)}", flush=True)
        return sentences

    return generate


def collect_pairs(
    *,
    records: Sequence[MutationRecord],
    question_by_query: Mapping[str, str],
) -> tuple[CacheKey, ...]:
    """Every (question, answer) the probe will ask for, deduplicated, in first-seen order.

    Records whose query is missing are skipped because no key can be formed for them; that
    matches what `build_probe_pairs` does. Records whose DOCUMENTS are missing are NOT skipped —
    the probe drops those, so the cache is a deliberate superset. A superset costs a few seconds
    of login-node time; a subset costs a Qa2dCacheMiss hours later on a compute node.
    """
    pairs: dict[CacheKey, None] = {}
    for record in records:
        question = question_by_query.get(record.query_id)
        if question is None:
            continue
        for answer in (record.gold_value, record.replacement_value):
            pairs.setdefault(cache_key(question, answer), None)
    return tuple(pairs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pre-generate the QA2D hypothesis cache")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="QA2D cache jsonl")
    parser.add_argument(
        "--model",
        required=True,
        help="QA2D converter checkpoint id; no default on purpose, see the module docstring",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    bundle = JsonlDatasetAdapter.load(arguments.manifest)
    records = read_provenance(arguments.provenance)
    question_by_query = {query.query_id: query.text for query in bundle.queries}

    pairs = collect_pairs(records=records, question_by_query=question_by_query)
    generate = load_generator(arguments.model)
    sentences = generate(pairs)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        "".join(
            json.dumps(
                {
                    QUESTION_FIELD: question,
                    ANSWER_FIELD: answer,
                    DECLARATIVE_FIELD: sentence.strip(),
                },
                sort_keys=True,
            )
            + "\n"
            for (question, answer), sentence in zip(pairs, sentences, strict=True)
        ),
        encoding="utf-8",
    )
    usable = [record for record in records if record.query_id in question_by_query]
    report = {
        "model": arguments.model,
        "n_pairs": len(usable) * ANSWERS_PER_RECORD,
        "n_unique": len(pairs),
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
