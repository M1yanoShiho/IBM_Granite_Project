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
converter would produce fluent sentences that are simply not the QA2D transform. The verified
ids, and the input format each one was verified in, are in INPUT_TEMPLATE below; anything else
is refused rather than guessed at.

The sentences written here are normalised in exactly one respect — see
`normalise_terminal_space` — so the arm does not differ from the other two ablation rungs on
tokenizer spacing, which is not the variable under test.
"""

import argparse
import importlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.materializer.provenance import MutationRecord, read_provenance
from evidence_rag.relations.qa2d import (
    ANSWER_FIELD,
    DECLARATIVE_FIELD,
    MODEL_FIELD,
    QUESTION_FIELD,
    TEMPLATE_FIELD,
    CacheKey,
    cache_key,
)

Qa2dGenerator = Callable[[Sequence[CacheKey]], Sequence[str]]

# QA2D converters differ in how they join the question and the answer (a task prefix, a separator
# token, a bare space) and the format is per checkpoint. A wrong one does not crash: it degrades
# into fluent sentences that are not the QA2D transform, and no downstream number can tell. So
# unknown ids are rejected rather than guessed, exactly as with LABEL_ORDER in cli/gate0b.py.
#
# The entry below was verified against the real checkpoint on 2026-08-03: the format is taken
# verbatim from the model card's usage example ("question: what day is it today? answer:
# Tuesday"), and 30 real (question, answer) pairs from this project's own probe were run through
# it on the cluster in that format, e.g.
#   ("when did they stop making the half dollar", "2002")
#       -> "They stopped making the half dollar in 2002 ."
#   ("which layer of the atmosphere is the ozone layer located", "stratosphere")
#       -> "The ozone layer is located in the stratosphere ."
# Re-verify against the card AND on real pairs before adding any new checkpoint.
#
# Rejected, recorded so it is not re-proposed: "domenicrosati/question_converter-3b" (T5-3B,
# format "{question} </s> {answer}") has no declared licence, no published evaluation numbers,
# 22.8 GB of weights, and does not load under this repo's pinned torch < 2.6 (it requires
# torch >= 2.6, and upgrading breaks safetensors loading elsewhere here).
INPUT_TEMPLATE: dict[str, str] = {
    # BART-base (~140M), AFL-3.0, BLEU 78.878 on the QA2D testset (vs 74.05 for the 2019
    # pointer-generator baseline).
    "MarkS/bart-base-qa2d": "question: {question} answer: {answer}",
}

BATCH_SIZE = 16
MAX_NEW_TOKENS = 64
# Each MutationRecord contributes the gold claim and the replacement claim, and the probe builds
# both from the record's own question.
ANSWERS_PER_RECORD = 2


def qa2d_input(question: str, answer: str, template: str) -> str:
    """The string handed to the converter, in the format its checkpoint was verified in.

    The template is a parameter rather than a constant because it belongs to the checkpoint, and
    the only place to get one is INPUT_TEMPLATE. A mismatched format does not crash, it degrades
    into fluent sentences that are not the QA2D transform.
    """
    return template.format(question=question.strip(), answer=answer.strip())


# Anchored to the end on purpose. BART's detokenisation writes "... in 2002 ." while the two
# deterministic rungs in relations/claims.py write "... is 2002.", and that spacing is a surface
# artefact of the tokenizer, not the variable this ablation manipulates (sentence form). An
# unanchored rule would also rewrite detokenised abbreviations such as "D . C ." mid-sentence,
# which IS content.
_TERMINAL_SPACE = re.compile(r"\s+([.!?])$")


def normalise_terminal_space(sentence: str) -> str:
    """Delete whitespace immediately before the sentence's final `.`, `!` or `?`, and nothing
    else.

    Deliberately not a tidy-up: no lowercasing, no re-casing, no stripping, no spacing fixes
    anywhere but that one position. The generated sentences are experimental data, so every
    edit to them has to be declarable in a single sentence in the protocol record.
    """
    return _TERMINAL_SPACE.sub(r"\1", sentence)


def input_template(model_id: str) -> str:
    """The verified input format for a checkpoint, or a loud failure.

    Guards every path that needs the template, not just generation: a caller holding an
    unregistered id has no verified format, and a wrong one yields fluent sentences that are not
    the QA2D transform — undetectable downstream.
    """
    if model_id not in INPUT_TEMPLATE:
        raise ValueError(
            f"no verified input template for {model_id!r}; take the format from the model card, "
            "confirm it on real pairs, and add it to INPUT_TEMPLATE before generating a cache"
        )
    return INPUT_TEMPLATE[model_id]


def load_generator(model_id: str) -> Qa2dGenerator:
    """Load the seq2seq converter. Seam for testing: the only place torch/transformers are
    imported, so the unit tests substitute this and never touch a model.

    Fails loudly on an unverified model id — see INPUT_TEMPLATE. The check precedes the imports
    so an unregistered id costs nothing and never reaches the Hub.
    """
    input_template(model_id)
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")

    template = INPUT_TEMPLATE[model_id]
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
                [qa2d_input(question, answer, template) for question, answer in batch],
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
    template = input_template(arguments.model)
    generate = load_generator(arguments.model)
    sentences = [normalise_terminal_space(sentence.strip()) for sentence in generate(pairs)]

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        "".join(
            json.dumps(
                {
                    QUESTION_FIELD: question,
                    ANSWER_FIELD: answer,
                    DECLARATIVE_FIELD: sentence,
                    MODEL_FIELD: arguments.model,
                    TEMPLATE_FIELD: template,
                },
                sort_keys=True,
            )
            + "\n"
            for (question, answer), sentence in zip(pairs, sentences, strict=True)
        ),
        encoding="utf-8",
    )
    usable = [record for record in records if record.query_id in question_by_query]
    # R012b 判读纪律 5: questions and answers both arrive lowercased, so any capital past
    # position 0 was introduced by the converter. A high rate means rung 3 varies casing on top
    # of sentence form and its increment cannot be attributed to form alone. Emitted here so the
    # obligation is discharged by running the tool, not by remembering to check.
    recased = sum(1 for sentence in sentences if any(char.isupper() for char in sentence[1:]))
    report = {
        "model": arguments.model,
        "n_pairs": len(usable) * ANSWERS_PER_RECORD,
        "n_unique": len(pairs),
        "non_initial_uppercase_rate": round(recased / len(sentences), 4) if sentences else 0.0,
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
