"""Materialise leakage-safe F006 clean/mixed key-fact LoRA examples.

Only the NIAH training pool is read.  Any query ID present in NIAH dev is
excluded before eligibility is assessed.  A target is accepted only when a
reference answer is an exact normalised substring of a labelled-relevant TopK10
candidate.  No sealed or held-out source is accepted by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from evidence_rag.contracts.models import EvidenceCandidate
from evidence_rag.generator.key_facts import NOTES_PROMPT, detect_question_slots

TRAIN_QUERIES = 1000
VALIDATION_QUERIES = 100
MAX_MIXED_EVIDENCE = 9


def _jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        yield value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", text))


def _stable_order(query_id: str) -> str:
    return hashlib.sha256(f"full-flow-f006-v1\n{query_id}".encode()).hexdigest()


def _context(candidates: Sequence[EvidenceCandidate]) -> str:
    return "\n".join(f"[{index}] {item.text}" for index, item in enumerate(candidates, start=1))


def _prompt(question: str, slot: str, candidates: Sequence[EvidenceCandidate]) -> str:
    return NOTES_PROMPT.format(slots=slot, context=_context(candidates), question=question)


def _target(slot: str, answer: str, evidence_index: int) -> str:
    return json.dumps(
        {"facts": [{"slot": slot, "value": answer, "evidence": [evidence_index]}]},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_cases(
    train_queries: Sequence[Mapping[str, Any]],
    train_candidates: Sequence[Mapping[str, Any]],
    train_gold: Sequence[Mapping[str, Any]],
    dev_query_ids: set[str],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    questions = {str(row["query_id"]): str(row["text"]) for row in train_queries}
    candidates_by_id = {str(row["query_id"]): row["candidates"] for row in train_candidates}
    gold_by_id = {str(row["query_id"]): row for row in train_gold}
    if set(questions) != set(candidates_by_id) or set(questions) != set(gold_by_id):
        raise ValueError("NIAH train queries/candidates/gold query IDs differ")

    counts: Counter[str] = Counter()
    output: list[dict[str, object]] = []
    for query_id, question in questions.items():
        counts["train_pool_queries"] += 1
        if query_id in dev_query_ids:
            counts["excluded_train_dev_overlap"] += 1
            continue
        slots = detect_question_slots(question)
        if len(slots) != 1:
            counts["excluded_not_exactly_one_slot"] += 1
            continue
        slot = slots[0]
        candidates = tuple(
            EvidenceCandidate.model_validate(item) for item in candidates_by_id[query_id]
        )[:10]
        relevant_ids = {str(item) for item in gold_by_id[query_id]["relevant_document_ids"]}
        references = tuple(str(item) for item in gold_by_id[query_id]["reference_answers"])
        matches: list[tuple[EvidenceCandidate, str]] = []
        for candidate in candidates:
            if candidate.document_id not in relevant_ids:
                continue
            for reference in references:
                if _normalise(reference) and _normalise(reference) in _normalise(candidate.text):
                    matches.append((candidate, reference))
        if not matches:
            counts["excluded_no_exact_relevant_answer"] += 1
            continue
        distractors = tuple(item for item in candidates if item.document_id not in relevant_ids)
        if not distractors:
            counts["excluded_no_nonrelevant_distractor"] += 1
            continue

        # Prefer the earliest relevant candidate, then the shortest exact reference.
        clean, answer = min(
            matches,
            key=lambda item: (item[0].retrieval_rank, len(_normalise(item[1])), item[1]),
        )
        mixed_members = sorted(
            (clean, *distractors[: MAX_MIXED_EVIDENCE - 1]),
            key=lambda item: item.retrieval_rank,
        )
        mixed_index = mixed_members.index(clean) + 1
        clean_target = _target(slot, answer, 1)
        mixed_target = _target(slot, answer, mixed_index)
        output.append(
            {
                "schema_version": "full-flow-f006-case-v1",
                "query_id": query_id,
                "question": question,
                "slot": slot,
                "answer": answer,
                "clean_evidence_id": clean.evidence_id,
                "mixed_evidence_ids": [item.evidence_id for item in mixed_members],
                "clean_prompt": _prompt(question, slot, (clean,)),
                "mixed_prompt": _prompt(question, slot, mixed_members),
                "clean_target": clean_target,
                "mixed_target": mixed_target,
            }
        )
        counts["eligible"] += 1
        counts[f"eligible_{slot}"] += 1
    return sorted(output, key=lambda row: _stable_order(str(row["query_id"]))), dict(counts)


def materialize(
    *,
    train_queries: Path,
    train_candidates: Path,
    train_gold: Path,
    dev_queries: Path,
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("F006 data output directory must be absent or empty")
    dev_ids = {str(row["query_id"]) for row in _jsonl(dev_queries)}
    cases, counts = build_cases(
        tuple(_jsonl(train_queries)),
        tuple(_jsonl(train_candidates)),
        tuple(_jsonl(train_gold)),
        dev_ids,
    )
    required = TRAIN_QUERIES + VALIDATION_QUERIES
    if len(cases) < required:
        raise ValueError(f"only {len(cases)} eligible F006 cases; need {required}")
    validation = cases[:VALIDATION_QUERIES]
    training = cases[VALIDATION_QUERIES:required]
    if {row["query_id"] for row in training} & {row["query_id"] for row in validation}:
        raise AssertionError("F006 train/validation query IDs overlap")
    if {str(row["query_id"]) for row in training + validation} & dev_ids:
        raise AssertionError("F006 materialisation leaked NIAH dev query IDs")

    _write_jsonl(output_dir / "train_cases.jsonl", training)
    _write_jsonl(output_dir / "validation_cases.jsonl", validation)
    manifest: dict[str, object] = {
        "schema_version": "full-flow-f006-data-manifest-v1",
        "status": "COMPLETE",
        "source_role": "NIAH train only",
        "forbidden_roles_read": [],
        "dev_used_only_as_query_id_exclusion_set": True,
        "sealed_or_heldout_read": False,
        "selection_rule": "sha256(full-flow-f006-v1\\nquery_id)",
        "eligibility": (
            "exactly one question slot; reference exact-normalised substring of a labelled-"
            "relevant TopK10 candidate; at least one labelled-nonrelevant distractor"
        ),
        "counts": counts,
        "train_queries": len(training),
        "validation_queries": len(validation),
        "examples_per_training_query": 2,
        "clean_only_examples": 2 * len(training),
        "mixed_examples": 2 * len(training),
        "clean_only_definition": "clean prompt duplicated twice",
        "mixed_definition": "one clean prompt plus one mixed prompt",
        "train_cases_sha256": _sha256(output_dir / "train_cases.jsonl"),
        "validation_cases_sha256": _sha256(output_dir / "validation_cases.jsonl"),
        "input_sha256": {
            "train_queries": _sha256(train_queries),
            "train_candidates": _sha256(train_candidates),
            "train_gold": _sha256(train_gold),
            "dev_queries_exclusion_only": _sha256(dev_queries),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-queries", required=True, type=Path)
    parser.add_argument("--train-candidates", required=True, type=Path)
    parser.add_argument("--train-gold", required=True, type=Path)
    parser.add_argument("--dev-queries", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = materialize(
        train_queries=args.train_queries,
        train_candidates=args.train_candidates,
        train_gold=args.train_gold,
        dev_queries=args.dev_queries,
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
