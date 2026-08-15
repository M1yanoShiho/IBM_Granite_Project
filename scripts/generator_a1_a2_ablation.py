"""G-A12: 2x2 ablation of the historical A1 prompt and A2 claim splitter.

This is an experiment-only runner. It reconstructs the exact behavioural delta
between parent commit 11c03849 and xzy's eda7065 commit without changing the
production Generator. Each old/new A1 answer is passed to both A2 variants, so
the A2 comparison is paired on byte-identical answer text.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import g3_baseline_comparison as g3  # noqa: E402, I001
from evidence_rag.contracts.models import Query, SelectedEvidenceSet  # noqa: E402
from evidence_rag.generator.granite import (  # noqa: E402
    GraniteGenerationConfig,
    GraniteLLMClient,
    TextGenerator,
)
from evidence_rag.generator.json_parsing import parse_json_object  # noqa: E402
from evidence_rag.generator.models import Claim, ClaimSpan  # noqa: E402

OLD_COMMIT = "11c03849b3bcb21bf83447b4726714d4bc762725"
NEW_COMMIT = "eda7065d8005e317ce951e2730f58996cd4757ec"
ARMS = ("old_old", "old_new", "new_old", "new_new")

OLD_DRAFT_PROMPT = (
    "Answer the question using only the evidence below.\n"
    "Cover every part of the question the evidence supports. If the question has "
    "more than one reasonable reading, address each reading you can.\n"
    "End every sentence with the bracketed number(s) of the evidence that supports "
    "it, for example: The sky is blue [1][3].\n\n"
    "If the evidence does not contain the answer, say: I don't know.\n\n"
    "Evidence:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

NEW_DRAFT_PROMPT = (
    "Answer the question using only the evidence below.\n"
    "Cover every part of the question the evidence supports. If the question has "
    "more than one reasonable reading, address each reading you can.\n"
    "Write one independently verifiable factual claim per sentence. Keep every "
    "sentence self-contained and do not combine separate facts into one sentence.\n"
    "Name the relevant entities, dates, quantities, and conditions explicitly. "
    "Avoid unresolved pronouns such as it, they, this, or that.\n"
    "Do not describe the answer, the evidence, or the sources. Do not include "
    "unsupported intermediate reasoning.\n"
    "End every factual sentence with the bracketed number(s) of the evidence that "
    "supports it, for example: The sky is blue [1][3]. Use only the evidence "
    "numbers shown below.\n\n"
    "If the evidence does not contain the answer, say: I don't know.\n\n"
    "Evidence:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

SPLIT_PROMPT = (
    "Split the answer into atomic, self-contained factual claims that could each "
    "be checked against source documents. Resolve pronouns without adding "
    "information. Skip any sentence that only describes the answer itself or where "
    "its information came from (e.g. 'this is based on...', 'the information is "
    "sourced from...') rather than asserting a fact about the world. Return JSON "
    "only in this shape: "
    '{{"claims":[{{"source_text":"exact contiguous text from the answer",'
    '"text":"self-contained claim"}}]}}.\n\nAnswer:\n{answer}'
)

FAITHFULNESS_PROMPT = (
    "Check whether each rewritten claim has exactly the same meaning as its source text. "
    "Return JSON only in this shape: "
    '{{"results":[{{"claim_id":"claim-1","faithful":true}}]}}.\n\nItems:\n{items}'
)

UNKNOWN_ANSWERS = {
    "",
    "unknown",
    "i don't know",
    "i do not know",
    "not in the evidence",
    "not contained in the evidence",
}
STOPWORDS = frozenset(
    "a an the of in on at to for and or is are was were be been by with as that this it its "
    "from has have had also his her their he she they there which who whom but not into than "
    "then when while about over under after before".split()
)
OLD_SENTENCE_RE = re.compile(r"[^.!?]*[.!?]+|\S[^.!?]*$")
MIN_SENTENCE_OVERLAP = 0.3
META_NARRATIVE = re.compile(
    r"\b(?:"
    r"the (?:answer|information|evidence|passage|document|source|text|context)s?\b"
    r"|this (?:answer|information|statement)\b"
    r"|(?:is|was|are|were) (?:sourced|derived|taken) from"
    r"|comes from the"
    r"|according to the (?:evidence|passage|document|source|text|context)"
    r"|mentioned in the (?:answer|evidence|passage|document|text)"
    r")",
    re.IGNORECASE,
)
UNRESOLVED_SUBJECT = re.compile(
    r"^(?:it|its|this|that|these|those|they|them|their|he|him|his|she|her)\b",
    re.IGNORECASE,
)
NON_TERMINAL_ABBREVIATIONS = frozenset(
    {"dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc"}
)
REDUNDANT_COVERAGE = 0.8


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


@dataclass(frozen=True)
class CallContext:
    query_id: str
    stage: str
    arm: str


class RecordingLLM:
    """Record every raw response with the experiment context that produced it."""

    def __init__(self, inner: TextGenerator) -> None:
        self.inner = inner
        self.context: CallContext | None = None
        self.records: list[dict[str, Any]] = []

    def set_context(self, query_id: str, stage: str, arm: str) -> None:
        self.context = CallContext(query_id=query_id, stage=stage, arm=arm)

    def generate(self, prompt: str) -> str:
        if self.context is None:
            raise RuntimeError("recording LLM context must be set before generation")
        response = self.inner.generate(prompt)
        self.records.append(
            {
                "call_index": len(self.records) + 1,
                "query_id": self.context.query_id,
                "stage": self.context.stage,
                "arm": self.context.arm,
                "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                "prompt": prompt,
                "response": response,
            }
        )
        return response


def generate_draft(
    llm: RecordingLLM,
    query: Query,
    selected: SelectedEvidenceSet,
    *,
    prompt_version: str,
) -> str:
    if query.query_id != selected.query_id:
        raise ValueError("query and selected evidence query IDs differ")
    if not selected.evidence:
        return ""
    prompt_template = OLD_DRAFT_PROMPT if prompt_version == "old" else NEW_DRAFT_PROMPT
    context = "\n".join(
        f"[{index}] ({item.evidence_id}) {item.text}"
        for index, item in enumerate(selected.evidence, start=1)
    )
    llm.set_context(query.query_id, "a1-draft", prompt_version)
    answer = llm.generate(prompt_template.format(context=context, question=query.text)).strip()
    normalized = answer.lower().strip(".!?\"' ")
    return "" if normalized in UNKNOWN_ANSWERS else answer


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    }


def old_sentence_spans(text: str) -> list[tuple[int, int]]:
    return [
        (match.start(), match.end())
        for match in OLD_SENTENCE_RE.finditer(text)
        if match.group().strip()
    ]


def new_sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for index, character in enumerate(text):
        if character not in ".!?":
            continue
        if character == ".":
            previous = text[index - 1] if index else ""
            following = text[index + 1] if index + 1 < len(text) else ""
            if previous.isdigit() and following.isdigit():
                continue
            if previous.isupper() and following.isupper():
                continue
            if re.search(r"(?:\b[A-Z]\.)+[A-Z]$", text[start:index]):
                continue
            word_match = re.search(r"([A-Za-z]+)$", text[start:index])
            if (
                word_match is not None
                and word_match.group(1).casefold() in NON_TERMINAL_ABBREVIATIONS
            ):
                continue
        end = index + 1
        if text[start:end].strip():
            spans.append((start, end))
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


class HistoricalClaimSplitter:
    """The old or new A2 behaviour at the frozen 11c03849/eda7065 boundary."""

    def __init__(self, llm: RecordingLLM, version: str) -> None:
        if version not in {"old", "new"}:
            raise ValueError("splitter version must be old or new")
        self.llm = llm
        self.version = version
        self.last_diagnostics: dict[str, int] = {}

    def _sentence_spans(self, text: str) -> list[tuple[int, int]]:
        return old_sentence_spans(text) if self.version == "old" else new_sentence_spans(text)

    def _locate_span(
        self,
        answer_text: str,
        source_text: str,
        claim_text: str,
        cursor: int,
    ) -> ClaimSpan | None:
        exact = answer_text.find(source_text, cursor)
        if exact >= 0:
            return ClaimSpan(start=exact, end=exact + len(source_text))
        needle = content_tokens(source_text) | content_tokens(claim_text)
        if not needle:
            return None
        best_overlap = 0.0
        best_span: tuple[int, int] | None = None
        for start, end in self._sentence_spans(answer_text):
            if self.version == "old" and end <= cursor:
                continue
            sentence_tokens = content_tokens(answer_text[start:end])
            if not sentence_tokens:
                continue
            overlap = len(needle & sentence_tokens) / len(needle)
            if overlap > best_overlap:
                best_overlap = overlap
                best_span = (start, end)
        if best_span is not None and best_overlap >= MIN_SENTENCE_OVERLAP:
            return ClaimSpan(start=best_span[0], end=best_span[1])
        return None

    def _filter_claims(
        self,
        pending: list[tuple[str, str, str, ClaimSpan]],
    ) -> list[tuple[str, str, str, ClaimSpan]]:
        kept = []
        meta = unresolved = 0
        for item in pending:
            if META_NARRATIVE.search(item[2]):
                meta += 1
            elif UNRESOLVED_SUBJECT.match(item[2].strip()):
                unresolved += 1
            else:
                kept.append(item)
        result: list[tuple[str, str, str, ClaimSpan]] = []
        duplicates = 0
        if self.version == "old":
            tokens = [content_tokens(item[2]) for item in kept]
            for index, item in enumerate(kept):
                mine = tokens[index]
                subsumed = bool(mine) and any(
                    other_index != index
                    and len(tokens[other_index]) > len(mine)
                    and len(mine & tokens[other_index]) / len(mine) >= REDUNDANT_COVERAGE
                    for other_index in range(len(kept))
                )
                if subsumed:
                    duplicates += 1
                else:
                    result.append(item)
        else:
            seen: set[tuple[str, ...]] = set()
            for item in kept:
                key = tuple(re.findall(r"[a-z0-9]+", item[2].casefold()))
                if key in seen:
                    duplicates += 1
                else:
                    seen.add(key)
                    result.append(item)
        self.last_diagnostics.update(
            {
                "meta_filtered": meta,
                "unresolved_filtered": unresolved,
                "deduplicated": duplicates,
                "claims_after_filter": len(result),
            }
        )
        return result

    def split(self, query_id: str, answer_text: str, arm: str) -> tuple[Claim, ...]:
        self.last_diagnostics = {
            "raw_claims": 0,
            "unlocatable": 0,
            "meta_filtered": 0,
            "unresolved_filtered": 0,
            "deduplicated": 0,
            "claims_after_filter": 0,
        }
        if not answer_text.strip():
            return ()
        self.llm.set_context(query_id, "a2-split", arm)
        split_data = parse_json_object(
            self.llm.generate(SPLIT_PROMPT.format(answer=answer_text))
        )
        raw_claims = split_data.get("claims")
        if not isinstance(raw_claims, list):
            raise ValueError("splitter output must contain a claims array")
        self.last_diagnostics["raw_claims"] = len(raw_claims)
        pending: list[tuple[str, str, str, ClaimSpan]] = []
        cursor = 0
        for index, item in enumerate(raw_claims, start=1):
            claim_id = f"claim-{index}"
            if not isinstance(item, dict):
                raise ValueError("each claim must contain nonempty source_text and text")
            source_text = item.get("source_text")
            claim_text = item.get("text")
            if not isinstance(source_text, str) or not source_text.strip():
                raise ValueError("each claim must contain nonempty source_text and text")
            if not isinstance(claim_text, str) or not claim_text.strip():
                raise ValueError("each claim must contain nonempty source_text and text")
            span = self._locate_span(answer_text, source_text, claim_text, cursor)
            if span is None:
                self.last_diagnostics["unlocatable"] += 1
                continue
            located_source = answer_text[span.start : span.end]
            faithful_source = source_text if self.version == "old" else located_source
            pending.append((claim_id, faithful_source, claim_text, span))
            cursor = span.end
        pending = self._filter_claims(pending)
        if not pending:
            return ()
        items = json.dumps(
            [
                {"claim_id": claim_id, "source_text": source, "claim_text": text}
                for claim_id, source, text, _span in pending
            ],
            ensure_ascii=False,
        )
        self.llm.set_context(query_id, "a2-faithfulness", arm)
        check_data = parse_json_object(
            self.llm.generate(FAITHFULNESS_PROMPT.format(items=items))
        )
        raw_results = check_data.get("results")
        if not isinstance(raw_results, list):
            raise ValueError("faithfulness output must contain a results array")
        faithful_by_id = {
            item["claim_id"]: item["faithful"]
            for item in raw_results
            if isinstance(item, dict) and "claim_id" in item and "faithful" in item
        }
        expected_ids = {claim_id for claim_id, _source, _text, _span in pending}
        if set(faithful_by_id) != expected_ids or len(faithful_by_id) != len(raw_results):
            raise ValueError("faithfulness results must cover exactly the split claims")
        return tuple(
            Claim(
                claim_id=claim_id,
                text=text,
                span=span,
                faithful_to_answer=faithful_by_id[claim_id],
            )
            for claim_id, _source, text, span in pending
        )


def claim_record(claim: Claim, answer: str) -> dict[str, Any]:
    return {
        "claim_id": claim.claim_id,
        "text": claim.text,
        "span": {"start": claim.span.start, "end": claim.span.end},
        "source_span_text": answer[claim.span.start : claim.span.end],
        "faithful_to_answer": claim.faithful_to_answer,
    }


def case_record(case: Any) -> dict[str, Any]:
    evidence = [
        {
            "evidence_id": item.evidence_id,
            "document_id": item.document_id,
            "retrieval_rank": item.retrieval_rank,
            "text": item.text,
            "text_sha256": sha256_bytes(item.text.encode("utf-8")),
        }
        for item in case.selected.evidence
    ]
    return {
        "query_id": case.query_id,
        "question": case.question,
        "required_facts": list(case.required_facts),
        "constraints": list(case.constraints),
        "evidence": evidence,
        "evidence_projection_sha256": sha256_bytes(canonical_json(evidence).encode("utf-8")),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def build_blind_packet(
    outputs: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    serial = 0
    seen_a1: set[tuple[str, str]] = set()
    for output in outputs:
        query_id = str(output["query_id"])
        arm = str(output["arm"])
        answer = str(output.get("answer", ""))
        a1_version = str(output["a1_version"])
        a1_identity = (query_id, a1_version)
        if a1_identity not in seen_a1:
            seen_a1.add(a1_identity)
            for sentence_index, (start, end) in enumerate(new_sentence_spans(answer), start=1):
                serial += 1
                item_id = f"item-{serial:06d}"
                rows.append(
                    {
                        "item_id": item_id,
                        "unit_type": "a1_sentence",
                        "question": output["question"],
                        "text": answer[start:end],
                        "source_span_text": "",
                        "atomic": "",
                        "self_contained": "",
                        "verifiable": "",
                        "unresolved_pronoun": "",
                        "span_aligned": "",
                        "rewrite_faithful": "",
                        "fact_coverage_notes": "",
                        "error_type": "",
                        "notes": "",
                    }
                )
                key.append(
                    {
                        "item_id": item_id,
                        "query_id": query_id,
                        "arm": a1_version,
                        "unit_type": "a1_sentence",
                        "unit_index": sentence_index,
                    }
                )
        for claim_index, claim in enumerate(output.get("claims", []), start=1):
            serial += 1
            item_id = f"item-{serial:06d}"
            rows.append(
                {
                    "item_id": item_id,
                    "unit_type": "a2_claim",
                    "question": output["question"],
                    "text": claim["text"],
                    "source_span_text": claim["source_span_text"],
                    "atomic": "",
                    "self_contained": "",
                    "verifiable": "",
                    "unresolved_pronoun": "",
                    "span_aligned": "",
                    "rewrite_faithful": "",
                    "fact_coverage_notes": "",
                    "error_type": "",
                    "notes": "",
                }
            )
            key.append(
                {
                    "item_id": item_id,
                    "query_id": query_id,
                    "arm": arm,
                    "unit_type": "a2_claim",
                    "unit_index": claim_index,
                }
            )
        serial += 1
        case_item_id = f"item-{serial:06d}"
        claims_text = "\n".join(f"- {claim['text']}" for claim in output.get("claims", []))
        rows.append(
            {
                "item_id": case_item_id,
                "unit_type": "a2_case",
                "question": output["question"],
                "text": answer,
                "source_span_text": claims_text,
                "atomic": "",
                "self_contained": "",
                "verifiable": "",
                "unresolved_pronoun": "",
                "span_aligned": "",
                "rewrite_faithful": "",
                "fact_coverage_notes": "",
                "error_type": "",
                "notes": "",
            }
        )
        key.append(
            {
                "item_id": case_item_id,
                "query_id": query_id,
                "arm": arm,
                "unit_type": "a2_case",
                "unit_index": 1,
            }
        )
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    return [rows[index] for index in order], key


def write_blind_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["item_id", "unit_type"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_cases(
    cases: list[Any],
    llm: RecordingLLM,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    diagnostics: dict[str, Counter[str]] = {arm: Counter() for arm in ARMS}
    started = time.perf_counter()
    for case_index, case in enumerate(cases, start=1):
        query = Query(query_id=case.query_id, text=case.question)
        answers: dict[str, str] = {}
        for a1_version in ("old", "new"):
            try:
                answers[a1_version] = generate_draft(
                    llm,
                    query,
                    case.selected,
                    prompt_version=a1_version,
                )
            except Exception as exc:  # noqa: BLE001 -- preserve paired run and raw failure
                failures[f"a1:{a1_version}:{type(exc).__name__}"] += 1
                answers[a1_version] = ""
                traceback.print_exc()
        for a1_version in ("old", "new"):
            answer = answers[a1_version]
            for a2_version in ("old", "new"):
                arm = f"{a1_version}_{a2_version}"
                splitter = HistoricalClaimSplitter(llm, a2_version)
                error: dict[str, str] | None = None
                claims: tuple[Claim, ...] = ()
                try:
                    claims = splitter.split(case.query_id, answer, arm)
                except Exception as exc:  # noqa: BLE001 -- failure is an experimental outcome
                    failures[f"a2:{arm}:{type(exc).__name__}"] += 1
                    error = {"type": type(exc).__name__, "message": str(exc)}
                    traceback.print_exc()
                diagnostics[arm].update(splitter.last_diagnostics)
                outputs.append(
                    {
                        "query_id": case.query_id,
                        "question": case.question,
                        "arm": arm,
                        "a1_version": a1_version,
                        "a2_version": a2_version,
                        "answer": answer,
                        "answer_sha256": sha256_bytes(answer.encode("utf-8")),
                        "selected_evidence_count": len(case.selected.evidence),
                        "claims": [claim_record(claim, answer) for claim in claims],
                        "diagnostics": splitter.last_diagnostics,
                        "error": error,
                    }
                )
        if case_index % 5 == 0:
            elapsed = time.perf_counter() - started
            print(f"[run] {case_index}/{len(cases)} ({elapsed / case_index:.1f}s/case)", flush=True)
    unique_a1: dict[tuple[str, str], dict[str, Any]] = {}
    for row in outputs:
        unique_a1.setdefault((str(row["query_id"]), str(row["a1_version"])), row)
    a1_summary: dict[str, Any] = {}
    citation_tail = re.compile(r"(?:\[\d+\])+\s*[.!?]?\s*$")
    citation_index = re.compile(r"\[(\d+)\]")
    for version in ("old", "new"):
        version_rows = [
            row for (_query_id, candidate), row in unique_a1.items() if candidate == version
        ]
        sentences = [
            (row, str(row["answer"])[start:end])
            for row in version_rows
            for start, end in new_sentence_spans(str(row["answer"]))
        ]
        compliant = 0
        out_of_range = 0
        for row, sentence in sentences:
            indices = [int(value) for value in citation_index.findall(sentence)]
            selected_count = int(row["selected_evidence_count"])
            in_range = all(1 <= value <= selected_count for value in indices)
            compliant += int(bool(indices) and in_range and bool(citation_tail.search(sentence)))
            out_of_range += int(any(not 1 <= value <= selected_count for value in indices))
        a1_summary[version] = {
            "answers": len(version_rows),
            "empty_answers": sum(not str(row["answer"]).strip() for row in version_rows),
            "sentences": len(sentences),
            "words": sum(len(re.findall(r"\b\w+\b", str(row["answer"]))) for row in version_rows),
            "citation_format_compliant_sentences": compliant,
            "citation_out_of_range_sentences": out_of_range,
        }
    summary = {
        "cases": len(cases),
        "records": len(outputs),
        "failures": dict(sorted(failures.items())),
        "a1": a1_summary,
        "by_arm": {
            arm: {
                "records": sum(row["arm"] == arm for row in outputs),
                "empty_answers": sum(
                    row["arm"] == arm and not str(row["answer"]).strip() for row in outputs
                ),
                "split_errors": sum(row["arm"] == arm and row["error"] is not None for row in outputs),
                "claims": sum(
                    len(row["claims"]) for row in outputs if row["arm"] == arm
                ),
                "diagnostics": dict(sorted(diagnostics[arm].items())),
            }
            for arm in ARMS
        },
    }
    return outputs, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--model-id", default="ibm-granite/granite-4.1-3b")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.limit <= 0 or args.top_k <= 0:
        raise SystemExit("--limit and --top-k must be positive")

    data = g3.vt.ensure_asqa()
    cases = g3.build_cases(data, args.limit, random.Random(args.seed), args.top_k)
    if len(cases) != args.limit:
        raise SystemExit(f"requested {args.limit} cases but built {len(cases)}")

    source_path = g3.vt.data_dir() / "asqa_eval_gtr_top100.json"
    case_rows = [case_record(case) for case in cases]
    query_ids = [case.query_id for case in cases]
    manifest = {
        "experiment": "G-A12",
        "status": "raw-unannotated",
        "git_head": git_head(),
        "historical_commits": {"old": OLD_COMMIT, "new": NEW_COMMIT},
        "dataset": {
            "name": "ALCE/ASQA calibration",
            "path": str(source_path),
            "sha256": sha256_bytes(source_path.read_bytes()),
        },
        "sampling": {
            "limit": args.limit,
            "top_k": args.top_k,
            "seed": args.seed,
            "query_ids": query_ids,
            "query_ids_sha256": sha256_bytes("\n".join(query_ids).encode("utf-8")),
        },
        "model": {
            "model_id": args.model_id,
            "max_new_tokens": 256,
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
        },
        "arms": list(ARMS),
        "a2_pairing_rule": [
            ["old_old", "old_new"],
            ["new_old", "new_new"],
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = GraniteLLMClient(
        model_id=args.model_id,
        config=GraniteGenerationConfig(max_new_tokens=256, temperature=0.0, top_p=1.0),
    )
    llm = RecordingLLM(client)
    outputs, summary = run_cases(cases, llm)
    blind_rows, annotation_key = build_blind_packet(outputs, seed=args.seed)

    write_jsonl(args.output_dir / "cases.jsonl", case_rows)
    write_jsonl(args.output_dir / "raw_responses.jsonl", llm.records)
    write_jsonl(args.output_dir / "outputs_by_arm.jsonl", outputs)
    write_blind_csv(args.output_dir / "annotation_blinded.csv", blind_rows)
    (args.output_dir / "annotation_key.json").write_text(
        json.dumps(annotation_key, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "automatic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["outputs"] = {
        path.name: {"sha256": sha256_bytes(path.read_bytes()), "bytes": path.stat().st_size}
        for path in sorted(args.output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[done] raw, manifest, and blind packet: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
