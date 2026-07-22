"""Poisoned-twin counterfactual injection (spec §3-§5).

Deterministic: eligible query -> copy the gold passage -> replace the single gold-alias
occurrence with a same-class bank value -> validate invariants -> emit twin Document +
MutationRecord. Passage granularity; no slicing (spec §11).
"""

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from evidence_rag.contracts.models import Document
from evidence_rag.infrastructure.datasets import DatasetBundle, DatasetManifest, GoldCase
from evidence_rag.materializer.answer_bank import AnswerBank, string_class
from evidence_rag.materializer.provenance import MutationRecord, write_provenance
from evidence_rag.selector.answer_norm import canonicalize_answer


@dataclass(frozen=True)
class InjectionTarget:
    query_id: str
    needle: Document
    alias_used: str
    span: tuple[int, int]
    gold_value: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class MaterializationResult:
    injected_documents: tuple[Document, ...]
    records: tuple[MutationRecord, ...]
    skipped_query_ids: tuple[str, ...]


def _occurrences(text: str, alias: str) -> list[tuple[int, int]]:
    pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
    return [(match.start(), match.end()) for match in pattern.finditer(text)]


def _has_any_alias(text: str, aliases: Sequence[str]) -> bool:
    return any(_occurrences(text, alias) for alias in aliases)


def find_injection_target(
    gold_case: GoldCase,
    documents: Mapping[str, Document],
) -> InjectionTarget | None:
    answers = gold_case.reference_answers or ()
    if not answers:
        return None
    normalized = {canonicalize_answer(answer) for answer in answers}
    if len(normalized) != 1:
        return None
    gold_value = next(iter(normalized))
    for document_id in gold_case.relevant_document_ids or ():
        document = documents.get(document_id)
        if document is None:
            continue
        for alias in answers:
            spans = _occurrences(document.text, alias)
            if len(spans) == 1:
                return InjectionTarget(
                    query_id=gold_case.query_id,
                    needle=document,
                    alias_used=alias,
                    span=spans[0],
                    gold_value=gold_value,
                    aliases=tuple(answers),
                )
    return None


def inject_counterfactual(
    target: InjectionTarget,
    bank: AnswerBank,
    *,
    seed: int = 42,
) -> tuple[Document, MutationRecord] | None:
    cls = string_class(target.alias_used)
    replacement = bank.select(target.gold_value, target.aliases, cls)
    if replacement is None:
        return None
    start, end = target.span
    new_text = target.needle.text[:start] + replacement + target.needle.text[end:]
    if _has_any_alias(new_text, target.aliases):
        return None
    blocked = {canonicalize_answer(alias) for alias in target.aliases}
    if canonicalize_answer(replacement) in blocked:
        return None
    restored = new_text[:start] + target.alias_used + new_text[start + len(replacement) :]
    if restored != target.needle.text:
        return None
    twin = Document(
        document_id=f"cf::{target.query_id}::{target.needle.document_id}",
        text=new_text,
        source_uri=f"synthetic://cf/{target.query_id}/{target.needle.document_id}",
    )
    record = MutationRecord(
        query_id=target.query_id,
        needle_document_id=target.needle.document_id,
        counterfactual_document_id=twin.document_id,
        gold_value=target.gold_value,
        gold_alias_used=target.alias_used,
        replacement_value=replacement,
        string_class=cls,
        seed=seed,
        char_span=(start, end),
        text_hash_before=sha256(target.needle.text.encode("utf-8")).hexdigest(),
        text_hash_after=sha256(new_text.encode("utf-8")).hexdigest(),
        answer_bank_hash=bank.content_hash,
    )
    return twin, record


def materialize_counterfactuals(
    bundle: DatasetBundle,
    bank: AnswerBank,
    *,
    seed: int = 42,
) -> MaterializationResult:
    documents = {document.document_id: document for document in bundle.documents}
    twins: list[Document] = []
    records: list[MutationRecord] = []
    skipped: list[str] = []
    for gold_case in bundle.gold_cases:
        target = find_injection_target(gold_case, documents)
        injection = inject_counterfactual(target, bank, seed=seed) if target is not None else None
        if injection is None:
            skipped.append(gold_case.query_id)
            continue
        twin, record = injection
        twins.append(twin)
        records.append(record)
    return MaterializationResult(
        injected_documents=tuple(bundle.documents) + tuple(twins),
        records=tuple(records),
        skipped_query_ids=tuple(skipped),
    )


def _write_jsonl(path: Path, records: Iterable[object]) -> None:
    lines = [
        json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        for record in records
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_injected_dataset(
    bundle: DatasetBundle,
    result: MaterializationResult,
    output_directory: Path,
    *,
    seed: int = 42,
) -> Path:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest = DatasetManifest(
        dataset_id=bundle.manifest.dataset_id,
        dataset_version=f"{bundle.manifest.dataset_version}+cf{seed}",
        split=bundle.manifest.split,
        documents_file="documents.jsonl",
        queries_file="queries.jsonl",
        gold_cases_file="gold_cases.jsonl",
    )
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    _write_jsonl(
        output_directory / manifest.documents_file,
        (document.model_dump(mode="json") for document in result.injected_documents),
    )
    _write_jsonl(
        output_directory / manifest.queries_file,
        (query.model_dump(mode="json") for query in bundle.queries),
    )
    _write_jsonl(
        output_directory / manifest.gold_cases_file,
        (gold_case.model_dump(mode="json") for gold_case in bundle.gold_cases),
    )
    write_provenance(output_directory / "provenance.jsonl", result.records)
    return manifest_path
