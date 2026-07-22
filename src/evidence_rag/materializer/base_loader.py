"""dpr-w100 NQ base loader with qrels-aware subsampling (spec §3-§5).

Mirrors infrastructure/benchmarks.py but streams the 21M-passage corpus and keeps only
gold passages plus a seeded distractor sample. Provider-injectable; no ir_datasets import
here (only the CLI pulls it in). No slicing — dpr-w100 is pre-chunked.
"""

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from evidence_rag.contracts.models import Document, Query
from evidence_rag.infrastructure.datasets import (
    DatasetManifest,
    GoldCase,
    JsonlDatasetAdapter,
    normalize_document,
    normalize_query,
)


class BaseDoc(Protocol):
    doc_id: str
    text: str


class BaseQuery(Protocol):
    query_id: str
    text: str


class BaseQrel(Protocol):
    query_id: str
    doc_id: str
    relevance: int


class BaseDataset(Protocol):
    def docs_iter(self) -> Iterable[BaseDoc]: ...
    def queries_iter(self) -> Iterable[BaseQuery]: ...
    def qrels_iter(self) -> Iterable[BaseQrel]: ...


@dataclass(frozen=True)
class BaseMaterialization:
    manifest_path: Path
    document_count: int
    query_count: int
    gold_case_count: int


def _document_text(doc: BaseDoc) -> str:
    title = getattr(doc, "title", "")
    parts = [part.strip() for part in (title, doc.text) if isinstance(part, str) and part.strip()]
    if not parts:
        raise ValueError(f"document text must not be blank: {doc.doc_id}")
    return "\n\n".join(parts)


def _write_jsonl(path: Path, records: Iterable[object]) -> None:
    lines = [
        json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        for record in records
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def materialize_niah_base(
    dataset: BaseDataset,
    output_directory: Path,
    *,
    corpus_size: int,
    query_limit: int | None = None,
    seed: int = 42,
) -> BaseMaterialization:
    if corpus_size <= 0:
        raise ValueError("corpus_size must be positive")

    answers: dict[str, tuple[str, ...] | None] = {}
    query_by_id: dict[str, Query] = {}
    for source_query in dataset.queries_iter():
        query = normalize_query(Query(query_id=source_query.query_id, text=source_query.text))
        query_by_id[query.query_id] = query
        raw = tuple(getattr(source_query, "answers", ()) or ())
        cleaned = tuple(sorted({answer.strip() for answer in raw if answer.strip()}))
        answers[query.query_id] = cleaned or None

    selected_ids = sorted(query_by_id)
    if query_limit is not None:
        selected_ids = selected_ids[:query_limit]
    selected = set(selected_ids)

    positive: dict[str, set[str]] = {qid: set() for qid in selected_ids}
    for qrel in dataset.qrels_iter():
        if int(qrel.relevance) > 0 and qrel.query_id in selected:
            positive[qrel.query_id].add(qrel.doc_id)
    kept_query_ids = [qid for qid in selected_ids if positive[qid]]
    gold_docs = {doc_id for qid in kept_query_ids for doc_id in positive[qid]}

    distractor_budget = max(0, corpus_size - len(gold_docs))
    rng = random.Random(seed)
    kept_docs: dict[str, Document] = {}
    reservoir: list[Document] = []
    seen_distractors = 0
    for source_doc in dataset.docs_iter():
        document = normalize_document(
            Document(
                document_id=source_doc.doc_id,
                text=_document_text(source_doc),
                source_uri=f"ir-datasets://dpr-w100/nq/document/{source_doc.doc_id}",
            )
        )
        if document.document_id in gold_docs:
            kept_docs[document.document_id] = document
            continue
        if distractor_budget == 0:
            continue
        seen_distractors += 1
        if len(reservoir) < distractor_budget:
            reservoir.append(document)
        else:
            index = rng.randrange(seen_distractors)
            if index < distractor_budget:
                reservoir[index] = document

    missing = sorted(gold_docs - set(kept_docs))
    if missing:
        raise ValueError(f"gold document missing from corpus: {missing[0]}")

    documents = tuple(kept_docs[doc_id] for doc_id in sorted(kept_docs)) + tuple(
        sorted(reservoir, key=lambda item: item.document_id)
    )
    queries = tuple(query_by_id[qid] for qid in kept_query_ids)
    gold_cases = tuple(
        GoldCase(
            query_id=qid,
            relevant_document_ids=tuple(sorted(positive[qid])),
            reference_answers=answers[qid],
        )
        for qid in kept_query_ids
    )

    manifest = DatasetManifest(
        dataset_id="niah/dpr-w100-nq",
        dataset_version=f"subsample-{corpus_size}",
        split="dev",
        documents_file="documents.jsonl",
        queries_file="queries.jsonl",
        gold_cases_file="gold_cases.jsonl",
    )
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    _write_jsonl(
        output_directory / manifest.documents_file,
        (document.model_dump(mode="json") for document in documents),
    )
    _write_jsonl(
        output_directory / manifest.queries_file,
        (query.model_dump(mode="json") for query in queries),
    )
    _write_jsonl(
        output_directory / manifest.gold_cases_file,
        (gold_case.model_dump(mode="json") for gold_case in gold_cases),
    )
    JsonlDatasetAdapter.load(manifest_path)
    return BaseMaterialization(
        manifest_path=manifest_path,
        document_count=len(documents),
        query_count=len(queries),
        gold_case_count=len(gold_cases),
    )
