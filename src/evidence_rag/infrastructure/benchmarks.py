from __future__ import annotations

import importlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
from typing import Protocol, TypeVar
from urllib.parse import quote

from evidence_rag.contracts.models import Document, Query
from evidence_rag.infrastructure.datasets import (
    DatasetManifest,
    GoldCase,
    JsonlDatasetAdapter,
    normalize_document,
    normalize_query,
)

RecordT = TypeVar("RecordT")


class BenchmarkDocument(Protocol):
    doc_id: str
    text: str


class BenchmarkQuery(Protocol):
    query_id: str
    text: str


class BenchmarkQrel(Protocol):
    query_id: str
    doc_id: str
    relevance: int


class BenchmarkDataset(Protocol):
    def docs_iter(self) -> Iterable[BenchmarkDocument]: ...

    def queries_iter(self) -> Iterable[BenchmarkQuery]: ...

    def qrels_iter(self) -> Iterable[BenchmarkQrel]: ...


class BenchmarkProvider(Protocol):
    version: str

    def load(self, dataset_id: str) -> BenchmarkDataset: ...


@dataclass(frozen=True)
class BenchmarkMaterialization:
    dataset_id: str
    split: str
    manifest_path: Path
    document_count: int
    query_count: int
    gold_case_count: int


class _IrDatasetsProvider:
    def __init__(self, module: ModuleType, package_version: str) -> None:
        self._module = module
        self.version = package_version

    @classmethod
    def create(cls) -> _IrDatasetsProvider:
        try:
            module = importlib.import_module("ir_datasets")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "benchmark materialization requires the optional benchmark dependency; "
                "install evidence-rag[benchmark]"
            ) from error
        return cls(module, version("ir-datasets"))

    def load(self, dataset_id: str) -> BenchmarkDataset:
        dataset: BenchmarkDataset = self._module.load(dataset_id)
        return dataset


def _required(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must not be blank")
    return normalized


def _insert_unique(
    records: dict[str, RecordT], identifier: str, record: RecordT, label: str
) -> None:
    if identifier in records:
        raise ValueError(f"duplicate {label} ID after boundary normalization: {identifier}")
    records[identifier] = record


def _source_uri(name: str, split: str, document_id: str) -> str:
    return "/".join(
        (
            "ir-datasets://beir",
            quote(name, safe=""),
            quote(split, safe=""),
            "document",
            quote(document_id, safe=""),
        )
    )


def _query_answers(query: BenchmarkQuery) -> tuple[str, ...] | None:
    raw_answers = getattr(query, "answers", None)
    if not raw_answers:
        return None
    answers = {_required(answer, "reference answer") for answer in raw_answers}
    return tuple(sorted(answers))


def _document_text(document: BenchmarkDocument) -> str:
    title = getattr(document, "title", "")
    body = document.text
    if not isinstance(title, str) or not isinstance(body, str):
        raise ValueError("document title and text must be strings")
    parts = tuple(part.strip() for part in (title, body) if part.strip())
    if not parts:
        raise ValueError("document text must not be blank")
    return "\n\n".join(parts)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Iterable[object]) -> None:
    lines = [
        json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        for record in records
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def materialize_beir_dataset(
    *,
    name: str,
    split: str,
    output_directory: Path,
    query_limit: int | None = None,
    provider: BenchmarkProvider | None = None,
) -> BenchmarkMaterialization:
    benchmark_name = _required(name, "benchmark name")
    dataset_split = _required(split, "dataset split")
    if "/" in benchmark_name:
        raise ValueError("benchmark name must be a BEIR dataset name without slashes")
    if "/" in dataset_split:
        raise ValueError("dataset split must not contain slashes")
    if query_limit is not None and query_limit <= 0:
        raise ValueError("query limit must be greater than zero")

    active_provider = provider or _IrDatasetsProvider.create()
    provider_dataset_id = f"beir/{benchmark_name}/{dataset_split}"
    dataset = active_provider.load(provider_dataset_id)

    queries: dict[str, Query] = {}
    answers: dict[str, tuple[str, ...] | None] = {}
    for source_query in dataset.queries_iter():
        query = normalize_query(
            Query(query_id=source_query.query_id, text=source_query.text)
        )
        _insert_unique(queries, query.query_id, query, "query")
        answers[query.query_id] = _query_answers(source_query)

    sorted_query_ids = sorted(queries)
    if query_limit is not None:
        if query_limit > len(sorted_query_ids):
            raise ValueError(
                f"query limit {query_limit} exceeds available query count {len(sorted_query_ids)}"
            )
        sorted_query_ids = sorted_query_ids[:query_limit]
    selected_query_ids = set(sorted_query_ids)

    positive_qrels: dict[str, set[str]] = {query_id: set() for query_id in sorted_query_ids}
    for qrel in dataset.qrels_iter():
        query_id = _required(qrel.query_id, "qrel query ID")
        document_id = _required(qrel.doc_id, "qrel document ID")
        if int(qrel.relevance) <= 0 or query_id not in selected_query_ids:
            continue
        positive_qrels[query_id].add(document_id)

    for query_id, document_ids in positive_qrels.items():
        if not document_ids:
            raise ValueError(f"query has no positive qrels: {query_id}")

    documents: dict[str, Document] = {}
    for source_document in dataset.docs_iter():
        document_id = _required(source_document.doc_id, "document ID")
        document = normalize_document(
            Document(
                document_id=document_id,
                text=_document_text(source_document),
                source_uri=_source_uri(benchmark_name, dataset_split, document_id),
            )
        )
        _insert_unique(documents, document.document_id, document, "document")

    gold_document_ids = {
        document_id for document_ids in positive_qrels.values() for document_id in document_ids
    }
    missing_document_ids = sorted(gold_document_ids.difference(documents))
    if missing_document_ids:
        raise ValueError(
            "gold document is missing from the benchmark corpus: "
            + ", ".join(missing_document_ids)
        )

    gold_cases = [
        GoldCase(
            query_id=query_id,
            relevant_document_ids=tuple(sorted(positive_qrels[query_id])),
            reference_answers=answers[query_id],
        )
        for query_id in sorted_query_ids
    ]
    manifest = DatasetManifest(
        dataset_id=f"beir/{benchmark_name}",
        dataset_version=f"ir-datasets-{active_provider.version}",
        split=dataset_split,
        documents_file="documents.jsonl",
        queries_file="queries.jsonl",
        gold_cases_file="gold_cases.jsonl",
    )

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    _write_json(manifest_path, manifest.model_dump(mode="json"))
    _write_jsonl(
        destination / manifest.documents_file,
        (documents[identifier].model_dump(mode="json") for identifier in sorted(documents)),
    )
    _write_jsonl(
        destination / manifest.queries_file,
        (queries[identifier].model_dump(mode="json") for identifier in sorted_query_ids),
    )
    _write_jsonl(
        destination / manifest.gold_cases_file,
        (gold_case.model_dump(mode="json") for gold_case in gold_cases),
    )
    JsonlDatasetAdapter.load(manifest_path)

    return BenchmarkMaterialization(
        dataset_id=manifest.dataset_id,
        split=manifest.split,
        manifest_path=manifest_path,
        document_count=len(documents),
        query_count=len(sorted_query_ids),
        gold_case_count=len(gold_cases),
    )
