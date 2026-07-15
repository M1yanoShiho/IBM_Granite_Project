import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from evidence_rag.infrastructure.benchmarks import materialize_beir_dataset
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter


@dataclass(frozen=True)
class FakeDocument:
    doc_id: str
    text: str
    title: str = ""


@dataclass(frozen=True)
class FakeQuery:
    query_id: str
    text: str
    answers: tuple[str, ...] = ()


@dataclass(frozen=True)
class FakeQrel:
    query_id: str
    doc_id: str
    relevance: int


class FakeDataset:
    def __init__(
        self,
        *,
        documents: tuple[FakeDocument, ...],
        queries: tuple[FakeQuery, ...],
        qrels: tuple[FakeQrel, ...],
    ) -> None:
        self._documents = documents
        self._queries = queries
        self._qrels = qrels

    def docs_iter(self) -> Iterator[FakeDocument]:
        return iter(self._documents)

    def queries_iter(self) -> Iterator[FakeQuery]:
        return iter(self._queries)

    def qrels_iter(self) -> Iterator[FakeQrel]:
        return iter(self._qrels)


class FakeProvider:
    version = "0.5.11"

    def __init__(self, dataset: FakeDataset) -> None:
        self.dataset = dataset
        self.loaded_ids: list[str] = []

    def load(self, dataset_id: str) -> FakeDataset:
        self.loaded_ids.append(dataset_id)
        return self.dataset


def fake_dataset() -> FakeDataset:
    return FakeDataset(
        documents=(
            FakeDocument("doc-z", "  Distractor  "),
            FakeDocument(" doc-b ", "  Gold B  "),
            FakeDocument("doc-a", "Gold  A"),
        ),
        queries=(
            FakeQuery("q-2", "Second question"),
            FakeQuery(" q-1 ", "  First  question  ", ("  Answer  one  ",)),
        ),
        qrels=(
            FakeQrel("q-1", "doc-b", 2),
            FakeQrel("q-1", "doc-z", 0),
            FakeQrel("q-1", "doc-a", 1),
            FakeQrel("q-2", "doc-z", 1),
        ),
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_materializer_writes_deterministic_typed_reference_dataset(tmp_path: Path) -> None:
    provider = FakeProvider(fake_dataset())

    summary = materialize_beir_dataset(
        name=" scifact ",
        split=" test ",
        output_directory=tmp_path,
        query_limit=1,
        provider=provider,
    )

    assert provider.loaded_ids == ["beir/scifact/test"]
    assert summary.document_count == 3
    assert summary.query_count == 1
    assert summary.gold_case_count == 1
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == {
        "dataset_id": "beir/scifact",
        "dataset_version": "ir-datasets-0.5.11",
        "documents_file": "documents.jsonl",
        "gold_cases_file": "gold_cases.jsonl",
        "queries_file": "queries.jsonl",
        "schema_version": "1.0",
        "split": "test",
    }
    assert read_jsonl(tmp_path / "documents.jsonl") == [
        {
            "document_id": "doc-a",
            "schema_version": "1.0",
            "source_uri": "ir-datasets://beir/scifact/test/document/doc-a",
            "text": "Gold  A",
        },
        {
            "document_id": "doc-b",
            "schema_version": "1.0",
            "source_uri": "ir-datasets://beir/scifact/test/document/doc-b",
            "text": "Gold B",
        },
        {
            "document_id": "doc-z",
            "schema_version": "1.0",
            "source_uri": "ir-datasets://beir/scifact/test/document/doc-z",
            "text": "Distractor",
        },
    ]
    assert read_jsonl(tmp_path / "queries.jsonl") == [
        {"query_id": "q-1", "schema_version": "1.0", "text": "First  question"}
    ]
    assert read_jsonl(tmp_path / "gold_cases.jsonl") == [
        {
            "query_id": "q-1",
            "reference_answers": ["Answer  one"],
            "relevant_document_ids": ["doc-a", "doc-b"],
        }
    ]
    bundle = JsonlDatasetAdapter.load(tmp_path / "manifest.json")
    assert tuple(document.document_id for document in bundle.documents) == (
        "doc-a",
        "doc-b",
        "doc-z",
    )


def test_materializer_output_is_byte_identical_for_different_source_order(
    tmp_path: Path,
) -> None:
    original = fake_dataset()
    reversed_dataset = FakeDataset(
        documents=tuple(reversed(original._documents)),
        queries=tuple(reversed(original._queries)),
        qrels=tuple(reversed(original._qrels)),
    )

    materialize_beir_dataset(
        name="scifact",
        split="test",
        output_directory=tmp_path / "first",
        provider=FakeProvider(original),
    )
    materialize_beir_dataset(
        name="scifact",
        split="test",
        output_directory=tmp_path / "second",
        provider=FakeProvider(reversed_dataset),
    )

    for filename in ("manifest.json", "documents.jsonl", "queries.jsonl", "gold_cases.jsonl"):
        assert (tmp_path / "first" / filename).read_bytes() == (
            tmp_path / "second" / filename
        ).read_bytes()


def test_materializer_combines_beir_title_and_body(tmp_path: Path) -> None:
    dataset = FakeDataset(
        documents=(
            FakeDocument(
                "doc-a",
                "  The body keeps  internal spacing.  ",
                "  A substantive title  ",
            ),
        ),
        queries=(FakeQuery("q-1", "Question"),),
        qrels=(FakeQrel("q-1", "doc-a", 1),),
    )

    materialize_beir_dataset(
        name="scifact",
        split="test",
        output_directory=tmp_path,
        provider=FakeProvider(dataset),
    )

    assert read_jsonl(tmp_path / "documents.jsonl")[0]["text"] == (
        "A substantive title\n\nThe body keeps  internal spacing."
    )


@pytest.mark.parametrize("query_limit", [0, -1, 3])
def test_materializer_rejects_invalid_query_limits(
    tmp_path: Path,
    query_limit: int,
) -> None:
    with pytest.raises(ValueError, match="query limit"):
        materialize_beir_dataset(
            name="scifact",
            split="test",
            output_directory=tmp_path,
            query_limit=query_limit,
            provider=FakeProvider(fake_dataset()),
        )


def test_materializer_fails_when_a_positive_qrel_document_is_missing(tmp_path: Path) -> None:
    dataset = FakeDataset(
        documents=(FakeDocument("doc-a", "Text"),),
        queries=(FakeQuery("q-1", "Question"),),
        qrels=(FakeQrel("q-1", "missing", 1),),
    )

    with pytest.raises(ValueError, match="gold document is missing.*missing"):
        materialize_beir_dataset(
            name="scifact",
            split="test",
            output_directory=tmp_path,
            provider=FakeProvider(dataset),
        )


def test_materializer_rejects_blank_source_text(tmp_path: Path) -> None:
    dataset = FakeDataset(
        documents=(FakeDocument("doc-a", " \n "),),
        queries=(FakeQuery("q-1", "Question"),),
        qrels=(FakeQrel("q-1", "doc-a", 1),),
    )

    with pytest.raises(ValueError, match="document text must not be blank"):
        materialize_beir_dataset(
            name="scifact",
            split="test",
            output_directory=tmp_path,
            provider=FakeProvider(dataset),
        )
