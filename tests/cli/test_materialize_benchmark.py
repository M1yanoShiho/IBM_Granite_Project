import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from evidence_rag.cli.materialize_benchmark import main


@dataclass(frozen=True)
class FakeDocument:
    doc_id: str
    text: str


@dataclass(frozen=True)
class FakeQuery:
    query_id: str
    text: str


@dataclass(frozen=True)
class FakeQrel:
    query_id: str
    doc_id: str
    relevance: int


class FakeDataset:
    def docs_iter(self) -> Iterator[FakeDocument]:
        return iter((FakeDocument("doc-1", "Document"),))

    def queries_iter(self) -> Iterator[FakeQuery]:
        return iter((FakeQuery("q-1", "Question"),))

    def qrels_iter(self) -> Iterator[FakeQrel]:
        return iter((FakeQrel("q-1", "doc-1", 1),))


class FakeProvider:
    version = "0.5.11"

    def load(self, dataset_id: str) -> FakeDataset:
        assert dataset_id == "beir/scifact/test"
        return FakeDataset()


def test_cli_materializes_with_an_injected_provider(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "scifact",
            "--split",
            "test",
            "--output",
            str(tmp_path),
            "--query-limit",
            "1",
        ],
        provider=FakeProvider(),
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "dataset_id": "beir/scifact",
        "document_count": 1,
        "gold_case_count": 1,
        "manifest": str(tmp_path / "manifest.json"),
        "query_count": 1,
        "split": "test",
    }
