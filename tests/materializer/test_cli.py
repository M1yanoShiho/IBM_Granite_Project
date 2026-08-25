from pathlib import Path

from evidence_rag.contracts.models import Document, Query
from evidence_rag.infrastructure.datasets import DatasetManifest, GoldCase, JsonlDatasetAdapter
from evidence_rag.materializer.cli import main


def _write_base(tmp_path: Path) -> Path:
    manifest = DatasetManifest(
        dataset_id="niah/nq",
        dataset_version="v1",
        split="dev",
        documents_file="documents.jsonl",
        queries_file="queries.jsonl",
        gold_cases_file="gold_cases.jsonl",
    )
    (tmp_path / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    (tmp_path / "documents.jsonl").write_text(
        Document(
            document_id="d1", text="Margin was 18% overall.", source_uri="x://d1"
        ).model_dump_json()
        + "\n"
        + Document(
            document_id="d2", text="Revenue was 23 last year.", source_uri="x://d2"
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "queries.jsonl").write_text(
        Query(query_id="q1", text="margin?").model_dump_json()
        + "\n"
        + Query(query_id="q2", text="revenue?").model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "gold_cases.jsonl").write_text(
        GoldCase(
            query_id="q1", relevant_document_ids=("d1",), reference_answers=("18%",)
        ).model_dump_json()
        + "\n"
        + GoldCase(
            query_id="q2", relevant_document_ids=("d2",), reference_answers=("23",)
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    return tmp_path / "manifest.json"


def test_cli_materializes_and_validates(tmp_path: Path) -> None:
    base = _write_base(tmp_path)
    out = tmp_path / "injected"
    exit_code = main(("--base-manifest", str(base), "--output", str(out), "--seed", "42"))
    assert exit_code == 0
    reloaded = JsonlDatasetAdapter.load(out / "manifest.json")
    assert any(document.document_id == "cf::q1::d1" for document in reloaded.documents)
