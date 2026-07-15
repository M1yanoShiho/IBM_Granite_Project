import json
from pathlib import Path

import pytest

from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter


def write_dataset(root: Path, *, document_text: str = "Revenue increased.") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_id": "annual-reports",
        "dataset_version": "2026-01",
        "split": "test",
        "documents_file": "documents.jsonl",
        "queries_file": "queries.jsonl",
        "gold_cases_file": "gold.jsonl",
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "documents.jsonl").write_text(
        json.dumps(
            {
                "document_id": "annual-report",
                "text": document_text,
                "source_uri": "fixture://annual-report",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "queries.jsonl").write_text(
        json.dumps({"query_id": "q-1", "text": "What changed?"}) + "\n",
        encoding="utf-8",
    )
    (root / "gold.jsonl").write_text(
        json.dumps(
            {
                "query_id": "q-1",
                "relevant_document_ids": ["annual-report"],
                "reference_answers": ["Revenue increased."],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root / "manifest.json"


def test_adapter_loads_typed_tuples_from_manifest_and_jsonl(tmp_path: Path) -> None:
    bundle = JsonlDatasetAdapter.load(write_dataset(tmp_path))

    assert bundle.manifest.dataset_id == "annual-reports"
    assert bundle.documents[0].document_id == "annual-report"
    assert bundle.queries[0].query_id == "q-1"
    assert bundle.gold_cases[0].relevant_document_ids == ("annual-report",)
    assert isinstance(bundle.documents, tuple)
    assert bundle.dataset_signature


def test_adapter_normalizes_manifest_identity_boundaries(tmp_path: Path) -> None:
    manifest_path = write_dataset(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "dataset_id": "  annual-reports  ",
            "dataset_version": "  2026-01  ",
            "split": "  test  ",
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    bundle = JsonlDatasetAdapter.load(manifest_path)

    assert bundle.manifest.dataset_id == "annual-reports"
    assert bundle.manifest.dataset_version == "2026-01"
    assert bundle.manifest.split == "test"


def test_adapter_rejects_blank_manifest_dataset_id(tmp_path: Path) -> None:
    manifest_path = write_dataset(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_id"] = " \t "
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset ID must not be blank"):
        JsonlDatasetAdapter.load(manifest_path)


def test_identical_relative_datasets_have_path_independent_signatures(tmp_path: Path) -> None:
    first = JsonlDatasetAdapter.load(write_dataset(tmp_path / "checkout-a" / "dataset"))
    second = JsonlDatasetAdapter.load(write_dataset(tmp_path / "checkout-b" / "dataset"))

    assert first.dataset_signature == second.dataset_signature


def test_changed_content_changes_dataset_signature(tmp_path: Path) -> None:
    first = JsonlDatasetAdapter.load(write_dataset(tmp_path / "first"))
    second = JsonlDatasetAdapter.load(
        write_dataset(tmp_path / "second", document_text="Revenue declined.")
    )

    assert first.dataset_signature != second.dataset_signature


def test_duplicate_document_ids_fail_clearly(tmp_path: Path) -> None:
    manifest_path = write_dataset(tmp_path)
    with (tmp_path / "documents.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "document_id": "annual-report",
                    "text": "Duplicate.",
                    "source_uri": "fixture://duplicate",
                }
            )
            + "\n"
        )

    with pytest.raises(ValueError, match="duplicate document ID: annual-report"):
        JsonlDatasetAdapter.load(manifest_path)


def test_unknown_labelled_document_fails_clearly(tmp_path: Path) -> None:
    manifest_path = write_dataset(tmp_path)
    (tmp_path / "gold.jsonl").write_text(
        json.dumps({"query_id": "q-1", "relevant_document_ids": ["missing"]}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown relevant document ID: missing"):
        JsonlDatasetAdapter.load(manifest_path)


def test_malformed_jsonl_reports_source_path_and_line(tmp_path: Path) -> None:
    manifest_path = write_dataset(tmp_path)
    (tmp_path / "queries.jsonl").write_text("\n{not json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"queries\.jsonl:2"):
        JsonlDatasetAdapter.load(manifest_path)


def test_gold_case_is_available_from_its_existing_evaluation_import_path() -> None:
    from evidence_rag.evaluation.models import GoldCase as EvaluationGoldCase
    from evidence_rag.infrastructure.datasets import GoldCase

    assert EvaluationGoldCase is GoldCase


def test_legacy_gold_case_import_normalizes_list_fields_to_tuples() -> None:
    from evidence_rag.evaluation.models import GoldCase

    gold_case = GoldCase(
        query_id="q-1",
        relevant_document_ids=["annual-report"],
        reference_answers=["Revenue increased."],
    )

    assert gold_case.relevant_document_ids == ("annual-report",)
    assert gold_case.reference_answers == ("Revenue increased.",)


def test_adapter_strips_boundary_whitespace_without_changing_internal_whitespace(
    tmp_path: Path,
) -> None:
    manifest_path = write_dataset(tmp_path)
    (tmp_path / "documents.jsonl").write_text(
        json.dumps(
            {
                "document_id": "  annual-report  ",
                "text": "  Revenue  increased.\nNext line.  ",
                "source_uri": "  fixture://annual-report  ",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "queries.jsonl").write_text(
        json.dumps({"query_id": "  q-1  ", "text": "  What  changed?  "}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "gold.jsonl").write_text(
        json.dumps(
            {
                "query_id": "  q-1  ",
                "relevant_document_ids": ["  annual-report  "],
                "reference_answers": ["  Revenue  increased.  "],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = JsonlDatasetAdapter.load(manifest_path)

    assert bundle.documents[0].document_id == "annual-report"
    assert bundle.documents[0].text == "Revenue  increased.\nNext line."
    assert bundle.documents[0].source_uri == "fixture://annual-report"
    assert bundle.queries[0].query_id == "q-1"
    assert bundle.queries[0].text == "What  changed?"
    assert bundle.gold_cases[0].query_id == "q-1"
    assert bundle.gold_cases[0].relevant_document_ids == ("annual-report",)
    assert bundle.gold_cases[0].reference_answers == ("Revenue  increased.",)


@pytest.mark.parametrize(
    ("filename", "record", "message"),
    [
        (
            "documents.jsonl",
            {"document_id": "doc", "text": " \n\t ", "source_uri": "fixture://doc"},
            "document text must not be blank",
        ),
        (
            "documents.jsonl",
            {"document_id": " \t ", "text": "Text", "source_uri": "fixture://doc"},
            "document ID must not be blank",
        ),
        (
            "queries.jsonl",
            {"query_id": "query", "text": " \n ", "schema_version": "1.0"},
            "query text must not be blank",
        ),
        (
            "queries.jsonl",
            {"query_id": "  ", "text": "Question", "schema_version": "1.0"},
            "query ID must not be blank",
        ),
        (
            "gold.jsonl",
            {
                "query_id": "q-1",
                "relevant_document_ids": ["annual-report"],
                "reference_answers": [" \t "],
            },
            "reference answer must not be blank",
        ),
        (
            "gold.jsonl",
            {
                "query_id": "q-1",
                "relevant_document_ids": [" \t "],
                "reference_answers": ["Answer"],
            },
            "relevant document ID must not be blank",
        ),
    ],
)
def test_adapter_rejects_whitespace_only_boundary_values(
    tmp_path: Path,
    filename: str,
    record: dict[str, object],
    message: str,
) -> None:
    manifest_path = write_dataset(tmp_path)
    (tmp_path / filename).write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        JsonlDatasetAdapter.load(manifest_path)
