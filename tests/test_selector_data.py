"""Tests for official selector-dataset audits and leakage-safe splits."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import hashlib
import json
from pathlib import Path

import pytest

from eval.prepare_selector_data import build_manifests, main
from src.retrieval.selector_data import (
    FinanceBenchRecord,
    NIAHMetadataRecord,
    assign_niah_grouped_splits,
    build_financebench_nested_folds,
    load_contractnli,
    load_financebench,
    load_ramdocs,
    sha256_file,
    validate_no_cross_split_leakage,
    validate_unique_ids,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, values: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def _finance_question(index: int, company: str) -> dict[str, object]:
    return {
        "financebench_id": index,
        "company": company,
        "doc_name": f"{company}_2025_10K",
        "question": f"SECRET_FINANCE_QUESTION_{index}",
        "answer": f"SECRET_FINANCE_ANSWER_{index}",
        "evidence": [
            {
                "evidence_text": f"SECRET_FINANCE_EVIDENCE_{index}",
                "evidence_doc_name": f"{company}_2025_10K",
                "evidence_page_num": 0,
                "evidence_text_full_page": f"SECRET_FINANCE_PAGE_{index}",
            }
        ],
    }


def _contract_payload(split: str, document_id: int) -> dict[str, object]:
    return {
        "documents": [
            {
                "id": document_id,
                "text": f"SECRET_CONTRACT_TEXT_{split}",
                "spans": [[0, 6], [7, 15]],
                "annotation_sets": [
                    {
                        "annotations": {
                            "nda-1": {"choice": "Entailment", "spans": [0]},
                            "nda-2": {"choice": "NotMentioned", "spans": []},
                        }
                    }
                ],
            }
        ],
        "labels": {
            "nda-1": {"hypothesis": "SECRET_CONTRACT_HYPOTHESIS_ONE"},
            "nda-2": {"hypothesis": "SECRET_CONTRACT_HYPOTHESIS_TWO"},
        },
    }


def _ramdocs_example(index: int) -> dict[str, object]:
    return {
        "question": f"SECRET_RAMDOCS_QUESTION_{index}",
        "documents": [
            {
                "text": f"SECRET_RAMDOCS_CORRECT_{index}",
                "type": "correct",
                "answer": f"SECRET_RAMDOCS_GOLD_{index}",
            },
            {
                "text": f"SECRET_RAMDOCS_MISINFO_{index}",
                "type": "misinfo",
                "answer": f"SECRET_RAMDOCS_WRONG_{index}",
            },
            {
                "text": f"SECRET_RAMDOCS_NOISE_{index}",
                "type": "noise",
                "answer": "unknown",
            },
        ],
        "gold_answers": [f"SECRET_RAMDOCS_GOLD_{index}"],
        "wrong_answers": [f"SECRET_RAMDOCS_WRONG_{index}"],
    }


def _official_layout(root: Path) -> Path:
    finance_dir = root / "financebench"
    questions = [_finance_question(i, f"Company-{i}") for i in range(10)]
    documents = [
        {
            "doc_name": question["doc_name"],
            "company": question["company"],
            "doc_type": "10K",
            "doc_period": 2025,
            "doc_link": f"https://example.test/{i}.pdf",
        }
        for i, question in enumerate(questions)
    ]
    _write_jsonl(finance_dir / "data/financebench_open_source.jsonl", questions)
    _write_jsonl(
        finance_dir / "data/financebench_document_information.jsonl", documents
    )
    pdf_dir = finance_dir / "pdfs"
    pdf_dir.mkdir(parents=True)
    for index in range(3):
        (pdf_dir / f"report-{index}.pdf").write_bytes(f"PDF {index}".encode())

    contract_dir = root / "contract-nli/resources/contract-nli/contract-nli"
    for index, split in enumerate(("train", "dev", "test"), start=1):
        _write_json(contract_dir / f"{split}.json", _contract_payload(split, index))

    _write_jsonl(
        root / "RAMDocs/RAMDocs_test.jsonl",
        [_ramdocs_example(0), _ramdocs_example(1)],
    )
    return root


def _finance_paths(root: Path) -> tuple[Path, Path, Path]:
    finance = root / "financebench"
    return (
        finance / "data/financebench_open_source.jsonl",
        finance / "data/financebench_document_information.jsonl",
        finance / "pdfs",
    )


def _contract_root(root: Path) -> Path:
    return root / "contract-nli/resources/contract-nli/contract-nli"


def test_loaders_return_immutable_metadata_only_records_and_audits(tmp_path: Path) -> None:
    root = _official_layout(tmp_path / "raw")

    finance_records, finance_audit = load_financebench(*_finance_paths(root))
    contract_records, contract_audit = load_contractnli(_contract_root(root))
    ramdocs_records, ramdocs_audit = load_ramdocs(
        root / "RAMDocs/RAMDocs_test.jsonl"
    )

    assert finance_audit == finance_audit.__class__(
        question_count=10,
        company_count=10,
        document_count=10,
        document_row_count=10,
        unique_document_count=10,
        duplicate_document_name_count=0,
        metadata_conflict_count=0,
        evidence_count=10,
        pdf_count=3,
    )
    assert contract_audit.split("train").document_count == 1
    assert contract_audit.split("train").hypothesis_count == 2
    assert contract_audit.split("train").annotation_choice_counts == {
        "Contradiction": 0,
        "Entailment": 1,
        "NotMentioned": 1,
    }
    assert contract_audit.split("train").evidence_span_count == 1
    assert ramdocs_audit.example_count == 2
    assert ramdocs_audit.document_count == 6
    assert ramdocs_audit.min_documents_per_query == 3
    assert ramdocs_audit.max_documents_per_query == 3
    assert ramdocs_audit.document_type_counts == {
        "correct": 2,
        "misinfo": 2,
        "noise": 2,
    }
    assert ramdocs_records[0].query_id == "ramdocs-000000"
    assert ramdocs_records[0].documents[0].document_id == "ramdocs-000000-doc-000"
    assert set(contract_records) == {"train", "dev", "test"}
    assert not {
        "question",
        "answer",
        "evidence",
        "text",
        "hypothesis",
        "gold_answers",
        "wrong_answers",
    } & {field.name for field in fields(finance_records[0])}
    with pytest.raises(FrozenInstanceError):
        finance_records[0].company = "changed"  # type: ignore[misc]


def test_cli_writes_exact_deterministic_manifest_shape(tmp_path: Path) -> None:
    root = _official_layout(tmp_path / "raw")
    out_one = tmp_path / "one"
    out_two = tmp_path / "two"
    common = [
        "--data-root",
        str(root),
        "--financebench-commit",
        "finance-sha",
        "--contractnli-commit",
        "contract-sha",
        "--ramdocs-commit",
        "ramdocs-sha",
        "--seed",
        "42",
    ]

    main([*common, "--out-dir", str(out_one)])
    main([*common, "--out-dir", str(out_two)])

    for name in ("dataset_manifest.json", "split_manifest.json"):
        first = (out_one / name).read_bytes()
        second = (out_two / name).read_bytes()
        assert first == second
        assert first.endswith(b"\n")

    dataset = json.loads((out_one / "dataset_manifest.json").read_text())
    splits = json.loads((out_one / "split_manifest.json").read_text())
    assert set(dataset) == {"schema_version", "datasets"}
    assert dataset["schema_version"] == "1.0"
    assert set(dataset["datasets"]) == {
        "contractnli",
        "financebench",
        "ramdocs",
    }
    for metadata in dataset["datasets"].values():
        assert set(metadata) == {
            "audit",
            "license",
            "official_url",
            "raw_files",
            "repository_commit",
        }
        assert set(metadata["license"]) == {"identifier", "source"}
        assert all(set(raw_file) == {"path", "sha256"} for raw_file in metadata["raw_files"])
        assert all(not Path(raw_file["path"]).is_absolute() for raw_file in metadata["raw_files"])
    assert dataset["datasets"]["financebench"]["audit"] == {
        "company_count": 10,
        "document_count": 10,
        "document_row_count": 10,
        "duplicate_document_name_count": 0,
        "evidence_count": 10,
        "metadata_conflict_count": 0,
        "pdf_count": 3,
        "question_count": 10,
        "unique_document_count": 10,
    }
    assert set(splits) == {"schema_version", "seed", "datasets"}
    assert splits["schema_version"] == "1.0"
    assert splits["seed"] == 42
    assert set(splits["datasets"]) == {
        "contractnli",
        "financebench",
        "niah",
        "ramdocs",
    }
    assert splits["datasets"]["niah"] == {
        "status": "pending_raw_data",
        "targets": {"dev": 300, "test": 300, "train": 2000},
        "pilot_train_target": 500,
    }
    assert splits["datasets"]["ramdocs"]["protocols"] == {
        "adapted": {
            "candidate_pool_size": 20,
            "context_size": 10,
            "status": "pending_mining",
        },
        "official": {
            "candidate_pool": "official",
            "context_size": 3,
            "status": "ready",
        },
    }


def test_financebench_nested_folds_prevent_company_leakage(tmp_path: Path) -> None:
    root = _official_layout(tmp_path / "raw")
    records, _ = load_financebench(*_finance_paths(root))

    folds = build_financebench_nested_folds(records, seed=42)

    assert len(folds) == 5
    for outer in folds:
        assert set(outer["train_companies"]).isdisjoint(outer["test_companies"])
        assert len(outer["inner_folds"]) == 5
        outer_train = set(outer["train_companies"])
        for inner in outer["inner_folds"]:
            assert inner["validation_companies"]
            assert set(inner["train_companies"]).isdisjoint(
                inner["validation_companies"]
            )
            assert set(inner["train_companies"]) | set(
                inner["validation_companies"]
            ) == outer_train


def test_financebench_greedy_folds_balance_uneven_company_sizes() -> None:
    sizes = [9, 8, 7, 6, 5, 4, 3, 2, 1, 1]
    records = tuple(
        FinanceBenchRecord(
            financebench_id=f"q-{company}-{index}",
            company=f"company-{company}",
            doc_name=f"doc-{company}",
            evidence_count=1,
        )
        for company, size in enumerate(sizes)
        for index in range(size)
    )

    folds = build_financebench_nested_folds(records, seed=42)
    outer_sizes = [len(fold["test_question_ids"]) for fold in folds]

    assert max(outer_sizes) - min(outer_sizes) <= max(sizes)
    assert folds == build_financebench_nested_folds(records, seed=42)


def test_financebench_rejects_outer_training_partition_too_small_for_inner_folds() -> None:
    records = tuple(
        FinanceBenchRecord(
            financebench_id=f"q-{company}",
            company=f"company-{company}",
            doc_name=f"doc-{company}",
            evidence_count=1,
        )
        for company in range(6)
    )

    with pytest.raises(
        ValueError,
        match=r"outer fold .* leaves 4 training companies.*at least 5",
    ):
        build_financebench_nested_folds(records, seed=42, n_folds=5)


def test_financebench_minimum_valid_company_count_fills_every_inner_fold() -> None:
    records = tuple(
        FinanceBenchRecord(
            financebench_id=f"q-{company}",
            company=f"company-{company}",
            doc_name=f"doc-{company}",
            evidence_count=1,
        )
        for company in range(7)
    )

    folds = build_financebench_nested_folds(records, seed=42, n_folds=5)

    assert all(
        inner["validation_companies"]
        for outer in folds
        for inner in outer["inner_folds"]
    )


def test_financebench_rejects_duplicate_official_id(tmp_path: Path) -> None:
    root = _official_layout(tmp_path / "raw")
    question_path, document_path, pdf_dir = _finance_paths(root)
    duplicate = _finance_question(0, "OtherCo")
    with question_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(duplicate) + "\n")

    with pytest.raises(ValueError, match=r"duplicate financebench_id.*0"):
        load_financebench(question_path, document_path, pdf_dir)


def test_financebench_accepts_and_audits_official_style_period_conflict(
    tmp_path: Path,
) -> None:
    root = _official_layout(tmp_path / "raw")
    question_path, document_path, pdf_dir = _finance_paths(root)
    duplicate = {
        "doc_name": "Company-0_2025_10K",
        "company": "Company-0",
        "doc_type": "10K",
        "doc_period": 2024,
        "doc_link": "https://example.test/0.pdf",
    }
    with document_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(duplicate) + "\n")

    _, audit = load_financebench(question_path, document_path, pdf_dir)

    assert audit.document_count == 10
    assert audit.document_row_count == 11
    assert audit.unique_document_count == 10
    assert audit.duplicate_document_name_count == 1
    assert audit.metadata_conflict_count == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("company", "Different Company"),
        ("doc_link", "https://example.test/different.pdf"),
    ],
)
def test_financebench_rejects_duplicate_document_identity_conflict(
    tmp_path: Path, field: str, value: str
) -> None:
    root = _official_layout(tmp_path / "raw")
    question_path, document_path, pdf_dir = _finance_paths(root)
    duplicate = {
        "doc_name": "Company-0_2025_10K",
        "company": "Company-0",
        "doc_type": "10K",
        "doc_period": 2025,
        "doc_link": "https://example.test/0.pdf",
    }
    duplicate[field] = value
    with document_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(duplicate) + "\n")

    with pytest.raises(
        ValueError,
        match=rf"duplicate doc_name.*Company-0_2025_10K.*conflicting {field}",
    ):
        load_financebench(question_path, document_path, pdf_dir)


def test_financebench_rejects_question_with_missing_document_metadata(
    tmp_path: Path,
) -> None:
    root = _official_layout(tmp_path / "raw")
    question_path, document_path, pdf_dir = _finance_paths(root)
    metadata_rows = [json.loads(line) for line in document_path.read_text().splitlines()]
    _write_jsonl(document_path, metadata_rows[1:])

    with pytest.raises(
        ValueError,
        match=r"question doc_name.*Company-0_2025_10K.*missing from document metadata",
    ):
        load_financebench(question_path, document_path, pdf_dir)


def test_contractnli_rejects_document_overlap_across_official_splits(tmp_path: Path) -> None:
    root = _official_layout(tmp_path / "raw")
    _write_json(_contract_root(root) / "dev.json", _contract_payload("dev", 1))

    with pytest.raises(ValueError, match=r"document ID.*train.*dev"):
        load_contractnli(_contract_root(root))


def test_contractnli_rejects_mismatched_label_keys(tmp_path: Path) -> None:
    root = _official_layout(tmp_path / "raw")
    path = _contract_root(root) / "dev.json"
    payload = _contract_payload("dev", 2)
    payload["labels"].pop("nda-2")  # type: ignore[union-attr]
    payload["documents"][0]["annotation_sets"][0]["annotations"].pop("nda-2")  # type: ignore[index,union-attr]
    _write_json(path, payload)

    with pytest.raises(ValueError, match=r"label keys.*dev.*train"):
        load_contractnli(_contract_root(root))


def test_contractnli_rejects_invalid_annotation_choice(tmp_path: Path) -> None:
    root = _official_layout(tmp_path / "raw")
    path = _contract_root(root) / "train.json"
    payload = _contract_payload("train", 1)
    annotation = payload["documents"][0]["annotation_sets"][0]["annotations"]["nda-1"]  # type: ignore[index]
    annotation["choice"] = "Maybe"  # type: ignore[index]
    _write_json(path, payload)

    with pytest.raises(ValueError, match=r"choice.*Maybe.*Entailment"):
        load_contractnli(_contract_root(root))


def test_contractnli_rejects_out_of_range_evidence_span(tmp_path: Path) -> None:
    root = _official_layout(tmp_path / "raw")
    path = _contract_root(root) / "test.json"
    payload = _contract_payload("test", 3)
    annotation = payload["documents"][0]["annotation_sets"][0]["annotations"]["nda-1"]  # type: ignore[index]
    annotation["spans"] = [2]  # type: ignore[index]
    _write_json(path, payload)

    with pytest.raises(ValueError, match=r"span index 2.*out of range"):
        load_contractnli(_contract_root(root))


def test_contractnli_rejects_second_annotation_set(tmp_path: Path) -> None:
    root = _official_layout(tmp_path / "raw")
    path = _contract_root(root) / "train.json"
    payload = _contract_payload("train", 1)
    annotation_sets = payload["documents"][0]["annotation_sets"]  # type: ignore[index]
    annotation_sets.append(annotation_sets[0])  # type: ignore[union-attr]
    _write_json(path, payload)

    with pytest.raises(
        ValueError,
        match=r"annotation_sets must contain exactly one annotation set; got 2",
    ):
        load_contractnli(_contract_root(root))


def test_ramdocs_rejects_invalid_official_document_type(tmp_path: Path) -> None:
    path = tmp_path / "RAMDocs_test.jsonl"
    example = _ramdocs_example(0)
    example["documents"][0]["type"] = "counterfactual"  # type: ignore[index]
    _write_jsonl(path, [example])

    with pytest.raises(ValueError, match=r"documents\[0\]\.type.*counterfactual.*correct"):
        load_ramdocs(path)


def test_manifests_never_serialize_raw_text_answers_or_hypotheses(tmp_path: Path) -> None:
    root = _official_layout(tmp_path / "raw")
    dataset, splits = build_manifests(
        root,
        financebench_commit="finance-sha",
        contractnli_commit="contract-sha",
        ramdocs_commit="ramdocs-sha",
        seed=42,
    )
    serialized = json.dumps({"dataset": dataset, "splits": splits})
    lower_keys = {key.lower() for key in _all_keys(dataset) | _all_keys(splits)}

    assert not any("SECRET_" in value for value in _all_string_values(serialized))
    assert not {
        "question",
        "answer",
        "answers",
        "evidence",
        "text",
        "hypothesis",
        "gold_answers",
        "wrong_answers",
    } & lower_keys


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_all_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value), set())
    return set()


def _all_string_values(serialized: str) -> list[str]:
    value = json.loads(serialized)
    output: list[str] = []

    def visit(item: object) -> None:
        if isinstance(item, str):
            output.append(item)
        elif isinstance(item, dict):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    return output


def test_niah_connected_components_prevent_parent_and_family_leakage() -> None:
    records = (
        NIAHMetadataRecord("q0", "parent-a", "family-a"),
        NIAHMetadataRecord("q1", "parent-a", "family-b"),
        NIAHMetadataRecord("q2", "parent-b", "family-b"),
        NIAHMetadataRecord("q3", "parent-c", "family-c"),
        NIAHMetadataRecord("q4", "parent-d", "family-d"),
    )

    splits = assign_niah_grouped_splits(
        records, targets={"train": 3, "dev": 1, "test": 1}, seed=42
    )

    by_query = {
        query_id: split for split, query_ids in splits.items() for query_id in query_ids
    }
    assert by_query["q0"] == by_query["q1"] == by_query["q2"]
    for field_name in ("parent_page_id", "synthetic_family_id"):
        memberships: dict[str, set[str]] = {}
        for record in records:
            memberships.setdefault(getattr(record, field_name), set()).add(
                by_query[record.query_id]
            )
        assert all(len(split_names) == 1 for split_names in memberships.values())
    assert splits == assign_niah_grouped_splits(
        records, targets={"train": 3, "dev": 1, "test": 1}, seed=42
    )


def test_niah_grouping_fails_clearly_when_there_are_too_few_groups() -> None:
    records = (
        NIAHMetadataRecord("q0", "same-parent", "family-a"),
        NIAHMetadataRecord("q1", "same-parent", "family-b"),
        NIAHMetadataRecord("q2", "same-parent", "family-c"),
    )

    with pytest.raises(ValueError, match=r"insufficient independent groups"):
        assign_niah_grouped_splits(
            records, targets={"train": 1, "dev": 1, "test": 1}, seed=42
        )


def _niah_records_for_component_sizes(sizes: list[int]) -> tuple[NIAHMetadataRecord, ...]:
    records: list[NIAHMetadataRecord] = []
    query_index = 0
    for component_index, size in enumerate(sizes):
        for _ in range(size):
            records.append(
                NIAHMetadataRecord(
                    query_id=f"q-{query_index:05d}",
                    parent_page_id=f"parent-{component_index}",
                    synthetic_family_id=f"family-{component_index}",
                )
            )
            query_index += 1
    return tuple(records)


def test_niah_exact_fallback_scales_to_intended_2600_records() -> None:
    records = _niah_records_for_component_sizes([1500, 400, 300, 200] + [1] * 200)

    splits = assign_niah_grouped_splits(
        records,
        targets={"train": 2000, "dev": 300, "test": 300},
        seed=42,
    )

    assert {split: len(query_ids) for split, query_ids in splits.items()} == {
        "train": 2000,
        "dev": 300,
        "test": 300,
    }
    assert splits == assign_niah_grouped_splits(
        records,
        targets={"train": 2000, "dev": 300, "test": 300},
        seed=42,
    )


def test_niah_exact_fallback_handles_more_than_100_adversarial_components() -> None:
    records = _niah_records_for_component_sizes(
        [1500, 800, 700, 400, 300] + [1] * 101
    )

    splits = assign_niah_grouped_splits(
        records,
        targets={"train": 334, "dev": 2334, "test": 1133},
        seed=42,
    )

    assert {split: len(query_ids) for split, query_ids in splits.items()} == {
        "train": 334,
        "dev": 2334,
        "test": 1133,
    }
    assert splits == assign_niah_grouped_splits(
        records,
        targets={"train": 334, "dev": 2334, "test": 1133},
        seed=42,
    )
    query_split = {
        query_id: split for split, query_ids in splits.items() for query_id in query_ids
    }
    offset = 0
    for size in [1500, 800, 700, 400, 300]:
        component_splits = {
            query_split[f"q-{index:05d}"] for index in range(offset, offset + size)
        }
        assert len(component_splits) == 1
        offset += size


def test_generic_validators_reject_duplicates_and_cross_split_leakage() -> None:
    with pytest.raises(ValueError, match=r"duplicate query ID.*q1"):
        validate_unique_ids(["q1", "q1"], id_kind="query")
    with pytest.raises(ValueError, match=r"document ID.*train.*test"):
        validate_no_cross_split_leakage(
            {"train": ["d1"], "dev": ["d2"], "test": ["d1"]},
            id_kind="document",
        )


def test_json_and_type_errors_are_actionable(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"question": ', encoding="utf-8")
    with pytest.raises(ValueError, match=r"malformed\.jsonl line 1.*invalid JSON"):
        load_ramdocs(malformed)

    wrong_type = tmp_path / "wrong-type.jsonl"
    example = _ramdocs_example(0)
    example["gold_answers"] = "not-a-list"
    _write_jsonl(wrong_type, [example])
    with pytest.raises(ValueError, match=r"line 1.*gold_answers.*list"):
        load_ramdocs(wrong_type)


def test_streaming_sha256_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    payload = (b"selector-audit\x00" * 100_000) + b"tail"
    path.write_bytes(payload)

    assert sha256_file(path, chunk_size=37) == hashlib.sha256(payload).hexdigest()
