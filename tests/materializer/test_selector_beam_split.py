import json
from pathlib import Path

import pytest

from evidence_rag.materializer.selector_beam_split import (
    NiahSelectorAssignment,
    build_splits,
    load_assignments,
)


def _assignment(
    query: str,
    *,
    parent: str,
    family: str,
) -> NiahSelectorAssignment:
    return NiahSelectorAssignment(
        query_id=query,
        required_document_ids=(f"gold-{query}",),
        harmful_document_id=f"cf-{query}",
        source_parent_ids=(parent,),
        synthetic_family=family,
    )


def test_build_splits_removes_all_three_leakage_axes() -> None:
    sealed = (_assignment("sealed", parent="sealed-page", family="sealed-family"),)
    dev = (
        _assignment("dev", parent="dev-page", family="dev-family"),
        _assignment("sealed", parent="other", family="other"),
    )
    train = (
        _assignment("train", parent="train-page", family="train-family"),
        _assignment("query-overlap", parent="unique-1", family="unique-1"),
        _assignment("parent-overlap", parent="dev-page", family="unique-2"),
        _assignment("family-overlap", parent="unique-3", family="dev-family"),
    )
    dev = (*dev, _assignment("query-overlap", parent="dev-extra", family="dev-extra"))

    train_kept, dev_kept, report = build_splits(train=train, dev=dev, sealed=sealed)

    assert [item.query_id for item in train_kept] == ["train"]
    assert [item.query_id for item in dev_kept] == ["dev", "query-overlap"]
    assert report["overlap_after_filtering"] == {
        "train_dev": {"query_id": 0, "source_parent_id": 0, "synthetic_family": 0},
        "train_sealed": {"query_id": 0, "source_parent_id": 0, "synthetic_family": 0},
        "dev_sealed": {"query_id": 0, "source_parent_id": 0, "synthetic_family": 0},
    }


def _write_bundle(path: Path, *, omit_counterfactual: bool = False) -> None:
    path.mkdir()
    (path / "provenance.jsonl").write_text(
        json.dumps(
            {
                "query_id": "q1",
                "needle_document_id": "gold",
                "counterfactual_document_id": "cf",
                "gold_value": "right",
                "gold_alias_used": "Right",
                "replacement_value": "wrong",
                "string_class": "noun-1",
                "seed": 42,
                "char_span": [0, 5],
                "text_hash_before": "a",
                "text_hash_after": "b",
                "answer_bank_hash": "c",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "gold_cases.jsonl").write_text(
        '{"query_id":"q1","relevant_document_ids":["gold"]}\n'
        '{"query_id":"not-injected","relevant_document_ids":["gold"]}\n',
        encoding="utf-8",
    )
    rows = [
        {"document_id": "gold", "text": "Page\n\nright", "source_uri": "x"},
    ]
    if not omit_counterfactual:
        rows.append({"document_id": "cf", "text": "Page\n\nwrong", "source_uri": "x"})
    (path / "documents.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def test_load_assignments_uses_real_documents_and_family(tmp_path: Path) -> None:
    _write_bundle(tmp_path / "bundle")
    assignments = load_assignments(tmp_path / "bundle")
    assert assignments[0].source_parent_ids == ("page",)
    assert assignments[0].synthetic_family == "right|wrong|noun-1"


def test_load_assignments_refuses_missing_labeled_document(tmp_path: Path) -> None:
    _write_bundle(tmp_path / "bundle", omit_counterfactual=True)
    with pytest.raises(ValueError, match="missing labeled IDs"):
        load_assignments(tmp_path / "bundle")
