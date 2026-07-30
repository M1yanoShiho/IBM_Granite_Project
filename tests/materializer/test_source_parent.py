import json
from pathlib import Path

import pytest

from evidence_rag.materializer.source_parent import (
    ParentIndex,
    normalize_parent,
    parse_parent,
    read_parent_index,
    write_parent_index,
)


def test_parse_parent_takes_the_title_paragraph() -> None:
    assert parse_parent("John F. Kennedy\n\nHe was elected in 1960.") == "john f. kennedy"


def test_parse_parent_normalizes_case_and_whitespace() -> None:
    assert parse_parent("  John   F.  Kennedy \n\nbody") == "john f. kennedy"


def test_parse_parent_returns_none_without_a_title_paragraph() -> None:
    assert parse_parent("a single paragraph with no blank line") is None


def test_parse_parent_returns_none_for_a_blank_title() -> None:
    assert parse_parent("\n\nbody only") is None


def test_parse_parent_keeps_only_the_first_paragraph() -> None:
    assert parse_parent("Title\n\nbody one\n\nbody two") == "title"


def test_normalize_parent_is_idempotent() -> None:
    once = normalize_parent("John  F. Kennedy")
    assert normalize_parent(once) == once


def test_parent_index_falls_back_to_document_id_and_counts_it() -> None:
    index = ParentIndex(parent_by_document={"d1": "page a"})
    assert index.parent_of("d1") == "page a"
    assert index.parent_of("d2") == "d2"
    assert index.n_unresolved(("d1", "d2", "d3")) == 2


def test_counterfactual_twin_shares_the_needle_parent() -> None:
    """cf::<qid>::needle is a copy of the gold passage, so it inherits the same title."""
    text = "John F. Kennedy\n\nHe won in 1960."
    assert parse_parent(text) == parse_parent(text.replace("1960", "1964"))


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "source_parent.jsonl"
    write_parent_index(path, {"d1": "page a", "d2": "page a"})
    index = read_parent_index(path)
    assert index.parent_of("d1") == index.parent_of("d2") == "page a"


def test_write_parent_index_handles_an_empty_mapping(tmp_path: Path) -> None:
    path = tmp_path / "source_parent.jsonl"
    write_parent_index(path, {})
    assert path.read_text(encoding="utf-8") == ""
    assert read_parent_index(path).parent_by_document == {}


def test_read_parent_index_rejects_a_duplicate_document_id(tmp_path: Path) -> None:
    path = tmp_path / "source_parent.jsonl"
    path.write_text(
        json.dumps({"document_id": "d1", "source_parent_id": "a"}) + "\n"
        + json.dumps({"document_id": "d1", "source_parent_id": "b"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate document_id"):
        read_parent_index(path)
