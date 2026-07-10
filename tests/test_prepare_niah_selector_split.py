"""Tests for the leakage-safe NQ/DPR selector split."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.prepare_niah_selector_split import (
    NQGoldPassage,
    NQNeedleCandidate,
    build_niah_split_payload,
    choose_designated_needle,
    collect_nq_candidates,
    load_legacy_query_ids,
    normalize_parent_page,
)


def test_normalize_parent_page_is_stable() -> None:
    assert normalize_parent_page("  The   Old MAN and the Sea  ") == "the old man and the sea"


def test_choose_designated_needle_prefers_answer_bearing_highest_relevance() -> None:
    passages = [
        NQGoldPassage("10", "Related", "This is only background.", 2),
        NQGoldPassage("2", "Answer", "The singer was LINDA DAVIS.", 1),
        NQGoldPassage("3", "Answer", "Linda Davis performed the duet.", 2),
    ]

    chosen = choose_designated_needle(passages, ("Linda Davis",))

    assert chosen.doc_id == "3"
    assert chosen.title == "Answer"


def test_choose_designated_needle_falls_back_deterministically() -> None:
    passages = [
        NQGoldPassage("9", "B", "No literal alias.", 1),
        NQGoldPassage("2", "A", "Still no alias.", 2),
        NQGoldPassage("1", "A", "Still no alias.", 2),
    ]

    assert choose_designated_needle(passages, ("missing",)).doc_id == "1"


def _candidate(index: int, parent: str | None = None) -> NQNeedleCandidate:
    return NQNeedleCandidate(
        query_id=f"q{index:03d}",
        needle_doc_id=f"d{index:03d}",
        parent_page_id=parent or f"page-{index:03d}",
        synthetic_family_id=f"nq-family-{index:03d}",
    )


def test_build_payload_excludes_legacy_pages_and_assigns_exact_groups() -> None:
    candidates = [_candidate(i) for i in range(12)]
    candidates.extend([_candidate(20, "legacy-page"), _candidate(21, "legacy-page")])

    first = build_niah_split_payload(
        candidates,
        legacy_parent_pages={"legacy-page"},
        targets={"train": 6, "dev": 2, "test": 2},
        pilot_train_target=4,
        seed=42,
        dataset_fingerprint="sha256:dataset",
        legacy_query_fingerprint="sha256:legacy",
    )
    second = build_niah_split_payload(
        list(reversed(candidates)),
        legacy_parent_pages={"legacy-page"},
        targets={"train": 6, "dev": 2, "test": 2},
        pilot_train_target=4,
        seed=42,
        dataset_fingerprint="sha256:dataset",
        legacy_query_fingerprint="sha256:legacy",
    )

    assert first == second
    assert first["counts"] == {"train": 6, "dev": 2, "test": 2}
    assert len(first["pilot_train_query_ids"]) == 4
    assert first["excluded_legacy_parent_query_count"] == 2
    assert sum(len(rows) for rows in first["splits"].values()) == 10
    parent_owner: dict[str, str] = {}
    family_owner: dict[str, str] = {}
    for split, rows in first["splits"].items():
        for row in rows:
            assert row["parent_page_id"] != "legacy-page"
            assert parent_owner.setdefault(row["parent_page_id"], split) == split
            assert family_owner.setdefault(row["synthetic_family_id"], split) == split


def test_build_payload_keeps_shared_parent_in_one_split() -> None:
    candidates = [_candidate(i) for i in range(7)]
    candidates.extend([_candidate(30, "shared"), _candidate(31, "shared")])

    payload = build_niah_split_payload(
        candidates,
        legacy_parent_pages=set(),
        targets={"train": 5, "dev": 2, "test": 2},
        pilot_train_target=2,
        seed=13,
        dataset_fingerprint="data",
        legacy_query_fingerprint="legacy",
    )

    owners = {
        split
        for split, rows in payload["splits"].items()
        if any(row["parent_page_id"] == "shared" for row in rows)
    }
    assert len(owners) == 1


def test_build_payload_rejects_insufficient_eligible_queries() -> None:
    with pytest.raises(ValueError, match="eligible"):
        build_niah_split_payload(
            [_candidate(0)],
            legacy_parent_pages=set(),
            targets={"train": 1, "dev": 1, "test": 0},
            pilot_train_target=1,
            seed=42,
            dataset_fingerprint="data",
            legacy_query_fingerprint="legacy",
        )


def test_payload_contains_no_question_answer_or_passage_text() -> None:
    payload = build_niah_split_payload(
        [_candidate(i) for i in range(6)],
        legacy_parent_pages=set(),
        targets={"train": 2, "dev": 2, "test": 2},
        pilot_train_target=1,
        seed=42,
        dataset_fingerprint="data",
        legacy_query_fingerprint="legacy",
    )
    serialized = json.dumps(payload).lower()
    for forbidden in ("question", "answer", "passage", "document_text"):
        assert forbidden not in serialized


def test_load_legacy_query_ids_requires_unique_qid_column(tmp_path: Path) -> None:
    path = tmp_path / "legacy.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["qid", "score"])
        writer.writerow(["q1", 1])
        writer.writerow(["q2", 0])
    assert load_legacy_query_ids(path) == {"q1", "q2"}

    path.write_text("qid\nq1\nq1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate legacy query ID"):
        load_legacy_query_ids(path)


class _FakeDPRDataset:
    def queries_iter(self):
        yield SimpleNamespace(query_id="old", text="old question", answers=("answer",))
        yield SimpleNamespace(query_id="new", text="new question", answers=("answer",))

    def qrels_iter(self):
        yield SimpleNamespace(query_id="old", doc_id="d1", relevance=2)
        yield SimpleNamespace(query_id="old", doc_id="d2", relevance=1)
        yield SimpleNamespace(query_id="new", doc_id="d3", relevance=2)

    def docs_iter(self):
        yield SimpleNamespace(doc_id="d1", title="Old Page", text="answer")
        yield SimpleNamespace(doc_id="d2", title="Shared Page", text="answer")
        yield SimpleNamespace(doc_id="d3", title="New Page", text="answer")


def test_collection_excludes_every_gold_parent_from_legacy_queries() -> None:
    candidates, legacy_pages, _, audit = collect_nq_candidates(
        _FakeDPRDataset(), {"old"}
    )
    assert [candidate.query_id for candidate in candidates] == ["new"]
    assert legacy_pages == {"old page", "shared page"}
    assert audit["legacy_query_count"] == 1


def test_collection_rejects_legacy_ids_absent_from_dataset() -> None:
    with pytest.raises(ValueError, match="absent from the DPR split"):
        collect_nq_candidates(_FakeDPRDataset(), {"missing"})
