"""Tests for the RAMDocs evidence-selector adaptation."""

from __future__ import annotations

from eval.ramdocs_selector import ramdocs_utility_grade, select_mined_documents


def test_ramdocs_utility_grade_uses_only_official_evaluation_types() -> None:
    assert ramdocs_utility_grade("correct") == 4
    assert ramdocs_utility_grade("noise") == 1
    assert ramdocs_utility_grade("misinfo") == 0
    assert ramdocs_utility_grade("mined_external") == 1


def test_ramdocs_utility_grade_rejects_unknown_type() -> None:
    try:
        ramdocs_utility_grade("other")
    except ValueError as exc:
        assert "unsupported RAMDocs type" in str(exc)
    else:
        raise AssertionError("unknown types must be rejected")


def test_select_mined_documents_excludes_same_query_and_duplicate_text() -> None:
    documents = [
        {"origin_query_id": "q1", "candidate_id": "q1-a", "text": "own", "score": 0.99},
        {"origin_query_id": "q2", "candidate_id": "q2-a", "text": "duplicate", "score": 0.9},
        {"origin_query_id": "q3", "candidate_id": "q3-a", "text": "duplicate", "score": 0.8},
        {"origin_query_id": "q4", "candidate_id": "q4-a", "text": "new", "score": 0.7},
    ]

    selected = select_mined_documents(documents, query_id="q1", limit=2)

    assert [row["candidate_id"] for row in selected] == ["q2-a", "q4-a"]


def test_select_mined_documents_uses_score_then_id_tie_break() -> None:
    documents = [
        {"origin_query_id": "q2", "candidate_id": "b", "text": "b", "score": 0.5},
        {"origin_query_id": "q3", "candidate_id": "a", "text": "a", "score": 0.5},
    ]
    assert [
        row["candidate_id"]
        for row in select_mined_documents(documents, query_id="q1", limit=2)
    ] == ["a", "b"]
