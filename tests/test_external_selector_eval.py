"""Tests for external evidence-selector evaluation helpers."""

from __future__ import annotations

from eval.external_selector_eval import required_evidence_subset, selection_diagnostics


def test_selection_diagnostics_separates_coverage_harm_and_strict_success() -> None:
    groups = {
        "q1": [
            {"candidate_id": "gold", "utility_grade": 4, "score": 0.9},
            {"candidate_id": "bad", "utility_grade": 0, "score": 0.8},
        ],
        "q2": [
            {"candidate_id": "noise", "utility_grade": 1, "score": 0.9},
            {"candidate_id": "gold", "utility_grade": 4, "score": 0.8},
        ],
        "q3": [
            {"candidate_id": "bad", "utility_grade": 0, "score": 0.9},
            {"candidate_id": "noise", "utility_grade": 1, "score": 0.8},
        ],
    }

    metrics = selection_diagnostics(groups, score_key="score", k=1)

    assert metrics == {
        "required_query_coverage@1": 1 / 3,
        "harmful_query_exposure@1": 1 / 3,
        "strict_selection_success@1": 1 / 3,
    }


def test_selection_diagnostics_handles_empty_groups() -> None:
    assert selection_diagnostics({}, score_key="score", k=10) == {
        "required_query_coverage@10": 0.0,
        "harmful_query_exposure@10": 0.0,
        "strict_selection_success@10": 0.0,
    }


def test_required_evidence_subset_removes_retrieval_failures() -> None:
    groups = {
        "hit": [{"utility_grade": 3}],
        "miss": [{"utility_grade": 2}, {"utility_grade": 0}],
    }
    assert required_evidence_subset(groups) == {"hit": [{"utility_grade": 3}]}
