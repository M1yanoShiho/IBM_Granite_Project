"""Tests for reproducible controlled-NIAH artifact audits."""

from __future__ import annotations

from eval.audit_niah_selector_pilot import audit_feature_rows, audit_ranked_rows


def _candidate(candidate_id: str, grade: int, source: str = "dpr") -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "utility_grade": grade,
        "source": source,
        "source_parent_id": f"parent-{candidate_id}",
    }


def test_ranked_audit_reports_candidate_and_risk_coverage_by_split() -> None:
    rows = [
        {
            "query_id": "train-1",
            "split": "train",
            "counterfactual_valid": True,
            "non_answer_valid": True,
            "candidates": [
                _candidate("gold", 4),
                _candidate("wrong", 0, "counterfactual"),
                _candidate("background", 2, "generative_non_answer"),
            ],
        },
        {
            "query_id": "test-1",
            "split": "test",
            "counterfactual_valid": True,
            "non_answer_valid": False,
            "candidates": [_candidate("noise", 1)],
        },
    ]

    audit = audit_ranked_rows(rows, expected_candidates=None)

    assert audit["query_count"] == 2
    assert audit["candidate_count_min"] == 1
    assert audit["candidate_count_max"] == 3
    assert audit["required_evidence_recall_at_pool"] == 0.5
    assert audit["harmful_query_exposure_rate"] == 0.5
    assert audit["counterfactual_entry_rate_when_generated"] == 0.5
    assert audit["non_answer_entry_rate_when_generated"] == 1.0
    assert audit["splits"]["train"]["query_count"] == 1
    assert audit["splits"]["test"]["required_evidence_recall_at_pool"] == 0.0


def test_ranked_audit_rejects_duplicate_query_or_candidate_ids() -> None:
    duplicate_query = [
        {"query_id": "q", "split": "train", "candidates": [_candidate("a", 4)]},
        {"query_id": "q", "split": "dev", "candidates": [_candidate("b", 4)]},
    ]
    try:
        audit_ranked_rows(duplicate_query, expected_candidates=None)
    except ValueError as exc:
        assert "duplicate query_id" in str(exc)
    else:
        raise AssertionError("duplicate query IDs must be rejected")

    duplicate_candidate = [
        {
            "query_id": "q",
            "split": "train",
            "candidates": [_candidate("a", 4), _candidate("a", 1)],
        }
    ]
    try:
        audit_ranked_rows(duplicate_candidate, expected_candidates=None)
    except ValueError as exc:
        assert "duplicate candidate_id" in str(exc)
    else:
        raise AssertionError("duplicate candidate IDs must be rejected")


def test_ranked_audit_enforces_frozen_pool_size() -> None:
    rows = [
        {"query_id": "q", "split": "train", "candidates": [_candidate("a", 4)]}
    ]
    try:
        audit_ranked_rows(rows, expected_candidates=20)
    except ValueError as exc:
        assert "expected 20 candidates" in str(exc)
    else:
        raise AssertionError("unexpected candidate pool size must be rejected")


def test_feature_audit_reports_extraction_and_vote_quality() -> None:
    rows = [
        {
            "query_id": "q1",
            "split": "test",
            "parametric_answer": "answer",
            "candidates": [
                {
                    **_candidate("a", 4),
                    "extracted_answer": "answer",
                    "extraction_failure": 0.0,
                    "exact_vote_count": 2.0,
                },
                {
                    **_candidate("b", 1),
                    "extracted_answer": "NONE",
                    "extraction_failure": 1.0,
                    "exact_vote_count": 0.0,
                },
            ],
        }
    ]

    audit = audit_feature_rows(rows, expected_candidates=2)

    assert audit["candidate_count"] == 2
    assert audit["candidate_extraction_failure_rate"] == 0.5
    assert audit["parametric_extraction_failure_rate"] == 0.0
    assert audit["candidate_with_vote_rate"] == 0.5
    assert audit["mean_exact_vote_count"] == 1.0
    assert audit["utility_grade_counts"] == {"0": 0, "1": 1, "2": 0, "3": 0, "4": 1}
