"""Tests for FinanceBench page-level selector preparation."""

from __future__ import annotations

from eval.financebench_selector import finance_utility_grade, token_coverage


def test_token_coverage_ignores_layout_and_case() -> None:
    assert token_coverage("Net income: $5,363", "NET\nINCOME 5 363") == 1.0


def test_token_coverage_returns_zero_for_empty_reference() -> None:
    assert token_coverage("", "anything") == 0.0


def test_finance_grade_prioritizes_official_pages() -> None:
    grade, harm = finance_utility_grade(
        candidate_id="report-a::page:7",
        candidate_doc="report-a",
        candidate_company="A",
        target_doc="report-a",
        target_company="A",
        official_page_ids={"report-a::page:7"},
        required_page_count=1,
    )
    assert (grade, harm) == (4, None)


def test_finance_grade_marks_multi_page_evidence_as_required_partial() -> None:
    grade, harm = finance_utility_grade(
        candidate_id="report-a::page:7",
        candidate_doc="report-a",
        candidate_company="A",
        target_doc="report-a",
        target_company="A",
        official_page_ids={"report-a::page:7", "report-a::page:8"},
        required_page_count=2,
    )
    assert (grade, harm) == (3, None)


def test_finance_grade_distinguishes_same_report_noise_and_wrong_period() -> None:
    common = {
        "candidate_id": "candidate",
        "target_doc": "report-a-2025",
        "target_company": "A",
        "official_page_ids": {"gold"},
        "required_page_count": 1,
    }
    assert finance_utility_grade(
        **common, candidate_doc="report-a-2025", candidate_company="A"
    ) == (1, None)
    assert finance_utility_grade(
        **common, candidate_doc="report-a-2022", candidate_company="A"
    ) == (0, "wrong_period")
    assert finance_utility_grade(
        **common, candidate_doc="report-b-2025", candidate_company="B"
    ) == (0, "wrong_entity")
