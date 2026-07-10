"""Tests for ContractNLI evidence candidate labels."""

from __future__ import annotations

from eval.contractnli_selector import contract_utility_grade


def test_contract_single_official_span_is_direct_support() -> None:
    assert contract_utility_grade(
        choice="Entailment", evidence_spans=[3], candidate_span=3
    ) == 4


def test_contract_multi_span_set_marks_each_required_partial() -> None:
    assert contract_utility_grade(
        choice="Contradiction", evidence_spans=[3, 5], candidate_span=3
    ) == 3
    assert contract_utility_grade(
        choice="Contradiction", evidence_spans=[3, 5], candidate_span=4
    ) == 1


def test_contract_not_mentioned_has_no_positive_evidence() -> None:
    assert contract_utility_grade(
        choice="NotMentioned", evidence_spans=[], candidate_span=0
    ) == 1


def test_contract_grade_rejects_unknown_choice() -> None:
    try:
        contract_utility_grade(choice="Unknown", evidence_spans=[], candidate_span=0)
    except ValueError as exc:
        assert "unsupported ContractNLI choice" in str(exc)
    else:
        raise AssertionError("unknown choices must be rejected")
