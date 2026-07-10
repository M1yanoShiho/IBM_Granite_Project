"""Tests for controlled-NIAH label-audit helpers."""

from __future__ import annotations

from eval.niah_label_audit import parse_grade, quadratic_weighted_kappa, select_query_groups


def test_parse_grade_accepts_one_explicit_grade_and_rejects_ambiguous_output() -> None:
    assert parse_grade("GRADE: 4") == 4
    assert parse_grade("```\n2\n```") == 2
    assert parse_grade("The choice is either 2 or 3") is None
    assert parse_grade("unsupported") is None


def test_quadratic_weighted_kappa_is_one_for_identical_labels() -> None:
    assert quadratic_weighted_kappa([0, 2, 4], [0, 2, 4]) == 1.0


def test_quadratic_weighted_kappa_penalizes_opposite_labels() -> None:
    assert quadratic_weighted_kappa([0, 0, 4, 4], [4, 4, 0, 0]) < 0.0


def test_select_query_groups_is_deterministic_and_stratified() -> None:
    rows = [
        {"query_id": f"{split}-{index}", "split": split}
        for split, count in (("train", 8), ("dev", 6), ("test", 6))
        for index in range(count)
    ]
    quotas = {"train": 4, "dev": 3, "test": 3}

    first = select_query_groups(rows, quotas=quotas, seed=42)
    second = select_query_groups(rows, quotas=quotas, seed=42)

    assert first == second
    assert {split: sum(row["split"] == split for row in first) for split in quotas} == quotas
