"""Pure tests for the controlled NIAH ML-selector pilot."""

from __future__ import annotations

import math

from eval.niah_selector_pilot import (
    add_group_features,
    minmax,
    non_answer_passes_validator,
    parse_reliability_judgment,
    rank_candidates,
    selector_metrics,
    utility_grade,
)


def test_utility_grade_maps_official_and_synthetic_sources() -> None:
    assert utility_grade(source="dpr", relevance=2) == 4
    assert utility_grade(source="dpr", relevance=1) == 3
    assert utility_grade(source="dpr", relevance=0) == 1
    assert utility_grade(source="dpr", relevance=-1) == 2
    assert utility_grade(source="counterfactual", relevance=None) == 0
    assert utility_grade(source="generative_non_answer", relevance=None) == 2


def test_minmax_handles_constant_values_without_nan() -> None:
    assert minmax([3.0, 3.0]) == [0.0, 0.0]
    assert minmax([1.0, 3.0, 2.0]) == [0.0, 1.0, 0.5]


def test_rank_candidates_uses_score_then_original_rank_then_id() -> None:
    rows = [
        {"candidate_id": "b", "relevance_score": 0.5, "original_rank": 2},
        {"candidate_id": "a", "relevance_score": 0.5, "original_rank": 2},
        {"candidate_id": "c", "relevance_score": 0.5, "original_rank": 1},
    ]
    ranked = rank_candidates(rows, "relevance_score")
    assert [row["candidate_id"] for row in ranked] == ["c", "a", "b"]


def test_add_group_features_counts_exact_and_source_deduplicated_votes() -> None:
    rows = [
        {
            "candidate_id": "a",
            "source_parent_id": "page-1",
            "relevance_score": 0.9,
            "original_rank": 1,
            "extracted_answer": "The Old Man",
        },
        {
            "candidate_id": "b",
            "source_parent_id": "page-1",
            "relevance_score": 0.8,
            "original_rank": 2,
            "extracted_answer": "old man",
        },
        {
            "candidate_id": "c",
            "source_parent_id": "page-2",
            "relevance_score": 0.7,
            "original_rank": 3,
            "extracted_answer": "old man",
        },
        {
            "candidate_id": "d",
            "source_parent_id": "page-3",
            "relevance_score": 0.6,
            "original_rank": 4,
            "extracted_answer": "NONE",
        },
    ]
    enriched = add_group_features(rows, parametric_answer="Old Man")
    by_id = {row["candidate_id"]: row for row in enriched}
    assert by_id["a"]["exact_vote_count"] == 3.0
    assert by_id["a"]["source_dedup_vote_count"] == 2.0
    assert by_id["a"]["parametric_agreement"] == 1.0
    assert by_id["d"]["extraction_failure"] == 1.0
    assert by_id["d"]["exact_vote_count"] == 0.0


def test_selector_metrics_measure_quality_harm_and_required_recall() -> None:
    groups = {
        "q1": [
            {"candidate_id": "g", "utility_grade": 4, "score": 0.9},
            {"candidate_id": "h", "utility_grade": 0, "score": 0.8},
            {"candidate_id": "n", "utility_grade": 1, "score": 0.7},
        ],
        "q2": [
            {"candidate_id": "p1", "utility_grade": 3, "score": 0.7},
            {"candidate_id": "p2", "utility_grade": 3, "score": 0.6},
            {"candidate_id": "x", "utility_grade": 1, "score": 0.9},
        ],
    }
    metrics = selector_metrics(groups, score_key="score", k=2)
    assert 0.0 <= metrics["ndcg@2"] <= 1.0
    assert metrics["harmful_rate@2"] == 0.25
    assert metrics["direct_support_precision@2"] == 0.25
    assert math.isclose(metrics["required_evidence_recall@2"], 0.75)


def test_parse_reliability_judgment_accepts_fenced_json_and_normalizes_scores() -> None:
    parsed = parse_reliability_judgment(
        "```json\n"
        '{"direct_support": 2, "condition_coverage": 1, "evidence_sufficiency": 0}'
        "\n```"
    )

    assert parsed == {
        "judge_direct_support": 1.0,
        "judge_condition_coverage": 0.5,
        "judge_evidence_sufficiency": 0.0,
        "judge_parse_failure": 0.0,
    }


def test_parse_reliability_judgment_fails_closed_on_invalid_output() -> None:
    assert parse_reliability_judgment("This passage looks useful.") == {
        "judge_direct_support": 0.0,
        "judge_condition_coverage": 0.0,
        "judge_evidence_sufficiency": 0.0,
        "judge_parse_failure": 1.0,
    }


def test_parse_reliability_judgment_rejects_out_of_range_scores() -> None:
    parsed = parse_reliability_judgment(
        '{"direct_support": 3, "condition_coverage": 1, "evidence_sufficiency": 2}'
    )
    assert parsed["judge_parse_failure"] == 1.0


def test_non_answer_validator_rejects_paraphrased_answers_found_by_granite() -> None:
    assert non_answer_passes_validator(
        "The topic concerns the history of legislative chambers.",
        aliases=["ancient Roman Senate"],
        validator_output="NONE",
    )
    assert not non_answer_passes_validator(
        "It was inspired by the ancient Roman model.",
        aliases=["ancient Roman Senate"],
        validator_output="ancient Roman model",
    )


def test_non_answer_validator_also_rejects_literal_gold_aliases() -> None:
    assert not non_answer_passes_validator(
        "The answer is the Ancient Roman Senate.",
        aliases=["ancient Roman Senate"],
        validator_output="NONE",
    )
