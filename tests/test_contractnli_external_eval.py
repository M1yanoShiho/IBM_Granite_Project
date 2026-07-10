"""Tests for ContractNLI cross-domain selector feature materialization."""

from __future__ import annotations

from eval.contractnli_external_eval import materialize_contract_features


def test_materialize_contract_features_keeps_only_eligible_test_groups() -> None:
    rows = [
        {
            "query_id": "kept",
            "split": "test",
            "eligible_for_training": True,
            "candidates": [
                {
                    "candidate_id": "a",
                    "relevance_score": 0.8,
                    "original_rank": 1,
                    "utility_grade": 4,
                    "source": "contractnli_clause",
                },
                {
                    "candidate_id": "b",
                    "relevance_score": 0.2,
                    "original_rank": 2,
                    "utility_grade": 1,
                    "source": "contractnli_clause",
                },
            ],
        },
        {
            "query_id": "not-mentioned",
            "split": "test",
            "eligible_for_training": False,
            "candidates": [],
        },
        {
            "query_id": "train",
            "split": "train",
            "eligible_for_training": True,
            "candidates": [],
        },
    ]

    output = materialize_contract_features(rows)

    assert [row["query_id"] for row in output] == ["kept"]
    assert output[0]["candidates"][0]["relevance_normalized"] == 1.0
    assert output[0]["candidates"][1]["relevance_normalized"] == 0.0
    assert output[0]["candidates"][0]["reciprocal_rank"] == 1.0
    assert output[0]["candidates"][0]["extraction_failure"] == 1.0
    assert output[0]["candidates"][0]["judge_parse_failure"] == 1.0

