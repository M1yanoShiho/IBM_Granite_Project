from __future__ import annotations

import hashlib

import pytest

from evidence_rag.evaluation.system_scorer import (
    SCORER_SCHEMA_VERSION,
    ScorerContractError,
    answer_score,
    citation_f1,
    score_bundle,
)


def _gold(query_id: str = "q1", *, dataset: str = "hotpotqa") -> dict[str, object]:
    text = "Revealed development support."
    return {
        "schema_version": SCORER_SCHEMA_VERSION,
        "dataset": dataset,
        "query_id": query_id,
        "gold_answer_aliases": ["The Blue River"],
        "support_units": [
            {
                "unit_id": "p000:u000",
                "source_id": "p000",
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        ],
        "component_id": f"component-{query_id}",
    }


def _perfect(query_id: str = "q1") -> dict[str, object]:
    return {
        "query_id": query_id,
        "retrieved_unit_ids": ["p000:u000"],
        "selected_unit_ids": ["p000:u000"],
        "selected_source_ids": ["p000"],
        "answer": "Blue River [1]",
        "citation_indices": [1],
        "minicheck": {"precision": 1.0, "recall": 1.0},
    }


def test_five_metrics_score_known_revealed_fixture() -> None:
    report = score_bundle([_gold()], [_perfect()])

    assert report["n_queries"] == 1
    assert report["denominators"] == {
        "ret": 1,
        "sel": 1,
        "ans": 1,
        "cit": 1,
        "rar": 1,
    }
    assert report["aggregate"] == {
        "ret": 1.0,
        "sel": 1.0,
        "ans": 1.0,
        "cit": 1.0,
        "rar": 1.0,
    }


def test_answer_protocol_uses_token_f1_for_multihop_and_accuracy_for_rgb() -> None:
    assert answer_score("hotpotqa", "blue stream", ["blue river"]) == 0.5
    assert answer_score("musique-answerable", "blue stream", ["blue river"]) == 0.5
    assert answer_score("rgb-noise", "The Blue River [2]", ["blue river"]) == 1.0
    assert answer_score("rgb-noise", "blue river extra", ["blue river"]) == 0.0


def test_citation_metric_is_per_query_harmonic_f1() -> None:
    assert citation_f1(1.0, 0.5) == pytest.approx(2.0 / 3.0)
    assert citation_f1(0.0, 0.0) == 0.0


@pytest.mark.parametrize(
    ("change", "reason", "zero_metrics"),
    [
        ({"answer": "", "citation_indices": []}, "empty_answer", ("ans", "cit", "rar")),
        ({"citation_indices": [2]}, "invalid_citation", ("cit", "rar")),
        ({"failure_stage": "generation"}, "generation_failure", ("ans", "cit", "rar")),
        ({"failure_stage": "scoring"}, "scoring_failure", ("ans", "cit", "rar")),
    ],
)
def test_error_samples_remain_in_denominator_and_score_zero(
    change: dict[str, object], reason: str, zero_metrics: tuple[str, ...]
) -> None:
    outcome = {**_perfect(), **change}
    report = score_bundle([_gold()], [outcome])
    row = report["per_query"][0]

    assert report["denominators"] == {key: 1 for key in ("ret", "sel", "ans", "cit", "rar")}
    assert row["failure_reason"] == reason
    assert all(row["metrics"][key] == 0.0 for key in zero_metrics)


def test_missing_output_is_an_explicit_all_zero_row() -> None:
    report = score_bundle([_gold()], [])

    assert report["n_queries"] == 1
    assert report["failure_counts"] == {"missing_output": 1}
    assert report["per_query"][0]["metrics"] == {
        "ret": 0.0,
        "sel": 0.0,
        "ans": 0.0,
        "cit": 0.0,
        "rar": 0.0,
    }


def test_duplicate_or_unexpected_outputs_fail_instead_of_changing_denominator() -> None:
    with pytest.raises(ScorerContractError, match="duplicate outcome"):
        score_bundle([_gold()], [_perfect(), _perfect()])
    with pytest.raises(ScorerContractError, match="outside the frozen denominator"):
        score_bundle([_gold()], [_perfect("another-query")])
