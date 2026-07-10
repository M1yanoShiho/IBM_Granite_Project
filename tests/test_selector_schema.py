"""Contract tests for the ML evidence selector protocol and labels."""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.retrieval.selector_schema import (
    LabelRecord,
    Protocol,
    load_label_schema,
    load_protocol,
    validate_label_record,
)


def valid_label_record() -> dict[str, object]:
    return {
        "query_id": "q-001",
        "candidate_id": "doc-001",
        "source_parent_id": "source-001",
        "answer_cluster_id": "cluster-001",
        "required_fact_set_id": "facts-001",
        "support_level": "direct",
        "factual_status": "correct",
        "entity_match": "match",
        "time_match": "match",
        "condition_match": "not_applicable",
        "answerability": "answer",
        "source_quality": "strong",
        "conflict_to_gold": "no",
        "label_provenance": "official",
        "utility_grade": 4,
        "label_confidence": 0.95,
        "harm_type": "",
        "is_harmful": False,
        "is_direct_support": True,
        "is_required_support": True,
    }


def test_default_manifests_load_outside_repository(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    protocol = load_protocol()
    schema = load_label_schema()

    assert isinstance(protocol, Protocol)
    assert schema.schema_version == "1.0"


def test_protocol_freezes_every_required_decision() -> None:
    protocol = load_protocol()

    assert protocol.schema_version == "1.0"
    assert protocol.branch == "week5_MLSelector"
    assert protocol.first_stage == "q2d_granite"
    assert protocol.candidate_pool_size == 20
    assert protocol.generator_context_size == 10
    assert protocol.max_tokens_per_candidate == 384
    assert protocol.total_evidence_token_budget == 4096
    assert protocol.corroboration_relevance_alpha == 0.6
    assert protocol.random_seeds == (13, 42, 73)
    assert protocol.primary_selector_metric == "ndcg@10"
    assert protocol.harmful_rate_improvement_gate == -0.02
    assert protocol.ndcg_improvement_gate == 0.02
    assert protocol.required_evidence_non_inferiority_margin == -0.01
    assert protocol.deterministic_tie_break == ("first_stage_rank", "doc_id_ascending")
    assert protocol.utility_grades == (0, 1, 2, 3, 4)
    assert protocol.label_gains == (0, 1, 3, 7, 15)
    assert protocol.legacy_nq_300_role == "legacy_replication"
    assert protocol.sealed_niah_role == "confirmatory_test"
    assert protocol.ramdocs_official_context_size == 3
    assert protocol.ramdocs_adapted_candidate_pool_size == 20
    assert protocol.ramdocs_adapted_context_size == 10


def test_protocol_is_immutable() -> None:
    protocol = load_protocol()

    with pytest.raises(FrozenInstanceError):
        protocol.branch = "other"  # type: ignore[misc]


def test_valid_complete_label_record_is_normalized_to_immutable_record() -> None:
    record = validate_label_record(valid_label_record())

    assert isinstance(record, LabelRecord)
    assert record.query_id == "q-001"
    assert record.utility_grade == 4
    assert record.is_direct_support is True
    with pytest.raises(FrozenInstanceError):
        record.utility_grade = 3  # type: ignore[misc]


def test_optional_identifiers_accept_null() -> None:
    raw = valid_label_record()
    raw["answer_cluster_id"] = None
    raw["required_fact_set_id"] = None

    record = validate_label_record(raw)

    assert record.answer_cluster_id is None
    assert record.required_fact_set_id is None


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("support_level", "unsupported"),
        ("factual_status", "unsupported"),
        ("entity_match", "unsupported"),
        ("time_match", "unsupported"),
        ("condition_match", "unsupported"),
        ("answerability", "unsupported"),
        ("source_quality", "unsupported"),
        ("conflict_to_gold", "unsupported"),
        ("label_provenance", "unsupported"),
    ],
)
def test_controlled_vocabularies_reject_unknown_values(field: str, invalid_value: str) -> None:
    raw = valid_label_record()
    raw[field] = invalid_value

    with pytest.raises(ValueError, match=field):
        validate_label_record(raw)


@pytest.mark.parametrize("utility_grade", [-1, 5, True])
def test_utility_grade_outside_canonical_range_is_rejected(utility_grade: object) -> None:
    raw = valid_label_record()
    raw["utility_grade"] = utility_grade

    with pytest.raises(ValueError, match="utility_grade"):
        validate_label_record(raw)


@pytest.mark.parametrize("label_confidence", [-0.01, 1.01, True])
def test_confidence_outside_closed_unit_interval_is_rejected(label_confidence: object) -> None:
    raw = valid_label_record()
    raw["label_confidence"] = label_confidence

    with pytest.raises(ValueError, match="label_confidence"):
        validate_label_record(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("is_harmful", True),
        ("is_direct_support", False),
        ("is_required_support", False),
    ],
)
def test_inconsistent_derived_flags_are_rejected(field: str, value: bool) -> None:
    raw = valid_label_record()
    raw[field] = value

    with pytest.raises(ValueError, match=field):
        validate_label_record(raw)


@pytest.mark.parametrize("field", ["query_id", "candidate_id", "source_parent_id"])
def test_missing_required_identifiers_are_rejected(field: str) -> None:
    raw = valid_label_record()
    del raw[field]

    with pytest.raises(ValueError, match=field):
        validate_label_record(raw)


def test_label_schema_defines_all_required_allowed_values() -> None:
    schema = load_label_schema()

    assert schema.support_levels == ("direct", "partial", "none")
    assert schema.factual_statuses == ("correct", "incorrect", "disputed", "unknown")
    assert schema.entity_matches == ("match", "mismatch", "unknown", "not_applicable")
    assert schema.time_matches == ("match", "mismatch", "unknown", "not_applicable", "outdated")
    assert schema.condition_matches == ("match", "mismatch", "unknown", "not_applicable")
    assert schema.answerability_values == ("answer", "non_answer", "extraction_failure")
    assert schema.source_quality_values == ("strong", "weak", "unknown")
    assert schema.conflict_to_gold_values == ("yes", "no", "unknown")
    assert schema.label_provenance_values == (
        "official",
        "human",
        "deterministic_rule",
        "independent_teacher",
    )
    assert schema.utility_grade_minimum == 0
    assert schema.utility_grade_maximum == 4
    assert schema.label_confidence_minimum == 0.0
    assert schema.label_confidence_maximum == 1.0


@pytest.mark.parametrize("manifest_name", ["protocol_manifest.json", "label_schema.json"])
def test_manifests_contain_no_placeholder_or_todo_fields(manifest_name: str) -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "selector" / manifest_name
    manifest = json.loads(path.read_text(encoding="utf-8"))

    def assert_no_placeholders(value: object) -> None:
        if isinstance(value, dict):
            for key, nested_value in value.items():
                assert "todo" not in key.lower()
                assert_no_placeholders(nested_value)
        elif isinstance(value, list):
            for nested_value in value:
                assert_no_placeholders(nested_value)
        elif isinstance(value, str):
            assert "todo" not in value.lower()
            assert "placeholder" not in value.lower()

    assert_no_placeholders(manifest)
