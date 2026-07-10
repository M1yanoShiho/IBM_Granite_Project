"""Immutable protocol and label-schema helpers for the ML evidence selector."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SELECTOR_DIR = _REPOSITORY_ROOT / "data" / "selector"


@dataclass(frozen=True)
class Protocol:
    """The fixed, machine-readable M0 evaluation protocol."""

    schema_version: str
    branch: str
    first_stage: str
    candidate_pool_size: int
    generator_context_size: int
    max_tokens_per_candidate: int
    total_evidence_token_budget: int
    corroboration_relevance_alpha: float
    random_seeds: tuple[int, ...]
    primary_selector_metric: str
    harmful_rate_improvement_gate: float
    ndcg_improvement_gate: float
    required_evidence_non_inferiority_margin: float
    deterministic_tie_break: tuple[str, ...]
    utility_grades: tuple[int, ...]
    label_gains: tuple[int, ...]
    legacy_nq_300_role: str
    sealed_niah_role: str
    ramdocs_official_context_size: int
    ramdocs_adapted_candidate_pool_size: int
    ramdocs_adapted_context_size: int


@dataclass(frozen=True)
class LabelSchema:
    """Controlled vocabularies and numeric limits for selector labels."""

    schema_version: str
    support_levels: tuple[str, ...]
    factual_statuses: tuple[str, ...]
    entity_matches: tuple[str, ...]
    time_matches: tuple[str, ...]
    condition_matches: tuple[str, ...]
    answerability_values: tuple[str, ...]
    source_quality_values: tuple[str, ...]
    conflict_to_gold_values: tuple[str, ...]
    label_provenance_values: tuple[str, ...]
    utility_grade_minimum: int
    utility_grade_maximum: int
    label_confidence_minimum: float
    label_confidence_maximum: float
    required_identifier_fields: tuple[str, ...]
    optional_identifier_fields: tuple[str, ...]
    harm_type_allowed_values: tuple[str, ...] | None


@dataclass(frozen=True)
class LabelRecord:
    """A normalized, validated selector training or evaluation label."""

    query_id: str
    candidate_id: str
    source_parent_id: str
    answer_cluster_id: str | None
    required_fact_set_id: str | None
    support_level: str
    factual_status: str
    entity_match: str
    time_match: str
    condition_match: str
    answerability: str
    source_quality: str
    conflict_to_gold: str
    label_provenance: str
    utility_grade: int
    label_confidence: float
    harm_type: str | None
    is_harmful: bool
    is_direct_support: bool
    is_required_support: bool


def _load_json(path: str | Path) -> dict[str, object]:
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to load JSON manifest at {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"Manifest at {path} must contain a JSON object")
    return loaded


def _matches_exact_json_value(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _matches_exact_json_value(item, expected_item)
            for item, expected_item in zip(actual, expected)
        )
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _matches_exact_json_value(actual[key], expected_value)
            for key, expected_value in expected.items()
        )
    return actual == expected


def _exact_value(data: Mapping[str, object], field: str, expected: object) -> object:
    actual = data.get(field)
    if not _matches_exact_json_value(actual, expected):
        raise ValueError(f"{field} must be {expected!r}; got {actual!r}")
    return actual


def load_protocol(path: str | Path | None = None) -> Protocol:
    """Load and validate the fixed M0 protocol manifest."""

    data = _load_json(path or _DEFAULT_SELECTOR_DIR / "protocol_manifest.json")
    return Protocol(
        schema_version=_exact_value(data, "schema_version", "1.0"),  # type: ignore[arg-type]
        branch=_exact_value(data, "branch", "week5_MLSelector"),  # type: ignore[arg-type]
        first_stage=_exact_value(data, "first_stage", "q2d_granite"),  # type: ignore[arg-type]
        candidate_pool_size=_exact_value(data, "candidate_pool_size", 20),  # type: ignore[arg-type]
        generator_context_size=_exact_value(data, "generator_context_size", 10),  # type: ignore[arg-type]
        max_tokens_per_candidate=_exact_value(data, "max_tokens_per_candidate", 384),  # type: ignore[arg-type]
        total_evidence_token_budget=_exact_value(data, "total_evidence_token_budget", 4096),  # type: ignore[arg-type]
        corroboration_relevance_alpha=_exact_value(data, "corroboration_relevance_alpha", 0.6),  # type: ignore[arg-type]
        random_seeds=tuple(_exact_value(data, "random_seeds", [13, 42, 73])),  # type: ignore[arg-type]
        primary_selector_metric=_exact_value(data, "primary_selector_metric", "ndcg@10"),  # type: ignore[arg-type]
        harmful_rate_improvement_gate=_exact_value(data, "harmful_rate_improvement_gate", -0.02),  # type: ignore[arg-type]
        ndcg_improvement_gate=_exact_value(data, "ndcg_improvement_gate", 0.02),  # type: ignore[arg-type]
        required_evidence_non_inferiority_margin=_exact_value(data, "required_evidence_non_inferiority_margin", -0.01),  # type: ignore[arg-type]
        deterministic_tie_break=tuple(
            _exact_value(data, "deterministic_tie_break", ["first_stage_rank", "doc_id_ascending"])
        ),  # type: ignore[arg-type]
        utility_grades=tuple(_exact_value(data, "utility_grades", [0, 1, 2, 3, 4])),  # type: ignore[arg-type]
        label_gains=tuple(_exact_value(data, "label_gains", [0, 1, 3, 7, 15])),  # type: ignore[arg-type]
        legacy_nq_300_role=_exact_value(data, "legacy_nq_300_role", "legacy_replication"),  # type: ignore[arg-type]
        sealed_niah_role=_exact_value(data, "sealed_niah_role", "confirmatory_test"),  # type: ignore[arg-type]
        ramdocs_official_context_size=_exact_value(data, "ramdocs_official_context_size", 3),  # type: ignore[arg-type]
        ramdocs_adapted_candidate_pool_size=_exact_value(data, "ramdocs_adapted_candidate_pool_size", 20),  # type: ignore[arg-type]
        ramdocs_adapted_context_size=_exact_value(data, "ramdocs_adapted_context_size", 10),  # type: ignore[arg-type]
    )


def _allowed_values(
    fields: Mapping[str, object], field: str, expected: list[str]
) -> tuple[str, ...]:
    definition = fields.get(field)
    if not isinstance(definition, dict):
        raise ValueError(f"label schema field {field} must be an object")
    values = definition.get("allowed_values")
    if not _matches_exact_json_value(values, expected):
        raise ValueError(f"label schema {field}.allowed_values must be {expected!r}; got {values!r}")
    return tuple(expected)


def _numeric_bounds(
    fields: Mapping[str, object], field: str, expected_type: str, minimum: object, maximum: object
) -> tuple[object, object]:
    definition = fields.get(field)
    if not isinstance(definition, dict):
        raise ValueError(f"label schema field {field} must be an object")
    if not _matches_exact_json_value(definition.get("type"), expected_type):
        raise ValueError(f"label schema {field}.type must be {expected_type!r}")
    if not _matches_exact_json_value(definition.get("minimum"), minimum) or not _matches_exact_json_value(
        definition.get("maximum"), maximum
    ):
        raise ValueError(f"label schema {field} bounds must be [{minimum!r}, {maximum!r}]")
    return minimum, maximum


def _exact_field_declaration(
    fields: Mapping[str, object], field: str, expected: dict[str, object]
) -> None:
    definition = fields.get(field)
    if not _matches_exact_json_value(definition, expected):
        raise ValueError(f"label schema {field} declaration must be {expected!r}; got {definition!r}")


def load_label_schema(path: str | Path | None = None) -> LabelSchema:
    """Load and validate the canonical selector label schema."""

    data = _load_json(path or _DEFAULT_SELECTOR_DIR / "label_schema.json")
    if not _matches_exact_json_value(data.get("schema_version"), "1.0"):
        raise ValueError("label schema schema_version must be '1.0'")
    fields = data.get("fields")
    if not isinstance(fields, dict):
        raise ValueError("label schema fields must be an object")
    utility_minimum, utility_maximum = _numeric_bounds(fields, "utility_grade", "integer", 0, 4)
    confidence_minimum, confidence_maximum = _numeric_bounds(fields, "label_confidence", "float", 0.0, 1.0)
    required_identifier_fields = ("query_id", "candidate_id", "source_parent_id")
    optional_identifier_fields = ("answer_cluster_id", "required_fact_set_id")
    for field in required_identifier_fields:
        _exact_field_declaration(fields, field, {"type": "string", "required": True})
    for field in optional_identifier_fields:
        _exact_field_declaration(fields, field, {"type": "string", "required": False, "nullable": True})
    _exact_field_declaration(
        fields,
        "harm_type",
        {"type": "string", "required": False, "nullable": True, "allowed_values": None},
    )
    derived_flags = data.get("derived_flags")
    expected_derived_flags = {
        "is_harmful": "utility_grade == 0",
        "is_direct_support": "utility_grade == 4",
        "is_required_support": "utility_grade >= 3",
    }
    if not _matches_exact_json_value(derived_flags, expected_derived_flags):
        raise ValueError("label schema derived_flags must define the canonical utility-grade rules")
    return LabelSchema(
        schema_version="1.0",
        support_levels=_allowed_values(fields, "support_level", ["direct", "partial", "none"]),
        factual_statuses=_allowed_values(fields, "factual_status", ["correct", "incorrect", "disputed", "unknown"]),
        entity_matches=_allowed_values(fields, "entity_match", ["match", "mismatch", "unknown", "not_applicable"]),
        time_matches=_allowed_values(fields, "time_match", ["match", "mismatch", "unknown", "not_applicable", "outdated"]),
        condition_matches=_allowed_values(fields, "condition_match", ["match", "mismatch", "unknown", "not_applicable"]),
        answerability_values=_allowed_values(fields, "answerability", ["answer", "non_answer", "extraction_failure"]),
        source_quality_values=_allowed_values(fields, "source_quality", ["strong", "weak", "unknown"]),
        conflict_to_gold_values=_allowed_values(fields, "conflict_to_gold", ["yes", "no", "unknown"]),
        label_provenance_values=_allowed_values(
            fields, "label_provenance", ["official", "human", "deterministic_rule", "independent_teacher"]
        ),
        utility_grade_minimum=utility_minimum,  # type: ignore[arg-type]
        utility_grade_maximum=utility_maximum,  # type: ignore[arg-type]
        label_confidence_minimum=confidence_minimum,  # type: ignore[arg-type]
        label_confidence_maximum=confidence_maximum,  # type: ignore[arg-type]
        required_identifier_fields=required_identifier_fields,
        optional_identifier_fields=optional_identifier_fields,
        harm_type_allowed_values=None,
    )


def _required_identifier(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_identifier(record: Mapping[str, object], field: str) -> str | None:
    value = record.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string or null")
    return value


def _vocabulary_value(record: Mapping[str, object], field: str, allowed: tuple[str, ...]) -> str:
    value = record.get(field)
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{field} must be one of {allowed!r}; got {value!r}")
    return value


def _required_bool(record: Mapping[str, object], field: str) -> bool:
    value = record.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def validate_label_record(record: Mapping[str, object]) -> LabelRecord:
    """Validate a label record and return its immutable normalized form."""

    if not isinstance(record, Mapping):
        raise ValueError("record must be a mapping")
    schema = load_label_schema()
    utility_grade = record.get("utility_grade")
    if isinstance(utility_grade, bool) or not isinstance(utility_grade, int):
        raise ValueError("utility_grade must be an integer")
    if not schema.utility_grade_minimum <= utility_grade <= schema.utility_grade_maximum:
        raise ValueError("utility_grade must be between 0 and 4")
    label_confidence = record.get("label_confidence")
    if isinstance(label_confidence, bool) or not isinstance(label_confidence, float):
        raise ValueError("label_confidence must be a float")
    if not schema.label_confidence_minimum <= label_confidence <= schema.label_confidence_maximum:
        raise ValueError("label_confidence must be between 0.0 and 1.0")
    harm_type = record.get("harm_type")
    if harm_type is not None and not isinstance(harm_type, str):
        raise ValueError("harm_type must be a string or null")
    if harm_type is not None and schema.harm_type_allowed_values is not None:
        if harm_type not in schema.harm_type_allowed_values:
            raise ValueError(f"harm_type must be one of {schema.harm_type_allowed_values!r}; got {harm_type!r}")
    is_harmful = _required_bool(record, "is_harmful")
    is_direct_support = _required_bool(record, "is_direct_support")
    is_required_support = _required_bool(record, "is_required_support")
    expected_flags = {
        "is_harmful": utility_grade == 0,
        "is_direct_support": utility_grade == 4,
        "is_required_support": utility_grade >= 3,
    }
    actual_flags = {
        "is_harmful": is_harmful,
        "is_direct_support": is_direct_support,
        "is_required_support": is_required_support,
    }
    for field, expected in expected_flags.items():
        if actual_flags[field] != expected:
            raise ValueError(f"{field} must equal {expected} when utility_grade is {utility_grade}")
    return LabelRecord(
        query_id=_required_identifier(record, "query_id"),
        candidate_id=_required_identifier(record, "candidate_id"),
        source_parent_id=_required_identifier(record, "source_parent_id"),
        answer_cluster_id=_optional_identifier(record, "answer_cluster_id"),
        required_fact_set_id=_optional_identifier(record, "required_fact_set_id"),
        support_level=_vocabulary_value(record, "support_level", schema.support_levels),
        factual_status=_vocabulary_value(record, "factual_status", schema.factual_statuses),
        entity_match=_vocabulary_value(record, "entity_match", schema.entity_matches),
        time_match=_vocabulary_value(record, "time_match", schema.time_matches),
        condition_match=_vocabulary_value(record, "condition_match", schema.condition_matches),
        answerability=_vocabulary_value(record, "answerability", schema.answerability_values),
        source_quality=_vocabulary_value(record, "source_quality", schema.source_quality_values),
        conflict_to_gold=_vocabulary_value(record, "conflict_to_gold", schema.conflict_to_gold_values),
        label_provenance=_vocabulary_value(record, "label_provenance", schema.label_provenance_values),
        utility_grade=utility_grade,
        label_confidence=label_confidence,
        harm_type=harm_type,
        is_harmful=is_harmful,
        is_direct_support=is_direct_support,
        is_required_support=is_required_support,
    )
