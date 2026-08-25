"""Five-metric scorer for the frozen Experiment 04 system evaluation.

This module is scorer-only: it consumes a gold sidecar after system outputs are
frozen.  MiniCheck inference is intentionally outside this pure scorer; its
per-query precision and recall are supplied as frozen judge outputs.
"""

from __future__ import annotations

import hashlib
import re
import string
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

SCORER_SCHEMA_VERSION = "experiment04.scorer.v1"
SIDECAR_KEYS = {
    "schema_version",
    "dataset",
    "query_id",
    "gold_answer_aliases",
    "support_units",
    "component_id",
}
SUPPORT_UNIT_KEYS = {"unit_id", "source_id", "text", "text_sha256"}
METRIC_KEYS = ("ret", "sel", "ans", "cit", "rar")
TOKEN_F1_DATASETS = {"hotpotqa", "musique-answerable"}
ACCURACY_DATASETS = {"rgb-noise"}

_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_CITATION = re.compile(r"\[(?:\s*\d+\s*,?\s*)+\]")


class ScorerContractError(ValueError):
    """Raised for malformed scorer inputs rather than silently dropping a query."""


def _normalise_answer(text: str) -> str:
    without_citations = _CITATION.sub(" ", text)
    without_punctuation = "".join(
        " " if character in string.punctuation else character
        for character in without_citations.casefold()
    )
    without_articles = _ARTICLES.sub(" ", without_punctuation)
    return _WHITESPACE.sub(" ", without_articles).strip()


def token_f1(answer: str, alias: str) -> float:
    predicted = _normalise_answer(answer).split()
    gold = _normalise_answer(alias).split()
    if not predicted or not gold:
        return float(predicted == gold and bool(gold))
    overlap = sum((Counter(predicted) & Counter(gold)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2.0 * precision * recall / (precision + recall)


def answer_score(dataset: str, answer: str, aliases: Iterable[str]) -> float:
    materialized = tuple(aliases)
    if not materialized:
        raise ScorerContractError("gold answer aliases may not be empty")
    if dataset in TOKEN_F1_DATASETS:
        return max(token_f1(answer, alias) for alias in materialized)
    if dataset in ACCURACY_DATASETS:
        predicted = _normalise_answer(answer)
        return float(bool(predicted) and any(predicted == _normalise_answer(alias) for alias in materialized))
    raise ScorerContractError(f"unsupported dataset for Ans.: {dataset}")


def answer_exact_match(answer: str, aliases: Iterable[str]) -> bool:
    predicted = _normalise_answer(answer)
    return bool(predicted) and any(predicted == _normalise_answer(alias) for alias in aliases)


def citation_f1(precision: float, recall: float) -> float:
    if not 0.0 <= precision <= 1.0 or not 0.0 <= recall <= 1.0:
        raise ScorerContractError("MiniCheck precision/recall must be in [0, 1]")
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def validate_sidecar_record(record: Mapping[str, Any], *, dataset: str | None = None) -> None:
    observed = set(record)
    if observed != SIDECAR_KEYS:
        raise ScorerContractError(
            f"sidecar keys differ: missing={sorted(SIDECAR_KEYS - observed)}, "
            f"unexpected={sorted(observed - SIDECAR_KEYS)}"
        )
    if record["schema_version"] != SCORER_SCHEMA_VERSION:
        raise ScorerContractError("sidecar schema version is not frozen v1")
    if dataset is not None and record["dataset"] != dataset:
        raise ScorerContractError("sidecar dataset does not match its bundle")
    if not isinstance(record["query_id"], str) or not record["query_id"].strip():
        raise ScorerContractError("sidecar query_id must be a non-blank string")
    aliases = record["gold_answer_aliases"]
    if not isinstance(aliases, list) or not aliases or not all(
        isinstance(alias, str) and alias.strip() for alias in aliases
    ):
        raise ScorerContractError("gold_answer_aliases must be non-empty strings")
    units = record["support_units"]
    if not isinstance(units, list) or not units:
        raise ScorerContractError("support_units must be a non-empty list")
    unit_ids: set[str] = set()
    for index, unit in enumerate(units):
        if not isinstance(unit, Mapping) or set(unit) != SUPPORT_UNIT_KEYS:
            raise ScorerContractError(f"support unit {index} has the wrong schema")
        if not all(isinstance(unit[key], str) and unit[key] for key in SUPPORT_UNIT_KEYS):
            raise ScorerContractError(f"support unit {index} contains an empty field")
        if unit["unit_id"] in unit_ids:
            raise ScorerContractError("sidecar contains duplicate support unit IDs")
        if not str(unit["unit_id"]).startswith(f"{unit['source_id']}:u"):
            raise ScorerContractError("support unit is not local to its source")
        if not re.fullmatch(r"[0-9a-f]{64}", str(unit["text_sha256"])):
            raise ScorerContractError("support unit text_sha256 is malformed")
        observed_hash = hashlib.sha256(str(unit["text"]).encode()).hexdigest()
        if unit["text_sha256"] != observed_hash:
            raise ScorerContractError("support unit text_sha256 does not match its text")
        unit_ids.add(str(unit["unit_id"]))
    if not isinstance(record["component_id"], str) or not record["component_id"].strip():
        raise ScorerContractError("component_id must be a non-blank string")


def _string_set(outcome: Mapping[str, Any], key: str) -> set[str]:
    value = outcome.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ScorerContractError(f"{key} must be a list of strings")
    return set(value)


def _citation_indices(outcome: Mapping[str, Any]) -> tuple[int, ...]:
    value = outcome.get("citation_indices", [])
    if not isinstance(value, list) or not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        raise ScorerContractError("citation_indices must be a list of integers")
    return tuple(value)


def score_query(gold: Mapping[str, Any], outcome: Mapping[str, Any] | None) -> dict[str, Any]:
    """Score one query; failures are explicit zero-valued rows, never omissions."""

    validate_sidecar_record(gold)
    support_ids = {str(unit["unit_id"]) for unit in gold["support_units"]}
    if outcome is None:
        return {
            "query_id": gold["query_id"],
            "component_id": gold["component_id"],
            "metrics": {key: 0.0 for key in METRIC_KEYS},
            "failure_reason": "missing_output",
        }
    if outcome.get("query_id") != gold["query_id"]:
        raise ScorerContractError("outcome query_id does not match sidecar query_id")

    retrieved = _string_set(outcome, "retrieved_unit_ids")
    selected = _string_set(outcome, "selected_unit_ids")
    if not selected <= retrieved:
        raise ScorerContractError("selected_unit_ids must be a subset of retrieved_unit_ids")
    ret = len(retrieved & support_ids) / len(support_ids)
    sel = len(selected & support_ids) / len(support_ids)
    failure_stage = outcome.get("failure_stage")
    if failure_stage is not None and failure_stage not in {
        "retrieval",
        "selection",
        "generation",
        "scoring",
        "system",
    }:
        raise ScorerContractError("failure_stage is not a registered machine-readable value")

    answer = outcome.get("answer", "")
    if not isinstance(answer, str):
        raise ScorerContractError("answer must be a string")
    if failure_stage is not None:
        return {
            "query_id": gold["query_id"],
            "component_id": gold["component_id"],
            "metrics": {"ret": ret, "sel": sel, "ans": 0.0, "cit": 0.0, "rar": 0.0},
            "failure_reason": f"{failure_stage}_failure",
        }
    if not _normalise_answer(answer):
        return {
            "query_id": gold["query_id"],
            "component_id": gold["component_id"],
            "metrics": {"ret": ret, "sel": sel, "ans": 0.0, "cit": 0.0, "rar": 0.0},
            "failure_reason": "empty_answer",
        }

    ans = answer_score(str(gold["dataset"]), answer, gold["gold_answer_aliases"])
    selected_sources = outcome.get("selected_source_ids", [])
    if not isinstance(selected_sources, list) or not all(isinstance(item, str) for item in selected_sources):
        raise ScorerContractError("selected_source_ids must be a list of strings")
    indices = _citation_indices(outcome)
    citations_valid = bool(indices) and all(1 <= index <= len(selected_sources) for index in indices)
    if not citations_valid:
        return {
            "query_id": gold["query_id"],
            "component_id": gold["component_id"],
            "metrics": {"ret": ret, "sel": sel, "ans": ans, "cit": 0.0, "rar": 0.0},
            "failure_reason": "invalid_citation",
        }

    judge = outcome.get("minicheck")
    if not isinstance(judge, Mapping) or set(judge) != {"precision", "recall"}:
        return {
            "query_id": gold["query_id"],
            "component_id": gold["component_id"],
            "metrics": {"ret": ret, "sel": sel, "ans": ans, "cit": 0.0, "rar": 0.0},
            "failure_reason": "citation_scoring_failure",
        }
    precision = judge["precision"]
    recall = judge["recall"]
    if not isinstance(precision, int | float) or not isinstance(recall, int | float):
        raise ScorerContractError("MiniCheck precision/recall must be numeric")
    precision_value = float(precision)
    recall_value = float(recall)
    cit = citation_f1(precision_value, recall_value)
    rar = float(
        answer_exact_match(answer, gold["gold_answer_aliases"])
        and precision_value == 1.0
        and recall_value == 1.0
    )
    return {
        "query_id": gold["query_id"],
        "component_id": gold["component_id"],
        "metrics": {"ret": ret, "sel": sel, "ans": ans, "cit": cit, "rar": rar},
        "failure_reason": None,
    }


def score_bundle(
    sidecar_records: Iterable[Mapping[str, Any]],
    outcomes: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score on the sidecar's ordered query set, preserving the common denominator."""

    gold_rows = list(sidecar_records)
    if not gold_rows:
        raise ScorerContractError("cannot score an empty sidecar")
    gold_ids = [str(row.get("query_id", "")) for row in gold_rows]
    if len(gold_ids) != len(set(gold_ids)):
        raise ScorerContractError("sidecar contains duplicate query_id values")

    by_id: dict[str, Mapping[str, Any]] = {}
    for outcome in outcomes:
        query_id = outcome.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise ScorerContractError("outcome has no query_id")
        if query_id in by_id:
            raise ScorerContractError(f"duplicate outcome query_id: {query_id}")
        by_id[query_id] = outcome
    unexpected = sorted(set(by_id) - set(gold_ids))
    if unexpected:
        raise ScorerContractError(f"outcomes contain IDs outside the frozen denominator: {len(unexpected)}")

    per_query = [score_query(gold, by_id.get(str(gold["query_id"]))) for gold in gold_rows]
    n_queries = len(per_query)
    aggregate = {
        key: sum(float(row["metrics"][key]) for row in per_query) / n_queries
        for key in METRIC_KEYS
    }
    failures = Counter(
        str(row["failure_reason"])
        for row in per_query
        if row["failure_reason"] is not None
    )
    return {
        "n_queries": n_queries,
        "denominators": {key: n_queries for key in METRIC_KEYS},
        "aggregate": aggregate,
        "failure_counts": dict(sorted(failures.items())),
        "per_query": per_query,
    }
