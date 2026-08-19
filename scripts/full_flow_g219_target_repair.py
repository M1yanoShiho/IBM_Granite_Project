"""Apply the G219 controlled target repair.

G219 consumes the G218 pre-sample bundle and the G212M3 sample review rows.
It repairs only the failure classes exposed by G212M3:

* 2Wiki answerable targets whose support sentence is anchored but does not state
  the relation needed by the question.
* NIAH QA2D targets with deterministic title/entity truncation or malformed
  word order.

This stage writes a new PRE_AUDIT bundle.  It does not train, generate utility
labels, read held-out/dev data, or change Retriever/TRUE thresholds.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from full_flow_g200_v2 import SCHEMA_CASE, _sha256_text
from full_flow_g218_target_repair import (
    TRAIN_ROLE,
    VALIDATION_ROLE,
    _anchor_twowiki_sentence,
    _case_counts,
    _context_for_variant,
    _example_updates,
    _json,
    _jsonl,
    _normalise,
    _require_empty,
    _review_inputs,
    _sha256,
    _split_leakage,
    _write_json,
    _write_jsonl,
)
from full_flow_g212m_sample_adjudication import FAIL

SCHEMA_INPUT_FINAL_MANIFEST = "full-flow-g210-v2-final-manifest-v1"
SCHEMA_G219_MANIFEST = "full-flow-g219-target-repair-manifest-v1"
SCHEMA_ORDERED_IDS = "full-flow-g219-ordered-ids-v1"
TWOWIKI_TARGET_CONSTRUCTION = "relation_explicit_answer_target_v3"
NIAH_TARGET_CONSTRUCTION = "qa2d_single_claim_reuse_or_g219_filtered"

RELATION_EXPLICIT_REWRITE = {
    "country of origin",
    "country of citizenship",
    "place of birth",
}
US_ALIASES = {"u s", "u.s.", "us", "usa", "u.s", "united states", "american"}
COUNTRY_ADJECTIVES = {
    "american",
    "australian",
    "belgian",
    "bosnian",
    "british",
    "cambodian",
    "canadian",
    "chinese",
    "czech",
    "danish",
    "english",
    "finnish",
    "french",
    "german",
    "greek",
    "hungarian",
    "indian",
    "iranian",
    "irish",
    "italian",
    "japanese",
    "latvian",
    "moldovan",
    "norwegian",
    "pakistani",
    "polish",
    "romanian",
    "russian",
    "scottish",
    "slovak",
    "spanish",
    "swedish",
    "syrian",
    "thai",
    "vietnamese",
    "yemeni",
}
WHAT_SEASON_HAS = re.compile(r"^what season of (?P<title>.+?) has (?P<entity>.+)$")


def _clean_text(value: object) -> str:
    return " ".join(str(value).strip().split())


def _official_evidence(fact: Mapping[str, Any]) -> tuple[str, str, str] | None:
    raw = fact.get("official_evidence")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 3:
        return None
    subject = _clean_text(raw[0])
    relation = _clean_text(raw[1]).casefold()
    obj = _clean_text(raw[2])
    if not subject or not relation or not obj:
        return None
    return subject, relation, obj


def _is_country_adjective(value: str) -> bool:
    normalized = _normalise(value)
    if normalized in US_ALIASES:
        return True
    return normalized in COUNTRY_ADJECTIVES


def _country_phrase(value: str) -> str:
    normalized = _normalise(value)
    if normalized in US_ALIASES:
        return "the United States"
    return value


def _relation_explicit_sentence(subject: str, relation: str, obj: str) -> str | None:
    if relation == "place of birth":
        return f"{subject} was born in {obj}."
    if relation == "country of citizenship":
        if _is_country_adjective(obj) and _normalise(obj) not in {"u s", "u s"}:
            return f"{subject} is {obj}."
        return f"{subject} is from {_country_phrase(obj)}."
    if relation == "country of origin":
        if _is_country_adjective(obj) and _normalise(obj) not in {"u s", "u s"}:
            return f"{subject} is {obj}."
        return f"{subject} is from {_country_phrase(obj)}."
    return None


def _g219_twowiki_sentence(sentence: object, fact: Mapping[str, Any]) -> tuple[str, str]:
    evidence = _official_evidence(fact)
    if evidence is not None:
        subject, relation, obj = evidence
        if relation in RELATION_EXPLICIT_REWRITE:
            rewritten = _relation_explicit_sentence(subject, relation, obj)
            if rewritten is not None:
                return rewritten, f"relation_explicit::{relation.replace(' ', '_')}"
    anchored, mode = _anchor_twowiki_sentence(str(sentence), fact)
    return anchored, f"g218_anchor::{mode}"


def _rebuild_variant_targets(row: dict[str, Any], semantic_sentences: Sequence[str]) -> None:
    from full_flow_g200_v2 import _target_chain

    variants = row.get("variants")
    if not isinstance(variants, Mapping):
        raise ValueError(f"case lacks variants: {row.get('case_id')}")
    for variant in variants.values():
        if not isinstance(variant, dict):
            raise ValueError(f"invalid variant for {row.get('case_id')}")
        support_ids = [str(item) for item in variant.get("support_evidence_ids", [])]
        variant["target"] = _target_chain(
            semantic_sentences,
            support_ids,
            _context_for_variant(variant),
        )


def _repair_twowiki_case(row: Mapping[str, Any]) -> tuple[dict[str, Any] | None, Counter[str]]:
    repaired = json.loads(json.dumps(row, ensure_ascii=False))
    counts: Counter[str] = Counter()
    facts = repaired.get("official_supporting_facts")
    sentences = repaired.get("semantic_sentences")
    if not isinstance(facts, list) or not isinstance(sentences, list) or len(facts) != len(sentences):
        counts["filtered_invalid_twowiki_metadata"] += 1
        return None, counts
    new_sentences: list[str] = []
    new_facts: list[dict[str, Any]] = []
    changed = False
    for fact, sentence in zip(facts, sentences, strict=True):
        if not isinstance(fact, Mapping):
            counts["filtered_invalid_twowiki_metadata"] += 1
            return None, counts
        rewritten, mode = _g219_twowiki_sentence(sentence, fact)
        counts[f"twowiki_sentence_mode::{mode}"] += 1
        changed = changed or rewritten != _clean_text(sentence)
        copied_fact = dict(fact)
        copied_fact["g219_repair_mode"] = mode
        copied_fact["g219_semantic_sentence"] = rewritten
        new_facts.append(copied_fact)
        new_sentences.append(rewritten)
    semantic_target = " ".join(new_sentences)
    repaired["semantic_sentences"] = new_sentences
    repaired["semantic_target"] = semantic_target
    repaired["semantic_target_sha256"] = _sha256_text(semantic_target)
    repaired["target_construction"] = TWOWIKI_TARGET_CONSTRUCTION
    repaired["official_supporting_facts"] = new_facts
    _rebuild_variant_targets(repaired, new_sentences)
    counts["twowiki_cases_changed" if changed else "twowiki_cases_unchanged"] += 1
    return repaired, counts


def _review_failures(paths: Sequence[Path]) -> dict[str, list[str]]:
    failures: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        for row in _jsonl(path):
            if str(row.get("review_decision", "")) != FAIL:
                continue
            case_id = str(row.get("case_id", ""))
            dataset = str(row.get("dataset", ""))
            reason = str(row.get("review_reason", "")).strip()
            if case_id and dataset == "niah":
                failures[case_id].append(reason or "g212m3_failed_niah_row")
    return failures


def _contains_phrase(haystack: str, phrase: str) -> bool:
    normalized_phrase = _normalise(phrase)
    normalized_haystack = _normalise(haystack)
    return bool(normalized_phrase) and f" {normalized_phrase} " in f" {normalized_haystack} "


def _niah_filter_reasons(row: Mapping[str, Any], review_failures: Mapping[str, Sequence[str]]) -> list[str]:
    if row.get("dataset") != "niah" or not row.get("answerable"):
        return []
    case_id = str(row.get("case_id", ""))
    question = str(row.get("question", ""))
    target = str(row.get("semantic_target", ""))
    normalized_question = _normalise(question)
    normalized_target = _normalise(target)
    reasons: list[str] = []

    if case_id in review_failures:
        reasons.append("g212m3_failed_niah_qa2d_malformation")
    if "doctor who" in normalized_question and "doctor who" not in normalized_target:
        reasons.append("acted_on_title_truncation")
    season_match = WHAT_SEASON_HAS.match(normalized_question)
    if season_match is not None:
        entity = season_match.group("entity")
        if entity and not _contains_phrase(target, entity):
            reasons.append("has_entity_phrase_lost")
    if " used to define " in f" {normalized_question} " and "define for" in normalized_target:
        reasons.append("qa2d_define_for_malformed")
    if normalized_question.startswith("meatloaf duet ") and normalized_target.startswith("meatloaf duet "):
        reasons.append("qa2d_duet_malformed")
    return sorted(set(reasons))


def _load_bundle(
    *,
    manifest_path: Path,
    train_cases_path: Path,
    validation_cases_path: Path,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    manifest = _json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_INPUT_FINAL_MANIFEST:
        raise ValueError("G219 expects a finalized pre-sample manifest")
    if manifest.get("status") != "PRE_MANUAL_PASS":
        raise ValueError("G219 requires a PRE_MANUAL_PASS input bundle")
    if manifest.get("training_started") is not False:
        raise ValueError("G219 input must record training_started=false")
    if manifest.get("sealed_or_heldout_read") is not False or manifest.get("dev_read") is not False:
        raise ValueError("G219 input must not have read dev/sealed/held-out data")
    if str(manifest.get("train_cases_sha256", "")) != _sha256(train_cases_path):
        raise ValueError("train cases hash differs from source manifest")
    if str(manifest.get("validation_cases_sha256", "")) != _sha256(validation_cases_path):
        raise ValueError("validation cases hash differs from source manifest")
    train_rows = _jsonl(train_cases_path)
    validation_rows = _jsonl(validation_cases_path)
    seen: set[str] = set()
    for row in [*train_rows, *validation_rows]:
        case_id = str(row.get("case_id", ""))
        if not case_id or case_id in seen or row.get("schema_version") != SCHEMA_CASE:
            raise ValueError(f"invalid or duplicate case: {case_id!r}")
        seen.add(case_id)
    return manifest, train_rows, validation_rows


def repair(
    *,
    manifest_path: Path,
    train_cases_path: Path,
    validation_cases_path: Path,
    review_rows_paths: Sequence[Path],
    output_dir: Path,
    min_niah_train_groups: int = 400,
    min_niah_modelval_groups: int = 100,
    min_twowiki_train_groups: int = 400,
    min_twowiki_modelval_groups: int = 100,
    unsupported_ratio_min: float = 0.10,
    unsupported_ratio_max: float = 0.15,
) -> dict[str, object]:
    _require_empty(output_dir, "G219 target repair")
    input_manifest, train_rows, validation_rows = _load_bundle(
        manifest_path=manifest_path,
        train_cases_path=train_cases_path,
        validation_cases_path=validation_cases_path,
    )
    review_failures = _review_failures(review_rows_paths)
    repaired_train: list[Mapping[str, Any]] = []
    repaired_validation: list[Mapping[str, Any]] = []
    removed_cases: list[dict[str, object]] = []
    repair_counts: Counter[str] = Counter()

    for split, source_rows, output_rows in (
        ("train", train_rows, repaired_train),
        ("validation", validation_rows, repaired_validation),
    ):
        for row in source_rows:
            case_id = str(row.get("case_id", ""))
            if row.get("dataset") == "2wiki" and row.get("answerable"):
                repaired, counts = _repair_twowiki_case(row)
                repair_counts.update(counts)
                if repaired is None:
                    removed_cases.append(
                        {
                            "case_id": case_id,
                            "split": split,
                            "dataset": "2wiki",
                            "role": str(row.get("role", "")),
                            "reason": "invalid_twowiki_metadata",
                        }
                    )
                    continue
                output_rows.append(repaired)
                continue
            niah_reasons = _niah_filter_reasons(row, review_failures)
            if niah_reasons:
                repair_counts["niah_cases_filtered"] += 1
                for reason in niah_reasons:
                    repair_counts[f"niah_filter_reason::{reason}"] += 1
                removed_cases.append(
                    {
                        "case_id": case_id,
                        "split": split,
                        "dataset": "niah",
                        "role": str(row.get("role", "")),
                        "reason": ",".join(niah_reasons),
                    }
                )
                continue
            output_rows.append(json.loads(json.dumps(row, ensure_ascii=False)))

    answerable_train = [row for row in repaired_train if row.get("answerable")]
    unsupported_train = [row for row in repaired_train if not row.get("answerable")]
    answerable_updates = _example_updates(answerable_train)
    unsupported_updates = _example_updates(unsupported_train)
    unsupported_ratio = unsupported_updates / (answerable_updates + unsupported_updates)
    leakage = _split_leakage(repaired_train, repaired_validation)
    counts = _case_counts([*repaired_train, *repaired_validation])

    niah_train = counts.get("niah", {}).get(f"{TRAIN_ROLE}_answerable_groups", 0)
    niah_validation = counts.get("niah", {}).get(f"{VALIDATION_ROLE}_answerable_groups", 0)
    twiki_train = counts.get("2wiki", {}).get(f"{TRAIN_ROLE}_answerable_groups", 0)
    twiki_validation = counts.get("2wiki", {}).get(f"{VALIDATION_ROLE}_answerable_groups", 0)
    gates = {
        "niah_train_groups": niah_train >= min_niah_train_groups,
        "niah_new_modelval_groups": niah_validation >= min_niah_modelval_groups,
        "twowiki_train_groups": twiki_train >= min_twowiki_train_groups,
        "twowiki_modelval_groups": twiki_validation >= min_twowiki_modelval_groups,
        "unsupported_update_ratio": unsupported_ratio_min <= unsupported_ratio <= unsupported_ratio_max,
        "split_group_overlap_zero": leakage["group_overlap"] == 0,
        "split_component_overlap_zero": leakage["component_overlap"] == 0,
    }

    train_path = output_dir / "train_cases.jsonl"
    validation_path = output_dir / "validation_cases.jsonl"
    ordered_path = output_dir / "ordered_ids.json"
    _write_jsonl(train_path, repaired_train)
    _write_jsonl(validation_path, repaired_validation)
    _write_json(
        ordered_path,
        {
            "schema_version": SCHEMA_ORDERED_IDS,
            "train_case_ids": [str(row["case_id"]) for row in repaired_train],
            "validation_case_ids": [str(row["case_id"]) for row in repaired_validation],
        },
    )
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_G219_MANIFEST,
        "status": "PRE_AUDIT" if all(gates.values()) else "FAIL",
        "repair_stage": "G219",
        "repair_reason": "controlled_relation_explicit_targets_and_qa2d_malformation_filtering",
        "requires_structural_true_rerun": True,
        "sample_review_required_before_training": True,
        "training_started": False,
        "utility_labels_started": False,
        "sealed_or_heldout_read": False,
        "dev_read": False,
        "gold_use": "offline target repair only; runtime prompts contain evidence text and question only",
        "source_manifest_schema_version": str(input_manifest.get("schema_version", "")),
        "target_construction": {
            "2wiki": TWOWIKI_TARGET_CONSTRUCTION,
            "niah": NIAH_TARGET_CONSTRUCTION,
        },
        "review_failure_case_ids": sorted(review_failures),
        "repair_counts": dict(repair_counts),
        "removed_cases": removed_cases,
        "removed_case_count": len(removed_cases),
        "review_inputs": _review_inputs(review_rows_paths),
        "counts": counts,
        "answerable_train_updates": answerable_updates,
        "unsupported_updates": unsupported_updates,
        "unsupported_update_ratio": unsupported_ratio,
        "split_leakage": leakage,
        "gates": gates,
        "train_cases": len(repaired_train),
        "validation_cases": len(repaired_validation),
        "train_cases_sha256": _sha256(train_path),
        "validation_cases_sha256": _sha256(validation_path),
        "ordered_ids_sha256": _sha256(ordered_path),
        "input_sha256": {
            "source_manifest": _sha256(manifest_path),
            "source_train_cases": _sha256(train_cases_path),
            "source_validation_cases": _sha256(validation_cases_path),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--train-cases", required=True, type=Path)
    parser.add_argument("--validation-cases", required=True, type=Path)
    parser.add_argument("--review-rows", action="append", default=[], type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = repair(
        manifest_path=args.manifest,
        train_cases_path=args.train_cases,
        validation_cases_path=args.validation_cases,
        review_rows_paths=args.review_rows,
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
