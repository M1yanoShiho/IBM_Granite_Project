"""Apply the G218 systematic target repair.

G218 consumes the G216 pre-sample bundle and rewrites 2Wiki answerable targets
with deterministic entity anchoring.  It also filters narrowly detectable NIAH
QA2D semantic mismatches exposed by fixed sample review.

This stage writes a new PRE_AUDIT bundle.  Because 2Wiki target text can change,
structural and TRUE audit must be rerun before any freeze or training step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from full_flow_g200_v2 import SCHEMA_CASE, UNKNOWN_ANSWER, _sha256_text, _target_chain

SCHEMA_G216_MANIFEST = "full-flow-g216-sample-review-repair-manifest-v1"
SCHEMA_G218_MANIFEST = "full-flow-g218-target-repair-manifest-v1"
TRAIN_ROLE = "train-fit"
VALIDATION_ROLE = "train-modelval"
TWOWIKI_TARGET_CONSTRUCTION = "title_anchored_support_sentence_v2"
NIAH_TARGET_CONSTRUCTION = "qa2d_single_claim_reuse_or_frozen_qa2d_g218_filtered"

LEADING_PRONOUN = re.compile(
    r"^(?P<lead>he|she|it|they|this|that|these|those|his|her|its|their)\b(?P<rest>.*)$",
    re.IGNORECASE,
)
LEADING_GENERIC = re.compile(
    r"^(?P<lead>the film|the movie|the village|the stream|the river|the mountain|"
    r"the album|the song|the series|the town|the city|the company|the school|"
    r"the club|the station|the airport|the book|the novel|the county|the district)"
    r"\b(?P<rest>.*)$",
    re.IGNORECASE,
)
AFTER_RELEASE_IT = re.compile(r"^after release it\b(?P<rest>.*)$", re.IGNORECASE)
LEADING_BORN = re.compile(r"^born\b(?P<rest>.*)$", re.IGNORECASE)
TOKEN = re.compile(r"\w+")
POSSESSIVE_PRONOUNS = {"his", "her", "its", "their"}


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_empty(path: Path, label: str) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"{label} output directory must be absent or empty")


def _normalise(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(TOKEN.findall(value))


def _contains_normalised(needle: str, haystack: str) -> bool:
    normalized_needle = _normalise(needle)
    normalized_haystack = _normalise(haystack)
    return bool(normalized_needle) and f" {normalized_needle} " in f" {normalized_haystack} "


def _token_spans(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0).casefold(), match.start(), match.end()) for match in TOKEN.finditer(text)]


def _suffix_token_sequences(value: str) -> list[list[str]]:
    tokens = [token for token, _start, _end in _token_spans(value)]
    output: list[list[str]] = []
    for width in range(min(5, len(tokens)), 0, -1):
        suffix = tokens[-width:]
        if len(suffix) == 1 and (len(suffix[0]) < 3 or suffix[0] in {"the", "and"}):
            continue
        output.append(suffix)
    return output


def _replace_leading_token_sequence(sentence: str, replacement: str, prefixes: Sequence[Sequence[str]]) -> str | None:
    spans = _token_spans(sentence)
    if not spans:
        return None
    sentence_tokens = [token for token, _start, _end in spans]
    for prefix in prefixes:
        width = len(prefix)
        if width and sentence_tokens[:width] == list(prefix):
            end = spans[width - 1][2]
            return f"{replacement}{sentence[end:]}"
    return None


def _clean_anchor(value: str, fallback: str) -> str:
    anchor = " ".join(str(value).strip().split())
    if anchor:
        return anchor
    fallback = " ".join(str(fallback).strip().split())
    if not fallback:
        raise ValueError("blank 2Wiki anchoring subject and title")
    return fallback


def _official_evidence(fact: Mapping[str, Any]) -> tuple[str, str, str] | None:
    raw = fact.get("official_evidence")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 3:
        return None
    subject = str(raw[0]).strip()
    relation = str(raw[1]).strip()
    obj = str(raw[2]).strip()
    if not subject or not relation or not obj:
        return None
    return subject, relation, obj


def _anchor_twowiki_sentence(sentence: str, fact: Mapping[str, Any]) -> tuple[str, str]:
    text = " ".join(sentence.strip().split())
    evidence = _official_evidence(fact)
    title = str(fact.get("title", "")).strip()
    if not text or evidence is None:
        return text, "unchanged_invalid_fact"
    subject, _relation, _obj = evidence
    anchor = _clean_anchor(subject, title)
    if _contains_normalised(subject, text) or (title and _contains_normalised(title, text)):
        return text, "unchanged_already_anchored"

    after_release = AFTER_RELEASE_IT.match(text)
    if after_release is not None:
        return f"After release {anchor}{after_release.group('rest')}", "replace_after_release_it"

    born = LEADING_BORN.match(text)
    if born is not None:
        return f"{anchor} was born{born.group('rest')}", "prepend_subject_to_born"

    pronoun = LEADING_PRONOUN.match(text)
    if pronoun is not None:
        lead = pronoun.group("lead").casefold()
        rest = pronoun.group("rest")
        replacement = f"{anchor}'s" if lead in POSSESSIVE_PRONOUNS else anchor
        return f"{replacement}{rest}", "replace_leading_pronoun"

    generic = LEADING_GENERIC.match(text)
    if generic is not None:
        return f"{anchor}{generic.group('rest')}", "replace_leading_generic_noun"

    prefixes = [*_suffix_token_sequences(subject), *(_suffix_token_sequences(title) if title else [])]
    replaced = _replace_leading_token_sequence(text, anchor, prefixes)
    if replaced is not None and replaced != text:
        return replaced, "replace_leading_suffix_anchor"

    return f"{anchor}: {text}", "prefix_anchor_label"


def _context_for_variant(variant: Mapping[str, Any]) -> list[dict[str, str]]:
    evidence_ids = variant.get("evidence_ids")
    if not isinstance(evidence_ids, list):
        raise ValueError("variant lacks evidence_ids list")
    return [{"evidence_id": str(item)} for item in evidence_ids]


def _rebuild_variant_targets(row: dict[str, Any], semantic_sentences: Sequence[str]) -> None:
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
        anchored, mode = _anchor_twowiki_sentence(str(sentence), fact)
        counts[f"twowiki_sentence_mode::{mode}"] += 1
        changed = changed or anchored != str(sentence).strip()
        copied_fact = dict(fact)
        copied_fact["g218_repair_mode"] = mode
        copied_fact["g218_semantic_sentence"] = anchored
        new_facts.append(copied_fact)
        new_sentences.append(anchored)
    semantic_target = " ".join(new_sentences)
    repaired["semantic_sentences"] = new_sentences
    repaired["semantic_target"] = semantic_target
    repaired["semantic_target_sha256"] = _sha256_text(semantic_target)
    repaired["target_construction"] = TWOWIKI_TARGET_CONSTRUCTION
    repaired["official_supporting_facts"] = new_facts
    _rebuild_variant_targets(repaired, new_sentences)
    counts["twowiki_cases_changed" if changed else "twowiki_cases_unchanged"] += 1
    return repaired, counts


def _looks_like_year(answer: str) -> bool:
    return re.fullmatch(r"\d{3,4}", _normalise(answer)) is not None


def _niah_filter_reasons(row: Mapping[str, Any]) -> list[str]:
    if row.get("dataset") != "niah" or not row.get("answerable"):
        return []
    if row.get("source") != "G200-v2-new-parent-disjoint-modelval":
        return []
    question = str(row.get("question", ""))
    answer = str(row.get("answer", ""))
    target = str(row.get("semantic_target", ""))
    normalized_question = _normalise(question)
    target_tokens = set(_normalise(target).split())
    reasons: list[str] = []
    padded_question = f" {normalized_question} "
    if any(
        marker in padded_question
        for marker in (" and what ", " and who ", " and where ", " and when ")
    ):
        reasons.append("multi_slot_question")
    if normalized_question.startswith("who ") and _looks_like_year(answer):
        reasons.append("who_question_numeric_answer")
    question_tokens = normalized_question.split()
    if (
        len(question_tokens) >= 5
        and question_tokens[0] == "who"
        and question_tokens[1] in {"played", "plays"}
        and "on" in question_tokens
    ):
        on_index = max(index for index, token in enumerate(question_tokens) if token == "on")
        title_tokens = [
            token for token in question_tokens[on_index + 1 :] if token not in {"the", "a", "an"}
        ]
        if len(title_tokens) >= 3 and any(token not in target_tokens for token in title_tokens):
            reasons.append("acted_on_title_truncation")
    return reasons


def _case_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    seen: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        dataset = str(row.get("dataset", ""))
        role = str(row.get("role", ""))
        kind = "answerable" if row.get("answerable") else "unsupported"
        seen[(dataset, role, kind)].add(str(row.get("group_id", "")))
    for (dataset, role, kind), groups in seen.items():
        output.setdefault(dataset, {})[f"{role}_{kind}_groups"] = len(groups)
    return output


def _example_updates(rows: Sequence[Mapping[str, Any]]) -> int:
    total = 0
    for row in rows:
        variants = row.get("variants")
        if not isinstance(variants, Mapping):
            raise ValueError(f"case lacks variants: {row.get('case_id')}")
        total += len(variants)
    return total


def _split_leakage(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    train_groups = {(str(row.get("dataset")), str(row.get("group_id"))) for row in train_rows}
    validation_groups = {
        (str(row.get("dataset")), str(row.get("group_id"))) for row in validation_rows
    }
    train_components = {
        (str(row.get("dataset")), str(row.get("component_id"))) for row in train_rows
    }
    validation_components = {
        (str(row.get("dataset")), str(row.get("component_id"))) for row in validation_rows
    }
    return {
        "group_overlap": len(train_groups & validation_groups),
        "component_overlap": len(train_components & validation_components),
    }


def _load_bundle(
    *,
    manifest_path: Path,
    train_cases_path: Path,
    validation_cases_path: Path,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    manifest = _json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_G216_MANIFEST:
        raise ValueError("G218 expects the G216 repair manifest")
    if manifest.get("status") != "PRE_MANUAL_PASS":
        raise ValueError("G218 requires a PRE_MANUAL_PASS input bundle")
    if manifest.get("training_started") is not False:
        raise ValueError("G218 input must record training_started=false")
    if manifest.get("sealed_or_heldout_read") is not False or manifest.get("dev_read") is not False:
        raise ValueError("G218 input must not have read dev/sealed/held-out data")
    if str(manifest.get("train_cases_sha256", "")) != _sha256(train_cases_path):
        raise ValueError("train cases hash differs from G216 manifest")
    if str(manifest.get("validation_cases_sha256", "")) != _sha256(validation_cases_path):
        raise ValueError("validation cases hash differs from G216 manifest")
    train_rows = _jsonl(train_cases_path)
    validation_rows = _jsonl(validation_cases_path)
    seen: set[str] = set()
    for row in [*train_rows, *validation_rows]:
        case_id = str(row.get("case_id", ""))
        if not case_id or case_id in seen or row.get("schema_version") != SCHEMA_CASE:
            raise ValueError(f"invalid or duplicate case: {case_id!r}")
        seen.add(case_id)
    return manifest, train_rows, validation_rows


def _review_inputs(paths: Sequence[Path]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for path in paths:
        rows = _jsonl(path)
        decisions = Counter(str(row.get("review_decision", "")) for row in rows)
        output.append(
            {
                "path": str(path),
                "rows": len(rows),
                "decisions": dict(decisions),
                "sha256": _sha256(path),
            }
        )
    return output


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
    _require_empty(output_dir, "G218 target repair")
    input_manifest, train_rows, validation_rows = _load_bundle(
        manifest_path=manifest_path,
        train_cases_path=train_cases_path,
        validation_cases_path=validation_cases_path,
    )
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
            niah_reasons = _niah_filter_reasons(row)
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

    removed_train_group_ids = {
        str(item["case_id"]).removeprefix("2wiki::")
        for item in removed_cases
        if item.get("split") == "train" and item.get("dataset") == "2wiki"
    }
    if removed_train_group_ids:
        repaired_train = [
            row
            for row in repaired_train
            if not (
                row.get("dataset") == "2wiki"
                and not row.get("answerable")
                and str(row.get("group_id", "")) in removed_train_group_ids
            )
        ]

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
            "schema_version": "full-flow-g218-ordered-ids-v1",
            "train_case_ids": [str(row["case_id"]) for row in repaired_train],
            "validation_case_ids": [str(row["case_id"]) for row in repaired_validation],
        },
    )
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_G218_MANIFEST,
        "status": "PRE_AUDIT" if all(gates.values()) else "FAIL",
        "repair_stage": "G218",
        "repair_reason": "systematic_target_self_containment_and_qa2d_filtering",
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
