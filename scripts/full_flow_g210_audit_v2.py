"""Audit and freeze the G200 v2 generator data bundle.

G210 is deliberately split into explicit commands:

* ``structural`` validates hashes, split isolation, answer-alias policy,
  citation remaps, unsupported support removal, and emits the TRUE worklist.
* ``true-audit`` runs the frozen TRUE judge over that worklist.
* ``finalize`` filters cases using structural + TRUE results and writes a
  frozen pre-manual bundle.  Manual audit must still pass before G300 training.

No command trains a model or reads sealed/system-heldout data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from full_flow_g200_v2 import SCHEMA_CASE, UNKNOWN_ANSWER, _triple_sentence

from evidence_rag.generator.nli import DEFAULT_BINARY_ENTAIL_THRESHOLD, TrueNLIModel

SCHEMA_STRUCTURAL_SUMMARY = "full-flow-g210-v2-structural-summary-v1"
SCHEMA_STRUCTURAL_ROW = "full-flow-g210-v2-structural-row-v1"
SCHEMA_TRUE_WORKLIST_ROW = "full-flow-g210-v2-true-worklist-row-v1"
SCHEMA_TRUE_AUDIT_ROW = "full-flow-g210-v2-true-audit-row-v1"
SCHEMA_TRUE_AUDIT_MANIFEST = "full-flow-g210-v2-true-audit-manifest-v1"
SCHEMA_FINAL_MANIFEST = "full-flow-g210-v2-final-manifest-v1"
TRAIN_ROLE = "train-fit"
VALIDATION_ROLE = "train-modelval"
EVIDENCE_LINE = re.compile(r"^\[(\d+)\]\s+\(([^)]+)\)\s+(.*)$")
CITATION = re.compile(r"\[(\d+)\]")
YES_NO = frozenset({"yes", "no"})


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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_empty(path: Path, label: str) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"{label} output directory must be absent or empty")


def _normalise(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", value))


def _contains_normalised(needle: str, haystack: str) -> bool:
    normalized_needle = _normalise(needle)
    normalized_haystack = _normalise(haystack)
    return bool(normalized_needle) and f" {normalized_needle} " in f" {normalized_haystack} "


def _load_cases(train_cases_path: Path, validation_cases_path: Path) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    train = _jsonl(train_cases_path)
    validation = _jsonl(validation_cases_path)
    seen: set[str] = set()
    for row in [*train, *validation]:
        case_id = str(row.get("case_id", ""))
        if not case_id or case_id in seen or row.get("schema_version") != SCHEMA_CASE:
            raise ValueError(f"invalid or duplicate G200 v2 case: {case_id!r}")
        seen.add(case_id)
    return train, validation


def _evidence_text_by_id(variant: Mapping[str, Any]) -> dict[str, str]:
    prompt = str(variant.get("prompt", ""))
    evidence: dict[str, str] = {}
    for line in prompt.splitlines():
        match = EVIDENCE_LINE.match(line)
        if match is None:
            continue
        evidence_id = match.group(2)
        evidence[evidence_id] = match.group(3)
    return evidence


def _citation_indices(target: str) -> list[int]:
    return [int(item) for item in CITATION.findall(target)]


def _variant_citation_remap_ok(row: Mapping[str, Any]) -> tuple[bool, int, str | None]:
    variants = row.get("variants")
    if not isinstance(variants, Mapping) or not variants:
        return False, 0, "missing variants"
    checked = 0
    for name, variant in variants.items():
        if not isinstance(variant, Mapping):
            return False, checked, f"{name} is not an object"
        evidence_ids = [str(item) for item in variant.get("evidence_ids", [])]
        support_ids = {str(item) for item in variant.get("support_evidence_ids", [])}
        target = str(variant.get("target", ""))
        citations = _citation_indices(target)
        checked += 1
        if bool(row.get("answerable")):
            if not citations:
                return False, checked, f"{name} lacks citations"
            for citation in citations:
                if citation < 1 or citation > len(evidence_ids):
                    return False, checked, f"{name} citation {citation} is out of range"
                if evidence_ids[citation - 1] not in support_ids:
                    return False, checked, f"{name} citation {citation} does not map to support"
        else:
            removed = {str(item) for item in row.get("removed_support_evidence_ids", [])}
            if citations:
                return False, checked, f"{name} unsupported target contains citations"
            if target.strip() != UNKNOWN_ANSWER:
                return False, checked, f"{name} unsupported target is not fixed"
            if support_ids:
                return False, checked, f"{name} unsupported variant has support ids"
            if not removed:
                return False, checked, f"{name} unsupported row lacks removed support"
            if removed & set(evidence_ids):
                return False, checked, f"{name} removed support remains visible"
    return True, checked, None


def _answer_alias(row: Mapping[str, Any]) -> tuple[bool, str]:
    if not bool(row.get("answerable")):
        return True, "unsupported_fixed_unknown"
    answer = str(row.get("answer", "")).strip()
    semantic_target = str(row.get("semantic_target", "")).strip()
    dataset = str(row.get("dataset", ""))
    target_kind = str(row.get("target_kind", ""))
    if not answer or not semantic_target:
        return False, "blank_answer_or_target"
    normalized_answer = _normalise(answer)
    if dataset == "2wiki" and normalized_answer in YES_NO:
        evidences = row.get("official_evidences")
        if target_kind == "twowiki_evidence_chain" and isinstance(evidences, list) and len(evidences) >= 2:
            return True, "twowiki_yes_no_evidence_chain"
        return False, "twowiki_yes_no_without_chain"
    if _contains_normalised(answer, semantic_target):
        return True, "literal_answer_alias"
    return False, "literal_answer_missing"


def _support_only_variant(row: Mapping[str, Any]) -> Mapping[str, Any]:
    variants = row.get("variants")
    if not isinstance(variants, Mapping):
        raise ValueError(f"case lacks variants: {row.get('case_id')}")
    support_only = variants.get("support_only")
    if not isinstance(support_only, Mapping):
        raise ValueError(f"case lacks support_only variant: {row.get('case_id')}")
    return support_only


def _true_pairs(row: Mapping[str, Any]) -> list[dict[str, object]]:
    if not bool(row.get("answerable")):
        return []
    if str(row.get("source", "")) == "G200-v1-reviewed-train-fit-reuse":
        return []
    support_only = _support_only_variant(row)
    evidence_text = _evidence_text_by_id(support_only)
    support_ids = [str(item) for item in support_only.get("support_evidence_ids", [])]
    case_id = str(row["case_id"])
    pairs: list[dict[str, object]] = []
    if str(row.get("target_kind", "")) == "niah_qa2d_single_claim":
        if len(support_ids) != 1:
            raise ValueError(f"NIAH target must bind one support evidence: {case_id}")
        hypotheses = [str(row.get("semantic_target", ""))]
    elif str(row.get("target_kind", "")) == "twowiki_evidence_chain":
        evidences = row.get("official_evidences")
        if not isinstance(evidences, list) or len(evidences) != len(support_ids):
            raise ValueError(f"2Wiki evidence/support count mismatch: {case_id}")
        semantic_sentences = row.get("semantic_sentences")
        if isinstance(semantic_sentences, list) and len(semantic_sentences) == len(support_ids):
            hypotheses = [str(item).strip() for item in semantic_sentences]
            if any(not item for item in hypotheses):
                raise ValueError(f"2Wiki semantic sentence is blank: {case_id}")
        else:
            hypotheses = [_triple_sentence(item) for item in evidences]
    else:
        raise ValueError(f"unknown answerable target kind: {row.get('target_kind')}")
    for index, (support_id, hypothesis) in enumerate(zip(support_ids, hypotheses, strict=True)):
        premise = evidence_text.get(support_id)
        if premise is None:
            raise ValueError(f"support evidence text missing for {case_id}/{support_id}")
        audit_id = _sha256_text(f"{case_id}\0{index}\0{support_id}\0{hypothesis}")[:24]
        pairs.append(
            {
                "schema_version": SCHEMA_TRUE_WORKLIST_ROW,
                "audit_id": audit_id,
                "case_id": case_id,
                "dataset": str(row.get("dataset", "")),
                "role": str(row.get("role", "")),
                "group_id": str(row.get("group_id", "")),
                "target_kind": str(row.get("target_kind", "")),
                "support_evidence_id": support_id,
                "hypothesis": hypothesis,
                "premise": premise,
                "pair_index": index,
            }
        )
    return pairs


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


def _split_leakage(train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
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


def structural_audit(
    *,
    data_manifest_path: Path,
    train_cases_path: Path,
    validation_cases_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    _require_empty(output_dir, "G210 structural audit")
    manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PRE_AUDIT":
        raise ValueError("G210 structural audit expects G200 PRE_AUDIT manifest")
    if manifest.get("train_cases_sha256") != _sha256(train_cases_path):
        raise ValueError("train_cases hash disagrees with G200 manifest")
    if manifest.get("validation_cases_sha256") != _sha256(validation_cases_path):
        raise ValueError("validation_cases hash disagrees with G200 manifest")
    train_rows, validation_rows = _load_cases(train_cases_path, validation_cases_path)
    leakage = _split_leakage(train_rows, validation_rows)
    structural_rows: list[dict[str, object]] = []
    worklist: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        for row in rows:
            case_id = str(row["case_id"])
            citation_ok, variants_checked, citation_error = _variant_citation_remap_ok(row)
            answer_ok, answer_mode = _answer_alias(row)
            true_reuse = (
                bool(row.get("answerable"))
                and str(row.get("source", "")) == "G200-v1-reviewed-train-fit-reuse"
            )
            pairs = _true_pairs(row)
            case_pass = citation_ok and answer_ok
            if bool(row.get("answerable")) and not true_reuse and not pairs:
                case_pass = False
                citation_error = citation_error or "answerable case has no TRUE pairs"
            row_counts_key = (
                f"{split}:{row.get('dataset')}:{row.get('target_kind')}:"
                f"{'pass' if case_pass else 'fail'}"
            )
            counts[row_counts_key] += 1
            true_ids = [str(item["audit_id"]) for item in pairs]
            structural_rows.append(
                {
                    "schema_version": SCHEMA_STRUCTURAL_ROW,
                    "case_id": case_id,
                    "split": split,
                    "dataset": str(row.get("dataset", "")),
                    "role": str(row.get("role", "")),
                    "group_id": str(row.get("group_id", "")),
                    "component_id": str(row.get("component_id", "")),
                    "answerable": bool(row.get("answerable")),
                    "target_kind": str(row.get("target_kind", "")),
                    "structural_pass": case_pass,
                    "citation_remap_pass": citation_ok,
                    "citation_error": citation_error,
                    "variants_checked": variants_checked,
                    "answer_alias_pass": answer_ok,
                    "answer_alias_mode": answer_mode,
                    "true_reuse": "G200-v1-target-audit" if true_reuse else None,
                    "true_audit_ids": true_ids,
                }
            )
            if case_pass:
                worklist.extend(pairs)
    structural_path = output_dir / "structural_audit_rows.jsonl"
    worklist_path = output_dir / "true_worklist.jsonl"
    _write_jsonl(structural_path, structural_rows)
    _write_jsonl(worklist_path, worklist)
    summary: dict[str, object] = {
        "schema_version": SCHEMA_STRUCTURAL_SUMMARY,
        "status": "PASS" if all(row["structural_pass"] for row in structural_rows) and not any(leakage.values()) else "FAIL",
        "training_started": False,
        "sealed_or_heldout_read": False,
        "dev_read": False,
        "input_sha256": {
            "data_manifest": _sha256(data_manifest_path),
            "train_cases": _sha256(train_cases_path),
            "validation_cases": _sha256(validation_cases_path),
        },
        "cases": len(structural_rows),
        "true_worklist_rows": len(worklist),
        "counts": dict(counts),
        "split_leakage": leakage,
        "structural_audit_rows_sha256": _sha256(structural_path),
        "true_worklist_sha256": _sha256(worklist_path),
    }
    _write_json(output_dir / "structural_summary.json", summary)
    return summary


TrueScorer = Callable[[str, str], float]


def true_audit(
    *,
    true_worklist_path: Path,
    true_snapshot: Path,
    output_dir: Path,
    scorer: TrueScorer | None = None,
) -> dict[str, object]:
    _require_empty(output_dir, "G210 TRUE audit")
    worklist = _jsonl(true_worklist_path)
    if scorer is None:
        judge = TrueNLIModel(model_id=str(true_snapshot.resolve()))
        scorer = judge.score
    rows: list[dict[str, object]] = []
    entailed = 0
    for index, row in enumerate(worklist, start=1):
        audit_id = str(row.get("audit_id", ""))
        premise = str(row.get("premise", ""))
        hypothesis = str(row.get("hypothesis", ""))
        score = float(scorer(premise, hypothesis))
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"invalid TRUE score for {audit_id}: {score}")
        passed = score >= DEFAULT_BINARY_ENTAIL_THRESHOLD
        entailed += int(passed)
        rows.append(
            {
                "schema_version": SCHEMA_TRUE_AUDIT_ROW,
                "audit_id": audit_id,
                "case_id": str(row.get("case_id", "")),
                "dataset": str(row.get("dataset", "")),
                "role": str(row.get("role", "")),
                "target_kind": str(row.get("target_kind", "")),
                "support_evidence_id": str(row.get("support_evidence_id", "")),
                "threshold": DEFAULT_BINARY_ENTAIL_THRESHOLD,
                "true_entailment_score": score,
                "entailed": passed,
            }
        )
        if index % 100 == 0 or index == len(worklist):
            print(f"[G210 TRUE] {index}/{len(worklist)}", flush=True)
    audit_path = output_dir / "true_audit_rows.jsonl"
    _write_jsonl(audit_path, rows)
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_TRUE_AUDIT_MANIFEST,
        "status": "COMPLETE",
        "judge": "google/t5_xxl_true_nli_mixture",
        "judge_snapshot": str(true_snapshot.resolve()),
        "threshold": DEFAULT_BINARY_ENTAIL_THRESHOLD,
        "worklist_rows": len(worklist),
        "entailed": entailed,
        "not_entailed": len(worklist) - entailed,
        "input_sha256": {"true_worklist": _sha256(true_worklist_path)},
        "true_audit_rows_sha256": _sha256(audit_path),
        "training_started": False,
        "sealed_or_heldout_read": False,
    }
    _write_json(output_dir / "true_audit_manifest.json", manifest)
    return manifest


def _load_structural_rows(path: Path) -> dict[str, Mapping[str, Any]]:
    rows = _jsonl(path)
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if case_id in output:
            raise ValueError(f"duplicate structural row: {case_id}")
        output[case_id] = row
    return output


def _load_true_rows(path: Path) -> dict[str, Mapping[str, Any]]:
    rows = _jsonl(path)
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        audit_id = str(row.get("audit_id", ""))
        if audit_id in output:
            raise ValueError(f"duplicate TRUE row: {audit_id}")
        output[audit_id] = row
    return output


def _case_true_pass(structural_row: Mapping[str, Any], true_rows: Mapping[str, Mapping[str, Any]]) -> bool:
    audit_ids = structural_row.get("true_audit_ids")
    if not isinstance(audit_ids, list):
        raise ValueError(f"structural row lacks true_audit_ids: {structural_row.get('case_id')}")
    if not audit_ids:
        return True
    for audit_id in audit_ids:
        true_row = true_rows.get(str(audit_id))
        if true_row is None or not bool(true_row.get("entailed")):
            return False
    return True


def finalize(
    *,
    train_cases_path: Path,
    validation_cases_path: Path,
    structural_rows_path: Path,
    true_audit_rows_path: Path,
    output_dir: Path,
    min_niah_train_groups: int = 400,
    min_niah_modelval_groups: int = 100,
    min_twowiki_train_groups: int = 400,
    min_twowiki_modelval_groups: int = 100,
    unsupported_ratio_min: float = 0.10,
    unsupported_ratio_max: float = 0.15,
) -> dict[str, object]:
    _require_empty(output_dir, "G210 finalize")
    train_rows, validation_rows = _load_cases(train_cases_path, validation_cases_path)
    structural = _load_structural_rows(structural_rows_path)
    true_rows = _load_true_rows(true_audit_rows_path)

    excluded: Counter[str] = Counter()

    def keep(row: Mapping[str, Any]) -> bool:
        case_id = str(row["case_id"])
        structural_row = structural.get(case_id)
        if structural_row is None:
            excluded["missing_structural"] += 1
            return False
        if not bool(structural_row.get("structural_pass")):
            excluded["structural_fail"] += 1
            return False
        if not _case_true_pass(structural_row, true_rows):
            excluded["true_fail"] += 1
            return False
        return True

    frozen_train = [row for row in train_rows if keep(row)]
    frozen_validation = [row for row in validation_rows if keep(row)]
    answerable_train = [row for row in frozen_train if row.get("answerable")]
    unsupported_train = [row for row in frozen_train if not row.get("answerable")]
    answerable_updates = _example_updates(answerable_train)
    unsupported_updates = _example_updates(unsupported_train)
    unsupported_ratio = unsupported_updates / (answerable_updates + unsupported_updates)
    leakage = _split_leakage(frozen_train, frozen_validation)
    counts = _case_counts([*frozen_train, *frozen_validation])

    niah_train = counts.get("niah", {}).get("train-fit_answerable_groups", 0)
    niah_validation = counts.get("niah", {}).get("train-modelval_answerable_groups", 0)
    twiki_train = counts.get("2wiki", {}).get("train-fit_answerable_groups", 0)
    twiki_validation = counts.get("2wiki", {}).get("train-modelval_answerable_groups", 0)
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
    _write_jsonl(train_path, frozen_train)
    _write_jsonl(validation_path, frozen_validation)
    ordered = {
        "train_case_ids": [str(row["case_id"]) for row in frozen_train],
        "validation_case_ids": [str(row["case_id"]) for row in frozen_validation],
    }
    ordered_path = output_dir / "ordered_ids.json"
    _write_json(ordered_path, ordered)
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_FINAL_MANIFEST,
        "status": "PRE_MANUAL_PASS" if all(gates.values()) else "FAIL",
        "manual_audit_required_before_training": True,
        "training_started": False,
        "sealed_or_heldout_read": False,
        "dev_read": False,
        "counts": counts,
        "excluded": dict(excluded),
        "true_rows": len(true_rows),
        "answerable_train_updates": answerable_updates,
        "unsupported_updates": unsupported_updates,
        "unsupported_update_ratio": unsupported_ratio,
        "split_leakage": leakage,
        "gates": gates,
        "train_cases": len(frozen_train),
        "validation_cases": len(frozen_validation),
        "train_cases_sha256": _sha256(train_path),
        "validation_cases_sha256": _sha256(validation_path),
        "ordered_ids_sha256": _sha256(ordered_path),
        "input_sha256": {
            "source_train_cases": _sha256(train_cases_path),
            "source_validation_cases": _sha256(validation_cases_path),
            "structural_rows": _sha256(structural_rows_path),
            "true_audit_rows": _sha256(true_audit_rows_path),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    structural = commands.add_parser("structural")
    structural.add_argument("--data-manifest", required=True, type=Path)
    structural.add_argument("--train-cases", required=True, type=Path)
    structural.add_argument("--validation-cases", required=True, type=Path)
    structural.add_argument("--output-dir", required=True, type=Path)

    true = commands.add_parser("true-audit")
    true.add_argument("--true-worklist", required=True, type=Path)
    true.add_argument("--true-snapshot", required=True, type=Path)
    true.add_argument("--output-dir", required=True, type=Path)

    final = commands.add_parser("finalize")
    final.add_argument("--train-cases", required=True, type=Path)
    final.add_argument("--validation-cases", required=True, type=Path)
    final.add_argument("--structural-rows", required=True, type=Path)
    final.add_argument("--true-audit-rows", required=True, type=Path)
    final.add_argument("--output-dir", required=True, type=Path)
    final.add_argument("--min-niah-train-groups", type=int, default=400)
    final.add_argument("--min-niah-modelval-groups", type=int, default=100)
    final.add_argument("--min-twowiki-train-groups", type=int, default=400)
    final.add_argument("--min-twowiki-modelval-groups", type=int, default=100)
    final.add_argument("--unsupported-ratio-min", type=float, default=0.10)
    final.add_argument("--unsupported-ratio-max", type=float, default=0.15)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "structural":
        report = structural_audit(
            data_manifest_path=args.data_manifest,
            train_cases_path=args.train_cases,
            validation_cases_path=args.validation_cases,
            output_dir=args.output_dir.resolve(),
        )
    elif args.command == "true-audit":
        report = true_audit(
            true_worklist_path=args.true_worklist,
            true_snapshot=args.true_snapshot,
            output_dir=args.output_dir.resolve(),
        )
    else:
        report = finalize(
            train_cases_path=args.train_cases,
            validation_cases_path=args.validation_cases,
            structural_rows_path=args.structural_rows,
            true_audit_rows_path=args.true_audit_rows,
            output_dir=args.output_dir.resolve(),
            min_niah_train_groups=args.min_niah_train_groups,
            min_niah_modelval_groups=args.min_niah_modelval_groups,
            min_twowiki_train_groups=args.min_twowiki_train_groups,
            min_twowiki_modelval_groups=args.min_twowiki_modelval_groups,
            unsupported_ratio_min=args.unsupported_ratio_min,
            unsupported_ratio_max=args.unsupported_ratio_max,
        )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
