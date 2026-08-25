"""Prepare the G212 manual sample packet and length audit.

G212 sits after G210R2 pre-manual finalize and before any G300 training.  It
does not train a model and does not read sealed, held-out, or official dev data.

The ``prepare`` command validates the G210R2 pre-manual bundle, audits every
prompt/target example with the frozen Granite tokenizer, and writes a fixed
stratified sample for manual review.  The output intentionally remains blocked
until reviewer decisions are supplied to a later finalize step.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_CASE = "full-flow-g200-case-v2"
SCHEMA_PRE_MANUAL = "full-flow-g210-v2-final-manifest-v1"
SCHEMA_G214_REPAIR = "full-flow-g214-length-repair-manifest-v1"
SCHEMA_G216_REPAIR = "full-flow-g216-sample-review-repair-manifest-v1"
SCHEMA_G220_REPAIR = "full-flow-g220-conservative-filter-manifest-v1"
SCHEMA_G221_REPAIR = "full-flow-g221-targeted-sample-failure-repair-manifest-v1"
ACCEPTED_PRE_MANUAL_SCHEMAS = frozenset(
    {
        SCHEMA_PRE_MANUAL,
        SCHEMA_G214_REPAIR,
        SCHEMA_G216_REPAIR,
        SCHEMA_G220_REPAIR,
        SCHEMA_G221_REPAIR,
    }
)
SCHEMA_LENGTH_AUDIT = "full-flow-g212-length-audit-v1"
SCHEMA_MANUAL_SAMPLE_ROW = "full-flow-g212-manual-sample-row-v1"
SCHEMA_PREPARE_MANIFEST = "full-flow-g212-prepare-manifest-v1"
UNKNOWN_ANSWER = "I don't know."
DEFAULT_SAMPLE_PER_STRATUM = 20
DEFAULT_MAX_LENGTH = 2304
LENGTH_THRESHOLDS = (1536, 1792, 2048, 2304, 2560, 3072, 4096)
EVIDENCE_LINE = re.compile(r"^\[(\d+)\]\s+\(([^)]+)\)\s+(.*)$")
CITATION = re.compile(r"\[(\d+)\]")


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


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_empty(path: Path, label: str) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"{label} output directory must be absent or empty")


def _percentile(values: Sequence[int], probability: float) -> int:
    if not values:
        raise ValueError("cannot take percentile of empty values")
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def _variant_items(row: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    variants = row.get("variants")
    if not isinstance(variants, Mapping) or not variants:
        raise ValueError(f"case lacks variants: {row.get('case_id')}")
    items: list[tuple[str, Mapping[str, Any]]] = []
    for name in sorted(str(key) for key in variants):
        variant = variants[name]
        if not isinstance(variant, Mapping):
            raise ValueError(f"variant is not an object: {row.get('case_id')}/{name}")
        prompt = str(variant.get("prompt", ""))
        target = str(variant.get("target", ""))
        if not prompt or not target:
            raise ValueError(f"blank prompt/target: {row.get('case_id')}/{name}")
        items.append((name, variant))
    return items


def _load_bundle(
    *,
    manifest_path: Path,
    train_cases_path: Path,
    validation_cases_path: Path,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    manifest = _json(manifest_path)
    if manifest.get("schema_version") not in ACCEPTED_PRE_MANUAL_SCHEMAS:
        raise ValueError(
            "G212 expects a G210R2, G214, G216, G220, or G221 pre-manual manifest schema"
        )
    if manifest.get("status") != "PRE_MANUAL_PASS":
        raise ValueError("G212 requires a PRE_MANUAL_PASS input bundle")
    if manifest.get("training_started") is not False:
        raise ValueError("G212 input manifest must record training_started=false")
    if manifest.get("sealed_or_heldout_read") is not False or manifest.get("dev_read") is not False:
        raise ValueError("G212 input manifest must not have read dev/sealed/held-out data")
    if str(manifest.get("train_cases_sha256", "")) != _sha256(train_cases_path):
        raise ValueError("train cases hash differs from G210R2 manifest")
    if str(manifest.get("validation_cases_sha256", "")) != _sha256(validation_cases_path):
        raise ValueError("validation cases hash differs from G210R2 manifest")
    train_rows = _jsonl(train_cases_path)
    validation_rows = _jsonl(validation_cases_path)
    seen: set[str] = set()
    for row in [*train_rows, *validation_rows]:
        case_id = str(row.get("case_id", ""))
        if not case_id or case_id in seen or row.get("schema_version") != SCHEMA_CASE:
            raise ValueError(f"invalid or duplicate G212 input case: {case_id!r}")
        _variant_items(row)
        seen.add(case_id)
    return manifest, train_rows, validation_rows


def _chat_prompt_ids(tokenizer: Any, prompt: str) -> list[int]:
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return list(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])


def example_length(tokenizer: Any, prompt: str, target: str) -> int:
    prompt_ids = _chat_prompt_ids(tokenizer, prompt)
    target_ids = list(tokenizer(target, add_special_tokens=False)["input_ids"])
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Granite tokenizer has no EOS token")
    return len(prompt_ids) + len(target_ids) + 1


def length_audit_rows(
    *,
    tokenizer: Any,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    max_length: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    lengths: list[int] = []
    stratum_lengths: dict[str, list[int]] = defaultdict(list)
    for split, cases in (("train", train_rows), ("validation", validation_rows)):
        for case in cases:
            stratum = _stratum(case)
            for variant_name, variant in _variant_items(case):
                length = example_length(
                    tokenizer,
                    str(variant["prompt"]),
                    str(variant["target"]),
                )
                lengths.append(length)
                stratum_lengths[stratum].append(length)
                rows.append(
                    {
                        "case_id": str(case["case_id"]),
                        "split": split,
                        "dataset": str(case.get("dataset", "")),
                        "role": str(case.get("role", "")),
                        "target_kind": str(case.get("target_kind", "")),
                        "answerable": bool(case.get("answerable")),
                        "variant": variant_name,
                        "length": length,
                        "over_max_length": length > max_length,
                    }
                )
    over = sum(row["over_max_length"] is True for row in rows)
    summary: dict[str, object] = {
        "schema_version": SCHEMA_LENGTH_AUDIT,
        "status": "PASS" if over == 0 else "FAIL",
        "max_length": max_length,
        "examples": len(rows),
        "cases": len(train_rows) + len(validation_rows),
        "min": min(lengths),
        "p50": _percentile(lengths, 0.50),
        "p90": _percentile(lengths, 0.90),
        "p95": _percentile(lengths, 0.95),
        "p99": _percentile(lengths, 0.99),
        "max": max(lengths),
        "over_max_length": over,
        "truncation_rate_if_encoded_at_max_length": over / len(rows),
        "above_threshold": {
            str(threshold): sum(value > threshold for value in lengths)
            for threshold in LENGTH_THRESHOLDS
        },
        "by_stratum": {},
    }
    summary["by_stratum"] = {
        stratum: {
            "examples": len(values),
            "p95": _percentile(values, 0.95),
            "max": max(values),
            "over_max_length": sum(value > max_length for value in values),
        }
        for stratum, values in sorted(stratum_lengths.items())
    }
    return summary, rows


def _stratum(row: Mapping[str, Any]) -> str:
    answerable = "answerable" if bool(row.get("answerable")) else "unsupported"
    return ":".join(
        [
            str(row.get("dataset", "")),
            str(row.get("role", "")),
            str(row.get("target_kind", "")),
            answerable,
        ]
    )


def _sample_key(seed: str, row: Mapping[str, Any]) -> str:
    return _hash_text(f"{seed}\0{row.get('case_id')}\0{_stratum(row)}")


def select_manual_sample(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    sample_per_stratum: int,
    seed: str,
) -> list[Mapping[str, Any]]:
    if sample_per_stratum <= 0:
        raise ValueError("sample_per_stratum must be positive")
    strata: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in [*train_rows, *validation_rows]:
        strata[_stratum(row)].append(row)
    selected: list[Mapping[str, Any]] = []
    for stratum in sorted(strata):
        candidates = sorted(strata[stratum], key=lambda row: (_sample_key(seed, row), str(row["case_id"])))
        selected.extend(candidates[: min(sample_per_stratum, len(candidates))])
    return selected


def _evidence_lines(prompt: str) -> list[dict[str, object]]:
    lines: list[dict[str, object]] = []
    for line in prompt.splitlines():
        match = EVIDENCE_LINE.match(line)
        if match is None:
            continue
        lines.append(
            {
                "citation_index": int(match.group(1)),
                "evidence_id": match.group(2),
                "text": match.group(3),
            }
        )
    return lines


def _sample_variant(row: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    variants = dict(_variant_items(row))
    preferred = "support_only" if bool(row.get("answerable")) else "support_removed"
    if preferred in variants:
        return preferred, variants[preferred]
    return next(iter(variants.items()))


def _citation_indices(target: str) -> list[int]:
    return [int(item) for item in CITATION.findall(target)]


def _manual_sample_row(row: Mapping[str, Any], *, index: int, seed: str) -> dict[str, object]:
    variant_name, variant = _sample_variant(row)
    prompt = str(variant["prompt"])
    target = str(variant["target"])
    evidence = _evidence_lines(prompt)
    support_ids = {str(item) for item in variant.get("support_evidence_ids", [])}
    citations = _citation_indices(target)
    cited_ids = [
        evidence[item - 1]["evidence_id"]
        for item in citations
        if 1 <= item <= len(evidence)
    ]
    audit_id = _hash_text(f"G212\0{seed}\0{row.get('case_id')}\0{variant_name}")[:24]
    return {
        "schema_version": SCHEMA_MANUAL_SAMPLE_ROW,
        "audit_id": audit_id,
        "sample_index": index,
        "sample_seed": seed,
        "sample_stratum": _stratum(row),
        "case_id": str(row["case_id"]),
        "dataset": str(row.get("dataset", "")),
        "role": str(row.get("role", "")),
        "target_kind": str(row.get("target_kind", "")),
        "answerable": bool(row.get("answerable")),
        "group_id": str(row.get("group_id", "")),
        "component_id": str(row.get("component_id", "")),
        "question": str(row.get("question", "")),
        "answer": str(row.get("answer", "")),
        "semantic_target": str(row.get("semantic_target", "")),
        "review_variant": variant_name,
        "target": target,
        "target_citation_indices": citations,
        "target_cited_evidence_ids": cited_ids,
        "support_evidence_ids": sorted(support_ids),
        "removed_support_evidence_ids": [str(item) for item in row.get("removed_support_evidence_ids", [])],
        "review_evidence": evidence,
        "review_required": {
            "answerable_target_is_supported_by_cited_evidence": bool(row.get("answerable")),
            "answer_alias_or_yes_no_chain_is_preserved": bool(row.get("answerable")),
            "unsupported_target_has_no_visible_support_and_says_idk": not bool(row.get("answerable")),
            "citation_indices_match_support_evidence": True,
            "no_obvious_target_or_prompt_corruption": True,
        },
        "review_decision": "PENDING",
        "reviewer": None,
        "review_reason": None,
    }


def manual_sample_rows(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    sample_per_stratum: int,
    seed: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    selected = select_manual_sample(
        train_rows=train_rows,
        validation_rows=validation_rows,
        sample_per_stratum=sample_per_stratum,
        seed=seed,
    )
    rows = [
        _manual_sample_row(row, index=index, seed=seed)
        for index, row in enumerate(selected, start=1)
    ]
    counts = Counter(str(row["sample_stratum"]) for row in rows)
    summary: dict[str, object] = {
        "schema_version": "full-flow-g212-manual-sample-summary-v1",
        "status": "MANUAL_REVIEW_PENDING",
        "sample_seed": seed,
        "sample_per_stratum": sample_per_stratum,
        "sample_rows": len(rows),
        "stratum_counts": dict(sorted(counts.items())),
        "review_rows_required": len(rows),
        "training_started": False,
        "sealed_or_heldout_read": False,
        "dev_read": False,
    }
    return summary, rows


def prepare(
    *,
    manifest_path: Path,
    train_cases_path: Path,
    validation_cases_path: Path,
    model_snapshot: Path,
    output_dir: Path,
    max_length: int = DEFAULT_MAX_LENGTH,
    sample_per_stratum: int = DEFAULT_SAMPLE_PER_STRATUM,
    sample_seed: str = "G212-v1-fixed-stratified-sample",
    tokenizer: Any | None = None,
) -> dict[str, object]:
    _require_empty(output_dir, "G212 prepare")
    started = time.time()
    manifest, train_rows, validation_rows = _load_bundle(
        manifest_path=manifest_path,
        train_cases_path=train_cases_path,
        validation_cases_path=validation_cases_path,
    )
    if tokenizer is None:
        transformers = importlib.import_module("transformers")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(model_snapshot.resolve()), local_files_only=True
        )
        transformers_version = str(transformers.__version__)
    else:
        transformers_version = "test-tokenizer"
    length_summary, length_rows = length_audit_rows(
        tokenizer=tokenizer,
        train_rows=train_rows,
        validation_rows=validation_rows,
        max_length=max_length,
    )
    manual_summary, manual_rows = manual_sample_rows(
        train_rows=train_rows,
        validation_rows=validation_rows,
        sample_per_stratum=sample_per_stratum,
        seed=sample_seed,
    )
    length_path = output_dir / "length_audit.json"
    length_rows_path = output_dir / "length_rows.jsonl"
    manual_summary_path = output_dir / "manual_sample_summary.json"
    manual_rows_path = output_dir / "manual_sample.jsonl"
    _write_json(length_path, length_summary)
    _write_jsonl(length_rows_path, length_rows)
    _write_json(manual_summary_path, manual_summary)
    _write_jsonl(manual_rows_path, manual_rows)

    status = (
        "LENGTH_PASS_MANUAL_REVIEW_PENDING"
        if length_summary["status"] == "PASS"
        else "FAIL_LENGTH"
    )
    config_path = model_snapshot / "config.json"
    manifest_out: dict[str, object] = {
        "schema_version": SCHEMA_PREPARE_MANIFEST,
        "status": status,
        "training_started": False,
        "utility_labels_started": False,
        "sealed_or_heldout_read": False,
        "dev_read": False,
        "manual_review_required_before_training": True,
        "manual_review_complete": False,
        "length_audit_pass": length_summary["status"] == "PASS",
        "max_length": max_length,
        "input_sha256": {
            "pre_manual_manifest": _sha256(manifest_path),
            "train_cases": _sha256(train_cases_path),
            "validation_cases": _sha256(validation_cases_path),
        },
        "input_counts": {
            "train_cases": len(train_rows),
            "validation_cases": len(validation_rows),
            "manifest_train_cases": int(manifest.get("train_cases", -1)),
            "manifest_validation_cases": int(manifest.get("validation_cases", -1)),
        },
        "model_snapshot": str(model_snapshot.resolve()),
        "model_snapshot_config_sha256": _sha256(config_path) if config_path.is_file() else None,
        "sample_seed": sample_seed,
        "sample_rows": len(manual_rows),
        "artifacts": {
            "length_audit": {"path": str(length_path), "sha256": _sha256(length_path)},
            "length_rows": {"path": str(length_rows_path), "sha256": _sha256(length_rows_path)},
            "manual_sample_summary": {
                "path": str(manual_summary_path),
                "sha256": _sha256(manual_summary_path),
            },
            "manual_sample": {"path": str(manual_rows_path), "sha256": _sha256(manual_rows_path)},
        },
        "versions": {"transformers": transformers_version},
        "elapsed_seconds": time.time() - started,
    }
    manifest_out_path = output_dir / "prepare_manifest.json"
    _write_json(manifest_out_path, manifest_out)
    return manifest_out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--manifest", required=True, type=Path)
    prepare_parser.add_argument("--train-cases", required=True, type=Path)
    prepare_parser.add_argument("--validation-cases", required=True, type=Path)
    prepare_parser.add_argument("--model-snapshot", required=True, type=Path)
    prepare_parser.add_argument("--output-dir", required=True, type=Path)
    prepare_parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    prepare_parser.add_argument("--sample-per-stratum", type=int, default=DEFAULT_SAMPLE_PER_STRATUM)
    prepare_parser.add_argument("--sample-seed", default="G212-v1-fixed-stratified-sample")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = prepare(
        manifest_path=args.manifest,
        train_cases_path=args.train_cases,
        validation_cases_path=args.validation_cases,
        model_snapshot=args.model_snapshot,
        output_dir=args.output_dir.resolve(),
        max_length=args.max_length,
        sample_per_stratum=args.sample_per_stratum,
        sample_seed=args.sample_seed,
    )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
