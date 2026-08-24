"""Materialize S110 Generator-aware utility labels.

S110 reuses the frozen S100 generation and scoring mechanics after S100 has
shown that the label signal is dense enough.  It expands the fixed GQ
full-context/leave-one-out procedure to the controlled train and model-val
answerable questions.  Generation task packets still exclude references and
support provenance; those are used only by ``score`` after generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import full_flow_s100_utility_pilot as s100

CONTEXT_VARIANT = s100.CONTEXT_VARIANT
DEFAULT_SAMPLE_SEED = "S110-v1-full-utility-2026-08-21"

SCHEMA_ORDERED_IDS = "full-flow-s110-ordered-ids-v1"
SCHEMA_ARCHIVE_MANIFEST = "full-flow-s110-archive-manifest-v1"
SCHEMA_SCORE_REPORT = "full-flow-s110-score-report-v1"
SCHEMA_SCORE_MANIFEST = "full-flow-s110-score-manifest-v1"
SCHEMA_SCORE_ROW = "full-flow-s110-scored-row-v1"
SCHEMA_LABEL_ROW = "full-flow-s110-utility-label-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def _validate_s100_ready(score_report_path: Path, score_manifest_path: Path) -> Mapping[str, Any]:
    report = _json(score_report_path)
    manifest = _json(score_manifest_path)
    if report.get("status") != "S100_PILOT_COMPLETE":
        raise ValueError("S110 requires completed S100 pilot")
    if report.get("s110_recommendation") != "S110_READY":
        raise ValueError("S110 requires S100 recommendation S110_READY")
    if manifest.get("status") != "COMPLETE":
        raise ValueError("S100 score manifest is not complete")
    if manifest.get("s100_status") != "S100_PILOT_COMPLETE":
        raise ValueError("S100 score manifest does not confirm pilot completion")
    if manifest.get("s110_recommendation") != "S110_READY":
        raise ValueError("S100 score manifest does not confirm S110_READY")
    if manifest.get("score_report_sha256") != _sha256(score_report_path):
        raise ValueError("S100 score report hash mismatch")
    if manifest.get("sealed_or_heldout_read") is not False:
        raise ValueError("S100 boundary invalid for S110")
    return report


def _role_for_split(split: str) -> str:
    if split == "train":
        return "train-fit"
    if split == "modelval":
        return "train-modelval"
    raise ValueError(f"unknown S110 split: {split}")


def _question_uid(*, split: str, dataset: str, case_id: str) -> str:
    return f"s110::{split}::{dataset}::{case_id}::{CONTEXT_VARIANT}"


def _candidate_question(row: Mapping[str, Any], *, split: str) -> dict[str, object] | None:
    if row.get("dataset") not in {"niah", "2wiki"}:
        return None
    if row.get("role") != _role_for_split(split) or row.get("answerable") is not True:
        return None
    variants = row.get("variants")
    if not isinstance(variants, Mapping) or not isinstance(variants.get(CONTEXT_VARIANT), Mapping):
        return None
    variant = cast(Mapping[str, Any], variants[CONTEXT_VARIANT])
    references = s100._reference_answers(row)
    if not references:
        return None
    evidence = s100.g310.parse_prompt_evidence(
        prompt=str(variant.get("prompt", "")),
        case_id=str(row["case_id"]),
        variant_name=CONTEXT_VARIANT,
    )
    if len(evidence) < 2:
        return None
    dataset = str(row["dataset"])
    case_id = str(row["case_id"])
    return {
        "split": split,
        "question_uid": _question_uid(split=split, dataset=dataset, case_id=case_id),
        "dataset": dataset,
        "case_id": case_id,
        "query_id": str(row.get("query_id", case_id)),
        "component_id": str(row.get("component_id", case_id)),
        "question": str(row.get("question") or s100.g310._extract_question(str(variant.get("prompt", "")))),
        "target_kind": str(row.get("target_kind", "")),
        "evidence": evidence,
        "reference_answers": references,
        "support_evidence_ids": s100._support_ids(variant),
    }


def _select_questions(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    dataset: str,
    limit: int | None,
    sample_seed: str,
) -> list[dict[str, object]]:
    candidates = [
        item
        for row in rows
        if (item := _candidate_question(row, split=split)) is not None and item["dataset"] == dataset
    ]
    ordered = sorted(
        candidates,
        key=lambda item: (
            s100._stable_value(sample_seed, split, dataset, str(item["question_uid"])),
            str(item["question_uid"]),
        ),
    )
    if limit is not None:
        if limit < 0:
            raise ValueError("S110 limits must be non-negative")
        if len(ordered) < limit:
            raise ValueError(f"S110 has only {len(ordered)} candidate {split}/{dataset} questions")
        return ordered[:limit]
    return ordered


def _build_tasks(question: Mapping[str, object]) -> list[s100.S100Task]:
    evidence = cast(tuple[s100.EvidenceCandidate, ...], question["evidence"])
    base = {
        "question_uid": str(question["question_uid"]),
        "dataset": str(question["dataset"]),
        "case_id": str(question["case_id"]),
        "query_id": str(question["query_id"]),
        "component_id": str(question["component_id"]),
        "context_variant": CONTEXT_VARIANT,
        "question": str(question["question"]),
        "target_kind": str(question.get("target_kind", "")),
        "original_evidence_count": len(evidence),
    }
    tasks = [
        s100.S100Task(
            task_id=f"{base['question_uid']}::full",
            variant_type="full",
            dropped_evidence_id=None,
            dropped_rank=None,
            evidence=evidence,
            **base,
        )
    ]
    for index, item in enumerate(evidence, 1):
        tasks.append(
            s100.S100Task(
                task_id=f"{base['question_uid']}::drop-rank{index}",
                variant_type="leave_one_out",
                dropped_evidence_id=item.evidence_id,
                dropped_rank=index,
                evidence=tuple(candidate for candidate in evidence if candidate.evidence_id != item.evidence_id),
                **base,
            )
        )
    return tasks


def _counts_by_split_dataset(questions: Sequence[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for question in questions:
        counts[str(question["split"])][str(question["dataset"])] += 1
    return {split: dict(counter) for split, counter in sorted(counts.items())}


def prepare(
    *,
    g430_manifest_path: Path,
    g223_manifest_path: Path,
    train_cases_path: Path,
    validation_cases_path: Path,
    ordered_ids_path: Path,
    s100_score_report_path: Path,
    s100_score_manifest_path: Path,
    output_dir: Path,
    sample_seed: str = DEFAULT_SAMPLE_SEED,
    train_niah_limit: int | None = None,
    train_twowiki_limit: int | None = None,
    modelval_niah_limit: int | None = None,
    modelval_twowiki_limit: int | None = None,
) -> dict[str, object]:
    g430 = s100._validate_g430_manifest(g430_manifest_path)
    g223 = s100._validate_g223_manifest(
        manifest_path=g223_manifest_path,
        train_cases_path=train_cases_path,
        ordered_ids_path=ordered_ids_path,
    )
    s100_ready = _validate_s100_ready(s100_score_report_path, s100_score_manifest_path)
    train_rows = _jsonl(train_cases_path)
    validation_rows = _jsonl(validation_cases_path)
    selected = [
        *_select_questions(
            train_rows,
            split="train",
            dataset="niah",
            limit=train_niah_limit,
            sample_seed=sample_seed,
        ),
        *_select_questions(
            train_rows,
            split="train",
            dataset="2wiki",
            limit=train_twowiki_limit,
            sample_seed=sample_seed,
        ),
        *_select_questions(
            validation_rows,
            split="modelval",
            dataset="niah",
            limit=modelval_niah_limit,
            sample_seed=sample_seed,
        ),
        *_select_questions(
            validation_rows,
            split="modelval",
            dataset="2wiki",
            limit=modelval_twowiki_limit,
            sample_seed=sample_seed,
        ),
    ]
    selected = sorted(
        selected,
        key=lambda item: (str(item["split"]), str(item["dataset"]), str(item["question_uid"])),
    )
    tasks = [task for question in selected for task in _build_tasks(question)]
    tasks = sorted(
        tasks,
        key=lambda task: (
            s100._stable_value(sample_seed, "s110-task-order", task.task_id),
            task.task_id,
        ),
    )
    references = [
        {
            "schema_version": s100.SCHEMA_REFERENCE,
            "stage": "S110",
            "split": str(question["split"]),
            "question_uid": str(question["question_uid"]),
            "dataset": str(question["dataset"]),
            "case_id": str(question["case_id"]),
            "query_id": str(question["query_id"]),
            "component_id": str(question["component_id"]),
            "reference_answers": list(cast(tuple[str, ...], question["reference_answers"])),
            "support_evidence_ids": list(cast(tuple[str, ...], question["support_evidence_ids"])),
            "full_evidence_ids": [
                item.evidence_id for item in cast(tuple[s100.EvidenceCandidate, ...], question["evidence"])
            ],
        }
        for question in selected
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = output_dir / "generation_tasks.jsonl"
    references_path = output_dir / "references.jsonl"
    ordered_ids_path_out = output_dir / "ordered_ids.json"
    _write_jsonl(tasks_path, [s100._task_row(task) for task in tasks])
    _write_jsonl(references_path, references)
    ordered_ids = {
        "schema_version": SCHEMA_ORDERED_IDS,
        "stage": "S110",
        "sample_seed": sample_seed,
        "question_uids": [str(item["question_uid"]) for item in selected],
        "task_ids_in_execution_order": [task.task_id for task in tasks],
        "split_dataset_question_counts": _counts_by_split_dataset(selected),
    }
    _write_json(ordered_ids_path_out, ordered_ids)
    dataset_counts = Counter(str(item["dataset"]) for item in selected)
    split_counts = Counter(str(item["split"]) for item in selected)
    evidence_counts = Counter(task.original_evidence_count for task in tasks if task.variant_type == "full")
    manifest = {
        "schema_version": s100.SCHEMA_PREPARE_MANIFEST,
        "s110_schema_version": "full-flow-s110-prepare-manifest-v1",
        "stage": "S110_PREPARE",
        "status": "COMPLETE",
        "gq_id": cast(Mapping[str, Any], g430["gq"])["id"],
        "sample_seed": sample_seed,
        "questions": len(selected),
        "dataset_question_counts": dict(dataset_counts),
        "split_question_counts": dict(split_counts),
        "split_dataset_question_counts": _counts_by_split_dataset(selected),
        "tasks": len(tasks),
        "evidence_count_distribution": {str(key): value for key, value in sorted(evidence_counts.items())},
        "expected_seed_generation_rows": len(tasks),
        "expected_total_generation_rows": len(tasks) * len(s100.ALLOWED_SEEDS),
        "tasks_sha256": _sha256(tasks_path),
        "references_sha256": _sha256(references_path),
        "ordered_ids_sha256": _sha256(ordered_ids_path_out),
        "task_identity_sha256": hashlib.sha256(
            s100._canonical_bytes([s100._task_identity(task) for task in tasks])
        ).hexdigest(),
        "source_sha256": {
            "script": _sha256(Path(__file__)),
            "s100_script": _sha256(Path(s100.__file__)),
            "g430_manifest": _sha256(g430_manifest_path),
            "g223_manifest": _sha256(g223_manifest_path),
            "train_cases": _sha256(train_cases_path),
            "validation_cases": _sha256(validation_cases_path),
            "ordered_ids": _sha256(ordered_ids_path),
            "s100_score_report": _sha256(s100_score_report_path),
            "s100_score_manifest": _sha256(s100_score_manifest_path),
        },
        "g223_status": g223.get("status"),
        "s100_status": s100_ready.get("status"),
        "s100_recommendation": s100_ready.get("s110_recommendation"),
        "generation_task_packet_contains_reference_answers": False,
        "generation_task_packet_contains_support_provenance": False,
        "score_uses_references_after_generation": True,
        "sealed_or_heldout_read": False,
        "selector_training_started": False,
        "full_system_started": False,
        "next_stage": "S110 run",
    }
    _write_json(output_dir / "prepare_manifest.json", manifest)
    return manifest


def run(
    *,
    seed: int,
    tasks_path: Path,
    prepare_manifest_path: Path,
    g430_manifest_path: Path,
    model_snapshot: Path,
    true_snapshot: Path,
    output_dir: Path,
    limit_tasks: int | None = None,
) -> dict[str, object]:
    manifest = s100.run(
        seed=seed,
        tasks_path=tasks_path,
        prepare_manifest_path=prepare_manifest_path,
        g430_manifest_path=g430_manifest_path,
        model_snapshot=model_snapshot,
        true_snapshot=true_snapshot,
        output_dir=output_dir,
        limit_tasks=limit_tasks,
    )
    manifest = dict(manifest)
    manifest["stage"] = "S110"
    manifest["s100_compatible_runner"] = "full_flow_s100_utility_pilot.run"
    manifest["s110_materialization_run"] = True
    s100._write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def _reference_splits(path: Path) -> dict[str, str]:
    splits: dict[str, str] = {}
    for row in _jsonl(path):
        question_uid = str(row["question_uid"])
        split = str(row.get("split", ""))
        if split not in {"train", "modelval"}:
            raise ValueError(f"{question_uid} lacks S110 split")
        splits[question_uid] = split
    return splits


def _enrich_jsonl_with_split(
    *,
    path: Path,
    splits: Mapping[str, str],
    schema_version: str,
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for row in _jsonl(path):
        enriched = dict(row)
        question_uid = str(enriched["question_uid"])
        enriched["schema_version"] = schema_version
        enriched["split"] = splits[question_uid]
        rows.append(enriched)
    _write_jsonl(path, rows)
    return rows


def _split_label_summary(labels: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    by_split_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    for row in labels:
        split = str(row["split"])
        dataset = str(row["dataset"])
        label = str(row["label"])
        by_split[split][label] += 1
        by_split_dataset[f"{split}:{dataset}"][label] += 1
    return {
        "label_counts_by_split": {key: dict(value) for key, value in sorted(by_split.items())},
        "label_counts_by_split_dataset": {
            key: dict(value) for key, value in sorted(by_split_dataset.items())
        },
    }


def _s200_recommendation(
    *,
    report: Mapping[str, Any],
    labels: Sequence[Mapping[str, Any]],
) -> str:
    if report.get("status") != "S100_PILOT_COMPLETE":
        return "STOP_TECHNICAL_FAILURE"
    if report.get("incomplete_questions"):
        return "STOP_INCOMPLETE_QUESTIONS"
    summary = cast(Mapping[str, Any], report.get("label_summary", {}))
    if int(summary.get("utility_labels", 0)) <= 0:
        return "STOP_NO_UTILITY_LABELS"
    split_utility: dict[str, int] = defaultdict(int)
    for row in labels:
        if row.get("label") in {"MUST_KEEP", "SAFE_DROP"}:
            split_utility[str(row["split"])] += 1
    if split_utility.get("train", 0) <= 0 or split_utility.get("modelval", 0) <= 0:
        return "STOP_SPLIT_WITHOUT_UTILITY_LABELS"
    return "S200_READY"


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    summary = cast(Mapping[str, Any], report["label_summary"])
    counts = cast(Mapping[str, object], summary.get("label_counts", {}))
    lines = [
        "# S110 Utility Materialization Report",
        "",
        f"**Status:** `{report['status']}`",
        f"**S200 recommendation:** `{report['s200_recommendation']}`",
        f"**Questions:** {report['questions']}",
        f"**Generation tasks:** {report['tasks']}",
        f"**Labels:** {summary.get('total_labels')}",
        "",
        "## Label Summary",
        "",
        "| label | count |",
        "|---|---:|",
    ]
    for label in ("MUST_KEEP", "SAFE_DROP", "NEUTRAL", "UNCERTAIN"):
        lines.append(f"| {label} | {counts.get(label, 0)} |")
    split_summary = cast(Mapping[str, Any], report.get("s110_split_label_summary", {}))
    split_counts = cast(Mapping[str, Mapping[str, int]], split_summary.get("label_counts_by_split", {}))
    if split_counts:
        lines += [
            "",
            "## Split Summary",
            "",
            "| split | MUST_KEEP | SAFE_DROP | NEUTRAL | UNCERTAIN |",
            "|---|---:|---:|---:|---:|",
        ]
        for split, values in sorted(split_counts.items()):
            lines.append(
                f"| {split} | {values.get('MUST_KEEP', 0)} | {values.get('SAFE_DROP', 0)} | "
                f"{values.get('NEUTRAL', 0)} | {values.get('UNCERTAIN', 0)} |"
            )
    lines += [
        "",
        f"- stable label rate: {float(summary.get('stable_label_rate', 0.0)):.4f}",
        f"- utility label rate: {float(summary.get('utility_label_rate', 0.0)):.4f}",
        f"- uncertain rate: {float(summary.get('uncertain_rate', 0.0)):.4f}",
        "",
        "## Boundary Check",
        "",
        "- generation runtime did not load reference answers or support provenance;",
        "- scoring used references only after generation;",
        "- sealed and held-out data were not read;",
        "- Selector training and full-system evaluation were not started.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def score(
    *,
    tasks_path: Path,
    references_path: Path,
    seed_generations: Mapping[int, Path],
    output_json: Path,
    output_rows: Path,
    output_labels: Path,
    output_report: Path,
    output_manifest: Path,
    minicheck_model_id: str = s100.FROZEN_MINICHECK_MODEL,
    minicheck_device: str = "cpu",
    entails: Callable[[str, str], bool] | None = None,
    allow_subset: bool = False,
) -> dict[str, object]:
    base_report = s100.score(
        tasks_path=tasks_path,
        references_path=references_path,
        seed_generations=seed_generations,
        output_json=output_json,
        output_rows=output_rows,
        output_labels=output_labels,
        output_report=output_report,
        output_manifest=None,
        minicheck_model_id=minicheck_model_id,
        minicheck_device=minicheck_device,
        entails=entails,
        allow_subset=allow_subset,
    )
    splits = _reference_splits(references_path)
    scored_rows = _enrich_jsonl_with_split(
        path=output_rows,
        splits=splits,
        schema_version=SCHEMA_SCORE_ROW,
    )
    labels = _enrich_jsonl_with_split(
        path=output_labels,
        splits=splits,
        schema_version=SCHEMA_LABEL_ROW,
    )
    report = dict(base_report)
    report["schema_version"] = SCHEMA_SCORE_REPORT
    report["stage"] = "S110"
    report["s100_compatible_score_status"] = base_report.get("status")
    report["status"] = (
        "S110_MATERIALIZATION_COMPLETE"
        if base_report.get("status") == "S100_PILOT_COMPLETE"
        and not base_report.get("technical_failures")
        and not base_report.get("incomplete_questions")
        else "S110_TECHNICAL_FAIL"
    )
    report["s200_recommendation"] = _s200_recommendation(report=base_report, labels=labels)
    report["s110_split_label_summary"] = _split_label_summary(labels)
    report["boundaries"] = {
        **cast(Mapping[str, Any], report.get("boundaries", {})),
        "generation_runtime_reference_answers_loaded": False,
        "generation_runtime_support_provenance_loaded": False,
        "score_uses_references_after_generation": True,
        "sealed_or_heldout_read": False,
        "selector_training_started": False,
        "full_system_started": False,
        "gq_changed": False,
    }
    _write_json(output_json, report)
    _write_report(output_report, report)
    manifest = {
        "schema_version": SCHEMA_SCORE_MANIFEST,
        "stage": "S110_SCORE",
        "status": "COMPLETE",
        "s110_status": report["status"],
        "s200_recommendation": report["s200_recommendation"],
        "score_report_sha256": _sha256(output_json),
        "scored_rows_sha256": _sha256(output_rows),
        "labels_sha256": _sha256(output_labels),
        "human_report_sha256": _sha256(output_report),
        "sealed_or_heldout_read": False,
        "selector_training_started": False,
        "full_system_started": False,
    }
    _write_json(output_manifest, manifest)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--g430-manifest", required=True, type=Path)
    prepare_parser.add_argument("--g223-manifest", required=True, type=Path)
    prepare_parser.add_argument("--train-cases", required=True, type=Path)
    prepare_parser.add_argument("--validation-cases", required=True, type=Path)
    prepare_parser.add_argument("--ordered-ids", required=True, type=Path)
    prepare_parser.add_argument("--s100-score-report", required=True, type=Path)
    prepare_parser.add_argument("--s100-score-manifest", required=True, type=Path)
    prepare_parser.add_argument("--output-dir", required=True, type=Path)
    prepare_parser.add_argument("--sample-seed", default=DEFAULT_SAMPLE_SEED)
    prepare_parser.add_argument("--train-niah-limit", type=int)
    prepare_parser.add_argument("--train-twowiki-limit", type=int)
    prepare_parser.add_argument("--modelval-niah-limit", type=int)
    prepare_parser.add_argument("--modelval-twowiki-limit", type=int)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--seed", required=True, type=int)
    run_parser.add_argument("--tasks", required=True, type=Path)
    run_parser.add_argument("--prepare-manifest", required=True, type=Path)
    run_parser.add_argument("--g430-manifest", required=True, type=Path)
    run_parser.add_argument("--model-snapshot", required=True, type=Path)
    run_parser.add_argument("--true-snapshot", required=True, type=Path)
    run_parser.add_argument("--output-dir", required=True, type=Path)
    run_parser.add_argument("--limit-tasks", type=int)

    score_parser = sub.add_parser("score")
    score_parser.add_argument("--tasks", required=True, type=Path)
    score_parser.add_argument("--references", required=True, type=Path)
    score_parser.add_argument("--seed13-generations", required=True, type=Path)
    score_parser.add_argument("--seed42-generations", required=True, type=Path)
    score_parser.add_argument("--seed73-generations", required=True, type=Path)
    score_parser.add_argument("--output-json", required=True, type=Path)
    score_parser.add_argument("--output-rows", required=True, type=Path)
    score_parser.add_argument("--output-labels", required=True, type=Path)
    score_parser.add_argument("--output-report", required=True, type=Path)
    score_parser.add_argument("--output-manifest", required=True, type=Path)
    score_parser.add_argument("--minicheck-model-id", default=s100.FROZEN_MINICHECK_MODEL)
    score_parser.add_argument("--minicheck-device", default="cpu")
    score_parser.add_argument("--allow-subset", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(
            json.dumps(
                prepare(
                    g430_manifest_path=args.g430_manifest,
                    g223_manifest_path=args.g223_manifest,
                    train_cases_path=args.train_cases,
                    validation_cases_path=args.validation_cases,
                    ordered_ids_path=args.ordered_ids,
                    s100_score_report_path=args.s100_score_report,
                    s100_score_manifest_path=args.s100_score_manifest,
                    output_dir=args.output_dir,
                    sample_seed=args.sample_seed,
                    train_niah_limit=args.train_niah_limit,
                    train_twowiki_limit=args.train_twowiki_limit,
                    modelval_niah_limit=args.modelval_niah_limit,
                    modelval_twowiki_limit=args.modelval_twowiki_limit,
                ),
                indent=2,
            )
        )
    elif args.command == "run":
        print(
            json.dumps(
                run(
                    seed=args.seed,
                    tasks_path=args.tasks,
                    prepare_manifest_path=args.prepare_manifest,
                    g430_manifest_path=args.g430_manifest,
                    model_snapshot=args.model_snapshot,
                    true_snapshot=args.true_snapshot,
                    output_dir=args.output_dir,
                    limit_tasks=args.limit_tasks,
                ),
                indent=2,
            )
        )
    elif args.command == "score":
        print(
            json.dumps(
                score(
                    tasks_path=args.tasks,
                    references_path=args.references,
                    seed_generations={
                        13: args.seed13_generations,
                        42: args.seed42_generations,
                        73: args.seed73_generations,
                    },
                    output_json=args.output_json,
                    output_rows=args.output_rows,
                    output_labels=args.output_labels,
                    output_report=args.output_report,
                    output_manifest=args.output_manifest,
                    minicheck_model_id=args.minicheck_model_id,
                    minicheck_device=args.minicheck_device,
                    allow_subset=args.allow_subset,
                ),
                indent=2,
            )
        )
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
