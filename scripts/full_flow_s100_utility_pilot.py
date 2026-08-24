"""Prepare, run, and score the S100 Generator-aware utility pilot.

S100 uses the frozen GQ teacher from G430.  The generation command consumes a
task packet that contains only question text and evidence text.  References and
support provenance are split into a separate packet and are used only by
``score`` after all generations have finished.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import full_flow_g310_seed13_screen as g310
import full_flow_g400_niah_qualification as g400
from alce_metrics import ScoredExample, compute_citation_metrics
from evidence_rag.contracts.models import (
    EvidenceCandidate,
    GenerationResult,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
    strip_annotations,
)
from evidence_rag.evaluation.scoring import answer_match
from evidence_rag.generator.granite import GraniteGenerationConfig, PeftGraniteLLMClient
from evidence_rag.generator.nli import MiniCheckNLIModel, TrueNLIModel

Seed = Literal[13, 42, 73]
Label = Literal["MUST_KEEP", "SAFE_DROP", "NEUTRAL", "UNCERTAIN"]

ALLOWED_SEEDS: tuple[Seed, ...] = (13, 42, 73)
CANDIDATE_ARM = "GQ"
CONTEXT_VARIANT = "topk"
DEFAULT_SAMPLE_SEED = "S100-v1-fixed-pilot-2026-08-20"
DEFAULT_NIAH_QUESTIONS = 50
DEFAULT_TWOWIKI_QUESTIONS = 50
MIN_UTILITY_LABELS_FOR_S110 = 20
MAX_UNCERTAIN_RATE_FOR_S110 = 0.50
MIN_DATASETS_WITH_UTILITY_FOR_S110 = 2

SCHEMA_G430 = "full-flow-g430-gq-freeze-v1"
SCHEMA_G223 = g310.SCHEMA_G223_MANIFEST
SCHEMA_PREPARE_MANIFEST = "full-flow-s100-prepare-manifest-v1"
SCHEMA_TASK = "full-flow-s100-generation-task-v1"
SCHEMA_REFERENCE = "full-flow-s100-reference-v1"
SCHEMA_ORDERED_IDS = "full-flow-s100-ordered-ids-v1"
SCHEMA_RUN_SPEC = "full-flow-s100-run-spec-v1"
SCHEMA_RUN_MANIFEST = "full-flow-s100-run-manifest-v1"
SCHEMA_GENERATION_ROW = "full-flow-s100-generation-row-v1"
SCHEMA_SCORE_REPORT = "full-flow-s100-score-report-v1"
SCHEMA_SCORE_ROW = "full-flow-s100-scored-row-v1"
SCHEMA_LABEL_ROW = "full-flow-s100-utility-label-v1"
SCHEMA_SCORE_MANIFEST = "full-flow-s100-score-manifest-v1"
FROZEN_MINICHECK_MODEL = g400.FROZEN_MINICHECK_MODEL
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class S100Task:
    task_id: str
    question_uid: str
    dataset: str
    case_id: str
    query_id: str
    component_id: str
    context_variant: str
    variant_type: str
    question: str
    target_kind: str
    original_evidence_count: int
    dropped_evidence_id: str | None
    dropped_rank: int | None
    evidence: tuple[EvidenceCandidate, ...]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _stable_value(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("::".join(parts).encode()).digest()[:8], "big")


def _validate_g430_manifest(path: Path) -> Mapping[str, Any]:
    manifest = _json(path)
    if manifest.get("schema_version") != SCHEMA_G430:
        raise ValueError("S100 requires the G430 GQ freeze manifest")
    if manifest.get("status") != "GQ_FROZEN_NEW_GRC":
        raise ValueError("S100 requires G430 status GQ_FROZEN_NEW_GRC")
    gq = manifest.get("gq")
    if not isinstance(gq, Mapping) or tuple(gq.get("seeds", ())) != ALLOWED_SEEDS:
        raise ValueError("S100 requires the frozen three-seed GQ family")
    boundaries = manifest.get("boundaries")
    if not isinstance(boundaries, Mapping) or boundaries.get("sealed_or_heldout_read") is not False:
        raise ValueError("G430 manifest boundary is invalid")
    return manifest


def _validate_g223_manifest(
    *,
    manifest_path: Path,
    train_cases_path: Path,
    ordered_ids_path: Path,
) -> Mapping[str, Any]:
    manifest = _json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_G223:
        raise ValueError("S100 expects the G223 controlled-continuation manifest")
    if manifest.get("status") != "CONTROLLED_CONTINUATION_READY":
        raise ValueError("S100 requires G223 status CONTROLLED_CONTINUATION_READY")
    if manifest.get("dev_read") is not False or manifest.get("sealed_or_heldout_read") is not False:
        raise ValueError("S100 input must not have read dev/sealed/held-out data")
    if manifest.get("train_cases_sha256") != _sha256(train_cases_path):
        raise ValueError("G223 train cases hash mismatch")
    if manifest.get("ordered_ids_sha256") != _sha256(ordered_ids_path):
        raise ValueError("G223 ordered IDs hash mismatch")
    return manifest


def _support_ids(variant: Mapping[str, Any]) -> tuple[str, ...]:
    raw = variant.get("support_evidence_ids")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return tuple(str(item) for item in raw)
    single = variant.get("support_evidence_id")
    return (str(single),) if single else ()


def _reference_answers(row: Mapping[str, Any]) -> tuple[str, ...]:
    values = {
        str(row.get("answer", "")).strip(),
        str(row.get("official_answer", "")).strip(),
    }
    return tuple(sorted(item for item in values if item and item != "None"))


def _question_uid(row: Mapping[str, Any]) -> str:
    return f"s100::{row['dataset']}::{row['case_id']}::{CONTEXT_VARIANT}"


def _task_identity(task: S100Task) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "question_uid": task.question_uid,
        "dataset": task.dataset,
        "case_id": task.case_id,
        "query_id": task.query_id,
        "component_id": task.component_id,
        "context_variant": task.context_variant,
        "variant_type": task.variant_type,
        "target_kind": task.target_kind,
        "original_evidence_count": task.original_evidence_count,
        "dropped_evidence_id": task.dropped_evidence_id,
        "dropped_rank": task.dropped_rank,
        "evidence_ids": [item.evidence_id for item in task.evidence],
    }


def _task_row(task: S100Task) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_TASK,
        **_task_identity(task),
        "question": task.question,
        "evidence": [item.model_dump(mode="json") for item in task.evidence],
    }


def _task_from_row(row: Mapping[str, Any]) -> S100Task:
    if row.get("schema_version") != SCHEMA_TASK:
        raise ValueError(f"{row.get('task_id')} is not an S100 task row")
    forbidden = {
        "answer",
        "official_answer",
        "reference_answers",
        "gold",
        "support_evidence_ids",
        "support_evidence_id",
    }
    if forbidden & set(row):
        raise ValueError(f"{row.get('task_id')} contains post-generation fields")
    raw_evidence = row.get("evidence")
    if not isinstance(raw_evidence, Sequence) or isinstance(raw_evidence, (str, bytes)):
        raise ValueError(f"{row.get('task_id')} has invalid evidence")
    return S100Task(
        task_id=str(row["task_id"]),
        question_uid=str(row["question_uid"]),
        dataset=str(row["dataset"]),
        case_id=str(row["case_id"]),
        query_id=str(row["query_id"]),
        component_id=str(row["component_id"]),
        context_variant=str(row["context_variant"]),
        variant_type=str(row["variant_type"]),
        question=str(row["question"]),
        target_kind=str(row.get("target_kind", "")),
        original_evidence_count=int(row["original_evidence_count"]),
        dropped_evidence_id=(
            str(row["dropped_evidence_id"]) if row.get("dropped_evidence_id") is not None else None
        ),
        dropped_rank=int(row["dropped_rank"]) if row.get("dropped_rank") is not None else None,
        evidence=tuple(EvidenceCandidate.model_validate(item) for item in raw_evidence),
    )


def _candidate_question(row: Mapping[str, Any]) -> dict[str, object] | None:
    if row.get("dataset") not in {"niah", "2wiki"}:
        return None
    if row.get("role") != "train-fit" or row.get("answerable") is not True:
        return None
    variants = row.get("variants")
    if not isinstance(variants, Mapping) or not isinstance(variants.get(CONTEXT_VARIANT), Mapping):
        return None
    variant = cast(Mapping[str, Any], variants[CONTEXT_VARIANT])
    references = _reference_answers(row)
    if not references:
        return None
    evidence = g310.parse_prompt_evidence(
        prompt=str(variant.get("prompt", "")),
        case_id=str(row["case_id"]),
        variant_name=CONTEXT_VARIANT,
    )
    if len(evidence) < 2:
        return None
    return {
        "question_uid": _question_uid(row),
        "dataset": str(row["dataset"]),
        "case_id": str(row["case_id"]),
        "query_id": str(row.get("query_id", row["case_id"])),
        "component_id": str(row.get("component_id", row["case_id"])),
        "question": str(row.get("question") or g310._extract_question(str(variant.get("prompt", "")))),
        "target_kind": str(row.get("target_kind", "")),
        "evidence": evidence,
        "reference_answers": references,
        "support_evidence_ids": _support_ids(variant),
    }


def _select_questions(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    count: int,
    sample_seed: str,
) -> list[dict[str, object]]:
    candidates = [
        item
        for row in rows
        if (item := _candidate_question(row)) is not None and item["dataset"] == dataset
    ]
    ordered = sorted(
        candidates,
        key=lambda item: (
            _stable_value(sample_seed, dataset, str(item["question_uid"])),
            str(item["question_uid"]),
        ),
    )
    if len(ordered) < count:
        raise ValueError(f"S100 has only {len(ordered)} candidate {dataset} questions")
    return ordered[:count]


def _build_tasks(question: Mapping[str, object]) -> list[S100Task]:
    evidence = cast(tuple[EvidenceCandidate, ...], question["evidence"])
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
        S100Task(
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
            S100Task(
                task_id=f"{base['question_uid']}::drop-rank{index}",
                variant_type="leave_one_out",
                dropped_evidence_id=item.evidence_id,
                dropped_rank=index,
                evidence=tuple(candidate for candidate in evidence if candidate.evidence_id != item.evidence_id),
                **base,
            )
        )
    return tasks


def prepare(
    *,
    g430_manifest_path: Path,
    g223_manifest_path: Path,
    train_cases_path: Path,
    ordered_ids_path: Path,
    output_dir: Path,
    niah_questions: int = DEFAULT_NIAH_QUESTIONS,
    twowiki_questions: int = DEFAULT_TWOWIKI_QUESTIONS,
    sample_seed: str = DEFAULT_SAMPLE_SEED,
) -> dict[str, object]:
    g430 = _validate_g430_manifest(g430_manifest_path)
    g223 = _validate_g223_manifest(
        manifest_path=g223_manifest_path,
        train_cases_path=train_cases_path,
        ordered_ids_path=ordered_ids_path,
    )
    train_rows = _jsonl(train_cases_path)
    selected = [
        *_select_questions(train_rows, dataset="niah", count=niah_questions, sample_seed=sample_seed),
        *_select_questions(train_rows, dataset="2wiki", count=twowiki_questions, sample_seed=sample_seed),
    ]
    selected = sorted(selected, key=lambda item: (str(item["dataset"]), str(item["question_uid"])))
    tasks = [task for question in selected for task in _build_tasks(question)]
    tasks = sorted(
        tasks,
        key=lambda task: (
            _stable_value(sample_seed, "s100-task-order", task.task_id),
            task.task_id,
        ),
    )
    references = [
        {
            "schema_version": SCHEMA_REFERENCE,
            "question_uid": str(question["question_uid"]),
            "dataset": str(question["dataset"]),
            "case_id": str(question["case_id"]),
            "query_id": str(question["query_id"]),
            "component_id": str(question["component_id"]),
            "reference_answers": list(cast(tuple[str, ...], question["reference_answers"])),
            "support_evidence_ids": list(cast(tuple[str, ...], question["support_evidence_ids"])),
            "full_evidence_ids": [
                item.evidence_id for item in cast(tuple[EvidenceCandidate, ...], question["evidence"])
            ],
        }
        for question in selected
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = output_dir / "generation_tasks.jsonl"
    references_path = output_dir / "references.jsonl"
    ordered_ids_path_out = output_dir / "ordered_ids.json"
    _write_jsonl(tasks_path, [_task_row(task) for task in tasks])
    _write_jsonl(references_path, references)
    ordered_ids = {
        "schema_version": SCHEMA_ORDERED_IDS,
        "stage": "S100",
        "sample_seed": sample_seed,
        "question_uids": [str(item["question_uid"]) for item in selected],
        "task_ids_in_execution_order": [task.task_id for task in tasks],
    }
    _write_json(ordered_ids_path_out, ordered_ids)
    dataset_counts = Counter(str(item["dataset"]) for item in selected)
    evidence_counts = Counter(task.original_evidence_count for task in tasks if task.variant_type == "full")
    manifest = {
        "schema_version": SCHEMA_PREPARE_MANIFEST,
        "stage": "S100_PREPARE",
        "status": "COMPLETE",
        "gq_id": cast(Mapping[str, Any], g430["gq"])["id"],
        "sample_seed": sample_seed,
        "questions": len(selected),
        "dataset_question_counts": dict(dataset_counts),
        "tasks": len(tasks),
        "evidence_count_distribution": {str(key): value for key, value in sorted(evidence_counts.items())},
        "expected_seed_generation_rows": len(tasks),
        "expected_total_generation_rows": len(tasks) * len(ALLOWED_SEEDS),
        "tasks_sha256": _sha256(tasks_path),
        "references_sha256": _sha256(references_path),
        "ordered_ids_sha256": _sha256(ordered_ids_path_out),
        "task_identity_sha256": _sha256_bytes(_canonical_bytes([_task_identity(task) for task in tasks])),
        "source_sha256": {
            "script": _sha256(Path(__file__)),
            "g430_manifest": _sha256(g430_manifest_path),
            "g223_manifest": _sha256(g223_manifest_path),
            "train_cases": _sha256(train_cases_path),
            "ordered_ids": _sha256(ordered_ids_path),
        },
        "g223_status": g223.get("status"),
        "generation_task_packet_contains_reference_answers": False,
        "generation_task_packet_contains_support_provenance": False,
        "score_uses_references_after_generation": True,
        "sealed_or_heldout_read": False,
        "selector_training_started": False,
        "full_system_started": False,
    }
    _write_json(output_dir / "prepare_manifest.json", manifest)
    return manifest


def _load_tasks(tasks_path: Path) -> list[S100Task]:
    tasks = [_task_from_row(row) for row in _jsonl(tasks_path)]
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("duplicate S100 task rows")
    return tasks


def _validate_prepare_manifest(path: Path, tasks_path: Path) -> Mapping[str, Any]:
    manifest = _json(path)
    if manifest.get("schema_version") != SCHEMA_PREPARE_MANIFEST:
        raise ValueError("S100 requires an S100 prepare manifest")
    if manifest.get("status") != "COMPLETE":
        raise ValueError("S100 prepare manifest is not complete")
    if manifest.get("tasks_sha256") != _sha256(tasks_path):
        raise ValueError("S100 task packet hash mismatch")
    if manifest.get("sealed_or_heldout_read") is not False:
        raise ValueError("S100 prepare must not read held-out data")
    return manifest


def validate_resume_prefix(rows: Sequence[Mapping[str, Any]], tasks: Sequence[S100Task]) -> None:
    if len(rows) > len(tasks):
        raise ValueError("S100 resume has more rows than frozen tasks")
    for index, row in enumerate(rows):
        task = tasks[index]
        if row.get("task_id") != task.task_id:
            raise ValueError(f"S100 resume diverges at row {index + 1}")
        arms = row.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != {CANDIDATE_ARM}:
            raise ValueError(f"S100 resume row {index + 1} has wrong arms")


def _gq_run_dir(g430_manifest: Mapping[str, Any], seed: int) -> Path:
    adapters = g430_manifest.get("adapters")
    if not isinstance(adapters, Mapping):
        raise ValueError("G430 manifest lacks adapters")
    item = adapters.get(str(seed))
    if not isinstance(item, Mapping):
        raise ValueError(f"G430 manifest lacks seed {seed}")
    adapter = Path(str(item["runtime_adapter"]))
    return adapter.parent


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
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"S100 seed must be one of {ALLOWED_SEEDS}")
    prepare_manifest = _validate_prepare_manifest(prepare_manifest_path, tasks_path)
    g430_manifest = _validate_g430_manifest(g430_manifest_path)
    tasks = _load_tasks(tasks_path)
    formal_task_count = len(tasks)
    if limit_tasks is not None:
        if limit_tasks <= 0:
            raise ValueError("--limit-tasks must be positive")
        tasks = tasks[:limit_tasks]
    model_config_sha256 = _sha256(model_snapshot / "config.json")
    grc_run_dir = _gq_run_dir(g430_manifest, seed)
    adapter_audit = g400._validate_g330_run(
        grc_run_dir,
        seed=seed,
        model_config_sha256=model_config_sha256,
    )
    spec: dict[str, object] = {
        "schema_version": SCHEMA_RUN_SPEC,
        "stage": "S100",
        "seed": seed,
        "arm": CANDIDATE_ARM,
        "formal": limit_tasks is None,
        "tasks": len(tasks),
        "formal_task_count": formal_task_count,
        "prepare_manifest_sha256": _sha256(prepare_manifest_path),
        "prepare_task_identity_sha256": prepare_manifest.get("task_identity_sha256"),
        "g430_manifest_sha256": _sha256(g430_manifest_path),
        "gq_id": cast(Mapping[str, Any], g430_manifest["gq"])["id"],
        "task_identity_sha256": _sha256_bytes(_canonical_bytes([_task_identity(task) for task in tasks])),
        "source_sha256": {
            "script": _sha256(Path(__file__)),
            "tasks": _sha256(tasks_path),
            "granite_config": model_config_sha256,
            "true_config": _sha256(true_snapshot / "config.json"),
        },
        "adapter_audit": adapter_audit,
        "decode": {
            "max_new_tokens": 256,
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
        },
        "gold_loaded_at_runtime": False,
        "reference_answers_loaded_at_runtime": False,
        "support_provenance_loaded_at_runtime": False,
        "sealed_or_heldout_read": False,
        "selector_training_started": False,
        "full_system_started": False,
        "adapter_scope": "draft generation call only",
        "claim_splitter_scope": "frozen Granite base with adapters disabled",
        "true_scope": "frozen verifier",
        "runtime_environment": g400._runtime_environment(),
        "git_commit": _git_commit(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / "run_spec.json"
    generations_path = output_dir / "generations.jsonl"
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        raise ValueError(f"S100 run is already complete: {manifest_path}")
    if spec_path.exists():
        if _json(spec_path) != spec:
            raise ValueError("S100 resume specification differs from frozen run")
    else:
        if generations_path.exists() and generations_path.stat().st_size:
            raise ValueError("S100 generations exist without a run specification")
        _write_json(spec_path, spec)
    existing = _jsonl(generations_path) if generations_path.exists() else []
    validate_resume_prefix(existing, tasks)

    client = PeftGraniteLLMClient(
        model_id=str(model_snapshot.resolve()),
        adapters={"grc": str((grc_run_dir / "adapter").resolve())},
        config=GraniteGenerationConfig(max_new_tokens=256, temperature=0.0, top_p=1.0),
    )
    nli = TrueNLIModel(model_id=str(true_snapshot.resolve()))
    generator = g400.build_generator(client=client, nli=nli)
    started = time.perf_counter()
    with generations_path.open("a", encoding="utf-8", newline="\n") as handle:
        for index, task in enumerate(tasks[len(existing) :], start=len(existing) + 1):
            query = Query(query_id=task.task_id, text=task.question)
            checklist = QueryChecklist(
                query_id=task.task_id,
                focus=task.question,
                required_facts=(),
            )
            selected = SelectedEvidenceSet(query_id=task.task_id, evidence=task.evidence)
            row = {
                "schema_version": SCHEMA_GENERATION_ROW,
                **_task_identity(task),
                "question": task.question,
                "evidence": [item.model_dump(mode="json") for item in task.evidence],
                "arm_order": [CANDIDATE_ARM],
                "arms": {
                    CANDIDATE_ARM: g400._run_one(generator, query, checklist, selected)
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 25 == 0 or index == len(tasks):
                elapsed = time.perf_counter() - started
                completed_now = index - len(existing)
                print(
                    f"[S100 seed-{seed}] {index}/{len(tasks)} "
                    f"{elapsed / max(completed_now, 1):.2f}s/new-task",
                    flush=True,
                )
    rows = _jsonl(generations_path)
    validate_resume_prefix(rows, tasks)
    errors = sum(
        bool(cast(Mapping[str, object], row["arms"])[CANDIDATE_ARM].get("error"))
        for row in rows
    )
    trace_missing = sum(
        cast(Mapping[str, object], row["arms"])[CANDIDATE_ARM].get("trace") is None
        and not cast(Mapping[str, object], row["arms"])[CANDIDATE_ARM].get("error")
        for row in rows
    )
    manifest = {
        "schema_version": SCHEMA_RUN_MANIFEST,
        "stage": "S100",
        "status": "INVALID_RUNTIME" if errors or trace_missing else "COMPLETE",
        "seed": seed,
        "arm": CANDIDATE_ARM,
        "formal": limit_tasks is None,
        "tasks": len(rows),
        "generations_sha256": _sha256(generations_path),
        "run_spec_sha256": _sha256(spec_path),
        "errors": errors,
        "trace_missing": trace_missing,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "gold_loaded_at_runtime": False,
        "reference_answers_loaded_at_runtime": False,
        "support_provenance_loaded_at_runtime": False,
        "sealed_or_heldout_read": False,
        "selector_training_started": False,
        "full_system_started": False,
    }
    _write_json(manifest_path, manifest)
    if errors or trace_missing:
        raise RuntimeError("S100 runtime has errors or missing traces")
    return manifest


def _load_references(path: Path, wanted_question_uids: set[str]) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in _jsonl(path):
        if row.get("schema_version") != SCHEMA_REFERENCE:
            raise ValueError(f"{row.get('question_uid')} is not an S100 reference row")
        question_uid = str(row["question_uid"])
        if question_uid not in wanted_question_uids:
            continue
        references = row.get("reference_answers")
        support = row.get("support_evidence_ids")
        if not isinstance(references, Sequence) or isinstance(references, (str, bytes)):
            raise ValueError(f"{question_uid} lacks reference answers")
        if not isinstance(support, Sequence) or isinstance(support, (str, bytes)):
            raise ValueError(f"{question_uid} lacks support evidence IDs")
        output[question_uid] = row
    if set(output) != wanted_question_uids:
        raise ValueError("references do not exactly cover S100 questions")
    return output


def _candidate_results(
    path: Path,
    *,
    seed: int,
) -> tuple[dict[str, Mapping[str, object]], Mapping[str, Any]]:
    manifest = _json(path.parent / "run_manifest.json")
    if (
        manifest.get("schema_version") != SCHEMA_RUN_MANIFEST
        or manifest.get("status") != "COMPLETE"
        or manifest.get("seed") != seed
        or manifest.get("arm") != CANDIDATE_ARM
    ):
        raise ValueError(f"invalid S100 seed-{seed} manifest for {path}")
    if manifest.get("generations_sha256") != _sha256(path):
        raise ValueError(f"S100 seed-{seed} generations hash differs")
    rows: dict[str, Mapping[str, object]] = {}
    for row in _jsonl(path):
        if row.get("schema_version") != SCHEMA_GENERATION_ROW:
            raise ValueError(f"{row.get('task_id')} is not an S100 generation row")
        task_id = str(row["task_id"])
        arms = row.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != {CANDIDATE_ARM}:
            raise ValueError(f"S100 task {task_id} has invalid arms")
        value = arms[CANDIDATE_ARM]
        if not isinstance(value, Mapping):
            raise ValueError(f"S100 task {task_id}/{CANDIDATE_ARM} is invalid")
        rows[task_id] = value
    return rows, manifest


def _citation_example(
    *,
    task_id: str,
    run_row: Mapping[str, object],
    evidence: tuple[EvidenceCandidate, ...],
) -> ScoredExample | None:
    generation = GenerationResult.model_validate(run_row["generation"])
    if not generation.answer.strip():
        return None
    routing = run_row.get("routing")
    if not isinstance(routing, list):
        if run_row.get("error"):
            return None
        raise ValueError(f"{task_id} has invalid routing")
    docs = {item.evidence_id: item.text for item in evidence}
    sentences: list[str] = []
    citations: list[tuple[str, ...]] = []
    for item in routing:
        if not isinstance(item, Mapping):
            raise ValueError(f"{task_id} has invalid routing item")
        sentence = strip_annotations(str(item.get("sentence", ""))).strip()
        if not sentence:
            continue
        citation = item.get("citation")
        refs = (str(citation),) if citation is not None and str(citation) in docs else ()
        sentences.append(sentence)
        citations.append(refs)
    if not sentences:
        return None
    return ScoredExample(
        example_id=task_id,
        sentences=tuple(sentences),
        citations=tuple(citations),
        docs=docs,
    )


def _build_minicheck_entailer(
    *,
    model_id: str,
    device: str,
) -> tuple[Callable[[str, str], bool], Callable[[], int]]:
    judge = MiniCheckNLIModel(model_id=model_id)
    if device != "cpu":
        _tokenizer, model = judge._ensure_loaded()
        judge._model = model.to(device)
    cache: dict[tuple[str, str], bool] = {}

    def entails(premise: str, hypothesis: str) -> bool:
        key = (premise, hypothesis)
        if key not in cache:
            cache[key] = (
                judge.classify(premise=premise, hypothesis=hypothesis) == "entailment"
            )
        return cache[key]

    return entails, lambda: len(cache)


def _minicheck_task_metrics(
    *,
    configs: Mapping[str, Mapping[str, Mapping[str, object]]],
    evidence: Mapping[str, tuple[EvidenceCandidate, ...]],
    entails: Callable[[str, str], bool],
) -> dict[str, dict[str, dict[str, float]]]:
    output: dict[str, dict[str, dict[str, float]]] = {}
    for config, rows in configs.items():
        examples = [
            example
            for task_id, run_row in rows.items()
            if (example := _citation_example(task_id=task_id, run_row=run_row, evidence=evidence[task_id]))
            is not None
        ]
        report = compute_citation_metrics(examples, entails)
        by_id = {str(item["example_id"]): item for item in report.per_example}
        for task_id in evidence:
            item = by_id.get(task_id)
            output.setdefault(task_id, {})[config] = {
                "minicheck_citation_precision": float(item["citation_prec"]) if item else 0.0,
                "minicheck_citation_recall": float(item["citation_rec"]) if item else 0.0,
                "minicheck_sentences": float(item["sentences"]) if item else 0.0,
                "minicheck_citations": float(item["citations"]) if item else 0.0,
            }
    return output


def _config_metrics(
    *,
    run_row: Mapping[str, object],
    references: Sequence[str],
    evidence_count: int,
    minicheck_metrics: Mapping[str, float],
) -> dict[str, float]:
    generation = GenerationResult.model_validate(run_row["generation"])
    error = bool(run_row.get("error"))
    answer = strip_annotations(generation.answer)
    coverage = float(bool(answer.strip()) and not g310._is_unknown(answer))
    value = answer_match(answer, references).value
    matched = float(value) if value is not None else 0.0
    invalid = g310._draft_invalid_citations(run_row.get("trace"), evidence_count)
    minicheck_precision = float(minicheck_metrics["minicheck_citation_precision"])
    minicheck_recall = float(minicheck_metrics["minicheck_citation_recall"])
    return {
        "runtime_error": float(error),
        "missing_trace": float(run_row.get("trace") is None and not error),
        "draft_invalid_citation": float(invalid > 0),
        "answer_match": matched,
        "coverage": coverage,
        "final_empty": float(coverage == 0.0),
        "minicheck_citation_precision": minicheck_precision,
        "minicheck_citation_recall": minicheck_recall,
        "minicheck_sentences": float(minicheck_metrics["minicheck_sentences"]),
        "minicheck_citations": float(minicheck_metrics["minicheck_citations"]),
        "correct_and_cited": float(
            matched == 1.0
            and coverage == 1.0
            and invalid == 0
            and minicheck_precision >= 1.0
            and minicheck_recall >= 1.0
        ),
    }


def _effect_label(
    full: Mapping[str, float],
    drop: Mapping[str, float],
    *,
    dropped_is_support: bool,
) -> Label:
    if max(
        full.get("runtime_error", 0.0),
        full.get("missing_trace", 0.0),
        drop.get("runtime_error", 0.0),
        drop.get("missing_trace", 0.0),
    ) > 0.0:
        return "UNCERTAIN"
    metrics = (
        "correct_and_cited",
        "answer_match",
        "coverage",
        "minicheck_citation_precision",
        "minicheck_citation_recall",
    )
    deltas = {metric: float(drop.get(metric, 0.0)) - float(full.get(metric, 0.0)) for metric in metrics}
    degraded = any(value < -1e-12 for value in deltas.values())
    improved = any(value > 1e-12 for value in deltas.values())
    if degraded and improved:
        return "UNCERTAIN"
    if degraded:
        return "MUST_KEEP"
    if improved:
        return "SAFE_DROP"
    return "NEUTRAL" if dropped_is_support else "SAFE_DROP"


def _combine_seed_labels(labels: Sequence[Label], *, dropped_is_support: bool) -> Label:
    if not labels or "UNCERTAIN" in labels:
        return "UNCERTAIN"
    unique = set(labels)
    if len(unique) == 1:
        return labels[0]
    strong = unique & {"MUST_KEEP", "SAFE_DROP"}
    if len(strong) > 1:
        return "UNCERTAIN"
    if strong == {"SAFE_DROP"} and not dropped_is_support and unique <= {"SAFE_DROP", "NEUTRAL"}:
        return "SAFE_DROP"
    return "UNCERTAIN"


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarize_metrics(
    *,
    tasks: Sequence[S100Task],
    per_task: Mapping[str, Mapping[str, Mapping[str, float]]],
    configs: Sequence[str],
) -> dict[str, dict[str, dict[str, float | int]]]:
    groups: dict[str, list[str]] = {
        "all": [task.task_id for task in tasks],
        "full": [task.task_id for task in tasks if task.variant_type == "full"],
        "leave_one_out": [task.task_id for task in tasks if task.variant_type == "leave_one_out"],
    }
    for dataset in sorted({task.dataset for task in tasks}):
        groups[f"dataset:{dataset}"] = [task.task_id for task in tasks if task.dataset == dataset]
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    for group, task_ids in groups.items():
        output[group] = {}
        for config in configs:
            metric_names = sorted(
                {metric for task_id in task_ids for metric in per_task[task_id][config]}
            )
            output[group][config] = {
                "tasks": len(task_ids),
                **{
                    metric: _mean(
                        [float(per_task[task_id][config][metric]) for task_id in task_ids]
                    )
                    for metric in metric_names
                },
            }
    return output


def _label_rows(
    *,
    tasks: Sequence[S100Task],
    references: Mapping[str, Mapping[str, Any]],
    per_task: Mapping[str, Mapping[str, Mapping[str, float]]],
    configs: Sequence[str],
) -> tuple[list[dict[str, object]], list[str]]:
    by_question: dict[str, list[S100Task]] = defaultdict(list)
    for task in tasks:
        by_question[task.question_uid].append(task)
    labels: list[dict[str, object]] = []
    incomplete: list[str] = []
    for question_uid, question_tasks in sorted(by_question.items()):
        full = next((task for task in question_tasks if task.variant_type == "full"), None)
        drops = sorted(
            [task for task in question_tasks if task.variant_type == "leave_one_out"],
            key=lambda task: int(task.dropped_rank or 0),
        )
        reference = references[question_uid]
        full_evidence_ids = [str(item) for item in reference.get("full_evidence_ids", [])]
        if full is None or len(drops) != len(full_evidence_ids):
            incomplete.append(question_uid)
            continue
        support = {str(item) for item in reference.get("support_evidence_ids", [])}
        for drop in drops:
            if drop.dropped_evidence_id is None or drop.dropped_rank is None:
                continue
            dropped_is_support = drop.dropped_evidence_id in support
            seed_labels: dict[str, str] = {}
            metric_deltas: dict[str, dict[str, float]] = {}
            for config in configs:
                full_metrics = per_task[full.task_id][config]
                drop_metrics = per_task[drop.task_id][config]
                label = _effect_label(
                    full_metrics,
                    drop_metrics,
                    dropped_is_support=dropped_is_support,
                )
                seed_labels[config] = label
                metric_deltas[config] = {
                    metric: float(drop_metrics.get(metric, 0.0)) - float(full_metrics.get(metric, 0.0))
                    for metric in (
                        "correct_and_cited",
                        "answer_match",
                        "coverage",
                        "minicheck_citation_precision",
                        "minicheck_citation_recall",
                    )
                }
            final_label = _combine_seed_labels(
                [cast(Label, value) for value in seed_labels.values()],
                dropped_is_support=dropped_is_support,
            )
            labels.append(
                {
                    "schema_version": SCHEMA_LABEL_ROW,
                    "question_uid": question_uid,
                    "dataset": drop.dataset,
                    "case_id": drop.case_id,
                    "query_id": drop.query_id,
                    "component_id": drop.component_id,
                    "context_variant": CONTEXT_VARIANT,
                    "evidence_id": drop.dropped_evidence_id,
                    "rank": drop.dropped_rank,
                    "is_official_support": dropped_is_support,
                    "label": final_label,
                    "seed_labels": seed_labels,
                    "metric_deltas_drop_minus_full": metric_deltas,
                }
            )
    return labels, incomplete


def _label_summary(labels: Sequence[Mapping[str, object]]) -> dict[str, object]:
    counts = Counter(str(row["label"]) for row in labels)
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    by_role: dict[str, Counter[str]] = defaultdict(Counter)
    by_rank_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    for row in labels:
        label = str(row["label"])
        by_dataset[str(row["dataset"])][label] += 1
        by_role["support" if row.get("is_official_support") else "distractor"][label] += 1
        rank = int(row.get("rank", 0))
        bucket = "rank_1_2" if rank <= 2 else "rank_3_5" if rank <= 5 else "rank_6_10"
        by_rank_bucket[bucket][label] += 1
    total = len(labels)
    utility = counts["MUST_KEEP"] + counts["SAFE_DROP"]
    stable = total - counts["UNCERTAIN"]
    uncertain_rate = counts["UNCERTAIN"] / total if total else 1.0
    datasets_with_utility = sorted(
        dataset
        for dataset, dataset_counts in by_dataset.items()
        if dataset_counts["MUST_KEEP"] + dataset_counts["SAFE_DROP"] > 0
    )
    return {
        "total_labels": total,
        "label_counts": dict(counts),
        "label_counts_by_dataset": {key: dict(value) for key, value in by_dataset.items()},
        "label_counts_by_role": {key: dict(value) for key, value in by_role.items()},
        "label_counts_by_rank_bucket": {key: dict(value) for key, value in by_rank_bucket.items()},
        "stable_label_rate": stable / total if total else 0.0,
        "utility_label_rate": utility / total if total else 0.0,
        "uncertain_rate": uncertain_rate,
        "utility_labels": utility,
        "datasets_with_utility": datasets_with_utility,
    }


def _recommendation(
    *,
    technical_failures: Sequence[str],
    incomplete_questions: Sequence[str],
    label_summary: Mapping[str, object],
) -> str:
    if technical_failures:
        return "STOP_TECHNICAL_FAILURE"
    if incomplete_questions:
        return "NO_DECISION_INCOMPLETE_QUESTIONS"
    if int(label_summary.get("utility_labels", 0)) < MIN_UTILITY_LABELS_FOR_S110:
        return "STOP_UTILITY_LABELS_TOO_SPARSE"
    if float(label_summary.get("uncertain_rate", 1.0)) > MAX_UNCERTAIN_RATE_FOR_S110:
        return "STOP_LABELS_TOO_UNSTABLE"
    if len(label_summary.get("datasets_with_utility", [])) < MIN_DATASETS_WITH_UTILITY_FOR_S110:
        return "STOP_SIGNAL_NOT_CROSS_DATA"
    return "S110_READY"


def _write_report(path: Path, report: Mapping[str, object]) -> None:
    summary = cast(Mapping[str, object], report["label_summary"])
    lines = [
        "# S100 Utility Pilot Report",
        "",
        f"**Status:** `{report['status']}`",
        f"**S110 recommendation:** `{report['s110_recommendation']}`",
        f"**Questions:** {report['questions']}",
        f"**Generation tasks:** {report['tasks']}",
        f"**Labels:** {summary.get('total_labels')}",
        "",
        "## Label Summary",
        "",
        "| label | count |",
        "|---|---:|",
    ]
    counts = cast(Mapping[str, object], summary.get("label_counts", {}))
    for label in ("MUST_KEEP", "SAFE_DROP", "NEUTRAL", "UNCERTAIN"):
        lines.append(f"| {label} | {counts.get(label, 0)} |")
    lines += [
        "",
        f"- stable label rate: {float(summary.get('stable_label_rate', 0.0)):.4f}",
        f"- utility label rate: {float(summary.get('utility_label_rate', 0.0)):.4f}",
        f"- uncertain rate: {float(summary.get('uncertain_rate', 0.0)):.4f}",
        f"- datasets with utility: {', '.join(cast(Sequence[str], summary.get('datasets_with_utility', [])))}",
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
    output_manifest: Path | None = None,
    minicheck_model_id: str = FROZEN_MINICHECK_MODEL,
    minicheck_device: str = "cpu",
    entails: Callable[[str, str], bool] | None = None,
    allow_subset: bool = False,
) -> dict[str, object]:
    if set(seed_generations) != set(ALLOWED_SEEDS):
        raise ValueError("S100 scoring requires seeds 13, 42, and 73")
    tasks = _load_tasks(tasks_path)
    wanted = {task.task_id for task in tasks}
    candidate_rows: dict[int, dict[str, Mapping[str, object]]] = {}
    candidate_task_ids: set[str] | None = None
    candidate_manifests: dict[str, Mapping[str, Any]] = {}
    for seed, path in sorted(seed_generations.items()):
        rows, manifest = _candidate_results(path, seed=seed)
        row_ids = set(rows)
        if candidate_task_ids is None:
            candidate_task_ids = row_ids
        elif row_ids != candidate_task_ids:
            raise ValueError("S100 candidate seed task sets differ")
        candidate_rows[seed] = rows
        candidate_manifests[f"seed{seed}"] = manifest
    if candidate_task_ids is None:
        raise ValueError("S100 candidate rows are empty")
    if candidate_task_ids != wanted:
        if not allow_subset:
            raise ValueError("S100 candidates do not exactly cover prepared tasks")
        if not candidate_task_ids < wanted:
            raise ValueError("S100 subset scoring requires a strict prepared-task subset")
        tasks = [task for task in tasks if task.task_id in candidate_task_ids]
        wanted = candidate_task_ids
    wanted_questions = {task.question_uid for task in tasks}
    references = _load_references(references_path, wanted_questions)
    configs: dict[str, dict[str, Mapping[str, object]]] = {
        f"GQ{seed}": candidate_rows[seed] for seed in sorted(seed_generations)
    }
    config_names = tuple(configs)
    evidence = {task.task_id: task.evidence for task in tasks}
    if entails is None:
        entails, judge_call_count = _build_minicheck_entailer(
            model_id=minicheck_model_id,
            device=minicheck_device,
        )
    else:
        cache: dict[tuple[str, str], bool] = {}
        raw_entails = entails

        def cached_entails(premise: str, hypothesis: str) -> bool:
            key = (premise, hypothesis)
            if key not in cache:
                cache[key] = raw_entails(premise, hypothesis)
            return cache[key]

        entails = cached_entails
        judge_call_count = lambda: len(cache)
    minicheck_by_task = _minicheck_task_metrics(
        configs=configs,
        evidence=evidence,
        entails=entails,
    )
    per_task: dict[str, dict[str, dict[str, float]]] = {}
    scored_rows: list[dict[str, object]] = []
    for task in tasks:
        refs = references[task.question_uid]["reference_answers"]
        if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)):
            raise ValueError(f"{task.question_uid} lacks references")
        per_task[task.task_id] = {
            config: _config_metrics(
                run_row=configs[config][task.task_id],
                references=[str(item) for item in refs],
                evidence_count=len(task.evidence),
                minicheck_metrics=minicheck_by_task[task.task_id][config],
            )
            for config in config_names
        }
        scored_rows.append(
            {
                "schema_version": SCHEMA_SCORE_ROW,
                **_task_identity(task),
                "arms": per_task[task.task_id],
            }
        )
    labels, incomplete_questions = _label_rows(
        tasks=tasks,
        references=references,
        per_task=per_task,
        configs=config_names,
    )
    label_summary = _label_summary(labels)
    technical_failures = sorted(
        {
            f"{config}:{task_id}:{metric}"
            for task_id, values in per_task.items()
            for config, metrics in values.items()
            for metric in ("runtime_error", "missing_trace", "draft_invalid_citation")
            if float(metrics.get(metric, 0.0)) > 0.0
        }
    )
    recommendation = _recommendation(
        technical_failures=technical_failures,
        incomplete_questions=incomplete_questions,
        label_summary=label_summary,
    )
    status = "S100_TECHNICAL_FAIL" if technical_failures else "S100_PILOT_COMPLETE"
    report: dict[str, object] = {
        "schema_version": SCHEMA_SCORE_REPORT,
        "stage": "S100",
        "status": status,
        "s110_recommendation": recommendation,
        "questions": len({task.question_uid for task in tasks}),
        "tasks": len(tasks),
        "labels": len(labels),
        "configs": list(config_names),
        "aggregate": _summarize_metrics(tasks=tasks, per_task=per_task, configs=config_names),
        "label_summary": label_summary,
        "incomplete_questions": incomplete_questions,
        "technical_failures": technical_failures,
        "candidate_manifests": candidate_manifests,
        "continuation_rule": {
            "purpose": "minimum pilot signal floor, not a final-success target",
            "min_utility_labels_for_s110": MIN_UTILITY_LABELS_FOR_S110,
            "max_uncertain_rate_for_s110": MAX_UNCERTAIN_RATE_FOR_S110,
            "min_datasets_with_utility_for_s110": MIN_DATASETS_WITH_UTILITY_FOR_S110,
        },
        "citation_judge": {
            "model": "MiniCheck-Flan-T5-Large",
            "model_id": minicheck_model_id,
            "device": minicheck_device,
            "unique_judge_calls": judge_call_count(),
            "production_true_excluded_from_judging": True,
        },
        "source_sha256": {
            "script": _sha256(Path(__file__)),
            "tasks": _sha256(tasks_path),
            "references": _sha256(references_path),
            **{
                f"seed{seed}_generations": _sha256(path)
                for seed, path in sorted(seed_generations.items())
            },
        },
        "boundaries": {
            "generation_runtime_reference_answers_loaded": False,
            "generation_runtime_support_provenance_loaded": False,
            "score_uses_references_after_generation": True,
            "sealed_or_heldout_read": False,
            "selector_training_started": False,
            "full_system_started": False,
            "gq_changed": False,
        },
    }
    _write_jsonl(output_rows, scored_rows)
    _write_jsonl(output_labels, labels)
    _write_json(output_json, report)
    _write_report(output_report, report)
    if output_manifest is not None:
        manifest = {
            "schema_version": SCHEMA_SCORE_MANIFEST,
            "stage": "S100_SCORE",
            "status": "COMPLETE",
            "s100_status": status,
            "s110_recommendation": recommendation,
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
    prepare_parser.add_argument("--ordered-ids", required=True, type=Path)
    prepare_parser.add_argument("--output-dir", required=True, type=Path)
    prepare_parser.add_argument("--niah-questions", type=int, default=DEFAULT_NIAH_QUESTIONS)
    prepare_parser.add_argument("--twowiki-questions", type=int, default=DEFAULT_TWOWIKI_QUESTIONS)
    prepare_parser.add_argument("--sample-seed", default=DEFAULT_SAMPLE_SEED)

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
    score_parser.add_argument("--output-manifest", type=Path)
    score_parser.add_argument("--minicheck-model-id", default=FROZEN_MINICHECK_MODEL)
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
                    ordered_ids_path=args.ordered_ids,
                    output_dir=args.output_dir,
                    niah_questions=args.niah_questions,
                    twowiki_questions=args.twowiki_questions,
                    sample_seed=args.sample_seed,
                ),
                ensure_ascii=False,
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
                ensure_ascii=False,
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
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
