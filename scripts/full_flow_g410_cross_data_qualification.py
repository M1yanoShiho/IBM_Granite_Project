"""Run and score G410 cross-data Generator qualification.

G410 starts with the registered 2Wiki internal cross-data screen from the G223
validation bundle.  The generation command consumes a prepared no-answer task
packet; references are read only by ``score`` after generation has completed.

This stage is not GQ freeze, not Selector utility materialization, and not
held-out.  The ASQA/QAMPARI revealed citation guard remains pending for G420
unless a separate pre-registered G410 guard packet is added.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from evidence_rag.generator.granite import (
    GraniteGenerationConfig,
    PeftGraniteLLMClient,
)
from evidence_rag.generator.nli import MiniCheckNLIModel, TrueNLIModel

ALLOWED_SEEDS = g400.ALLOWED_SEEDS
CANDIDATE_ARM = "GR-C"
CONTEXTS = ("topk", "support_only", "support_first", "support_middle", "support_last")
ALL_CONTEXT = "all_2wiki"
EXPECTED_TWOWIKI_CASES = 95
EXPECTED_TWOWIKI_TASKS = EXPECTED_TWOWIKI_CASES * len(CONTEXTS)
EXPECTED_TRAIN_GROUPS = g400.EXPECTED_TRAIN_GROUPS
EXPECTED_TRAINING_EXAMPLES = g400.EXPECTED_TRAINING_EXAMPLES
EXPECTED_OPTIMIZER_STEPS = g400.EXPECTED_OPTIMIZER_STEPS
EXPECTED_VALIDATION_GROUPS = g400.EXPECTED_VALIDATION_GROUPS
EXPECTED_TWOWIKI_MODELVAL_GROUPS = g400.EXPECTED_TWOWIKI_MODELVAL_GROUPS

SCHEMA_PREPARE_MANIFEST = "full-flow-g410-prepare-manifest-v1"
SCHEMA_TASK = "full-flow-g410-task-v1"
SCHEMA_REFERENCE = "full-flow-g410-reference-v1"
SCHEMA_RUN_SPEC = "full-flow-g410-run-spec-v1"
SCHEMA_RUN_MANIFEST = "full-flow-g410-run-manifest-v1"
SCHEMA_GENERATION_ROW = "full-flow-g410-generation-row-v1"
SCHEMA_SCORE_REPORT = "full-flow-g410-score-report-v1"
SCHEMA_SCORE_ROW = "full-flow-g410-scored-row-v1"
FROZEN_MINICHECK_MODEL = g400.FROZEN_MINICHECK_MODEL
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class G410Task:
    task_id: str
    scope: str
    dataset: str
    case_id: str
    query_id: str
    component_id: str
    context: str
    variant_name: str
    question: str
    target_kind: str
    evidence: tuple[EvidenceCandidate, ...]
    support_evidence_ids: tuple[str, ...]


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


def _runtime_environment() -> dict[str, object]:
    torch = __import__("torch")
    transformers = __import__("transformers")
    peft = __import__("peft")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
    }


def _task_identity(task: G410Task) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "scope": task.scope,
        "dataset": task.dataset,
        "case_id": task.case_id,
        "query_id": task.query_id,
        "component_id": task.component_id,
        "context": task.context,
        "variant_name": task.variant_name,
        "target_kind": task.target_kind,
        "evidence_ids": [item.evidence_id for item in task.evidence],
        "support_evidence_ids": list(task.support_evidence_ids),
    }


def _task_row(task: G410Task) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_TASK,
        **_task_identity(task),
        "question": task.question,
        "evidence": [item.model_dump(mode="json") for item in task.evidence],
    }


def _task_from_row(row: Mapping[str, Any]) -> G410Task:
    if row.get("schema_version") != SCHEMA_TASK:
        raise ValueError(f"{row.get('task_id')} is not a G410 task row")
    forbidden = {"answer", "official_answer", "reference_answers", "gold"}
    if forbidden & set(row):
        raise ValueError(f"{row.get('task_id')} contains reference fields")
    raw_evidence = row.get("evidence")
    if not isinstance(raw_evidence, Sequence) or isinstance(raw_evidence, (str, bytes)):
        raise ValueError(f"{row.get('task_id')} has invalid evidence")
    return G410Task(
        task_id=str(row["task_id"]),
        scope=str(row["scope"]),
        dataset=str(row["dataset"]),
        case_id=str(row["case_id"]),
        query_id=str(row["query_id"]),
        component_id=str(row["component_id"]),
        context=str(row["context"]),
        variant_name=str(row["variant_name"]),
        question=str(row["question"]),
        target_kind=str(row.get("target_kind", "")),
        evidence=tuple(EvidenceCandidate.model_validate(item) for item in raw_evidence),
        support_evidence_ids=tuple(str(item) for item in row.get("support_evidence_ids", []) or []),
    )


def _reference_row(task: G410Task, references: Sequence[str]) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_REFERENCE,
        "task_id": task.task_id,
        "query_id": task.query_id,
        "case_id": task.case_id,
        "reference_answers": sorted({item for item in references if item}),
        "support_evidence_ids": list(task.support_evidence_ids),
    }


def build_2wiki_tasks(
    validation_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[G410Task], list[dict[str, object]]]:
    tasks: list[G410Task] = []
    references: list[dict[str, object]] = []
    for row in validation_rows:
        if row.get("dataset") != "2wiki":
            continue
        if not bool(row.get("answerable")):
            raise ValueError("G410 expects 2Wiki validation rows to be answerable")
        variants = row.get("variants")
        if not isinstance(variants, Mapping):
            raise ValueError(f"{row.get('case_id')} lacks variants")
        refs = [
            str(row.get("answer", "")).strip(),
            str(row.get("official_answer", "")).strip(),
        ]
        refs = [item for item in refs if item]
        if not refs:
            raise ValueError(f"{row.get('case_id')} lacks references")
        for variant_name, raw_variant in variants.items():
            if str(variant_name) not in CONTEXTS:
                raise ValueError(f"unexpected G410 2Wiki variant {variant_name}")
            if not isinstance(raw_variant, Mapping):
                raise ValueError(f"{row.get('case_id')}/{variant_name} is invalid")
            prompt = str(raw_variant.get("prompt", ""))
            task = G410Task(
                task_id=f"{g310.ANSWERABLE_SCOPE}::{row['case_id']}::{variant_name}",
                scope="validation-2wiki-answerable",
                dataset="2wiki",
                case_id=str(row["case_id"]),
                query_id=str(row.get("query_id", row["case_id"])),
                component_id=str(row.get("component_id", row["case_id"])),
                context=str(variant_name),
                variant_name=str(variant_name),
                question=str(row.get("question") or g310._extract_question(prompt)),
                target_kind=str(row.get("target_kind", "")),
                evidence=g310.parse_prompt_evidence(
                    prompt=prompt,
                    case_id=str(row["case_id"]),
                    variant_name=str(variant_name),
                ),
                support_evidence_ids=g310._support_ids(raw_variant),
            )
            tasks.append(task)
            references.append(_reference_row(task, refs))
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("duplicate G410 task ID")
    return tasks, references


def prepare(
    *,
    data_manifest_path: Path,
    train_cases_path: Path,
    validation_cases_path: Path,
    ordered_ids_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    g223 = g310._load_g223_manifest(
        data_manifest_path=data_manifest_path,
        train_cases_path=train_cases_path,
        validation_cases_path=validation_cases_path,
        ordered_ids_path=ordered_ids_path,
    )
    validation_rows = _jsonl(validation_cases_path)
    tasks, references = build_2wiki_tasks(validation_rows)
    if len(tasks) != EXPECTED_TWOWIKI_TASKS:
        raise ValueError("G410 2Wiki task count differs from frozen G223 screen")
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = output_dir / "tasks.jsonl"
    references_path = output_dir / "references.jsonl"
    _write_jsonl(tasks_path, [_task_row(task) for task in tasks])
    _write_jsonl(references_path, references)
    manifest = {
        "schema_version": SCHEMA_PREPARE_MANIFEST,
        "stage": "G410_PREPARE",
        "status": "COMPLETE",
        "source": "G223 validation 2Wiki answerable model-val screen",
        "g223_manifest_sha256": _sha256(data_manifest_path),
        "g223_status": g223.get("status"),
        "twowiki_modelval_groups": EXPECTED_TWOWIKI_MODELVAL_GROUPS,
        "tasks": len(tasks),
        "case_groups": EXPECTED_TWOWIKI_CASES,
        "contexts": list(CONTEXTS),
        "tasks_sha256": _sha256(tasks_path),
        "references_sha256": _sha256(references_path),
        "task_identity_sha256": _sha256_bytes(
            _canonical_bytes([_task_identity(task) for task in tasks])
        ),
        "source_sha256": {
            "train_cases": _sha256(train_cases_path),
            "validation_cases": _sha256(validation_cases_path),
            "ordered_ids": _sha256(ordered_ids_path),
        },
        "generation_runtime_gold_or_reference_fields": False,
        "sealed_or_heldout_read": False,
        "utility_labels_started": False,
        "citation_regression_guard": "PENDING_G420_REVEALED_ASQA_QAMPARI_BOUNDARY_CONFIRMATION",
    }
    _write_json(output_dir / "prepare_manifest.json", manifest)
    return manifest


def _load_tasks(tasks_path: Path) -> list[G410Task]:
    tasks = [_task_from_row(row) for row in _jsonl(tasks_path)]
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("duplicate G410 task rows")
    return tasks


def _validate_prepare_manifest(path: Path, tasks_path: Path) -> Mapping[str, Any]:
    manifest = _json(path)
    if manifest.get("schema_version") != SCHEMA_PREPARE_MANIFEST:
        raise ValueError("G410 requires a G410 prepare manifest")
    if manifest.get("status") != "COMPLETE":
        raise ValueError("G410 prepare manifest is not complete")
    if manifest.get("tasks_sha256") != _sha256(tasks_path):
        raise ValueError("G410 task packet hash mismatch")
    if manifest.get("sealed_or_heldout_read") is not False:
        raise ValueError("G410 prepare must not read sealed/held-out data")
    return manifest


def validate_resume_prefix(rows: Sequence[Mapping[str, Any]], tasks: Sequence[G410Task]) -> None:
    if len(rows) > len(tasks):
        raise ValueError("G410 resume has more rows than frozen tasks")
    for index, row in enumerate(rows):
        task = tasks[index]
        if row.get("task_id") != task.task_id:
            raise ValueError(f"G410 resume diverges at row {index + 1}")
        arms = row.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != {CANDIDATE_ARM}:
            raise ValueError(f"G410 resume row {index + 1} has wrong arms")


def run(
    *,
    seed: int,
    tasks_path: Path,
    prepare_manifest_path: Path,
    model_snapshot: Path,
    true_snapshot: Path,
    grc_run_dir: Path,
    output_dir: Path,
    limit_tasks: int | None = None,
) -> dict[str, object]:
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"G410 seed must be one of {ALLOWED_SEEDS}")
    prepare_manifest = _validate_prepare_manifest(prepare_manifest_path, tasks_path)
    tasks = _load_tasks(tasks_path)
    formal_task_count = len(tasks)
    if limit_tasks is None:
        if formal_task_count != EXPECTED_TWOWIKI_TASKS:
            raise ValueError("formal G410 task count differs from frozen protocol")
    else:
        if limit_tasks <= 0:
            raise ValueError("--limit-tasks must be positive")
        tasks = tasks[:limit_tasks]
    model_config_sha256 = _sha256(model_snapshot / "config.json")
    adapter_audit = g400._validate_g330_run(
        grc_run_dir,
        seed=seed,
        model_config_sha256=model_config_sha256,
    )
    spec: dict[str, object] = {
        "schema_version": SCHEMA_RUN_SPEC,
        "stage": "G410",
        "seed": seed,
        "arms": [CANDIDATE_ARM],
        "formal": limit_tasks is None,
        "tasks": len(tasks),
        "formal_task_count": formal_task_count,
        "contexts": list(CONTEXTS),
        "prepare_manifest_sha256": _sha256(prepare_manifest_path),
        "prepare_task_identity_sha256": prepare_manifest.get("task_identity_sha256"),
        "task_identity_sha256": _sha256_bytes(
            _canonical_bytes([_task_identity(task) for task in tasks])
        ),
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
        "sealed_or_heldout_read": False,
        "utility_labels_started": False,
        "adapter_scope": "draft generation call only",
        "claim_splitter_scope": "frozen Granite base with adapters disabled",
        "true_scope": "frozen verifier",
        "runtime_environment": _runtime_environment(),
        "git_commit": _git_commit(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / "run_spec.json"
    generations_path = output_dir / "generations.jsonl"
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        raise ValueError(f"G410 run is already complete: {manifest_path}")
    if spec_path.exists():
        if _json(spec_path) != spec:
            raise ValueError("G410 resume specification differs from frozen run")
    else:
        if generations_path.exists() and generations_path.stat().st_size:
            raise ValueError("G410 generations exist without a run specification")
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
            if index % 10 == 0 or index == len(tasks):
                elapsed = time.perf_counter() - started
                completed_now = index - len(existing)
                print(
                    f"[G410 seed-{seed}] {index}/{len(tasks)} "
                    f"{elapsed / max(completed_now, 1):.2f}s/new-task",
                    flush=True,
                )
    rows = _jsonl(generations_path)
    validate_resume_prefix(rows, tasks)
    manifest = {
        "schema_version": SCHEMA_RUN_MANIFEST,
        "stage": "G410",
        "status": "COMPLETE",
        "seed": seed,
        "arms": [CANDIDATE_ARM],
        "formal": limit_tasks is None,
        "tasks": len(tasks),
        "contexts": list(CONTEXTS),
        "generations_sha256": _sha256(generations_path),
        "run_spec_sha256": _sha256(spec_path),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "gold_loaded_at_runtime": False,
        "reference_answers_loaded_at_runtime": False,
        "sealed_or_heldout_read": False,
        "utility_labels_started": False,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _load_references(path: Path, wanted: set[str]) -> dict[str, tuple[str, ...]]:
    output: dict[str, tuple[str, ...]] = {}
    for row in _jsonl(path):
        if row.get("schema_version") != SCHEMA_REFERENCE:
            raise ValueError(f"{row.get('task_id')} is not a G410 reference row")
        task_id = str(row["task_id"])
        if task_id not in wanted:
            continue
        references = row.get("reference_answers")
        if not isinstance(references, Sequence) or isinstance(references, (str, bytes)):
            raise ValueError(f"{task_id} lacks reference answers")
        output[task_id] = tuple(str(item) for item in references)
    if set(output) != wanted:
        raise ValueError("references do not exactly cover G410 tasks")
    return output


def _baseline_g0(path: Path, wanted: set[str]) -> dict[str, Mapping[str, object]]:
    rows: dict[str, Mapping[str, object]] = {}
    for row in _jsonl(path):
        task_id = str(row.get("task_id", ""))
        if task_id not in wanted:
            continue
        arms = row.get("arms")
        if not isinstance(arms, Mapping) or not isinstance(arms.get("G0"), Mapping):
            raise ValueError(f"G410 baseline {task_id} lacks G0")
        rows[task_id] = arms["G0"]
    if set(rows) != wanted:
        raise ValueError("G410 baseline G0 does not exactly cover prepared tasks")
    return rows


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
        or tuple(manifest.get("arms", ())) != (CANDIDATE_ARM,)
    ):
        raise ValueError(f"invalid G410 seed-{seed} manifest for {path}")
    if manifest.get("generations_sha256") != _sha256(path):
        raise ValueError(f"G410 seed-{seed} generations hash differs")
    rows: dict[str, Mapping[str, object]] = {}
    for row in _jsonl(path):
        if row.get("schema_version") != SCHEMA_GENERATION_ROW:
            raise ValueError(f"{row.get('task_id')} is not a G410 generation row")
        task_id = str(row["task_id"])
        arms = row.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != {CANDIDATE_ARM}:
            raise ValueError(f"candidate task {task_id} has invalid arms")
        value = arms[CANDIDATE_ARM]
        if not isinstance(value, Mapping):
            raise ValueError(f"candidate task {task_id}/{CANDIDATE_ARM} is invalid")
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
    if isinstance(run_row.get("routing"), list):
        routing = run_row["routing"]
    else:
        routing = g400._routing_from_trace(run_row)
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


def _minicheck_task_metrics(
    *,
    configs: Mapping[str, Mapping[str, Mapping[str, object]]],
    evidence: Mapping[str, tuple[EvidenceCandidate, ...]],
    entails: Callable[[str, str], bool],
) -> dict[str, dict[str, dict[str, float]]]:
    output: dict[str, dict[str, dict[str, float]]] = {}
    for name, rows in configs.items():
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
            output.setdefault(task_id, {})[name] = {
                "minicheck_citation_precision": float(item["citation_prec"]) if item else 0.0,
                "minicheck_citation_recall": float(item["citation_rec"]) if item else 0.0,
                "minicheck_sentences": float(item["sentences"]) if item else 0.0,
                "minicheck_citations": float(item["citations"]) if item else 0.0,
            }
    return output


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


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate(
    *,
    metadata: Mapping[str, Mapping[str, object]],
    per_task: Mapping[str, Mapping[str, Mapping[str, float]]],
    configs: Sequence[str],
) -> dict[str, dict[str, dict[str, float | int]]]:
    groups = (ALL_CONTEXT, *CONTEXTS)
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    for context in groups:
        task_ids = [
            task_id
            for task_id, item in metadata.items()
            if context == ALL_CONTEXT or str(item["context"]) == context
        ]
        if not task_ids:
            continue
        output[context] = {}
        for config in configs:
            metric_names = sorted(
                {
                    metric
                    for task_id in task_ids
                    for metric in per_task[task_id][config]
                }
            )
            output[context][config] = {
                "tasks": len(task_ids),
                **{
                    metric: _mean(
                        [float(per_task[task_id][config][metric]) for task_id in task_ids]
                    )
                    for metric in metric_names
                },
            }
    return output


def _delta(after: Mapping[str, float | int], before: Mapping[str, float | int], metric: str) -> float:
    return float(after.get(metric, 0.0)) - float(before.get(metric, 0.0))


def _deltas(
    aggregate: Mapping[str, Mapping[str, Mapping[str, float | int]]],
    *,
    configs: Sequence[str],
) -> dict[str, dict[str, dict[str, float]]]:
    metrics = (
        "answer_match",
        "coverage",
        "final_empty",
        "minicheck_citation_precision",
        "minicheck_citation_recall",
        "correct_and_cited",
    )
    output: dict[str, dict[str, dict[str, float]]] = {}
    for config in configs:
        if config == "G0":
            continue
        output[config] = {}
        for context, values in aggregate.items():
            output[config][context] = {
                metric: _delta(values[config], values["G0"], metric)
                for metric in metrics
            }
    return output


def _family_deltas(
    seed_deltas: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    seed_names = sorted(seed_deltas)
    for context in (ALL_CONTEXT, *CONTEXTS):
        present = [name for name in seed_names if context in seed_deltas[name]]
        if not present:
            continue
        metrics = sorted(
            {metric for name in present for metric in seed_deltas[name][context]}
        )
        output[context] = {
            metric: _mean([seed_deltas[name][context][metric] for name in present])
            for metric in metrics
        }
    return output


def _gate_decision(
    *,
    aggregate: Mapping[str, Mapping[str, Mapping[str, float | int]]],
    seed_deltas: Mapping[str, Mapping[str, Mapping[str, float]]],
    family_deltas: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    candidate_configs = sorted(seed_deltas)
    technical_failures: list[str] = []
    for context, values in aggregate.items():
        for config in candidate_configs:
            item = values[config]
            if float(item.get("runtime_error", 0.0)) > 0:
                technical_failures.append(f"{config}:{context}:runtime_error")
            if float(item.get("missing_trace", 0.0)) > 0:
                technical_failures.append(f"{config}:{context}:missing_trace")
            if float(item.get("draft_invalid_citation", 0.0)) > 0:
                technical_failures.append(f"{config}:{context}:invalid_citation")
    all_family = family_deltas.get(ALL_CONTEXT, {})
    responsibility = {
        "twowiki_correct_and_cited_delta_ge_minus_2pp": all_family.get(
            "correct_and_cited",
            0.0,
        )
        >= -0.02,
        "answer_delta_ge_minus_3pp": all_family.get("answer_match", 0.0) >= -0.03,
        "coverage_delta_ge_minus_2pp": all_family.get("coverage", 0.0) >= -0.02,
        "citation_precision_delta_ge_minus_3pp": all_family.get(
            "minicheck_citation_precision",
            0.0,
        )
        >= -0.03,
        "citation_recall_delta_ge_minus_3pp": all_family.get(
            "minicheck_citation_recall",
            0.0,
        )
        >= -0.03,
        "at_least_2_of_3_seeds_correct_and_cited_nonnegative": sum(
            seed_deltas[config].get(ALL_CONTEXT, {}).get("correct_and_cited", 0.0) >= 0.0
            for config in candidate_configs
        )
        >= 2,
    }
    tripwires: list[str] = []
    for context, values in family_deltas.items():
        if context == ALL_CONTEXT:
            continue
        if values.get("correct_and_cited", 0.0) < -0.05:
            tripwires.append(f"{context}:correct_and_cited_drop_gt_5pp")
    for config in candidate_configs:
        values = seed_deltas[config].get(ALL_CONTEXT, {})
        if values.get("minicheck_citation_precision", 0.0) < -0.05:
            tripwires.append(f"{config}:citation_precision_drop_gt_5pp")
        if values.get("minicheck_citation_recall", 0.0) < -0.05:
            tripwires.append(f"{config}:citation_recall_drop_gt_5pp")
    technical_pass = not technical_failures
    responsibility_pass = all(responsibility.values())
    pass_gate = technical_pass and responsibility_pass and not tripwires
    return {
        "status": "G410_CROSS_DATA_RESPONSIBILITY_PASS"
        if pass_gate
        else "G410_CROSS_DATA_RESPONSIBILITY_FAIL",
        "technical_pass": technical_pass,
        "technical_failures": technical_failures,
        "responsibility_pass": responsibility_pass,
        "responsibility_checks": responsibility,
        "tripwires": tripwires,
        "positive_signal": all_family.get("correct_and_cited", 0.0) > 0.0,
        "strong_claim": "PENDING_G420_STATISTICS",
        "citation_regression_guard": "PENDING_G420_REVEALED_ASQA_QAMPARI_BOUNDARY_CONFIRMATION",
    }


def score(
    *,
    tasks_path: Path,
    references_path: Path,
    baseline_g310_generations_path: Path,
    seed_generations: Mapping[int, Path],
    output_json: Path,
    output_rows: Path,
    output_report: Path,
    minicheck_model_id: str = FROZEN_MINICHECK_MODEL,
    minicheck_device: str = "cpu",
    entails: Callable[[str, str], bool] | None = None,
    allow_subset: bool = False,
) -> dict[str, object]:
    if set(seed_generations) != set(ALLOWED_SEEDS):
        raise ValueError("G410 scoring requires seeds 13, 42, and 73")
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
            raise ValueError("G410 candidate seed task sets differ")
        candidate_rows[seed] = rows
        candidate_manifests[f"seed{seed}"] = manifest
    if candidate_task_ids is None:
        raise ValueError("G410 candidate rows are empty")
    if candidate_task_ids != wanted:
        if not allow_subset:
            raise ValueError("G410 candidates do not exactly cover prepared tasks")
        if not candidate_task_ids < wanted:
            raise ValueError("G410 subset scoring requires a strict prepared-task subset")
        tasks = [task for task in tasks if task.task_id in candidate_task_ids]
        wanted = candidate_task_ids
    references = _load_references(references_path, wanted)
    metadata = {task.task_id: _task_identity(task) for task in tasks}
    evidence = {task.task_id: task.evidence for task in tasks}
    baseline = _baseline_g0(baseline_g310_generations_path, wanted)
    configs: dict[str, dict[str, Mapping[str, object]]] = {"G0": baseline}
    for seed, path in sorted(seed_generations.items()):
        config = f"GRC{seed}"
        configs[config] = candidate_rows[seed]
    config_names = tuple(configs)
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
        per_task[task.task_id] = {
            config: _config_metrics(
                run_row=configs[config][task.task_id],
                references=references[task.task_id],
                evidence_count=len(task.evidence),
                minicheck_metrics=minicheck_by_task[task.task_id][config],
            )
            for config in config_names
        }
        scored_rows.append(
            {
                "schema_version": SCHEMA_SCORE_ROW,
                "task_id": task.task_id,
                "dataset": task.dataset,
                "case_id": task.case_id,
                "query_id": task.query_id,
                "component_id": task.component_id,
                "context": task.context,
                "variant_name": task.variant_name,
                "arms": per_task[task.task_id],
            }
        )
    aggregate = _aggregate(metadata=metadata, per_task=per_task, configs=config_names)
    seed_deltas = _deltas(aggregate, configs=config_names)
    family_deltas = _family_deltas(seed_deltas)
    gate = _gate_decision(
        aggregate=aggregate,
        seed_deltas=seed_deltas,
        family_deltas=family_deltas,
    )
    report: dict[str, object] = {
        "schema_version": SCHEMA_SCORE_REPORT,
        "stage": "G410",
        "status": gate["status"],
        "score_type": "locked_2wiki_cross_data_internal_screen",
        "tasks": len(tasks),
        "case_groups": EXPECTED_TWOWIKI_CASES,
        "formal_task_subset": allow_subset,
        "configs": list(config_names),
        "contexts": [ALL_CONTEXT, *CONTEXTS],
        "aggregate": aggregate,
        "seed_deltas_vs_g0": seed_deltas,
        "family_deltas_vs_g0": family_deltas,
        "gate": gate,
        "candidate_manifests": candidate_manifests,
        "citation_judge": {
            "model": "MiniCheck-Flan-T5-Large",
            "model_id": minicheck_model_id,
            "device": minicheck_device,
            "unique_judge_calls": judge_call_count(),
            "production_true_excluded_from_judging": True,
        },
        "source_sha256": {
            "tasks": _sha256(tasks_path),
            "references": _sha256(references_path),
            "baseline_g310_generations": _sha256(baseline_g310_generations_path),
            **{
                f"seed{seed}_generations": _sha256(path)
                for seed, path in sorted(seed_generations.items())
            },
        },
        "boundaries": {
            "gold_loaded_at_runtime": False,
            "references_loaded_at_runtime": False,
            "score_uses_references_after_generation": True,
            "sealed_or_heldout_read": False,
            "utility_labels_started": False,
            "not_teacher_generator_freeze": True,
            "g420_final_gate_pending": True,
            "revealed_asqa_qampari_guard_pending": True,
        },
    }
    _write_json(output_json, report)
    _write_jsonl(output_rows, scored_rows)
    _write_markdown_report(output_report, report)
    return report


def _pct(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _write_markdown_report(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# G410 Cross-Data Qualification",
        "",
        f"**Status:** `{report['status']}`",
        "",
        "This is G410 2Wiki internal cross-data qualification only. It is not GQ freeze, not G420, and not held-out.",
        "",
        "## Family Deltas vs Fixed G0",
        "",
        "| Context | correct+cited | answer | coverage | citation precision | citation recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for context, values in report["family_deltas_vs_g0"].items():
        lines.append(
            "| "
            f"{context} | {_pct(values.get('correct_and_cited'))} | "
            f"{_pct(values.get('answer_match'))} | "
            f"{_pct(values.get('coverage'))} | "
            f"{_pct(values.get('minicheck_citation_precision'))} | "
            f"{_pct(values.get('minicheck_citation_recall'))} |"
        )
    lines.extend(["", "## Gate", "", "```json"])
    lines.append(json.dumps(report["gate"], ensure_ascii=False, indent=2))
    lines.extend(["```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--data-manifest", required=True, type=Path)
    prepare_parser.add_argument("--train-cases", required=True, type=Path)
    prepare_parser.add_argument("--validation-cases", required=True, type=Path)
    prepare_parser.add_argument("--ordered-ids", required=True, type=Path)
    prepare_parser.add_argument("--output-dir", required=True, type=Path)

    run_parser = commands.add_parser("run")
    run_parser.add_argument("--seed", required=True, type=int)
    run_parser.add_argument("--tasks", required=True, type=Path)
    run_parser.add_argument("--prepare-manifest", required=True, type=Path)
    run_parser.add_argument("--model-snapshot", required=True, type=Path)
    run_parser.add_argument("--true-snapshot", required=True, type=Path)
    run_parser.add_argument("--grc-run-dir", required=True, type=Path)
    run_parser.add_argument("--output-dir", required=True, type=Path)
    run_parser.add_argument("--limit-tasks", type=int)

    score_parser = commands.add_parser("score")
    score_parser.add_argument("--tasks", required=True, type=Path)
    score_parser.add_argument("--references", required=True, type=Path)
    score_parser.add_argument("--baseline-g310-generations", required=True, type=Path)
    score_parser.add_argument("--seed13-generations", required=True, type=Path)
    score_parser.add_argument("--seed42-generations", required=True, type=Path)
    score_parser.add_argument("--seed73-generations", required=True, type=Path)
    score_parser.add_argument("--output-json", required=True, type=Path)
    score_parser.add_argument("--output-rows", required=True, type=Path)
    score_parser.add_argument("--output-report", required=True, type=Path)
    score_parser.add_argument("--minicheck-model-id", default=FROZEN_MINICHECK_MODEL)
    score_parser.add_argument("--minicheck-device", default="cpu")
    score_parser.add_argument("--allow-subset", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(
            data_manifest_path=args.data_manifest,
            train_cases_path=args.train_cases,
            validation_cases_path=args.validation_cases,
            ordered_ids_path=args.ordered_ids,
            output_dir=args.output_dir,
        )
    elif args.command == "run":
        result = run(
            seed=args.seed,
            tasks_path=args.tasks,
            prepare_manifest_path=args.prepare_manifest,
            model_snapshot=args.model_snapshot,
            true_snapshot=args.true_snapshot,
            grc_run_dir=args.grc_run_dir,
            output_dir=args.output_dir,
            limit_tasks=args.limit_tasks,
        )
    else:
        result = score(
            tasks_path=args.tasks,
            references_path=args.references,
            baseline_g310_generations_path=args.baseline_g310_generations,
            seed_generations={
                13: args.seed13_generations,
                42: args.seed42_generations,
                73: args.seed73_generations,
            },
            output_json=args.output_json,
            output_rows=args.output_rows,
            output_report=args.output_report,
            minicheck_model_id=args.minicheck_model_id,
            minicheck_device=args.minicheck_device,
            allow_subset=args.allow_subset,
        )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
