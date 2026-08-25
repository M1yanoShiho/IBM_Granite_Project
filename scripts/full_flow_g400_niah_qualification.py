"""Run and score the G400 locked NIAH Generator qualification.

The ``run`` command has no gold input.  It reuses the frozen A002/B100 NIAH
runtime inputs and generates only the locked GR-C candidate for one seed.  G0 is
kept fixed from A002/B100 for answerable NIAH full/stress comparison.

Optional unsupported-safety tasks are read from the already materialized G223
train bundle.  Their G0 baseline is reused from the G310 formal screen during
``score``; no reference answers are needed for these unsupported tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import full_flow_b100 as b100
import full_flow_g230 as g230
import full_flow_g310_seed13_screen as g310
import full_flow_joint as joint
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
from evidence_rag.generator.claim_splitter import ClaimSplitter
from evidence_rag.generator.draft import DraftAnswerGenerator, DraftGenerator
from evidence_rag.generator.granite import (
    GraniteGenerationConfig,
    NamedAdapterTextGenerator,
    PeftGraniteLLMClient,
)
from evidence_rag.generator.nli import MiniCheckNLIModel, TrueNLIModel
from evidence_rag.generator.verify_annotate import VerifyAnnotateGenerator
from evidence_rag.infrastructure.datasets import GoldCase

Seed = Literal[13, 42, 73]

ALLOWED_SEEDS: tuple[Seed, ...] = (13, 42, 73)
CANDIDATE_ARM = "GR-C"
FULL_CONTEXT = g230.FULL_CONTEXT
STRESS_CONTEXTS = g230.STRESS_CONTEXTS
ANSWERABLE_CONTEXTS = g230.CONTEXTS
UNSUPPORTED_SCOPE = g310.UNSUPPORTED_SCOPE
UNSUPPORTED_CONTEXT = "train_unsupported_safety"
ALL_CONTEXTS = (*ANSWERABLE_CONTEXTS, UNSUPPORTED_CONTEXT)

EXPECTED_ANSWERABLE_TASKS = g230.EXPECTED_TASKS
EXPECTED_UNSUPPORTED_TASKS = 1044
EXPECTED_TRAIN_GROUPS = 2370
EXPECTED_TRAINING_EXAMPLES = 9207
EXPECTED_VALIDATION_GROUPS = 302
EXPECTED_OPTIMIZER_STEPS = 297
EXPECTED_TWOWIKI_MODELVAL_GROUPS = 95

SCHEMA_G300_TRAINING = g310.SCHEMA_G300_TRAINING
SCHEMA_RUN_SPEC = "full-flow-g400-run-spec-v1"
SCHEMA_RUN_MANIFEST = "full-flow-g400-run-manifest-v1"
SCHEMA_GENERATION_ROW = "full-flow-g400-generation-row-v1"
SCHEMA_SCORE_REPORT = "full-flow-g400-score-report-v1"
SCHEMA_SCORE_ROW = "full-flow-g400-scored-row-v1"
FROZEN_MINICHECK_MODEL = g310.FROZEN_MINICHECK_MODEL
REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_SOURCE_ROOT = REPO_ROOT / "src" / "evidence_rag" / "generator"


@dataclass(frozen=True, slots=True)
class G400Task:
    task_id: str
    scope: str
    context: str
    dataset: str
    case_id: str
    query_id: str
    component_id: str
    selector_changed: bool
    variant_name: str
    question: str
    answerable: bool
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


def _validate_g330_run(
    run_dir: Path,
    *,
    seed: int,
    model_config_sha256: str | None = None,
) -> dict[str, object]:
    manifest_path = run_dir / "training_manifest.json"
    adapter_dir = run_dir / "adapter"
    weights_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    manifest = _json(manifest_path)
    checks = {
        "schema": manifest.get("schema_version") == SCHEMA_G300_TRAINING,
        "status": manifest.get("status") == "COMPLETE",
        "run_kind": manifest.get("run_kind") == "formal",
        "recipe": manifest.get("recipe") == "gr-c",
        "seed": manifest.get("seed") == seed,
        "entry": manifest.get("entry_mode")
        == "controlled_continuation_limited_internal_screen",
        "clean_freeze": manifest.get("clean_freeze_ready") is False,
        "g223_status": manifest.get("g223_status") == "CONTROLLED_CONTINUATION_READY",
        "g223_controlled": manifest.get("g223_controlled_continuation_ready") is True,
        "g223_twowiki_modelval": manifest.get("g223_twowiki_modelval_groups")
        == EXPECTED_TWOWIKI_MODELVAL_GROUPS,
        "full_train_groups": manifest.get("full_data_train_groups")
        == EXPECTED_TRAIN_GROUPS,
        "full_validation_groups": manifest.get("full_data_validation_groups")
        == EXPECTED_VALIDATION_GROUPS,
        "training_examples": manifest.get("training_examples")
        == EXPECTED_TRAINING_EXAMPLES,
        "optimizer_steps": manifest.get("optimizer_steps") == EXPECTED_OPTIMIZER_STEPS,
        "model": model_config_sha256 is None
        or manifest.get("model_snapshot_config_sha256") == model_config_sha256,
        "dev_boundary": manifest.get("dev_read") is False
        and manifest.get("decision_dev_used") is False,
        "heldout_boundary": manifest.get("sealed_or_heldout_read") is False,
        "utility_boundary": manifest.get("utility_labels_started") is False,
        "adapter_scope": manifest.get("adapter_scope") == "draft generation call only",
        "splitter_scope": manifest.get("claim_splitter_scope")
        == "frozen Granite base with adapters disabled",
        "reload": cast(Mapping[str, object], manifest.get("reload_check", {})).get(
            "status"
        )
        == "PASS",
        "weights": weights_path.is_file()
        and manifest.get("adapter_weights_sha256") == _sha256(weights_path),
        "config": config_path.is_file()
        and manifest.get("adapter_config_sha256") == _sha256(config_path),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"invalid G330 GR-C seed-{seed} run: {failed}")
    return {
        "run_dir": str(run_dir.resolve()),
        "training_manifest_sha256": _sha256(manifest_path),
        "adapter_config_sha256": _sha256(config_path),
        "adapter_weights_sha256": _sha256(weights_path),
        "mean_group_weighted_loss": manifest.get("mean_group_weighted_loss"),
        "validation_mean_group_weighted_loss": manifest.get(
            "validation_mean_group_weighted_loss"
        ),
    }


def _from_g230_task(task: g230.EvalTask) -> G400Task:
    return G400Task(
        task_id=task.task_id,
        scope=task.scope,
        context=task.context,
        dataset="niah",
        case_id=task.query_id,
        query_id=task.query_id,
        component_id=task.component_id,
        selector_changed=task.selector_changed,
        variant_name=task.context,
        question=task.question,
        answerable=True,
        evidence=task.evidence,
    )


def _from_g310_task(task: g310.ScreenTask) -> G400Task:
    return G400Task(
        task_id=task.task_id,
        scope=UNSUPPORTED_SCOPE,
        context=UNSUPPORTED_CONTEXT,
        dataset=task.dataset,
        case_id=task.case_id,
        query_id=task.query_id,
        component_id=task.component_id,
        selector_changed=False,
        variant_name=task.variant_name,
        question=task.question,
        answerable=False,
        evidence=task.evidence,
    )


def build_tasks(
    *,
    full_cases: Sequence[joint.JointCase],
    stress_rows: Sequence[Mapping[str, object]],
    unsupported_train_rows: Sequence[Mapping[str, Any]] = (),
    include_unsupported: bool = False,
) -> list[G400Task]:
    answerable_tasks = [
        _from_g230_task(task) for task in g230.build_tasks(full_cases, stress_rows)
    ]
    unsupported_tasks: list[G400Task] = []
    if include_unsupported:
        unsupported_tasks = [
            _from_g310_task(task)
            for task in g310.build_tasks(
                train_rows=unsupported_train_rows,
                validation_rows=(),
                include_unsupported_train=True,
            )
            if task.scope == UNSUPPORTED_SCOPE
        ]
    tasks = [*answerable_tasks, *unsupported_tasks]
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("duplicate G400 task ID")
    return tasks


def _task_identity(task: G400Task) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "scope": task.scope,
        "context": task.context,
        "dataset": task.dataset,
        "case_id": task.case_id,
        "query_id": task.query_id,
        "component_id": task.component_id,
        "selector_changed": task.selector_changed,
        "variant_name": task.variant_name,
        "answerable": task.answerable,
        "evidence_ids": [item.evidence_id for item in task.evidence],
    }


def validate_resume_prefix(rows: Sequence[Mapping[str, Any]], tasks: Sequence[G400Task]) -> None:
    if len(rows) > len(tasks):
        raise ValueError("G400 resume has more rows than frozen tasks")
    for index, row in enumerate(rows):
        task = tasks[index]
        if row.get("task_id") != task.task_id:
            raise ValueError(f"G400 resume diverges at row {index + 1}")
        arms = row.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != {CANDIDATE_ARM}:
            raise ValueError(f"G400 resume row {index + 1} has wrong arms")


def _run_one(
    generator: VerifyAnnotateGenerator,
    query: Query,
    checklist: QueryChecklist,
    selected: SelectedEvidenceSet,
) -> dict[str, object]:
    started = time.perf_counter()
    try:
        generation = generator.generate(query, checklist, selected)
        trace = generator.last_trace
        if trace is None:
            raise RuntimeError("trace-enabled G400 Generator produced no trace")
    except Exception as error:  # noqa: BLE001 -- runtime errors are measured outcomes
        return {
            "generation": GenerationResult(
                query_id=query.query_id,
                answer="",
                cited_evidence_ids=(),
            ).model_dump(mode="json"),
            "trace": None,
            "routing": [],
            "error": f"{type(error).__name__}: {error}",
            "seconds": time.perf_counter() - started,
        }
    return {
        "generation": generation.model_dump(mode="json"),
        "trace": trace.model_dump(mode="json"),
        "routing": g310._routing_rows(generator),
        "error": None,
        "seconds": time.perf_counter() - started,
    }


def build_generator(
    *,
    client: PeftGraniteLLMClient,
    nli: TrueNLIModel,
) -> VerifyAnnotateGenerator:
    draft = DraftAnswerGenerator(
        draft_generator=DraftGenerator(
            llm=NamedAdapterTextGenerator(client, "grc"),
            trace_enabled=True,
        ),
        claim_splitter=ClaimSplitter(llm=client, trace_enabled=True),
        trace_enabled=True,
    )
    return VerifyAnnotateGenerator(
        draft_generator=draft,
        nli=nli,
        entity_gate="observe",
        abstain_when_unverified=False,
        trace_enabled=True,
    )


def run(
    *,
    seed: int,
    queries_path: Path,
    candidate_pool_path: Path,
    roles_path: Path,
    decision_trace_path: Path,
    stress_contexts_path: Path,
    a002_manifest_path: Path,
    b100_manifest_path: Path,
    model_snapshot: Path,
    true_snapshot: Path,
    grc_run_dir: Path,
    output_dir: Path,
    unsupported_train_cases_path: Path | None = None,
    limit_tasks: int | None = None,
    include_unsupported: bool = False,
) -> dict[str, object]:
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"G400 seed must be one of {ALLOWED_SEEDS}")
    if include_unsupported and unsupported_train_cases_path is None:
        raise ValueError("--include-unsupported requires --unsupported-train-cases")

    a002_manifest = _json(a002_manifest_path)
    b100_manifest = _json(b100_manifest_path)
    full_cases = joint.load_runtime_cases(
        queries_path=queries_path,
        candidate_pool_path=candidate_pool_path,
        roles_path=roles_path,
        decision_trace_path=decision_trace_path,
    )
    stress_rows = b100._runtime_contexts(stress_contexts_path)
    unsupported_rows = (
        _jsonl(unsupported_train_cases_path)
        if include_unsupported and unsupported_train_cases_path is not None
        else []
    )
    tasks = build_tasks(
        full_cases=full_cases,
        stress_rows=stress_rows,
        unsupported_train_rows=unsupported_rows,
        include_unsupported=include_unsupported,
    )
    if limit_tasks is None:
        if len(full_cases) != g230.EXPECTED_FULL_QUERIES:
            raise ValueError("formal G400 full-dev count differs from 739")
        if (
            len(stress_rows) != g230.EXPECTED_STRESS_QUERIES
            or sum(task.answerable for task in tasks) != EXPECTED_ANSWERABLE_TASKS
        ):
            raise ValueError("formal G400 NIAH answerable task count differs from frozen protocol")
        if include_unsupported and (
            sum(not task.answerable for task in tasks) != EXPECTED_UNSUPPORTED_TASKS
        ):
            raise ValueError("formal G400 unsupported task count differs from G223")
    else:
        if limit_tasks <= 0:
            raise ValueError("--limit-tasks must be positive")
        tasks = tasks[:limit_tasks]

    runtime_hash = _sha256_bytes(
        _canonical_bytes([case.identity_row() for case in full_cases])
    )
    if a002_manifest.get("status") != "COMPLETE" or a002_manifest.get(
        "runtime_input_sha256"
    ) != runtime_hash:
        raise ValueError("G400 full-dev runtime inputs differ from frozen A002")
    if b100_manifest.get("status") != "COMPLETE" or b100_manifest.get(
        "runtime_contexts_sha256"
    ) != _sha256(stress_contexts_path):
        raise ValueError("G400 stress contexts differ from frozen B100")

    model_config_sha256 = _sha256(model_snapshot / "config.json")
    adapter_audit = _validate_g330_run(
        grc_run_dir,
        seed=seed,
        model_config_sha256=model_config_sha256,
    )
    spec: dict[str, object] = {
        "schema_version": SCHEMA_RUN_SPEC,
        "stage": "G400",
        "seed": seed,
        "arms": [CANDIDATE_ARM],
        "formal": limit_tasks is None,
        "tasks": len(tasks),
        "answerable_tasks": sum(task.answerable for task in tasks),
        "unsupported_tasks": sum(not task.answerable for task in tasks),
        "include_unsupported": include_unsupported,
        "task_identity_sha256": _sha256_bytes(
            _canonical_bytes([_task_identity(task) for task in tasks])
        ),
        "source_sha256": {
            "script": _sha256(Path(__file__)),
            "g230_script": _sha256(Path(g230.__file__)),
            "g310_script": _sha256(Path(g310.__file__)),
            "draft_source": _sha256(GENERATOR_SOURCE_ROOT / "draft.py"),
            "granite_source": _sha256(GENERATOR_SOURCE_ROOT / "granite.py"),
            "claim_splitter_source": _sha256(GENERATOR_SOURCE_ROOT / "claim_splitter.py"),
            "verify_annotate_source": _sha256(GENERATOR_SOURCE_ROOT / "verify_annotate.py"),
            "nli_source": _sha256(GENERATOR_SOURCE_ROOT / "nli.py"),
            "queries": _sha256(queries_path),
            "candidate_pool": _sha256(candidate_pool_path),
            "roles": _sha256(roles_path),
            "decision_trace": _sha256(decision_trace_path),
            "stress_contexts": _sha256(stress_contexts_path),
            "a002_manifest": _sha256(a002_manifest_path),
            "b100_manifest": _sha256(b100_manifest_path),
            "unsupported_train_cases": (
                _sha256(unsupported_train_cases_path)
                if include_unsupported and unsupported_train_cases_path is not None
                else None
            ),
            "granite_config": model_config_sha256,
            "true_config": _sha256(true_snapshot / "config.json"),
        },
        "adapter_audit": {"grc": adapter_audit},
        "decode": {
            "max_new_tokens": 256,
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
        },
        "gold_loaded_at_runtime": False,
        "reference_answers_loaded_at_runtime": False,
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
        raise ValueError(f"G400 run is already complete: {manifest_path}")
    if spec_path.exists():
        if _json(spec_path) != spec:
            raise ValueError("G400 resume specification differs from frozen run")
    else:
        if generations_path.exists() and generations_path.stat().st_size:
            raise ValueError("G400 generations exist without a run specification")
        _write_json(spec_path, spec)
    existing = _jsonl(generations_path) if generations_path.exists() else []
    validate_resume_prefix(existing, tasks)

    client = PeftGraniteLLMClient(
        model_id=str(model_snapshot.resolve()),
        adapters={"grc": str((grc_run_dir / "adapter").resolve())},
        config=GraniteGenerationConfig(max_new_tokens=256, temperature=0.0, top_p=1.0),
    )
    nli = TrueNLIModel(model_id=str(true_snapshot.resolve()))
    generator = build_generator(client=client, nli=nli)

    started = time.perf_counter()
    with generations_path.open("a", encoding="utf-8", newline="\n") as handle:
        for index, task in enumerate(tasks[len(existing) :], start=len(existing) + 1):
            query = Query(query_id=task.query_id, text=task.question)
            checklist = QueryChecklist(
                query_id=task.query_id,
                focus=task.question,
                required_facts=(),
            )
            selected = SelectedEvidenceSet(query_id=task.query_id, evidence=task.evidence)
            arm_row = _run_one(generator, query, checklist, selected)
            row = {
                "schema_version": SCHEMA_GENERATION_ROW,
                **_task_identity(task),
                "question": task.question,
                "evidence": [item.model_dump(mode="json") for item in task.evidence],
                "arm_order": [CANDIDATE_ARM],
                "arms": {CANDIDATE_ARM: arm_row},
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 10 == 0 or index == len(tasks):
                elapsed = time.perf_counter() - started
                completed_now = index - len(existing)
                print(
                    f"[G400 seed-{seed}] {index}/{len(tasks)} "
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
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_RUN_MANIFEST,
        "stage": "G400",
        "status": "INVALID_RUNTIME" if errors or trace_missing else "COMPLETE",
        "seed": seed,
        "arms": [CANDIDATE_ARM],
        "tasks": len(rows),
        "answerable_tasks": sum(bool(row.get("answerable")) for row in rows),
        "unsupported_tasks": sum(not bool(row.get("answerable")) for row in rows),
        "formal": limit_tasks is None,
        "generations_sha256": _sha256(generations_path),
        "run_spec_sha256": _sha256(spec_path),
        "errors": errors,
        "trace_missing": trace_missing,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "gold_loaded_at_runtime": False,
        "reference_answers_loaded_at_runtime": False,
        "sealed_or_heldout_read": False,
        "utility_labels_started": False,
        "environment": _runtime_environment(),
    }
    _write_json(manifest_path, manifest)
    if errors or trace_missing:
        raise RuntimeError("G400 runtime has errors or missing traces")
    return manifest


def _routing_from_trace(run_row: Mapping[str, object]) -> list[dict[str, object]]:
    trace = run_row.get("trace")
    if not isinstance(trace, Mapping):
        if run_row.get("error"):
            return []
        raise ValueError("G0 run has neither a trace nor a runtime error")
    claims = trace.get("claims")
    if not isinstance(claims, Sequence):
        raise ValueError("G0 trace has no claim sequence")
    routing: list[dict[str, object]] = []
    for claim in claims:
        if not isinstance(claim, Mapping):
            raise ValueError("G0 trace contains an invalid claim")
        sentence = str(claim.get("final_sentence", ""))
        if not sentence:
            continue
        routing.append(
            {
                "sentence": sentence,
                "citation": claim.get("citation"),
                "outcome": claim.get("routing_outcome", ""),
            }
        )
    return routing


def _baseline_with_optional_unsupported(
    *,
    a002_generations_path: Path,
    b100_generations_path: Path,
    unsupported_baseline_generations_path: Path | None,
) -> tuple[dict[str, Mapping[str, object]], dict[str, str]]:
    baseline, components = g230._baseline_results(
        _jsonl(a002_generations_path),
        _jsonl(b100_generations_path),
    )
    if unsupported_baseline_generations_path is not None:
        for row in _jsonl(unsupported_baseline_generations_path):
            if row.get("scope") != UNSUPPORTED_SCOPE:
                continue
            arms = row.get("arms")
            if not isinstance(arms, Mapping) or not isinstance(arms.get("G0"), Mapping):
                raise ValueError(f"unsupported baseline {row.get('task_id')} lacks G0")
            task_id = str(row["task_id"])
            baseline[task_id] = cast(Mapping[str, object], arms["G0"])
            components[task_id] = str(row["component_id"])
    return baseline, components


def _candidate_results(
    path: Path,
    *,
    seed: int,
) -> tuple[
    dict[str, Mapping[str, object]],
    dict[str, tuple[EvidenceCandidate, ...]],
    dict[str, dict[str, object]],
    Mapping[str, Any],
]:
    manifest = _json(path.parent / "run_manifest.json")
    if (
        manifest.get("status") != "COMPLETE"
        or manifest.get("seed") != seed
        or tuple(manifest.get("arms", ())) != (CANDIDATE_ARM,)
    ):
        raise ValueError(f"invalid G400 seed-{seed} manifest for {path}")
    if manifest.get("generations_sha256") != _sha256(path):
        raise ValueError(f"G400 seed-{seed} generations hash differs")
    rows: dict[str, Mapping[str, object]] = {}
    evidence: dict[str, tuple[EvidenceCandidate, ...]] = {}
    metadata: dict[str, dict[str, object]] = {}
    for row in _jsonl(path):
        task_id = str(row["task_id"])
        arms = row.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != {CANDIDATE_ARM}:
            raise ValueError(f"candidate task {task_id} has invalid arms")
        value = arms[CANDIDATE_ARM]
        if not isinstance(value, Mapping):
            raise ValueError(f"candidate task {task_id}/{CANDIDATE_ARM} is invalid")
        rows[task_id] = value
        raw_evidence = row.get("evidence")
        if not isinstance(raw_evidence, Sequence) or isinstance(raw_evidence, (str, bytes)):
            raise ValueError(f"candidate task {task_id} has no evidence payload")
        evidence[task_id] = tuple(
            EvidenceCandidate.model_validate(item) for item in raw_evidence
        )
        metadata[task_id] = {
            "task_id": task_id,
            "scope": row["scope"],
            "context": row["context"],
            "dataset": row["dataset"],
            "case_id": row["case_id"],
            "query_id": row["query_id"],
            "component_id": row["component_id"],
            "selector_changed": row["selector_changed"],
            "variant_name": row["variant_name"],
            "answerable": row["answerable"],
        }
    return rows, evidence, metadata, manifest


def _load_gold(path: Path, wanted: set[str]) -> dict[str, GoldCase]:
    output: dict[str, GoldCase] = {}
    for row in _jsonl(path):
        gold = GoldCase.model_validate(row)
        if gold.query_id in wanted:
            output[gold.query_id] = gold
    if set(output) != wanted:
        raise ValueError("gold does not exactly cover G400 answerable query IDs")
    return output


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
        routing = _routing_from_trace(run_row)
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
    metadata: Mapping[str, object],
    evidence_count: int,
    minicheck_metrics: Mapping[str, float],
    gold: Mapping[str, GoldCase],
) -> dict[str, float]:
    generation = GenerationResult.model_validate(run_row["generation"])
    error = bool(run_row.get("error"))
    answer = strip_annotations(generation.answer)
    coverage = float(bool(answer.strip()) and not g310._is_unknown(answer))
    answerable = bool(metadata.get("answerable"))
    matched = 0.0
    if answerable:
        query_id = str(metadata["query_id"])
        value = answer_match(answer, gold[query_id].reference_answers).value
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
        "minicheck_citation_precision": minicheck_precision if answerable else 0.0,
        "minicheck_citation_recall": minicheck_recall if answerable else 0.0,
        "minicheck_sentences": float(minicheck_metrics["minicheck_sentences"]),
        "minicheck_citations": float(minicheck_metrics["minicheck_citations"]),
        "correct_and_cited": float(
            answerable
            and matched == 1.0
            and coverage == 1.0
            and invalid == 0
            and minicheck_precision >= 1.0
            and minicheck_recall >= 1.0
        ),
        "unsupported_ungrounded_assertion": float((not answerable) and coverage == 1.0),
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate(
    *,
    metadata: Mapping[str, Mapping[str, object]],
    per_task: Mapping[str, Mapping[str, Mapping[str, float]]],
    configs: Sequence[str],
) -> dict[str, dict[str, dict[str, float | int]]]:
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    for context in ALL_CONTEXTS:
        task_ids = [
            task_id
            for task_id, item in metadata.items()
            if str(item["context"]) == context
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
        "unsupported_ungrounded_assertion",
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
    for context in ALL_CONTEXTS:
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
    full_family = family_deltas.get(FULL_CONTEXT, {})
    unsupported_family = family_deltas.get(UNSUPPORTED_CONTEXT, {})
    seed_full = {
        config: seed_deltas[config].get(FULL_CONTEXT, {})
        for config in candidate_configs
    }
    responsibility = {
        "niah_correct_and_cited_delta_gt_0": full_family.get("correct_and_cited", 0.0) > 0,
        "answer_delta_ge_minus_2pp": full_family.get("answer_match", 0.0) >= -0.02,
        "coverage_delta_ge_minus_2pp": full_family.get("coverage", 0.0) >= -0.02,
        "citation_precision_delta_ge_minus_3pp": full_family.get(
            "minicheck_citation_precision",
            0.0,
        )
        >= -0.03,
        "citation_recall_delta_ge_minus_3pp": full_family.get(
            "minicheck_citation_recall",
            0.0,
        )
        >= -0.03,
        "unsupported_assertion_delta_le_plus_2pp": unsupported_family.get(
            "unsupported_ungrounded_assertion",
            0.0,
        )
        <= 0.02
        if UNSUPPORTED_CONTEXT in family_deltas
        else None,
        "at_least_2_of_3_seeds_correct_and_cited_nonnegative": sum(
            values.get("correct_and_cited", 0.0) >= 0.0
            for values in seed_full.values()
        )
        >= 2,
        "no_seed_citation_precision_or_recall_drop_gt_5pp": all(
            values.get("minicheck_citation_precision", 0.0) >= -0.05
            and values.get("minicheck_citation_recall", 0.0) >= -0.05
            for values in seed_full.values()
        ),
    }
    tripwires: list[str] = []
    for context in STRESS_CONTEXTS:
        if family_deltas.get(context, {}).get("correct_and_cited", 0.0) < -0.05:
            tripwires.append(f"{context}:family_correct_and_cited_drop_gt_5pp")
    if unsupported_family.get("unsupported_ungrounded_assertion", 0.0) > 0.05:
        tripwires.append("unsupported_assertion_increase_gt_5pp")
    for context in (FULL_CONTEXT, *STRESS_CONTEXTS):
        for metric in ("minicheck_citation_precision", "minicheck_citation_recall"):
            if (
                sum(
                    seed_deltas[config].get(context, {}).get(metric, 0.0) < -0.05
                    for config in candidate_configs
                )
                >= 2
            ):
                tripwires.append(f"{context}:{metric}:two_seed_drop_gt_5pp")

    technical_pass = not technical_failures
    responsibility_values = [
        value for value in responsibility.values() if isinstance(value, bool)
    ]
    responsibility_pass = technical_pass and not tripwires and all(responsibility_values)
    positive_signal = any(
        full_family.get(metric, 0.0) > 0.0
        for metric in ("correct_and_cited", "answer_match", "coverage")
    ) or any(
        family_deltas.get(context, {}).get("correct_and_cited", 0.0) > 0.0
        for context in STRESS_CONTEXTS
    ) or unsupported_family.get("unsupported_ungrounded_assertion", 0.0) < 0.0
    if not technical_pass:
        status = "INVALID_RUNTIME"
    elif tripwires:
        status = "FAIL_TRIPWIRE"
    elif responsibility_pass:
        status = "G400_NIAH_RESPONSIBILITY_PASS"
    elif positive_signal:
        status = "CONTROLLED_CONTINUATION_SIGNAL"
    else:
        status = "NO_POSITIVE_SIGNAL"
    return {
        "status": status,
        "technical_pass": technical_pass,
        "technical_failures": technical_failures,
        "responsibility_pass": responsibility_pass,
        "responsibility_checks": responsibility,
        "tripwires": tripwires,
        "positive_signal": positive_signal,
        "strong_claim": "PENDING_G420_STATISTICS",
        "cross_data_checks": "PENDING_G410",
    }


def score(
    *,
    a002_generations_path: Path,
    b100_generations_path: Path,
    seed_generations: Mapping[int, Path],
    gold_path: Path,
    output_json: Path,
    output_rows: Path,
    output_report: Path,
    unsupported_baseline_generations_path: Path | None = None,
    minicheck_model_id: str = FROZEN_MINICHECK_MODEL,
    minicheck_device: str = "cpu",
    entails: Callable[[str, str], bool] | None = None,
) -> dict[str, object]:
    if set(seed_generations) != set(ALLOWED_SEEDS):
        raise ValueError("G400 scoring requires exactly seeds 13, 42, and 73")
    baseline, components = _baseline_with_optional_unsupported(
        a002_generations_path=a002_generations_path,
        b100_generations_path=b100_generations_path,
        unsupported_baseline_generations_path=unsupported_baseline_generations_path,
    )
    configs: dict[str, Mapping[str, Mapping[str, object]]] = {"G0": baseline}
    canonical_evidence: dict[str, tuple[EvidenceCandidate, ...]] | None = None
    canonical_metadata: dict[str, dict[str, object]] | None = None
    candidate_manifests: dict[str, object] = {}
    for seed in ALLOWED_SEEDS:
        name = f"GRC{seed}"
        rows, evidence, metadata, manifest = _candidate_results(
            seed_generations[seed],
            seed=seed,
        )
        if canonical_evidence is None:
            canonical_evidence = evidence
            canonical_metadata = metadata
        elif evidence != canonical_evidence or metadata != canonical_metadata:
            raise ValueError(f"G400 seed-{seed} tasks differ from the first seed")
        configs[name] = rows
        candidate_manifests[f"seed{seed}"] = manifest
    assert canonical_evidence is not None and canonical_metadata is not None
    expected_tasks = set(canonical_evidence)
    missing_baseline = expected_tasks - set(baseline)
    if missing_baseline:
        missing = sorted(missing_baseline)
        raise ValueError(f"G400 fixed G0 baseline lacks tasks: {missing[:5]}")
    configs["G0"] = {task_id: baseline[task_id] for task_id in expected_tasks}
    for name, rows in configs.items():
        if set(rows) != expected_tasks:
            raise ValueError(f"G400 {name} does not exactly cover candidate tasks")
    if expected_tasks - set(components):
        missing = sorted(expected_tasks - set(components))
        raise ValueError(f"G400 baseline lacks components for tasks: {missing[:5]}")

    answerable_query_ids = {
        str(item["query_id"])
        for item in canonical_metadata.values()
        if bool(item["answerable"])
    }
    gold = _load_gold(gold_path, answerable_query_ids)
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

        def judge_call_count() -> int:
            return len(cache)
    minicheck = _minicheck_task_metrics(
        configs=configs,
        evidence=canonical_evidence,
        entails=entails,
    )
    config_order = ("G0", *[f"GRC{seed}" for seed in ALLOWED_SEEDS])
    per_task: dict[str, dict[str, dict[str, float]]] = {}
    scored_rows: list[dict[str, object]] = []
    for task_id in sorted(expected_tasks):
        meta = canonical_metadata[task_id]
        per_task[task_id] = {}
        for config in config_order:
            per_task[task_id][config] = _config_metrics(
                run_row=configs[config][task_id],
                metadata=meta,
                evidence_count=len(canonical_evidence[task_id]),
                minicheck_metrics=minicheck[task_id][config],
                gold=gold,
            )
        scored_rows.append(
            {
                "schema_version": SCHEMA_SCORE_ROW,
                **meta,
                "configs": per_task[task_id],
            }
        )
    aggregate = _aggregate(
        metadata=canonical_metadata,
        per_task=per_task,
        configs=config_order,
    )
    seed_deltas = _deltas(aggregate, configs=config_order)
    family_deltas = _family_deltas(seed_deltas)
    gate = _gate_decision(
        aggregate=aggregate,
        seed_deltas=seed_deltas,
        family_deltas=family_deltas,
    )
    report: dict[str, object] = {
        "schema_version": SCHEMA_SCORE_REPORT,
        "stage": "G400",
        "status": gate["status"],
        "score_type": "locked_niah_full_stress_with_optional_unsupported_safety",
        "tasks": len(expected_tasks),
        "answerable_tasks": sum(
            bool(item["answerable"]) for item in canonical_metadata.values()
        ),
        "unsupported_tasks": sum(
            not bool(item["answerable"]) for item in canonical_metadata.values()
        ),
        "configs": list(config_order),
        "contexts": [context for context in ALL_CONTEXTS if context in aggregate],
        "aggregate": aggregate,
        "seed_deltas_vs_g0": seed_deltas,
        "family_deltas_vs_g0": family_deltas,
        "gate": gate,
        "citation_judge": {
            "model": "MiniCheck-Flan-T5-Large",
            "model_id": minicheck_model_id,
            "device": minicheck_device,
            "unique_judge_calls": judge_call_count(),
            "production_true_excluded_from_judging": True,
        },
        "boundaries": {
            "gold_loaded_at_runtime": False,
            "score_uses_references_after_generation": True,
            "sealed_or_heldout_read": False,
            "utility_labels_started": False,
            "not_teacher_generator_freeze": True,
            "g410_cross_data_pending": True,
            "g420_final_gate_pending": True,
        },
        "source_sha256": {
            "a002_generations": _sha256(a002_generations_path),
            "b100_generations": _sha256(b100_generations_path),
            "unsupported_baseline_generations": (
                _sha256(unsupported_baseline_generations_path)
                if unsupported_baseline_generations_path is not None
                else None
            ),
            "gold": _sha256(gold_path),
            **{
                f"seed{seed}_generations": _sha256(seed_generations[seed])
                for seed in ALLOWED_SEEDS
            },
        },
        "candidate_manifests": candidate_manifests,
    }
    _write_json(output_json, report)
    _write_jsonl(output_rows, scored_rows)
    _write_markdown_report(output_report, report)
    return report


def _pct(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{100 * float(value):.2f}%"


def _write_markdown_report(path: Path, report: Mapping[str, Any]) -> None:
    aggregate = cast(Mapping[str, Mapping[str, Mapping[str, Any]]], report["aggregate"])
    family = cast(Mapping[str, Mapping[str, float]], report["family_deltas_vs_g0"])
    gate = cast(Mapping[str, Any], report["gate"])
    lines = [
        "# G400 Locked NIAH Qualification",
        "",
        f"**Status:** `{report['status']}`",
        "",
        "This is G400 only. It is not GQ freeze, not G410 cross-data qualification, and not held-out.",
        "",
        "## Family Deltas vs Fixed G0",
        "",
        "| Context | correct+cited | answer | coverage | citation precision | citation recall | unsupported assertion |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for context in report["contexts"]:
        item = family.get(str(context), {})
        lines.append(
            f"| {context} | {_pct(item.get('correct_and_cited'))} | "
            f"{_pct(item.get('answer_match'))} | {_pct(item.get('coverage'))} | "
            f"{_pct(item.get('minicheck_citation_precision'))} | "
            f"{_pct(item.get('minicheck_citation_recall'))} | "
            f"{_pct(item.get('unsupported_ungrounded_assertion'))} |"
        )
    lines.extend(["", "## Aggregate", ""])
    for context in report["contexts"]:
        lines.append(f"### {context}")
        lines.append("")
        lines.append("| Config | Tasks | correct+cited | answer | coverage | citation precision | citation recall | unsupported assertion |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for config, item in aggregate[str(context)].items():
            lines.append(
                f"| {config} | {int(item.get('tasks', 0))} | "
                f"{_pct(item.get('correct_and_cited'))} | "
                f"{_pct(item.get('answer_match'))} | "
                f"{_pct(item.get('coverage'))} | "
                f"{_pct(item.get('minicheck_citation_precision'))} | "
                f"{_pct(item.get('minicheck_citation_recall'))} | "
                f"{_pct(item.get('unsupported_ungrounded_assertion'))} |"
            )
        lines.append("")
    lines.extend(["## Gate", "", "```json"])
    lines.append(json.dumps(gate, ensure_ascii=False, indent=2))
    lines.extend(["```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run")
    run_parser.add_argument("--seed", required=True, type=int, choices=ALLOWED_SEEDS)
    run_parser.add_argument("--queries", required=True, type=Path)
    run_parser.add_argument("--candidate-pool", required=True, type=Path)
    run_parser.add_argument("--roles", required=True, type=Path)
    run_parser.add_argument("--decision-trace", required=True, type=Path)
    run_parser.add_argument("--stress-contexts", required=True, type=Path)
    run_parser.add_argument("--a002-manifest", required=True, type=Path)
    run_parser.add_argument("--b100-manifest", required=True, type=Path)
    run_parser.add_argument("--model-snapshot", required=True, type=Path)
    run_parser.add_argument("--true-snapshot", required=True, type=Path)
    run_parser.add_argument("--grc-run-dir", required=True, type=Path)
    run_parser.add_argument("--output-dir", required=True, type=Path)
    run_parser.add_argument("--unsupported-train-cases", type=Path)
    run_parser.add_argument("--include-unsupported", action="store_true")
    run_parser.add_argument("--limit-tasks", type=int)

    score_parser = commands.add_parser("score")
    score_parser.add_argument("--a002-generations", required=True, type=Path)
    score_parser.add_argument("--b100-generations", required=True, type=Path)
    for seed in ALLOWED_SEEDS:
        score_parser.add_argument(f"--seed{seed}-generations", required=True, type=Path)
    score_parser.add_argument("--gold", required=True, type=Path)
    score_parser.add_argument("--unsupported-baseline-generations", type=Path)
    score_parser.add_argument("--minicheck-model-id", default=FROZEN_MINICHECK_MODEL)
    score_parser.add_argument("--minicheck-device", default="cpu")
    score_parser.add_argument("--output-json", required=True, type=Path)
    score_parser.add_argument("--output-rows", required=True, type=Path)
    score_parser.add_argument("--output-report", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        result = run(
            seed=args.seed,
            queries_path=args.queries,
            candidate_pool_path=args.candidate_pool,
            roles_path=args.roles,
            decision_trace_path=args.decision_trace,
            stress_contexts_path=args.stress_contexts,
            a002_manifest_path=args.a002_manifest,
            b100_manifest_path=args.b100_manifest,
            model_snapshot=args.model_snapshot,
            true_snapshot=args.true_snapshot,
            grc_run_dir=args.grc_run_dir,
            output_dir=args.output_dir,
            unsupported_train_cases_path=args.unsupported_train_cases,
            include_unsupported=args.include_unsupported,
            limit_tasks=args.limit_tasks,
        )
    else:
        seed_generations = {
            seed: cast(Path, getattr(args, f"seed{seed}_generations"))
            for seed in ALLOWED_SEEDS
        }
        result = score(
            a002_generations_path=args.a002_generations,
            b100_generations_path=args.b100_generations,
            seed_generations=seed_generations,
            gold_path=args.gold,
            unsupported_baseline_generations_path=args.unsupported_baseline_generations,
            minicheck_model_id=args.minicheck_model_id,
            minicheck_device=args.minicheck_device,
            output_json=args.output_json,
            output_rows=args.output_rows,
            output_report=args.output_report,
        )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
