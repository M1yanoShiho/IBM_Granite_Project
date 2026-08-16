"""Run and score the frozen G230 Generator development gate.

The runtime command has no gold input.  It evaluates the complete 739-query
decision-dev distribution on TopK to isolate the Generator main effect, then
uses the frozen B100 matched subset for Selected, support-only, benign-noise,
harmful-noise, and support-last diagnostics.  G0 is reused from A002/B100;
this command generates GN once and matched GC/GM pairs for seeds 13/42/73.

Long runs are append-only and resumable.  A run specification binds every
input, adapter, prompt, and decode setting before the first generation row is
written.  Gold is joined only by the separate ``score`` command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import full_flow_b100 as b100
import full_flow_joint as joint

from evidence_rag.contracts.models import (
    EvidenceCandidate,
    GenerationResult,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
    strip_annotations,
)
from evidence_rag.evaluation.paired_metric import compare_paired
from evidence_rag.evaluation.scoring import answer_match
from evidence_rag.generator.claim_splitter import (
    FAITHFULNESS_PROMPT,
    SPLIT_PROMPT,
    ClaimSplitter,
)
from evidence_rag.generator.draft import (
    DRAFT_PROMPT,
    DraftAnswerGenerator,
    DraftGenerator,
    KeyFactDraftAnswerGenerator,
)
from evidence_rag.generator.granite import (
    GraniteGenerationConfig,
    NamedAdapterTextGenerator,
    PeftGraniteLLMClient,
)
from evidence_rag.generator.nli import TrueNLIModel
from evidence_rag.generator.verify_annotate import VerifyAnnotateGenerator
from evidence_rag.infrastructure.datasets import GoldCase

Mode = Literal["gn", "draft-pair"]
ALLOWED_SEEDS = (13, 42, 73)
FULL_CONTEXT = "K_topk"
STRESS_CONTEXTS = (
    "S_legacy_selected",
    "O_support_only",
    "OB_support_benign",
    "OH_support_harmful",
    "OP_support_last",
)
CONTEXTS = (FULL_CONTEXT, *STRESS_CONTEXTS)
EXPECTED_FULL_QUERIES = 739
EXPECTED_STRESS_QUERIES = 218
EXPECTED_TASKS = EXPECTED_FULL_QUERIES + EXPECTED_STRESS_QUERIES * len(
    STRESS_CONTEXTS
)
MAX_ERROR_RATE = 0.05
COVERAGE_NONINFERIORITY_MARGIN = -0.01
SUBSTANTIVE_FAILURE_REDUCTION = -0.01
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class EvalTask:
    task_id: str
    scope: Literal["full", "stress"]
    context: str
    query_id: str
    question: str
    component_id: str
    selector_changed: bool
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


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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
        "cudnn": torch.backends.cudnn.version(),
        "gpu_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
    }


def _stable_value(task_id: str, namespace: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"13:{namespace}:{task_id}".encode()).digest()[:8], "big"
    )


def arm_order(task_id: str, arms: Sequence[str]) -> tuple[str, ...]:
    if not arms:
        raise ValueError("G230 requires at least one arm")
    offset = _stable_value(task_id, "g230-arm-order") % len(arms)
    ordered = tuple(arms[offset:]) + tuple(arms[:offset])
    if _stable_value(task_id, "g230-arm-direction") % 2:
        ordered = tuple(reversed(ordered))
    return ordered


def build_tasks(
    full_cases: Sequence[joint.JointCase],
    stress_rows: Sequence[Mapping[str, object]],
) -> list[EvalTask]:
    tasks = [
        EvalTask(
            task_id=f"full::{FULL_CONTEXT}::{case.query_id}",
            scope="full",
            context=FULL_CONTEXT,
            query_id=case.query_id,
            question=case.question,
            component_id=case.component_id,
            selector_changed=case.selector_changed,
            evidence=case.topk10,
        )
        for case in full_cases
    ]
    for row in stress_rows:
        raw_contexts = row.get("contexts")
        if not isinstance(raw_contexts, Mapping):
            raise ValueError(f"stress query {row.get('query_id')} has no contexts")
        query_id = str(row["query_id"])
        for context in STRESS_CONTEXTS:
            raw_evidence = raw_contexts.get(context)
            if not isinstance(raw_evidence, Sequence):
                raise ValueError(f"stress query {query_id} lacks {context}")
            tasks.append(
                EvalTask(
                    task_id=f"stress::{context}::{query_id}",
                    scope="stress",
                    context=context,
                    query_id=query_id,
                    question=str(row["question"]),
                    component_id=str(row["component_id"]),
                    selector_changed=bool(row["selector_changed"]),
                    evidence=tuple(
                        EvidenceCandidate.model_validate(item) for item in raw_evidence
                    ),
                )
            )
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("duplicate G230 task ID")
    return tasks


def _validate_g220_run(
    run_dir: Path, *, arm: str, seed: int, model_config_sha256: str | None = None
) -> dict[str, object]:
    manifest_path = run_dir / "training_manifest.json"
    adapter_dir = run_dir / "adapter"
    weights_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    manifest = _json(manifest_path)
    checks = {
        "schema": manifest.get("schema_version")
        == "full-flow-g220-training-manifest-v1",
        "status": manifest.get("status") == "COMPLETE",
        "run_kind": manifest.get("run_kind") == "formal",
        "arm": manifest.get("arm") == arm,
        "seed": manifest.get("seed") == seed,
        "queries": manifest.get("queries") == 515,
        "training_examples": manifest.get("training_examples") == 4120,
        "validation_queries": manifest.get("validation_queries") == 62,
        "optimizer_steps": manifest.get("optimizer_steps") == 515,
        "max_length": manifest.get("max_length") == 2304,
        "truncation": manifest.get("truncated_examples") == 0,
        "model": model_config_sha256 is None
        or manifest.get("model_snapshot_config_sha256") == model_config_sha256,
        "dev_boundary": manifest.get("decision_dev_used") is False,
        "heldout_boundary": manifest.get("sealed_or_heldout_read") is False,
        "adapter_scope": manifest.get("adapter_scope") == "draft generation call only",
        "key_fact_scope": manifest.get("key_fact_extraction_scope") == "not used",
        "splitter_scope": manifest.get("claim_splitter_scope")
        == "frozen Granite base with adapter disabled",
        "reload": cast(Mapping[str, object], manifest.get("reload_check", {})).get(
            "status"
        )
        == "PASS",
        "weights": weights_path.is_file()
        and manifest.get("adapter_weights_sha256") == _sha256(weights_path),
        "adapter_config": config_path.is_file(),
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"invalid G220 {arm}/seed-{seed} run: {failed}")
    return {
        "run_dir": str(run_dir.resolve()),
        "training_manifest_sha256": _sha256(manifest_path),
        "adapter_config_sha256": _sha256(config_path),
        "adapter_weights_sha256": _sha256(weights_path),
    }


def _validate_gn_run(
    run_dir: Path,
    *,
    historical_eval_manifest_path: Path,
    model_config_sha256: str | None = None,
) -> dict[str, object]:
    training_manifest_path = run_dir / "training_manifest.json"
    adapter_dir = run_dir / "adapter"
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    training = _json(training_manifest_path)
    historical = _json(historical_eval_manifest_path)
    checks = {
        "schema": training.get("schema_version")
        == "full-flow-f006-training-manifest-v1",
        "status": training.get("status") == "COMPLETE",
        "arm": training.get("arm") == "mixed",
        "seed": training.get("seed") == 13,
        "model": model_config_sha256 is None
        or training.get("model_snapshot_config_sha256") == model_config_sha256,
        "dev_boundary": training.get("decision_dev_used") is False,
        "heldout_boundary": training.get("sealed_or_heldout_read") is False,
        "scope": historical.get("mixed_adapter_scope")
        in {"key-fact extraction only", "key-fact extraction call only"},
        "config": config_path.is_file()
        and historical.get("mixed_adapter_config_sha256") == _sha256(config_path),
        "weights": weights_path.is_file(),
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"invalid historical GN run: {failed}")
    return {
        "run_dir": str(run_dir.resolve()),
        "training_manifest_sha256": _sha256(training_manifest_path),
        "historical_eval_manifest_sha256": _sha256(historical_eval_manifest_path),
        "adapter_config_sha256": _sha256(config_path),
        "adapter_weights_sha256": _sha256(weights_path),
    }


def _routing_rows(generator: VerifyAnnotateGenerator) -> list[dict[str, object]]:
    return [
        {
            "claim_id": item.claim_id,
            "outcome": item.outcome,
            "sentence": item.sentence,
            "citation": item.citation,
            "claim_text": item.claim_text,
            "declared_indices": list(item.declared_indices),
            "declared_verified": item.declared_verified,
            "rescued_by_scan": item.rescued_by_scan,
            "gated_outcome": item.gated_outcome,
            "gated_citation": item.gated_citation,
            "review_flagged": item.review_flagged,
        }
        for item in generator.last_routings
    ]


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
            raise RuntimeError("trace-enabled G230 Generator produced no trace")
    except Exception as error:  # noqa: BLE001 -- runtime errors are measured outcomes
        traceback.print_exc()
        return {
            "generation": GenerationResult(
                query_id=query.query_id, answer="", cited_evidence_ids=()
            ).model_dump(mode="json"),
            "trace": None,
            "routing": [],
            "error": f"{type(error).__name__}: {error}",
            "seconds": time.perf_counter() - started,
        }
    return {
        "generation": generation.model_dump(mode="json"),
        "trace": trace.model_dump(mode="json"),
        "routing": _routing_rows(generator),
        "error": None,
        "seconds": time.perf_counter() - started,
    }


def build_generators(
    *,
    mode: Mode,
    client: PeftGraniteLLMClient,
    nli: TrueNLIModel,
) -> dict[str, VerifyAnnotateGenerator]:
    if mode == "gn":
        draft = KeyFactDraftAnswerGenerator(
            client,
            note_llm=NamedAdapterTextGenerator(client, "gn"),
            guided=False,
            trace_enabled=True,
        )
        return {
            "GN": VerifyAnnotateGenerator(
                draft_generator=draft,
                nli=nli,
                entity_gate="observe",
                abstain_when_unverified=False,
                trace_enabled=True,
            )
        }

    generators: dict[str, VerifyAnnotateGenerator] = {}
    for arm, adapter_name in (("GC", "gc"), ("GM", "gm")):
        draft = DraftAnswerGenerator(
            draft_generator=DraftGenerator(
                llm=NamedAdapterTextGenerator(client, adapter_name),
                trace_enabled=True,
            ),
            claim_splitter=ClaimSplitter(llm=client, trace_enabled=True),
            trace_enabled=True,
        )
        generators[arm] = VerifyAnnotateGenerator(
            draft_generator=draft,
            nli=nli,
            entity_gate="observe",
            abstain_when_unverified=False,
            trace_enabled=True,
        )
    return generators


def _task_identity(task: EvalTask) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "scope": task.scope,
        "context": task.context,
        "query_id": task.query_id,
        "component_id": task.component_id,
        "selector_changed": task.selector_changed,
        "evidence_ids": [item.evidence_id for item in task.evidence],
    }


def validate_resume_prefix(
    rows: Sequence[Mapping[str, Any]], tasks: Sequence[EvalTask], arms: Sequence[str]
) -> None:
    if len(rows) > len(tasks):
        raise ValueError("G230 resume has more rows than frozen tasks")
    for index, row in enumerate(rows):
        task = tasks[index]
        if row.get("task_id") != task.task_id:
            raise ValueError(f"G230 resume diverges at row {index + 1}")
        raw_arms = row.get("arms")
        if not isinstance(raw_arms, Mapping) or set(raw_arms) != set(arms):
            raise ValueError(f"G230 resume row {index + 1} has wrong arms")


def run(
    *,
    mode: Mode,
    seed: int | None,
    queries_path: Path,
    candidate_pool_path: Path,
    roles_path: Path,
    decision_trace_path: Path,
    stress_contexts_path: Path,
    a002_manifest_path: Path,
    b100_manifest_path: Path,
    model_snapshot: Path,
    true_snapshot: Path,
    gn_run_dir: Path | None,
    gn_eval_manifest_path: Path | None,
    gc_run_dir: Path | None,
    gm_run_dir: Path | None,
    output_dir: Path,
    limit_tasks: int | None = None,
) -> dict[str, object]:
    if mode == "gn" and seed is not None:
        raise ValueError("GN is the fixed historical seed-13 control; omit --seed")
    if mode == "draft-pair" and seed not in ALLOWED_SEEDS:
        raise ValueError(f"draft-pair seed must be one of {ALLOWED_SEEDS}")
    a002_manifest = _json(a002_manifest_path)
    b100_manifest = _json(b100_manifest_path)
    full_cases = joint.load_runtime_cases(
        queries_path=queries_path,
        candidate_pool_path=candidate_pool_path,
        roles_path=roles_path,
        decision_trace_path=decision_trace_path,
    )
    stress_rows = b100._runtime_contexts(stress_contexts_path)
    tasks = build_tasks(full_cases, stress_rows)
    if limit_tasks is None:
        if len(full_cases) != EXPECTED_FULL_QUERIES:
            raise ValueError("formal G230 full-dev count differs from 739")
        if len(stress_rows) != EXPECTED_STRESS_QUERIES or len(tasks) != EXPECTED_TASKS:
            raise ValueError("formal G230 stress/task counts differ from the frozen protocol")
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
        raise ValueError("G230 full-dev runtime inputs differ from frozen A002")
    if b100_manifest.get("status") != "COMPLETE" or b100_manifest.get(
        "runtime_contexts_sha256"
    ) != _sha256(stress_contexts_path):
        raise ValueError("G230 stress contexts differ from frozen B100")

    adapters: dict[str, str]
    adapter_audit: dict[str, object]
    model_config_sha256 = _sha256(model_snapshot / "config.json")
    if mode == "gn":
        if gn_run_dir is None or gn_eval_manifest_path is None:
            raise ValueError("GN requires --gn-run-dir and --gn-eval-manifest")
        adapter_audit = {
            "gn": _validate_gn_run(
                gn_run_dir,
                historical_eval_manifest_path=gn_eval_manifest_path,
                model_config_sha256=model_config_sha256,
            )
        }
        adapters = {"gn": str((gn_run_dir / "adapter").resolve())}
        expected_arms = ("GN",)
    else:
        assert seed is not None
        if gc_run_dir is None or gm_run_dir is None:
            raise ValueError("draft-pair requires --gc-run-dir and --gm-run-dir")
        adapter_audit = {
            "gc": _validate_g220_run(
                gc_run_dir,
                arm="gc",
                seed=seed,
                model_config_sha256=model_config_sha256,
            ),
            "gm": _validate_g220_run(
                gm_run_dir,
                arm="gm",
                seed=seed,
                model_config_sha256=model_config_sha256,
            ),
        }
        adapters = {
            "gc": str((gc_run_dir / "adapter").resolve()),
            "gm": str((gm_run_dir / "adapter").resolve()),
        }
        expected_arms = ("GC", "GM")

    spec: dict[str, object] = {
        "schema_version": "full-flow-g230-run-spec-v1",
        "mode": mode,
        "seed": seed,
        "arms": list(expected_arms),
        "tasks": len(tasks),
        "formal": limit_tasks is None,
        "task_identity_sha256": _sha256_bytes(
            _canonical_bytes([_task_identity(task) for task in tasks])
        ),
        "source_sha256": {
            "queries": _sha256(queries_path),
            "candidate_pool": _sha256(candidate_pool_path),
            "roles": _sha256(roles_path),
            "decision_trace": _sha256(decision_trace_path),
            "stress_contexts": _sha256(stress_contexts_path),
            "a002_manifest": _sha256(a002_manifest_path),
            "b100_manifest": _sha256(b100_manifest_path),
            "granite_config": model_config_sha256,
            "true_config": _sha256(true_snapshot / "config.json"),
        },
        "adapter_audit": adapter_audit,
        "prompt_sha256": {
            "draft": _sha256_bytes(DRAFT_PROMPT.encode()),
            "split": _sha256_bytes(SPLIT_PROMPT.encode()),
            "faithfulness": _sha256_bytes(FAITHFULNESS_PROMPT.encode()),
        },
        "decode": {
            "max_new_tokens": 256,
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
        },
        "gold_loaded_at_runtime": False,
        "adapter_scope": (
            "key-fact extraction call only"
            if mode == "gn"
            else "draft generation call only"
        ),
        "claim_splitter_scope": "frozen Granite base with adapters disabled",
        "true_scope": "frozen verifier",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / "run_spec.json"
    generations_path = output_dir / "generations.jsonl"
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        raise ValueError(f"G230 run is already complete: {manifest_path}")
    if spec_path.exists():
        if _json(spec_path) != spec:
            raise ValueError("G230 resume specification differs from the frozen run")
    else:
        if generations_path.exists() and generations_path.stat().st_size:
            raise ValueError("G230 generations exist without a run specification")
        _write_json(spec_path, spec)
    existing = _jsonl(generations_path) if generations_path.exists() else []
    validate_resume_prefix(existing, tasks, expected_arms)

    client = PeftGraniteLLMClient(
        model_id=str(model_snapshot.resolve()),
        adapters=adapters,
        config=GraniteGenerationConfig(max_new_tokens=256, temperature=0.0, top_p=1.0),
    )
    nli = TrueNLIModel(model_id=str(true_snapshot.resolve()))
    generators = build_generators(mode=mode, client=client, nli=nli)
    if tuple(generators) != expected_arms:
        raise AssertionError("G230 generator construction changed arm order")

    started = time.perf_counter()
    with generations_path.open("a", encoding="utf-8", newline="\n") as handle:
        for index, task in enumerate(tasks[len(existing) :], start=len(existing) + 1):
            query = Query(query_id=task.query_id, text=task.question)
            checklist = QueryChecklist(
                query_id=task.query_id, focus=task.question, required_facts=()
            )
            selected = SelectedEvidenceSet(query_id=task.query_id, evidence=task.evidence)
            order = arm_order(task.task_id, expected_arms)
            arms = {
                arm: _run_one(generators[arm], query, checklist, selected) for arm in order
            }
            row = {
                "schema_version": "full-flow-g230-generation-row-v1",
                **_task_identity(task),
                "question": task.question,
                "evidence": [item.model_dump(mode="json") for item in task.evidence],
                "arm_order": list(order),
                "arms": arms,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 10 == 0 or index == len(tasks):
                elapsed = time.perf_counter() - started
                completed_now = index - len(existing)
                print(
                    f"[G230 {mode}/{seed}] {index}/{len(tasks)} "
                    f"{elapsed / max(completed_now, 1):.2f}s/new-task",
                    flush=True,
                )

    rows = _jsonl(generations_path)
    validate_resume_prefix(rows, tasks, expected_arms)
    errors = {
        arm: sum(bool(cast(Mapping[str, object], row["arms"])[arm].get("error")) for row in rows)
        for arm in expected_arms
    }
    trace_missing = {
        arm: sum(
            cast(Mapping[str, object], row["arms"])[arm].get("trace") is None
            and not cast(Mapping[str, object], row["arms"])[arm].get("error")
            for row in rows
        )
        for arm in expected_arms
    }
    manifest: dict[str, object] = {
        "schema_version": "full-flow-g230-run-manifest-v1",
        "status": "COMPLETE",
        "git_commit": _git_commit(),
        "mode": mode,
        "seed": seed,
        "arms": list(expected_arms),
        "tasks": len(rows),
        "full_topk_tasks": sum(row["scope"] == "full" for row in rows),
        "stress_tasks_by_context": {
            context: sum(row["context"] == context for row in rows)
            for context in STRESS_CONTEXTS
        },
        "resumed_rows": len(existing),
        "attempts_by_arm": dict.fromkeys(expected_arms, len(rows)),
        "errors_by_arm": errors,
        "trace_missing_by_arm": trace_missing,
        "gold_loaded_at_runtime": False,
        "run_spec_sha256": _sha256(spec_path),
        "generations_sha256": _sha256(generations_path),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "environment": _runtime_environment(),
    }
    _write_json(manifest_path, manifest)
    broken = {
        arm: errors[arm] / len(rows)
        for arm in expected_arms
        if rows and errors[arm] / len(rows) > MAX_ERROR_RATE
    }
    if broken or any(trace_missing.values()):
        _write_json(
            output_dir / "INVALID_RUNTIME.json",
            {"excess_errors": broken, "trace_missing": trace_missing},
        )
        raise RuntimeError("G230 runtime exceeded its frozen failure boundary")
    return manifest


def _load_gold(path: Path, wanted: set[str]) -> dict[str, GoldCase]:
    output: dict[str, GoldCase] = {}
    for row in _jsonl(path):
        gold = GoldCase.model_validate(row)
        if gold.query_id in wanted:
            output[gold.query_id] = gold
    if set(output) != wanted:
        raise ValueError("gold does not exactly cover G230 query IDs")
    return output


def _baseline_results(
    a002_rows: Sequence[Mapping[str, Any]],
    b100_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, object]], dict[str, str]]:
    results: dict[str, Mapping[str, object]] = {}
    components: dict[str, str] = {}
    for row in a002_rows:
        query_id = str(row["query_id"])
        task_id = f"full::{FULL_CONTEXT}::{query_id}"
        arms = cast(Mapping[str, Sequence[Mapping[str, object]]], row["arms"])
        results[task_id] = arms["K_topk_base"][0]
        components[task_id] = str(row["component_id"])
    for row in b100_rows:
        query_id = str(row["query_id"])
        arms = cast(Mapping[str, Mapping[str, object]], row["arms"])
        for context in STRESS_CONTEXTS:
            task_id = f"stress::{context}::{query_id}"
            results[task_id] = arms[context]
            components[task_id] = str(row["component_id"])
    return results, components


def _candidate_results(
    path: Path, expected_arms: Sequence[str]
) -> tuple[dict[str, dict[str, Mapping[str, object]]], Mapping[str, Any]]:
    manifest = _json(path.parent / "run_manifest.json")
    if manifest.get("status") != "COMPLETE" or tuple(manifest.get("arms", ())) != tuple(
        expected_arms
    ):
        raise ValueError(f"invalid G230 candidate manifest for {path}")
    if manifest.get("generations_sha256") != _sha256(path):
        raise ValueError(f"G230 candidate generations hash differs for {path}")
    output: dict[str, dict[str, Mapping[str, object]]] = {
        arm: {} for arm in expected_arms
    }
    for row in _jsonl(path):
        task_id = str(row["task_id"])
        arms = row.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != set(expected_arms):
            raise ValueError(f"candidate task {task_id} has invalid arms")
        for arm in expected_arms:
            value = arms[arm]
            if not isinstance(value, Mapping):
                raise ValueError(f"candidate task {task_id}/{arm} is invalid")
            output[arm][task_id] = value
    return output, manifest


def _failure_flags(run_row: Mapping[str, object]) -> dict[str, float]:
    generation = GenerationResult.model_validate(run_row["generation"])
    trace = run_row.get("trace")
    if not isinstance(trace, Mapping):
        return {
            "coverage": float(bool(generation.answer.strip())),
            "draft_empty": 1.0,
            "zero_claims": 1.0,
            "final_empty": float(not generation.answer.strip()),
            "runtime_error": float(bool(run_row.get("error"))),
        }
    draft = trace.get("draft")
    normalized_draft = ""
    if isinstance(draft, Mapping):
        normalized_draft = str(draft.get("normalized_draft_text", ""))
    claims = trace.get("claims")
    claim_count = len(claims) if isinstance(claims, Sequence) else 0
    return {
        "coverage": float(bool(generation.answer.strip())),
        "draft_empty": float(not normalized_draft.strip()),
        "zero_claims": float(claim_count == 0),
        "final_empty": float(not generation.answer.strip()),
        "runtime_error": float(bool(run_row.get("error"))),
    }


def _scope_metrics(
    task_ids: Sequence[str],
    values: Mapping[str, Mapping[str, float]],
) -> dict[str, float | int]:
    metrics = ("answer_match", "coverage", "draft_empty", "zero_claims", "final_empty")
    return {
        "tasks": len(task_ids),
        **{
            metric: sum(values[task_id][metric] for task_id in task_ids) / len(task_ids)
            for metric in metrics
        },
        "runtime_errors": int(sum(values[task_id]["runtime_error"] for task_id in task_ids)),
    }


def _comparison(
    task_ids: Sequence[str],
    after: Mapping[str, Mapping[str, float]],
    before: Mapping[str, Mapping[str, float]],
    components: Mapping[str, str],
) -> dict[str, object]:
    output: dict[str, object] = {}
    for metric in ("answer_match", "coverage", "draft_empty", "zero_claims", "final_empty"):
        output[metric] = asdict(
            compare_paired(
                {task_id: after[task_id][metric] for task_id in task_ids},
                {task_id: before[task_id][metric] for task_id in task_ids},
                component_ids={task_id: components[task_id] for task_id in task_ids},
            )
        )
    output["answer_transitions"] = {
        "wrong_to_right": sum(
            before[task_id]["answer_match"] == 0.0
            and after[task_id]["answer_match"] == 1.0
            for task_id in task_ids
        ),
        "right_to_wrong": sum(
            before[task_id]["answer_match"] == 1.0
            and after[task_id]["answer_match"] == 0.0
            for task_id in task_ids
        ),
    }
    return output


def score(
    *,
    a002_generations_path: Path,
    b100_generations_path: Path,
    gn_generations_path: Path,
    seed_generations: Mapping[int, Path],
    gold_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if set(seed_generations) != set(ALLOWED_SEEDS):
        raise ValueError("G230 scoring requires exactly seeds 13, 42, and 73")
    baseline, components = _baseline_results(
        _jsonl(a002_generations_path), _jsonl(b100_generations_path)
    )
    expected_tasks = set(baseline)
    configs: dict[str, Mapping[str, Mapping[str, object]]] = {"G0": baseline}
    gn, gn_manifest = _candidate_results(gn_generations_path, ("GN",))
    configs["GN"] = gn["GN"]
    candidate_manifests: dict[str, object] = {"GN": gn_manifest}
    for seed in ALLOWED_SEEDS:
        candidates, manifest = _candidate_results(seed_generations[seed], ("GC", "GM"))
        if manifest.get("seed") != seed:
            raise ValueError(f"G230 candidate seed mismatch for {seed}")
        configs[f"GC{seed}"] = candidates["GC"]
        configs[f"GM{seed}"] = candidates["GM"]
        candidate_manifests[f"seed{seed}"] = manifest
    for name, rows in configs.items():
        if set(rows) != expected_tasks:
            raise ValueError(f"G230 {name} does not exactly cover frozen tasks")

    query_ids = {task_id.rsplit("::", 1)[-1] for task_id in expected_tasks}
    gold = _load_gold(gold_path, query_ids)
    values: dict[str, dict[str, dict[str, float]]] = {name: {} for name in configs}
    case_rows: list[dict[str, object]] = []
    for task_id in sorted(expected_tasks):
        query_id = task_id.rsplit("::", 1)[-1]
        task_values: dict[str, object] = {}
        for name, rows in configs.items():
            run_row = rows[task_id]
            generation = GenerationResult.model_validate(run_row["generation"])
            matched = answer_match(
                strip_annotations(generation.answer), gold[query_id].reference_answers
            ).value
            if matched is None:
                raise ValueError(f"query {query_id} has no scorable answer reference")
            values[name][task_id] = {
                "answer_match": float(matched),
                **_failure_flags(run_row),
            }
            task_values[name] = {
                **values[name][task_id],
                "answer": generation.answer,
                "cited_evidence_ids": list(generation.cited_evidence_ids),
            }
        scope, context, _query_id = task_id.split("::", 2)
        case_rows.append(
            {
                "schema_version": "full-flow-g230-scored-case-v1",
                "task_id": task_id,
                "scope": scope,
                "context": context,
                "query_id": query_id,
                "component_id": components[task_id],
                "configs": task_values,
            }
        )

    task_ids_by_context = {
        context: sorted(
            task_id
            for task_id in expected_tasks
            if task_id.split("::", 2)[1] == context
        )
        for context in CONTEXTS
    }
    aggregate = {
        name: {
            context: _scope_metrics(task_ids, values[name])
            for context, task_ids in task_ids_by_context.items()
        }
        for name in configs
    }
    comparisons: dict[str, object] = {}
    for seed in ALLOWED_SEEDS:
        for label, after_name, before_name in (
            (f"GM{seed}_minus_G0", f"GM{seed}", "G0"),
            (f"GC{seed}_minus_G0", f"GC{seed}", "G0"),
            (f"GM{seed}_minus_GC{seed}", f"GM{seed}", f"GC{seed}"),
            (f"GM{seed}_minus_GN", f"GM{seed}", "GN"),
        ):
            comparisons[label] = {
                context: _comparison(
                    task_ids,
                    values[after_name],
                    values[before_name],
                    components,
                )
                for context, task_ids in task_ids_by_context.items()
            }

    full = FULL_CONTEXT
    family_gate: dict[str, object] = {}
    for family in ("GC", "GM"):
        answer_direction = [
            cast(Mapping[str, Any], comparisons[f"{family}{seed}_minus_G0"])[full][
                "answer_match"
            ]["delta"]
            > 0
            for seed in ALLOWED_SEEDS
        ]
        coverage_noninferior = [
            cast(Mapping[str, Any], comparisons[f"{family}{seed}_minus_G0"])[full][
                "coverage"
            ]["ci_low"]
            >= COVERAGE_NONINFERIORITY_MARGIN
            for seed in ALLOWED_SEEDS
        ]
        failure_reduction = [
            any(
                cast(Mapping[str, Any], comparisons[f"{family}{seed}_minus_G0"])[
                    full
                ][metric]["delta"]
                <= SUBSTANTIVE_FAILURE_REDUCTION
                for metric in ("draft_empty", "zero_claims", "final_empty")
            )
            for seed in ALLOWED_SEEDS
        ]
        support_only = all(
            cast(Mapping[str, Any], comparisons[f"{family}{seed}_minus_G0"])[
                "O_support_only"
            ]["answer_match"]["delta"]
            > 0
            and cast(Mapping[str, Any], comparisons[f"{family}{seed}_minus_G0"])[
                "O_support_only"
            ]["final_empty"]["delta"]
            < 0
            for seed in ALLOWED_SEEDS
        )
        family_gate[family] = {
            "answer_point_above_g0_all_seeds": all(answer_direction),
            "answer_direction_by_seed": dict(
                zip(ALLOWED_SEEDS, answer_direction, strict=True)
            ),
            "coverage_ci_above_minus_1pp_all_seeds": all(coverage_noninferior),
            "coverage_noninferior_by_seed": dict(
                zip(ALLOWED_SEEDS, coverage_noninferior, strict=True)
            ),
            "major_failure_reduction_at_least_1pp_all_seeds": all(
                failure_reduction
            ),
            "failure_reduction_by_seed": dict(
                zip(ALLOWED_SEEDS, failure_reduction, strict=True)
            ),
            "support_only_answer_and_empty_improve_all_seeds": support_only,
        }
        cast(dict[str, object], family_gate[family])["pass_before_citation"] = all(
            (
                all(answer_direction),
                all(coverage_noninferior),
                all(failure_reduction),
                support_only,
            )
        )
    stress_direction = {
        context: [
            cast(Mapping[str, Any], comparisons[f"GM{seed}_minus_GC{seed}"])[context][
                "answer_match"
            ]["delta"]
            > 0
            for seed in ALLOWED_SEEDS
        ]
        for context in ("OB_support_benign", "OH_support_harmful", "OP_support_last")
    }
    gm_robustness_pass = all(all(flags) for flags in stress_direction.values())
    gm_pass = bool(cast(Mapping[str, object], family_gate["GM"])["pass_before_citation"])
    gc_pass = bool(cast(Mapping[str, object], family_gate["GC"])["pass_before_citation"])
    selected_candidate = "GM" if gm_pass and gm_robustness_pass else "GC" if gc_pass else None
    gate = {
        "family_gate": family_gate,
        "gm_beats_gc_stress_all_seeds": {
            context: all(flags) for context, flags in stress_direction.items()
        },
        "stress_direction_by_seed": stress_direction,
        "gm_mixed_context_robustness_pass": gm_robustness_pass,
        "candidate_before_citation": selected_candidate,
        "citation_gate": "PENDING_INDEPENDENT_MINICHECK",
    }
    gate["pre_citation_pass"] = selected_candidate is not None
    report: dict[str, object] = {
        "schema_version": "full-flow-g230-answer-report-v1",
        "status": "COMPLETE_PENDING_CITATION",
        "gold_loaded_at_runtime": False,
        "gold_loaded_by_post_generation_scorer": True,
        "full_distribution_context": FULL_CONTEXT,
        "stress_contexts": list(STRESS_CONTEXTS),
        "queries": len(query_ids),
        "tasks": len(expected_tasks),
        "aggregate": aggregate,
        "comparisons": comparisons,
        "development_gate": gate,
        "source_sha256": {
            "a002_generations": _sha256(a002_generations_path),
            "b100_generations": _sha256(b100_generations_path),
            "gn_generations": _sha256(gn_generations_path),
            **{
                f"seed{seed}_generations": _sha256(seed_generations[seed])
                for seed in ALLOWED_SEEDS
            },
            "gold": _sha256(gold_path),
        },
        "candidate_manifests": candidate_manifests,
    }
    return report, case_rows


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _markdown(report: Mapping[str, Any]) -> str:
    aggregate = cast(Mapping[str, Mapping[str, Mapping[str, Any]]], report["aggregate"])
    lines = [
        "# G230 Generator Development Gate",
        "",
        "## Full Decision-Dev (TopK)",
        "",
        "| Config | Answer | Coverage | Draft empty | Zero claims | Final empty | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("G0", "GN", "GC13", "GM13", "GC42", "GM42", "GC73", "GM73"):
        item = aggregate[name][FULL_CONTEXT]
        lines.append(
            f"| {name} | {_pct(item['answer_match'])} | {_pct(item['coverage'])} | "
            f"{_pct(item['draft_empty'])} | {_pct(item['zero_claims'])} | "
            f"{_pct(item['final_empty'])} | {item['runtime_errors']} |"
        )
    lines.extend(["", "## Pre-Citation Gate", "", "```json"])
    lines.append(
        json.dumps(report["development_gate"], ensure_ascii=False, indent=2)
    )
    lines.extend(["```", ""])
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--mode", required=True, choices=("gn", "draft-pair"))
    run_parser.add_argument("--seed", type=int, choices=ALLOWED_SEEDS)
    run_parser.add_argument("--queries", required=True, type=Path)
    run_parser.add_argument("--candidate-pool", required=True, type=Path)
    run_parser.add_argument("--roles", required=True, type=Path)
    run_parser.add_argument("--decision-trace", required=True, type=Path)
    run_parser.add_argument("--stress-contexts", required=True, type=Path)
    run_parser.add_argument("--a002-manifest", required=True, type=Path)
    run_parser.add_argument("--b100-manifest", required=True, type=Path)
    run_parser.add_argument("--model-snapshot", required=True, type=Path)
    run_parser.add_argument("--true-snapshot", required=True, type=Path)
    run_parser.add_argument("--gn-run-dir", type=Path)
    run_parser.add_argument("--gn-eval-manifest", type=Path)
    run_parser.add_argument("--gc-run-dir", type=Path)
    run_parser.add_argument("--gm-run-dir", type=Path)
    run_parser.add_argument("--output-dir", required=True, type=Path)
    run_parser.add_argument("--limit-tasks", type=int)

    score_parser = commands.add_parser("score")
    score_parser.add_argument("--a002-generations", required=True, type=Path)
    score_parser.add_argument("--b100-generations", required=True, type=Path)
    score_parser.add_argument("--gn-generations", required=True, type=Path)
    for seed in ALLOWED_SEEDS:
        score_parser.add_argument(f"--seed{seed}-generations", required=True, type=Path)
    score_parser.add_argument("--gold", required=True, type=Path)
    score_parser.add_argument("--output-json", required=True, type=Path)
    score_parser.add_argument("--output-cases", required=True, type=Path)
    score_parser.add_argument("--output-report", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        report = run(
            mode=args.mode,
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
            gn_run_dir=args.gn_run_dir,
            gn_eval_manifest_path=args.gn_eval_manifest,
            gc_run_dir=args.gc_run_dir,
            gm_run_dir=args.gm_run_dir,
            output_dir=args.output_dir.resolve(),
            limit_tasks=args.limit_tasks,
        )
        print(json.dumps(report, ensure_ascii=True, sort_keys=True))
        return 0

    seed_generations = {
        seed: cast(Path, getattr(args, f"seed{seed}_generations"))
        for seed in ALLOWED_SEEDS
    }
    report, cases = score(
        a002_generations_path=args.a002_generations,
        b100_generations_path=args.b100_generations,
        gn_generations_path=args.gn_generations,
        seed_generations=seed_generations,
        gold_path=args.gold,
    )
    _write_json(args.output_json, report)
    _write_jsonl(args.output_cases, cases)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
