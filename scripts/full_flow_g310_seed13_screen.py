"""Run and score the G310 seed-13 Generator recipe screen.

G310 compares only the two predeclared seed-13 recipes:

* GR-F: fresh draft LoRA from frozen Granite base.
* GR-C: continuation from the frozen GM13 adapter.

The run command uses G223 validation answerable cases plus the G223 train
unsupported cases as a safety screen.  It runs the full Generator path:
adapter-enabled draft call, frozen-base claim splitter, and frozen TRUE routing.
Gold/reference answers are not read during generation; scoring is a separate
command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

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

Arm = Literal["G0", "GR-F", "GR-C"]

ARMS: tuple[Arm, ...] = ("G0", "GR-F", "GR-C")
ANSWERABLE_SCOPE = "validation-answerable"
UNSUPPORTED_SCOPE = "train-unsupported-safety"
SCHEMA_G223_MANIFEST = "full-flow-g223-residual-sample-failure-candidate-manifest-v1"
SCHEMA_G300_TRAINING = "full-flow-g300-draft-lora-training-manifest-v1"
SCHEMA_TASK = "full-flow-g310-screen-task-v1"
SCHEMA_RUN_SPEC = "full-flow-g310-run-spec-v1"
SCHEMA_RUN_MANIFEST = "full-flow-g310-run-manifest-v1"
SCHEMA_GENERATION_ROW = "full-flow-g310-generation-row-v1"
SCHEMA_SCORE_REPORT = "full-flow-g310-score-report-v1"
SCHEMA_SCORE_ROW = "full-flow-g310-scored-row-v1"
FROZEN_MINICHECK_MODEL = "lytang/MiniCheck-Flan-T5-Large"
ALLOWED_SEED = 13
EXPECTED_TRAIN_GROUPS = 2370
EXPECTED_VALIDATION_GROUPS = 302
EXPECTED_TWOWIKI_MODELVAL_GROUPS = 95
CITATION_RE = re.compile(r"\[(\d+)\]")
EVIDENCE_BLOCK_RE = re.compile(r"\nEvidence:\n(?P<body>.*?)\n\nQuestion:", re.S)
QUESTION_RE = re.compile(r"\nQuestion:\s*(?P<question>.*?)\nAnswer:", re.S)
EVIDENCE_LINE_RE = re.compile(r"^\[(?P<rank>\d+)\]\s+\((?P<evidence_id>[^)]+)\)\s+(?P<text>.*)$")
UNKNOWN_ANSWERS = {
    "",
    "unknown",
    "i don't know",
    "i do not know",
    "don't know",
    "do not know",
    "not in the evidence",
    "not contained in the evidence",
    "not in the context",
    "not contained in the context",
}


@dataclass(frozen=True, slots=True)
class ScreenTask:
    task_id: str
    scope: str
    dataset: str
    case_id: str
    query_id: str
    component_id: str
    variant_name: str
    question: str
    prompt: str
    reference_answers: tuple[str, ...]
    answerable: bool
    expected_unknown: bool
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
        cwd=Path(__file__).resolve().parents[1],
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


def _load_g223_manifest(
    *,
    data_manifest_path: Path,
    train_cases_path: Path,
    validation_cases_path: Path,
    ordered_ids_path: Path,
) -> Mapping[str, Any]:
    manifest = _json(data_manifest_path)
    if manifest.get("schema_version") != SCHEMA_G223_MANIFEST:
        raise ValueError("G310 expects the G223 controlled-continuation manifest")
    if manifest.get("status") != "CONTROLLED_CONTINUATION_READY":
        raise ValueError("G310 requires G223 status CONTROLLED_CONTINUATION_READY")
    if manifest.get("g300_entry_mode") != "controlled_continuation_limited_internal_screen":
        raise ValueError("G310 requires G223 limited entry mode")
    if manifest.get("clean_freeze_ready") is not False:
        raise ValueError("G310 input must not be a clean freeze")
    if manifest.get("dev_read") is not False or manifest.get("sealed_or_heldout_read") is not False:
        raise ValueError("G310 input must not have read dev/sealed/held-out data")
    if str(manifest.get("train_cases_sha256", "")) != _sha256(train_cases_path):
        raise ValueError("G223 train cases hash mismatch")
    if str(manifest.get("validation_cases_sha256", "")) != _sha256(validation_cases_path):
        raise ValueError("G223 validation cases hash mismatch")
    if str(manifest.get("ordered_ids_sha256", "")) != _sha256(ordered_ids_path):
        raise ValueError("G223 ordered IDs hash mismatch")
    counts = manifest.get("counts", {})
    if isinstance(counts, Mapping):
        twowiki = counts.get("2wiki", {})
        if isinstance(twowiki, Mapping) and int(
            twowiki.get("train-modelval_answerable_groups", 0)
        ) != EXPECTED_TWOWIKI_MODELVAL_GROUPS:
            raise ValueError("G310 requires G223 2Wiki model-val screen=95")
    return manifest


def _validate_training_run(run_dir: Path, *, recipe: str) -> dict[str, object]:
    manifest_path = run_dir / "training_manifest.json"
    adapter_dir = run_dir / "adapter"
    weights_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    manifest = _json(manifest_path)
    checks = {
        "schema": manifest.get("schema_version") == SCHEMA_G300_TRAINING,
        "status": manifest.get("status") == "COMPLETE",
        "run_kind": manifest.get("run_kind") == "formal",
        "recipe": manifest.get("recipe") == recipe,
        "seed": manifest.get("seed") == ALLOWED_SEED,
        "entry": manifest.get("entry_mode")
        == "controlled_continuation_limited_internal_screen",
        "clean_freeze": manifest.get("clean_freeze_ready") is False,
        "full_train_groups": manifest.get("full_data_train_groups") == EXPECTED_TRAIN_GROUPS,
        "full_validation_groups": manifest.get("full_data_validation_groups")
        == EXPECTED_VALIDATION_GROUPS,
        "train_groups": manifest.get("train_groups") == EXPECTED_TRAIN_GROUPS,
        "formal_training": manifest.get("formal_training_started") is True,
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
        raise ValueError(f"invalid G310 {recipe} training run: {failed}")
    return {
        "run_dir": str(run_dir.resolve()),
        "training_manifest_sha256": _sha256(manifest_path),
        "adapter_config_sha256": _sha256(config_path),
        "adapter_weights_sha256": _sha256(weights_path),
        "mean_group_weighted_loss": manifest.get("mean_group_weighted_loss"),
        "validation_loss": manifest.get("validation_loss"),
    }


def _extract_question(prompt: str) -> str:
    match = QUESTION_RE.search(prompt)
    if match is None:
        raise ValueError("prompt lacks a Question block")
    return " ".join(match.group("question").split())


def parse_prompt_evidence(
    *,
    prompt: str,
    case_id: str,
    variant_name: str,
) -> tuple[EvidenceCandidate, ...]:
    match = EVIDENCE_BLOCK_RE.search(prompt)
    if match is None:
        raise ValueError(f"{case_id}/{variant_name} lacks an Evidence block")
    evidence: list[EvidenceCandidate] = []
    expected_rank = 1
    for line in match.group("body").splitlines():
        item = EVIDENCE_LINE_RE.match(line)
        if item is None:
            raise ValueError(f"{case_id}/{variant_name} has malformed evidence line")
        rank = int(item.group("rank"))
        if rank != expected_rank:
            raise ValueError(f"{case_id}/{variant_name} evidence ranks are not consecutive")
        evidence_id = item.group("evidence_id")
        evidence.append(
            EvidenceCandidate(
                evidence_id=evidence_id,
                document_id=f"g223::{evidence_id}",
                chunk_id=f"g223::{evidence_id}::chunk",
                text=item.group("text"),
                source_uri=f"g223://{case_id}/{variant_name}/{rank}",
                retrieval_score=float(1.0 / rank),
                retrieval_rank=rank,
            )
        )
        expected_rank += 1
    if not evidence:
        raise ValueError(f"{case_id}/{variant_name} has no evidence")
    return tuple(evidence)


def _support_ids(variant: Mapping[str, Any]) -> tuple[str, ...]:
    raw = variant.get("support_evidence_ids")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return tuple(str(item) for item in raw)
    single = variant.get("support_evidence_id")
    return (str(single),) if single else ()


def build_tasks(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    include_unsupported_train: bool = True,
) -> list[ScreenTask]:
    tasks: list[ScreenTask] = []
    for row in validation_rows:
        if not bool(row.get("answerable")):
            raise ValueError("G223 validation is expected to contain answerable rows only")
        variants = row.get("variants")
        if not isinstance(variants, Mapping):
            raise ValueError(f"{row.get('case_id')} lacks variants")
        references = {
            str(row.get("answer", "")).strip(),
            str(row.get("official_answer", "")).strip(),
        }
        references = {item for item in references if item}
        if not references:
            raise ValueError(f"{row.get('case_id')} lacks answer references")
        for variant_name, raw_variant in variants.items():
            if not isinstance(raw_variant, Mapping):
                raise ValueError(f"{row.get('case_id')}/{variant_name} is invalid")
            prompt = str(raw_variant.get("prompt", ""))
            evidence = parse_prompt_evidence(
                prompt=prompt,
                case_id=str(row["case_id"]),
                variant_name=str(variant_name),
            )
            tasks.append(
                ScreenTask(
                    task_id=f"{ANSWERABLE_SCOPE}::{row['case_id']}::{variant_name}",
                    scope=ANSWERABLE_SCOPE,
                    dataset=str(row.get("dataset", "")),
                    case_id=str(row["case_id"]),
                    query_id=str(row.get("query_id", row["case_id"])),
                    component_id=str(row.get("component_id", row["case_id"])),
                    variant_name=str(variant_name),
                    question=str(row.get("question") or _extract_question(prompt)),
                    prompt=prompt,
                    reference_answers=tuple(sorted(references)),
                    answerable=True,
                    expected_unknown=False,
                    target_kind=str(row.get("target_kind", "")),
                    evidence=evidence,
                    support_evidence_ids=_support_ids(raw_variant),
                )
            )
    if include_unsupported_train:
        for row in train_rows:
            if bool(row.get("answerable")):
                continue
            variants = row.get("variants")
            if not isinstance(variants, Mapping):
                raise ValueError(f"{row.get('case_id')} lacks variants")
            for variant_name, raw_variant in variants.items():
                if not isinstance(raw_variant, Mapping):
                    raise ValueError(f"{row.get('case_id')}/{variant_name} is invalid")
                prompt = str(raw_variant.get("prompt", ""))
                evidence = parse_prompt_evidence(
                    prompt=prompt,
                    case_id=str(row["case_id"]),
                    variant_name=str(variant_name),
                )
                tasks.append(
                    ScreenTask(
                        task_id=f"{UNSUPPORTED_SCOPE}::{row['case_id']}::{variant_name}",
                        scope=UNSUPPORTED_SCOPE,
                        dataset=str(row.get("dataset", "")),
                        case_id=str(row["case_id"]),
                        query_id=str(row.get("query_id", row["case_id"])),
                        component_id=str(row.get("component_id", row["case_id"])),
                        variant_name=str(variant_name),
                        question=str(row.get("question") or _extract_question(prompt)),
                        prompt=prompt,
                        reference_answers=(),
                        answerable=False,
                        expected_unknown=True,
                        target_kind=str(row.get("target_kind", "")),
                        evidence=evidence,
                        support_evidence_ids=(),
                    )
                )
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("duplicate G310 task ID")
    return tasks


def _task_identity(task: ScreenTask) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "scope": task.scope,
        "dataset": task.dataset,
        "case_id": task.case_id,
        "variant_name": task.variant_name,
        "component_id": task.component_id,
        "answerable": task.answerable,
        "evidence_ids": [item.evidence_id for item in task.evidence],
        "support_evidence_ids": list(task.support_evidence_ids),
        "reference_answers": list(task.reference_answers),
    }


def _stable_value(task_id: str, namespace: str) -> int:
    return int.from_bytes(hashlib.sha256(f"13:{namespace}:{task_id}".encode()).digest()[:8], "big")


def arm_order(task_id: str) -> tuple[Arm, ...]:
    arms = list(ARMS)
    offset = _stable_value(task_id, "g310-arm-order") % len(arms)
    ordered = arms[offset:] + arms[:offset]
    if _stable_value(task_id, "g310-arm-direction") % 2:
        ordered = list(reversed(ordered))
    return cast(tuple[Arm, ...], tuple(ordered))


def validate_resume_prefix(rows: Sequence[Mapping[str, Any]], tasks: Sequence[ScreenTask]) -> None:
    if len(rows) > len(tasks):
        raise ValueError("G310 resume has more rows than frozen tasks")
    for index, row in enumerate(rows):
        task = tasks[index]
        if row.get("task_id") != task.task_id:
            raise ValueError(f"G310 resume diverges at row {index + 1}")
        arms = row.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
            raise ValueError(f"G310 resume row {index + 1} has wrong arms")


def _routing_rows(generator: VerifyAnnotateGenerator) -> list[dict[str, object]]:
    return [
        {
            "claim_id": item.claim_id,
            "outcome": item.outcome,
            "sentence": item.sentence,
            "citation": item.citation,
            "claim_text": item.claim_text,
            "routing_hypothesis": item.routing_hypothesis,
            "declared_indices": list(item.declared_indices),
            "declared_verified": item.declared_verified,
            "rescued_by_scan": item.rescued_by_scan,
            "attachment_verified": item.outcome == "verified" and item.citation is not None,
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
            raise RuntimeError("trace-enabled G310 Generator produced no trace")
    except Exception as error:  # noqa: BLE001
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
        "routing": _routing_rows(generator),
        "error": None,
        "seconds": time.perf_counter() - started,
    }


def build_generators(
    *,
    client: PeftGraniteLLMClient,
    nli: TrueNLIModel,
) -> dict[Arm, VerifyAnnotateGenerator]:
    output: dict[Arm, VerifyAnnotateGenerator] = {}
    for arm, llm in (
        ("G0", client),
        ("GR-F", NamedAdapterTextGenerator(client, "grf")),
        ("GR-C", NamedAdapterTextGenerator(client, "grc")),
    ):
        draft = DraftAnswerGenerator(
            draft_generator=DraftGenerator(llm=llm, trace_enabled=True),
            claim_splitter=ClaimSplitter(llm=client, trace_enabled=True),
            trace_enabled=True,
        )
        output[cast(Arm, arm)] = VerifyAnnotateGenerator(
            draft_generator=draft,
            nli=nli,
            entity_gate="observe",
            abstain_when_unverified=False,
            trace_enabled=True,
        )
    return output


def run(
    *,
    model_snapshot: Path,
    true_snapshot: Path,
    data_manifest_path: Path,
    train_cases_path: Path,
    validation_cases_path: Path,
    ordered_ids_path: Path,
    grf_run_dir: Path,
    grc_run_dir: Path,
    output_dir: Path,
    limit_tasks: int | None = None,
    include_unsupported_train: bool = True,
) -> dict[str, object]:
    g223 = _load_g223_manifest(
        data_manifest_path=data_manifest_path,
        train_cases_path=train_cases_path,
        validation_cases_path=validation_cases_path,
        ordered_ids_path=ordered_ids_path,
    )
    train_rows = _jsonl(train_cases_path)
    validation_rows = _jsonl(validation_cases_path)
    tasks = build_tasks(
        train_rows=train_rows,
        validation_rows=validation_rows,
        include_unsupported_train=include_unsupported_train,
    )
    formal_task_count = len(tasks)
    if limit_tasks is not None:
        if limit_tasks <= 0:
            raise ValueError("--limit-tasks must be positive")
        tasks = tasks[:limit_tasks]
    grf_audit = _validate_training_run(grf_run_dir, recipe="gr-f")
    grc_audit = _validate_training_run(grc_run_dir, recipe="gr-c")

    spec: dict[str, object] = {
        "schema_version": SCHEMA_RUN_SPEC,
        "stage": "G310",
        "seed": ALLOWED_SEED,
        "arms": list(ARMS),
        "formal": limit_tasks is None,
        "tasks": len(tasks),
        "formal_task_count": formal_task_count,
        "include_unsupported_train": include_unsupported_train,
        "entry_mode": "controlled_continuation_limited_internal_screen",
        "clean_freeze_ready": False,
        "g223_manifest_sha256": _sha256(data_manifest_path),
        "g223_twowiki_modelval_groups": int(
            cast(Mapping[str, Any], cast(Mapping[str, Any], g223.get("counts", {})).get("2wiki", {})).get(
                "train-modelval_answerable_groups",
                0,
            )
        ),
        "task_identity_sha256": _sha256_bytes(_canonical_bytes([_task_identity(task) for task in tasks])),
        "source_sha256": {
            "script": _sha256(Path(__file__)),
            "train_cases": _sha256(train_cases_path),
            "validation_cases": _sha256(validation_cases_path),
            "ordered_ids": _sha256(ordered_ids_path),
            "granite_config": _sha256(model_snapshot / "config.json"),
            "true_config": _sha256(true_snapshot / "config.json"),
        },
        "adapter_audit": {
            "grf": grf_audit,
            "grc": grc_audit,
        },
        "decode": {
            "max_new_tokens": 256,
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
        },
        "gold_loaded_at_runtime": False,
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
        raise ValueError(f"G310 run is already complete: {manifest_path}")
    if spec_path.exists():
        if _json(spec_path) != spec:
            raise ValueError("G310 resume specification differs from frozen run")
    else:
        if generations_path.exists() and generations_path.stat().st_size:
            raise ValueError("G310 generations exist without a run specification")
        _write_json(spec_path, spec)
    existing = _jsonl(generations_path) if generations_path.exists() else []
    validate_resume_prefix(existing, tasks)

    client = PeftGraniteLLMClient(
        model_id=str(model_snapshot.resolve()),
        adapters={
            "grf": str((grf_run_dir / "adapter").resolve()),
            "grc": str((grc_run_dir / "adapter").resolve()),
        },
        config=GraniteGenerationConfig(max_new_tokens=256, temperature=0.0, top_p=1.0),
    )
    nli = TrueNLIModel(model_id=str(true_snapshot.resolve()))
    generators = build_generators(client=client, nli=nli)
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
            order = arm_order(task.task_id)
            arms = {
                arm: _run_one(generators[arm], query, checklist, selected) for arm in order
            }
            row = {
                "schema_version": SCHEMA_GENERATION_ROW,
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
                    f"[G310 run] {index}/{len(tasks)} "
                    f"{elapsed / max(completed_now, 1):.2f}s/new-task",
                    flush=True,
                )
    rows = _jsonl(generations_path)
    validate_resume_prefix(rows, tasks)
    manifest = {
        "schema_version": SCHEMA_RUN_MANIFEST,
        "stage": "G310",
        "status": "COMPLETE",
        "tasks": len(tasks),
        "formal": limit_tasks is None,
        "generations_sha256": _sha256(generations_path),
        "run_spec_sha256": _sha256(spec_path),
        "elapsed_seconds": time.perf_counter() - started,
        "gold_loaded_at_runtime": False,
        "dev_read": False,
        "sealed_or_heldout_read": False,
        "utility_labels_started": False,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _is_unknown(answer: str) -> bool:
    normalized = strip_annotations(answer).strip().lower().strip(".!?\"' ")
    return normalized in UNKNOWN_ANSWERS


def _draft_invalid_citations(trace: object, evidence_count: int) -> int:
    if not isinstance(trace, Mapping):
        return 0
    draft = trace.get("draft")
    if not isinstance(draft, Mapping):
        return 0
    raw = str(draft.get("raw_draft_text", ""))
    invalid = 0
    for match in CITATION_RE.findall(raw):
        index = int(match)
        if index < 1 or index > evidence_count:
            invalid += 1
    return invalid


def _citation_example(row: Mapping[str, Any], arm: Arm) -> ScoredExample | None:
    arms = row.get("arms")
    if not isinstance(arms, Mapping) or not isinstance(arms.get(arm), Mapping):
        raise ValueError(f"{row.get('task_id')} lacks arm {arm}")
    arm_row = cast(Mapping[str, Any], arms[arm])
    routing = arm_row.get("routing")
    if not isinstance(routing, list):
        if arm_row.get("error"):
            return None
        raise ValueError(f"{row.get('task_id')} {arm} has invalid routing")
    docs: dict[str, str] = {}
    raw_evidence = row.get("evidence")
    if not isinstance(raw_evidence, Sequence) or isinstance(raw_evidence, (str, bytes)):
        raise ValueError(f"{row.get('task_id')} lacks evidence payload")
    for item in raw_evidence:
        if not isinstance(item, Mapping):
            raise ValueError(f"{row.get('task_id')} has invalid evidence item")
        docs[str(item["evidence_id"])] = str(item.get("text", ""))
    sentences: list[str] = []
    citations: list[tuple[str, ...]] = []
    for item in routing:
        if not isinstance(item, Mapping):
            raise ValueError(f"{row.get('task_id')} {arm} has invalid routing item")
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
        example_id=f"{row['task_id']}::{arm}",
        sentences=tuple(sentences),
        citations=tuple(citations),
        docs=docs,
    )


def _minicheck_task_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    entails: Callable[[str, str], bool],
) -> dict[str, dict[Arm, dict[str, float]]]:
    output: dict[str, dict[Arm, dict[str, float]]] = {}
    for arm in ARMS:
        examples = [
            example
            for row in rows
            if (example := _citation_example(row, arm)) is not None
        ]
        report = compute_citation_metrics(examples, entails)
        by_id = {str(item["example_id"]): item for item in report.per_example}
        for row in rows:
            task_id = str(row["task_id"])
            item = by_id.get(f"{task_id}::{arm}")
            output.setdefault(task_id, {})[arm] = {
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


def _arm_metrics(
    row: Mapping[str, Any],
    arm: Arm,
    minicheck_metrics: Mapping[str, float],
) -> dict[str, float]:
    arms = row.get("arms")
    if not isinstance(arms, Mapping) or not isinstance(arms.get(arm), Mapping):
        raise ValueError(f"{row.get('task_id')} lacks arm {arm}")
    arm_row = cast(Mapping[str, Any], arms[arm])
    generation = GenerationResult.model_validate(arm_row["generation"])
    error = bool(arm_row.get("error"))
    answer = strip_annotations(generation.answer)
    coverage = float(bool(answer.strip()) and not _is_unknown(answer))
    answerable = bool(row.get("answerable"))
    references = row.get("reference_answers")
    reference_values = [str(item) for item in references] if isinstance(references, Sequence) else []
    matched = 0.0
    if answerable:
        value = answer_match(answer, reference_values).value
        matched = float(value) if value is not None else 0.0
    support = set(str(item) for item in row.get("support_evidence_ids", []) or [])
    cited = set(str(item) for item in generation.cited_evidence_ids)
    support_recall = (len(support & cited) / len(support)) if support else 0.0
    support_precision = (len(support & cited) / len(cited)) if cited else 0.0
    invalid = _draft_invalid_citations(
        arm_row.get("trace"),
        len(row.get("evidence", []) if isinstance(row.get("evidence"), Sequence) else []),
    )
    minicheck_precision = float(minicheck_metrics["minicheck_citation_precision"])
    minicheck_recall = float(minicheck_metrics["minicheck_citation_recall"])
    return {
        "runtime_error": float(error),
        "missing_trace": float(arm_row.get("trace") is None and not error),
        "draft_invalid_citation": float(invalid > 0),
        "answer_match": matched,
        "coverage": coverage,
        "final_empty": float(coverage == 0.0),
        "support_citation_recall": support_recall if answerable else 0.0,
        "support_citation_precision": support_precision if answerable else 0.0,
        "minicheck_citation_precision": minicheck_precision if answerable else 0.0,
        "minicheck_citation_recall": minicheck_recall if answerable else 0.0,
        "minicheck_sentences": float(minicheck_metrics["minicheck_sentences"]),
        "minicheck_citations": float(minicheck_metrics["minicheck_citations"]),
        "correct_and_cited": float(
            answerable
            and matched == 1.0
            and minicheck_precision >= 1.0
            and minicheck_recall >= 1.0
            and invalid == 0
            and coverage == 1.0
        ),
        "unsupported_ungrounded_assertion": float((not answerable) and coverage == 1.0),
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate_groups(
    rows: Sequence[Mapping[str, Any]],
    per_task: Mapping[str, Mapping[Arm, Mapping[str, float]]],
    *,
    predicate: Any,
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for arm in ARMS:
        by_case: dict[str, dict[str, list[float]]] = {}
        for row in rows:
            if not predicate(row):
                continue
            case_id = str(row["case_id"])
            metrics = per_task[str(row["task_id"])][arm]
            case_metrics = by_case.setdefault(case_id, defaultdict(list))
            for metric, value in metrics.items():
                case_metrics[metric].append(float(value))
        case_values = {
            case_id: {metric: _mean(values) for metric, values in metrics.items()}
            for case_id, metrics in by_case.items()
        }
        metrics_out: dict[str, float | int] = {"case_groups": len(case_values)}
        metric_names = sorted({metric for values in case_values.values() for metric in values})
        for metric in metric_names:
            metrics_out[metric] = _mean(
                [float(values[metric]) for values in case_values.values() if metric in values]
            )
        output[arm] = metrics_out
    return output


def _delta(after: Mapping[str, float | int], before: Mapping[str, float | int], key: str) -> float:
    return float(after.get(key, 0.0)) - float(before.get(key, 0.0))


def _screen_decision(aggregate: Mapping[str, Mapping[str, Mapping[str, float | int]]]) -> dict[str, object]:
    decisions: dict[str, object] = {}
    for arm in ("GR-F", "GR-C"):
        failures: list[str] = []
        answerable = aggregate["validation_answerable"][arm]
        baseline_answerable = aggregate["validation_answerable"]["G0"]
        unsupported = aggregate["train_unsupported_safety"][arm]
        baseline_unsupported = aggregate["train_unsupported_safety"]["G0"]
        if float(answerable.get("runtime_error", 0.0)) > 0:
            failures.append("runtime_error_nonzero")
        if float(answerable.get("missing_trace", 0.0)) > 0:
            failures.append("missing_trace_nonzero")
        if float(answerable.get("draft_invalid_citation", 0.0)) > 0:
            failures.append("invalid_citation_nonzero")
        if _delta(answerable, baseline_answerable, "minicheck_citation_precision") < -0.03:
            failures.append("citation_precision_regression_gt_3pp")
        if _delta(answerable, baseline_answerable, "minicheck_citation_recall") < -0.03:
            failures.append("citation_recall_regression_gt_3pp")
        if _delta(answerable, baseline_answerable, "answer_match") < -0.02:
            failures.append("answer_regression_gt_2pp")
        if _delta(answerable, baseline_answerable, "coverage") < -0.02:
            failures.append("coverage_regression_gt_2pp")
        if _delta(unsupported, baseline_unsupported, "unsupported_ungrounded_assertion") > 0.02:
            failures.append("unsupported_assertion_increase_gt_2pp")
        if _delta(answerable, baseline_answerable, "final_empty") > 0.02:
            failures.append("final_empty_increase_gt_2pp")
        decisions[arm] = {
            "survives": not failures,
            "failures": failures,
            "validation_correct_and_cited_delta": _delta(
                answerable, baseline_answerable, "correct_and_cited"
            ),
            "niah_correct_and_cited_delta": _delta(
                aggregate["validation_niah"][arm],
                aggregate["validation_niah"]["G0"],
                "correct_and_cited",
            ),
            "twowiki_correct_and_cited_delta": _delta(
                aggregate["validation_2wiki"][arm],
                aggregate["validation_2wiki"]["G0"],
                "correct_and_cited",
            ),
        }
    survivors = [
        arm
        for arm in ("GR-F", "GR-C")
        if cast(Mapping[str, object], decisions[arm])["survives"]
    ]
    if not survivors:
        return {"recipe": None, "status": "NO_SURVIVING_RECIPE", "decisions": decisions}
    ranked = sorted(
        survivors,
        key=lambda arm: (
            min(
                float(cast(Mapping[str, object], decisions[arm])["niah_correct_and_cited_delta"]),
                float(cast(Mapping[str, object], decisions[arm])["twowiki_correct_and_cited_delta"]),
            ),
            1 if arm == "GR-F" else 0,
        ),
        reverse=True,
    )
    return {"recipe": ranked[0], "status": "RECIPE_SELECTED", "decisions": decisions}


def score(
    *,
    generations_path: Path,
    output_json: Path,
    output_rows: Path,
    output_report: Path,
    minicheck_model_id: str = FROZEN_MINICHECK_MODEL,
    minicheck_device: str = "cpu",
    entails: Callable[[str, str], bool] | None = None,
) -> dict[str, object]:
    rows = _jsonl(generations_path)
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
    minicheck_by_task = _minicheck_task_metrics(rows, entails=entails)
    per_task: dict[str, dict[Arm, dict[str, float]]] = {}
    scored_rows: list[dict[str, object]] = []
    for row in rows:
        task_id = str(row["task_id"])
        per_task[task_id] = {
            arm: _arm_metrics(row, arm, minicheck_by_task[task_id][arm])
            for arm in ARMS
        }
        scored_rows.append(
            {
                "schema_version": SCHEMA_SCORE_ROW,
                "task_id": task_id,
                "scope": row["scope"],
                "dataset": row["dataset"],
                "case_id": row["case_id"],
                "variant_name": row["variant_name"],
                "arms": per_task[task_id],
            }
        )
    aggregate = {
        "validation_answerable": _aggregate_groups(
            rows, per_task, predicate=lambda row: row.get("scope") == ANSWERABLE_SCOPE
        ),
        "validation_niah": _aggregate_groups(
            rows,
            per_task,
            predicate=lambda row: row.get("scope") == ANSWERABLE_SCOPE
            and row.get("dataset") == "niah",
        ),
        "validation_2wiki": _aggregate_groups(
            rows,
            per_task,
            predicate=lambda row: row.get("scope") == ANSWERABLE_SCOPE
            and row.get("dataset") == "2wiki",
        ),
        "train_unsupported_safety": _aggregate_groups(
            rows, per_task, predicate=lambda row: row.get("scope") == UNSUPPORTED_SCOPE
        ),
    }
    decision = _screen_decision(aggregate)
    report: dict[str, object] = {
        "schema_version": SCHEMA_SCORE_REPORT,
        "stage": "G310",
        "status": decision["status"],
        "screen_type": "full_generator_true_routed_minicheck_citation_screen",
        "citation_judge": {
            "model": "MiniCheck-Flan-T5-Large",
            "model_id": minicheck_model_id,
            "device": minicheck_device,
            "unique_judge_calls": judge_call_count(),
            "production_true_excluded_from_judging": True,
        },
        "tasks": len(rows),
        "arms": list(ARMS),
        "aggregate": aggregate,
        "decision": decision,
        "boundaries": {
            "gold_loaded_at_runtime": False,
            "score_uses_references_after_generation": True,
            "sealed_or_heldout_read": False,
            "utility_labels_started": False,
            "not_a_final_generator_qualification": True,
        },
    }
    _write_json(output_json, report)
    _write_jsonl(output_rows, scored_rows)
    lines = [
        "# G310 seed13 screen report",
        "",
        f"**Status:** `{decision['status']}`",
        f"**Selected recipe:** `{decision.get('recipe')}`",
        "",
        "This is a model-val screen, not final Generator qualification and not held-out.",
        "",
        "## Aggregate",
        "",
    ]
    for scope, scope_values in aggregate.items():
        lines.append(f"### {scope}")
        lines.append("")
        lines.append("| arm | groups | answer | coverage | cited | MiniCheck precision | MiniCheck recall | unsupported assertion |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for arm in ARMS:
            item = scope_values[arm]
            lines.append(
                "| "
                f"{arm} | {int(item.get('case_groups', 0))} | "
                f"{float(item.get('answer_match', 0.0)):.4f} | "
                f"{float(item.get('coverage', 0.0)):.4f} | "
                f"{float(item.get('correct_and_cited', 0.0)):.4f} | "
                f"{float(item.get('minicheck_citation_precision', 0.0)):.4f} | "
                f"{float(item.get('minicheck_citation_recall', 0.0)):.4f} | "
                f"{float(item.get('unsupported_ungrounded_assertion', 0.0)):.4f} |"
            )
        lines.append("")
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run")
    run_parser.add_argument("--model-snapshot", required=True, type=Path)
    run_parser.add_argument("--true-snapshot", required=True, type=Path)
    run_parser.add_argument("--data-manifest", required=True, type=Path)
    run_parser.add_argument("--train-cases", required=True, type=Path)
    run_parser.add_argument("--validation-cases", required=True, type=Path)
    run_parser.add_argument("--ordered-ids", required=True, type=Path)
    run_parser.add_argument("--grf-run-dir", required=True, type=Path)
    run_parser.add_argument("--grc-run-dir", required=True, type=Path)
    run_parser.add_argument("--output-dir", required=True, type=Path)
    run_parser.add_argument("--limit-tasks", type=int)
    run_parser.add_argument("--skip-unsupported-train", action="store_true")

    score_parser = commands.add_parser("score")
    score_parser.add_argument("--generations", required=True, type=Path)
    score_parser.add_argument("--output-json", required=True, type=Path)
    score_parser.add_argument("--output-rows", required=True, type=Path)
    score_parser.add_argument("--output-report", required=True, type=Path)
    score_parser.add_argument("--minicheck-model-id", default=FROZEN_MINICHECK_MODEL)
    score_parser.add_argument("--minicheck-device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        result = run(
            model_snapshot=args.model_snapshot,
            true_snapshot=args.true_snapshot,
            data_manifest_path=args.data_manifest,
            train_cases_path=args.train_cases,
            validation_cases_path=args.validation_cases,
            ordered_ids_path=args.ordered_ids,
            grf_run_dir=args.grf_run_dir,
            grc_run_dir=args.grc_run_dir,
            output_dir=args.output_dir,
            limit_tasks=args.limit_tasks,
            include_unsupported_train=not args.skip_unsupported_train,
        )
    else:
        result = score(
            generations_path=args.generations,
            output_json=args.output_json,
            output_rows=args.output_rows,
            output_report=args.output_report,
            minicheck_model_id=args.minicheck_model_id,
            minicheck_device=args.minicheck_device,
        )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
