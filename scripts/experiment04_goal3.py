#!/usr/bin/env python3
"""Experiment 04 Goal 3 formal seven-arm execution and scorer-only postprocess."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import platform
import re
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from evidence_rag.contracts.models import Query, SelectedEvidenceSet  # noqa: E402
from evidence_rag.contracts.validation import validate_generation  # noqa: E402
from evidence_rag.evaluation.experiment04_goal3 import (  # noqa: E402
    BASELINE_ARMS,
    OURS_ARMS,
    PRIMARY_ARMS,
    PreparedQuery,
    SystemOutput,
    append_canonical_jsonl,
    citation_sentences_from_inline,
    citation_sentences_from_routings,
    freeze_generation_manifest,
    maximal_whole_evidence_prefix,
    ordered_prefix_count,
    paired_component_cluster_bootstrap,
    prepare_query,
    read_jsonl,
    score_frozen_arm,
    selected_evidence_set,
    system_output,
)
from evidence_rag.evaluation.sealed_runtime import (  # noqa: E402
    file_sha256,
    ordered_id_sha256,
    read_runtime_bundle,
)
from evidence_rag.evaluation.system_scorer import validate_sidecar_record  # noqa: E402
from evidence_rag.generator.claim_splitter import ClaimSplitter  # noqa: E402
from evidence_rag.generator.draft import (  # noqa: E402
    DRAFT_PROMPT,
    DraftAnswerGenerator,
    DraftGenerator,
)
from evidence_rag.generator.entity_check import (  # noqa: E402
    EntityConsistencyChecker,
    SpacyEntityExtractor,
)
from evidence_rag.generator.granite import (  # noqa: E402
    GraniteGenerationConfig,
    GraniteLLMClient,
    InlineCitationGraniteGenerator,
    NamedAdapterTextGenerator,
    PeftGraniteLLMClient,
)
from evidence_rag.generator.nli import MiniCheckNLIModel, TrueNLIModel  # noqa: E402
from evidence_rag.generator.verify_annotate import (  # noqa: E402
    CitationRoutedVerifier,
    VerifyAnnotateGenerator,
)
from evidence_rag.query_analysis import RuleBasedQueryAnalyzer  # noqa: E402
from evidence_rag.retriever.granite import GraniteEmbedder  # noqa: E402
from evidence_rag.retriever.rerank import GraniteCrossEncoderReranker  # noqa: E402
from evidence_rag.selector.dual_head import load_dual_head_checkpoint  # noqa: E402
from evidence_rag.selector.nli_dual_head import load_nli_dual_head_model  # noqa: E402
from evidence_rag.selector.nli_runtime import NliRiskControlledSelector  # noqa: E402
from evidence_rag.selector.provence import ProvencePassagePruner  # noqa: E402

EXPERIMENT = (
    ROOT
    / "docs/full-flow/experiments/04_frozen_three_module_system_evaluation_2026-08-21"
)
DEFAULT_MODEL_MANIFEST = EXPERIMENT / "artifacts/goal2_model_config_manifest.json"
DEFAULT_DATA_MANIFEST = EXPERIMENT / "artifacts/goal1_data_manifest.json"
EXPECTED_COUNTS = {"hotpotqa": 400, "musique-answerable": 400, "rgb-noise": 300}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _verify_hash(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing frozen resource: {label}")
    if _sha256(path) != expected:
        raise ValueError(f"frozen resource hash differs: {label}")


def _model_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != "experiment04.model_config_manifest.v1"
        or value.get("status") != "FROZEN_BEFORE_HELD_OUT"
    ):
        raise ValueError("Goal 2 model/config manifest is not the frozen v1 manifest")
    prompt_hash = hashlib.sha256(DRAFT_PROMPT.encode("utf-8")).hexdigest()
    if prompt_hash != value["shared_runtime_contract"]["prompt_sha256"]:
        raise ValueError("shared Experiment 04 prompt differs from the frozen hash")
    return value


def _download_snapshot(cache_dir: Path, identity: Mapping[str, Any]) -> Path:
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=str(identity["model_id"]),
            revision=str(identity["revision"]),
            cache_dir=cache_dir,
        )
    )
    _verify_hash(snapshot / "config.json", str(identity["config_sha256"]), label="model config")
    for weight in identity.get("weights", []):
        _verify_hash(
            snapshot / str(weight["file"]),
            str(weight["sha256"]),
            label=f"{identity['model_id']} weight",
        )
    expected_shards = identity.get("weight_sha256")
    if expected_shards is not None:
        candidates = sorted(
            path
            for path in snapshot.iterdir()
            if path.is_file()
            and path.name != "config.json"
            and path.suffix in {".bin", ".safetensors"}
        )
        observed = sorted(_sha256(path) for path in candidates)
        if observed != sorted(str(value) for value in expected_shards):
            raise ValueError(f"frozen sharded weights differ: {identity['model_id']}")
    return snapshot


def _runtime_integrity(
    *,
    runtime_path: Path,
    dataset: str,
    data_manifest_path: Path,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    entry = manifest["datasets"][dataset]
    runtime_entry = entry["runtime"]
    _verify_hash(runtime_path, str(runtime_entry["file_sha256"]), label=f"{dataset} runtime")
    records = read_runtime_bundle(runtime_path, dataset=dataset)
    query_ids = [str(record["query_id"]) for record in records]
    if (
        len(records) != EXPECTED_COUNTS[dataset]
        or len(records) != int(runtime_entry["count"])
        or ordered_id_sha256(query_ids) != runtime_entry["ordered_ids_sha256"]
    ):
        raise ValueError(f"{dataset} runtime count/ordered IDs differ from Goal 1")
    return records, entry


def _sidecar_integrity(
    *,
    sidecar_path: Path,
    dataset: str,
    data_manifest_path: Path,
    expected_ids: Sequence[str],
) -> list[dict[str, Any]]:
    manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    entry = manifest["datasets"][dataset]["scorer_only"]
    _verify_hash(sidecar_path, str(entry["file_sha256"]), label=f"{dataset} scorer sidecar")
    rows = read_jsonl(sidecar_path)
    for row in rows:
        validate_sidecar_record(row, dataset=dataset)
    observed_ids = [str(row["query_id"]) for row in rows]
    if observed_ids != list(expected_ids) or ordered_id_sha256(observed_ids) != entry[
        "ordered_ids_sha256"
    ]:
        raise ValueError(f"{dataset} scorer sidecar ordered IDs differ")
    return rows


def _error_code(stage: str, error: Exception) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", type(error).__name__.casefold()).strip("_")
    return f"{stage}_{name or 'error'}"


def _progress(stage: str, dataset: str, count: int, total: int, *, arm: str | None = None) -> None:
    payload: dict[str, Any] = {"stage": stage, "dataset": dataset, "completed": count, "total": total}
    if arm is not None:
        payload["arm"] = arm
    print(json.dumps(payload, sort_keys=True), flush=True)


def _cuda_cleanup() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_stage(args: argparse.Namespace) -> int:
    records, _entry = _runtime_integrity(
        runtime_path=args.runtime,
        dataset=args.dataset,
        data_manifest_path=args.data_manifest,
    )
    expected_ids = [str(record["query_id"]) for record in records]
    completed = ordered_prefix_count(
        args.prepared,
        expected_ids,
        PreparedQuery.model_validate,
    )
    if completed == len(records):
        _progress("prepare", args.dataset, completed, len(records))
        return 0

    manifest = _model_manifest(args.model_manifest)
    models = manifest["models"]
    snapshots = {
        name: _download_snapshot(args.cache_dir, models[name])
        for name in ("dense_embedder", "granite_reranker", "provence", "selector_backbone")
    }
    selector_identity = models["selector_backbone"]
    _verify_hash(
        args.selector_checkpoint,
        str(selector_identity["runtime_checkpoint_sha256"]),
        label="selector seed13 checkpoint",
    )
    embedder = GraniteEmbedder(model_id=str(snapshots["dense_embedder"]), device=args.device)
    reranker = GraniteCrossEncoderReranker(
        model_id=str(snapshots["granite_reranker"]),
        revision=str(models["granite_reranker"]["revision"]),
        device=args.device,
        local_files_only=True,
    )
    provence = ProvencePassagePruner(
        model_id=str(snapshots["provence"]),
        revision=str(models["provence"]["revision"]),
        threshold=float(models["provence"]["threshold"]),
        always_select_title=bool(models["provence"]["always_select_title"]),
        reorder=bool(models["provence"]["reorder"]),
        local_files_only=True,
    )
    provence.model = provence.model.to(args.device)
    provence.model.eval()
    selector_model = load_nli_dual_head_model(
        str(snapshots["selector_backbone"]),
        revision=str(selector_identity["revision"]),
        identity_model_id=str(selector_identity["model_id"]),
        local_files_only=True,
        device=args.device,
    )
    load_dual_head_checkpoint(selector_model, args.selector_checkpoint)
    selector_model.eval()
    selector = NliRiskControlledSelector(
        model=selector_model,
        safe_threshold=float(selector_identity["safe_threshold"]),
        max_delete=int(selector_identity["max_delete"]),
    )
    for index, record in enumerate(records[completed:], start=completed + 1):
        prepared = prepare_query(
            record,
            embedder=embedder,
            reranker=reranker,
            provence=provence,
            selector=selector,
        )
        append_canonical_jsonl(args.prepared, prepared)
        if index % args.log_every == 0 or index == len(records):
            _progress("prepare", args.dataset, index, len(records))
    return 0


def _load_prepared(path: Path, expected_ids: Sequence[str], dataset: str) -> list[PreparedQuery]:
    rows = [PreparedQuery.model_validate(row) for row in read_jsonl(path)]
    if [row.query_id for row in rows] != list(expected_ids):
        raise ValueError("prepared rows do not cover the frozen ordered IDs")
    if any(row.dataset != dataset for row in rows):
        raise ValueError("prepared dataset identity differs")
    return rows


def _generation_paths(output_dir: Path) -> dict[str, Path]:
    return {arm: output_dir / "generations" / f"{arm}.jsonl" for arm in PRIMARY_ARMS}


def _draft_prompt(query: Query, selected: SelectedEvidenceSet) -> str:
    context = "\n".join(
        f"[{index}] ({item.evidence_id}) {item.text}"
        for index, item in enumerate(selected.evidence, start=1)
    )
    return DRAFT_PROMPT.format(
        context=context,
        question=query.text,
        focus="",
        required_facts="none",
        constraints="none",
    )


def _fit_generation_context(
    llm: GraniteLLMClient,
    query: Query,
    selected: SelectedEvidenceSet,
) -> SelectedEvidenceSet:
    max_input_tokens = llm.config.max_input_tokens
    if max_input_tokens is None:
        return selected
    return maximal_whole_evidence_prefix(
        selected,
        render_prompt=lambda value: _draft_prompt(query, value),
        input_token_count=llm.input_token_count,
        max_input_tokens=max_input_tokens,
    )


def generate_direct_stage(args: argparse.Namespace) -> int:
    records, _entry = _runtime_integrity(
        runtime_path=args.runtime,
        dataset=args.dataset,
        data_manifest_path=args.data_manifest,
    )
    expected_ids = [str(record["query_id"]) for record in records]
    prepared = _load_prepared(args.prepared, expected_ids, args.dataset)
    paths = _generation_paths(args.output_dir)
    completed = {
        arm: ordered_prefix_count(paths[arm], expected_ids, SystemOutput.model_validate)
        for arm in BASELINE_ARMS
    }
    if all(value == len(records) for value in completed.values()):
        for arm in BASELINE_ARMS:
            _progress("generate-direct", args.dataset, len(records), len(records), arm=arm)
        return 0

    manifest = _model_manifest(args.model_manifest)
    identity = manifest["models"]["direct_and_grounded_base"]
    snapshot = _download_snapshot(args.cache_dir, identity)
    llm = GraniteLLMClient(
        model_id=str(snapshot),
        config=GraniteGenerationConfig(
            max_new_tokens=256,
            temperature=0.0,
            top_p=1.0,
            max_input_tokens=2304,
        ),
        device=args.device,
        dtype=args.dtype,
    )
    generator = InlineCitationGraniteGenerator(
        llm=llm,
        prompt_template=DRAFT_PROMPT,
        require_declared_citations=True,
    )
    analyzer = RuleBasedQueryAnalyzer()
    by_id = {str(record["query_id"]): record for record in records}
    for arm in BASELINE_ARMS:
        for index in range(completed[arm], len(prepared)):
            item = prepared[index]
            query = Query(query_id=item.query_id, text=str(by_id[item.query_id]["question"]))
            selected = _fit_generation_context(
                llm,
                query,
                selected_evidence_set(item, arm),
            )
            try:
                result = generator.generate(query, analyzer.analyze(query), selected)
                validate_generation(selected, result)
                sentences, declared = citation_sentences_from_inline(
                    generator.last_raw_output,
                    tuple(evidence.evidence_id for evidence in selected.evidence),
                )
                output = system_output(
                    prepared=item,
                    arm_id=arm,
                    selected_evidence=selected,
                    answer=result.answer,
                    citation_indices=declared,
                    citation_sentences=sentences,
                )
            except Exception as error:  # noqa: BLE001 - failures remain in denominator
                output = system_output(
                    prepared=item,
                    arm_id=arm,
                    selected_evidence=selected,
                    answer="",
                    citation_indices=(),
                    citation_sentences=(),
                    failure_stage="generation",
                    error_code=_error_code("generation", error),
                )
            append_canonical_jsonl(paths[arm], output)
            count = index + 1
            if count % args.log_every == 0 or count == len(prepared):
                _progress("generate-direct", args.dataset, count, len(prepared), arm=arm)
    return 0


def _verify_adapters(args: argparse.Namespace, manifest: Mapping[str, Any]) -> dict[str, Path]:
    paths = {"13": args.adapter13, "42": args.adapter42, "73": args.adapter73}
    for seed, path in paths.items():
        identity = manifest["grc_adapters"][seed]
        _verify_hash(
            path / "adapter_model.safetensors",
            str(identity["weights_sha256"]),
            label=f"GR-C seed{seed} adapter weights",
        )
        _verify_hash(
            path / "adapter_config.json",
            str(identity["config_sha256"]),
            label=f"GR-C seed{seed} adapter config",
        )
    return paths


def _grounded_generators(
    *,
    client: PeftGraniteLLMClient,
    verifier_model: TrueNLIModel,
) -> dict[str, VerifyAnnotateGenerator]:
    generators: dict[str, VerifyAnnotateGenerator] = {}
    for seed in (13, 42, 73):
        name = f"grc{seed}"
        draft = DraftAnswerGenerator(
            draft_generator=DraftGenerator(
                llm=NamedAdapterTextGenerator(client, name),
                trace_enabled=True,
            ),
            claim_splitter=ClaimSplitter(llm=client, trace_enabled=True),
            trace_enabled=True,
        )
        verifier = CitationRoutedVerifier(
            verifier_model,
            EntityConsistencyChecker(SpacyEntityExtractor()),
            entity_gate="observe",
        )
        generators[f"ours_seed{seed}"] = VerifyAnnotateGenerator(
            draft_generator=draft,
            verifier=verifier,
            abstain_when_unverified=False,
            entity_gate="observe",
            trace_enabled=True,
        )
    return generators


def generate_ours_stage(args: argparse.Namespace) -> int:
    records, _entry = _runtime_integrity(
        runtime_path=args.runtime,
        dataset=args.dataset,
        data_manifest_path=args.data_manifest,
    )
    expected_ids = [str(record["query_id"]) for record in records]
    prepared = _load_prepared(args.prepared, expected_ids, args.dataset)
    paths = _generation_paths(args.output_dir)
    completed = {
        arm: ordered_prefix_count(paths[arm], expected_ids, SystemOutput.model_validate)
        for arm in OURS_ARMS
    }
    if all(value == len(records) for value in completed.values()):
        for arm in OURS_ARMS:
            _progress("generate-ours", args.dataset, len(records), len(records), arm=arm)
        return 0


    manifest = _model_manifest(args.model_manifest)
    adapter_paths = _verify_adapters(args, manifest)
    base = _download_snapshot(args.cache_dir, manifest["models"]["direct_and_grounded_base"])
    true_snapshot = _download_snapshot(args.cache_dir, manifest["models"]["true_runtime_verifier"])
    client = PeftGraniteLLMClient(
        model_id=str(base),
        adapters={f"grc{seed}": str(adapter_paths[str(seed)]) for seed in (13, 42, 73)},
        config=GraniteGenerationConfig(
            max_new_tokens=256,
            temperature=0.0,
            top_p=1.0,
            max_input_tokens=2304,
        ),
        device=args.device,
        dtype=args.dtype,
    )
    true_model = TrueNLIModel(model_id=str(true_snapshot))
    generators = _grounded_generators(client=client, verifier_model=true_model)
    analyzer = RuleBasedQueryAnalyzer()
    by_id = {str(record["query_id"]): record for record in records}
    for arm in OURS_ARMS:
        generator = generators[arm]
        for index in range(completed[arm], len(prepared)):
            item = prepared[index]
            query = Query(query_id=item.query_id, text=str(by_id[item.query_id]["question"]))
            selected = _fit_generation_context(
                client,
                query,
                selected_evidence_set(item, arm),
            )
            selected_ids = tuple(evidence.evidence_id for evidence in selected.evidence)
            try:
                result = generator.generate(query, analyzer.analyze(query), selected)
                validate_generation(selected, result)
                citation_indices = tuple(
                    selected_ids.index(evidence_id) + 1
                    for evidence_id in result.cited_evidence_ids
                )
                sentences = citation_sentences_from_routings(
                    generator.last_routings,
                    selected_ids,
                )
                output = system_output(
                    prepared=item,
                    arm_id=arm,
                    selected_evidence=selected,
                    answer=result.answer,
                    citation_indices=citation_indices,
                    citation_sentences=sentences,
                )
            except Exception as error:  # noqa: BLE001 - failures remain in denominator
                output = system_output(
                    prepared=item,
                    arm_id=arm,
                    selected_evidence=selected,
                    answer="",
                    citation_indices=(),
                    citation_sentences=(),
                    failure_stage="generation",
                    error_code=_error_code("generation", error),
                )
            append_canonical_jsonl(paths[arm], output)
            count = index + 1
            if count % args.log_every == 0 or count == len(prepared):
                _progress("generate-ours", args.dataset, count, len(prepared), arm=arm)
    return 0


def freeze_stage(args: argparse.Namespace) -> int:
    records, _entry = _runtime_integrity(
        runtime_path=args.runtime,
        dataset=args.dataset,
        data_manifest_path=args.data_manifest,
    )
    expected_ids = [str(record["query_id"]) for record in records]
    manifest = freeze_generation_manifest(
        dataset=args.dataset,
        arm_paths=_generation_paths(args.output_dir),
        expected_query_ids=expected_ids,
        runtime_sha256=_sha256(args.runtime),
        attempt_id=args.attempt_id,
    )
    manifest["prepared_sha256"] = _sha256(args.prepared)
    manifest["model_manifest_sha256"] = _sha256(args.model_manifest)
    preflight_path = args.output_dir / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or preflight.get("dataset") != args.dataset:
        raise ValueError("dataset preflight is not PASS")
    manifest["preflight_sha256"] = _sha256(preflight_path)
    manifest["implementation_sha256"] = {
        "goal3_cli": _sha256(Path(__file__).resolve()),
        "goal3_contract": _sha256(
            ROOT / "src/evidence_rag/evaluation/experiment04_goal3.py"
        ),
        "citation_metrics": _sha256(
            ROOT / "src/evidence_rag/evaluation/citation_metrics.py"
        ),
        "dataset_slurm": _sha256(
            ROOT / "scripts/run_experiment04_goal3_dataset.slurm"
        ),
    }
    prepared_rows = [PreparedQuery.model_validate(row) for row in read_jsonl(args.prepared)]
    fail_open = sum(
        row.arms["ours_seed13"].selection_status == "fail_open_all"
        for row in prepared_rows
    )
    manifest["selector_fail_open_queries"] = fail_open
    manifest["selector_fail_open_rate"] = fail_open / len(prepared_rows)
    _write_json(args.output_dir / "generation_manifest.json", manifest)
    _progress("freeze", args.dataset, len(records), len(records))
    return 0 if manifest["bundle_status"] == "PASS" else 2


def _verify_frozen_generation(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "experiment04.generation_manifest.v1"
        or manifest.get("bundle_status") != "PASS"
        or manifest.get("outputs_frozen_before_scoring") is not True
    ):
        raise ValueError("generation bundle is not frozen PASS before scoring")
    for arm, path in _generation_paths(output_dir).items():
        if _sha256(path) != manifest["arms"][arm]["file_sha256"]:
            raise ValueError(f"frozen generation hash changed before scoring: {arm}")
    return manifest


def score_stage(args: argparse.Namespace) -> int:
    generation_manifest = _verify_frozen_generation(args.output_dir)
    records, _entry = _runtime_integrity(
        runtime_path=args.runtime,
        dataset=args.dataset,
        data_manifest_path=args.data_manifest,
    )
    expected_ids = [str(record["query_id"]) for record in records]
    if generation_manifest["ordered_ids_sha256"] != ordered_id_sha256(expected_ids):
        raise ValueError("generation manifest IDs differ from the sealed runtime")
    if _sha256(args.prepared) != generation_manifest["prepared_sha256"]:
        raise ValueError("prepared runtime-safe rows changed after generation freeze")
    sidecars = _sidecar_integrity(
        sidecar_path=args.sidecar,
        dataset=args.dataset,
        data_manifest_path=args.data_manifest,
        expected_ids=expected_ids,
    )
    prepared_rows = read_jsonl(args.prepared)

    manifest = _model_manifest(args.model_manifest)
    minicheck_identity = manifest["models"]["minicheck_scorer_only"]
    minicheck_snapshot = _download_snapshot(args.cache_dir, minicheck_identity)
    minicheck = MiniCheckNLIModel(model_id=str(minicheck_snapshot))
    _tokenizer, model = minicheck._ensure_loaded()
    if args.device != "cpu":
        minicheck._model = model.to(args.device)
    entail_cache: dict[tuple[str, str], bool] = {}

    def entails(premise: str, hypothesis: str) -> bool:
        key = (premise, hypothesis)
        if key not in entail_cache:
            entail_cache[key] = minicheck.classify(premise=premise, hypothesis=hypothesis) == "entailment"
        return entail_cache[key]

    score_dir = args.output_dir / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    arm_entries: dict[str, Any] = {}
    for arm in PRIMARY_ARMS:
        generation_path = _generation_paths(args.output_dir)[arm]
        score_path = score_dir / f"{arm}.json"
        if score_path.exists():
            existing = json.loads(score_path.read_text(encoding="utf-8"))
            if (
                existing.get("generation_sha256") == _sha256(generation_path)
                and existing.get("minicheck_config_sha256") == minicheck_identity["config_sha256"]
            ):
                arm_entries[arm] = {
                    "score_file": str(score_path),
                    "score_sha256": _sha256(score_path),
                    "generation_sha256": _sha256(generation_path),
                    "n_queries": existing["score"]["n_queries"],
                    "judge_calls": existing["judge_calls"],
                }
                _progress("score", args.dataset, int(existing["score"]["n_queries"]), len(records), arm=arm)
                continue
        score, judge_calls = score_frozen_arm(
            sidecar_records=sidecars,
            outputs=read_jsonl(generation_path),
            prepared_records=prepared_rows,
            arm_id=arm,
            entails=entails,
        )
        artifact = {
            "schema_version": "experiment04.arm_score.v1",
            "dataset": args.dataset,
            "arm_id": arm,
            "generation_sha256": _sha256(generation_path),
            "minicheck_config_sha256": minicheck_identity["config_sha256"],
            "judge_calls": judge_calls,
            "score": score,
        }
        _write_json(score_path, artifact)
        arm_entries[arm] = {
            "score_file": str(score_path),
            "score_sha256": _sha256(score_path),
            "generation_sha256": _sha256(generation_path),
            "n_queries": score["n_queries"],
            "judge_calls": judge_calls,
        }
        _progress("score", args.dataset, score["n_queries"], len(records), arm=arm)
    score_manifest = {
        "schema_version": "experiment04.dataset_score_manifest.v1",
        "dataset": args.dataset,
        "query_count": len(records),
        "ordered_ids_sha256": ordered_id_sha256(expected_ids),
        "generation_manifest_sha256": _sha256(args.output_dir / "generation_manifest.json"),
        "scorer_sidecar_sha256": _sha256(args.sidecar),
        "minicheck_config_sha256": minicheck_identity["config_sha256"],
        "generation_frozen_before_sidecar_read": True,
        "arms": arm_entries,
        "status": "PASS",
    }
    _write_json(args.output_dir / "score_manifest.json", score_manifest)
    return 0


def _score_artifacts(dataset_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = dataset_dir / "score_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "experiment04.dataset_score_manifest.v1":
        raise ValueError("dataset score manifest has the wrong schema")
    if manifest.get("status") != "PASS":
        raise ValueError("dataset score manifest is not PASS")
    scores: dict[str, dict[str, Any]] = {}
    for arm in PRIMARY_ARMS:
        path = dataset_dir / "scores" / f"{arm}.json"
        if _sha256(path) != manifest["arms"][arm]["score_sha256"]:
            raise ValueError(f"score artifact hash differs: {arm}")
        scores[arm] = json.loads(path.read_text(encoding="utf-8"))["score"]
    return manifest, scores


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}"


def _mean_sd(values: Sequence[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}


def _ours_table_row(scores: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    aggregates = [scores[arm]["aggregate"] for arm in OURS_ARMS]
    ret_values = [float(value["ret"]) for value in aggregates]
    sel_values = [float(value["sel"]) for value in aggregates]
    if max(ret_values) != min(ret_values) or max(sel_values) != min(sel_values):
        raise ValueError("Ours seeds do not share fixed upstream Ret./Sel. values")
    return {
        "system": "Ours",
        "ret": ret_values[0],
        "sel": sel_values[0],
        "ans": _mean_sd([float(value["ans"]) for value in aggregates]),
        "cit": _mean_sd([float(value["cit"]) for value in aggregates]),
        "rar": _mean_sd([float(value["rar"]) for value in aggregates]),
    }


def _table_markdown(table: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    lines = ["# Experiment 04 — Table 1", ""]
    for dataset in EXPECTED_COUNTS:
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| System | Ret. | Sel. | Ans. | Cit. | RAR |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in table[dataset]:
            if row["system"] == "Ours":
                formatted = [
                    _percent(float(row["ret"])),
                    _percent(float(row["sel"])),
                    *[
                        f"{_percent(float(row[key]['mean']))} ± {_percent(float(row[key]['sample_sd']))}"
                        for key in ("ans", "cit", "rar")
                    ],
                ]
            else:
                formatted = [_percent(float(row[key])) for key in ("ret", "sel", "ans", "cit", "rar")]
            lines.append(f"| {row['system']} | " + " | ".join(formatted) + " |")
        lines.append("")
    return "\n".join(lines)


def summarize_stage(args: argparse.Namespace) -> int:
    dataset_dirs = {
        dataset: path
        for dataset, path in (
            (item.split("=", 1)[0], Path(item.split("=", 1)[1]).resolve())
            for item in args.dataset_dir
        )
    }
    if set(dataset_dirs) != set(EXPECTED_COUNTS):
        raise ValueError("summarize requires exactly hotpotqa, musique-answerable, and rgb-noise")
    labels = {
        "dense_rag": "Dense RAG",
        "hybrid_rag": "Hybrid RAG",
        "granite_rerank_rag": "Granite Rerank RAG",
        "provence_rag": "Provence RAG",
    }
    table: dict[str, list[dict[str, Any]]] = {}
    bootstrap: dict[str, Any] = {
        "schema_version": "experiment04.paired_rar_bootstrap.v1",
        "resamples": 10_000,
        "seed": 13,
        "datasets": {},
    }
    manifests: dict[str, Any] = {}
    per_query_rows: list[dict[str, Any]] = []
    for dataset, dataset_dir in dataset_dirs.items():
        manifest, scores = _score_artifacts(dataset_dir)
        if int(manifest["query_count"]) != EXPECTED_COUNTS[dataset]:
            raise ValueError(f"{dataset} score count differs")
        manifests[dataset] = {
            "directory": str(dataset_dir),
            "score_manifest_sha256": _sha256(dataset_dir / "score_manifest.json"),
            "generation_manifest_sha256": manifest["generation_manifest_sha256"],
        }
        rows: list[dict[str, Any]] = []
        for arm in BASELINE_ARMS:
            rows.append({"system": labels[arm], **scores[arm]["aggregate"]})
        rows.append(_ours_table_row(scores))
        table[dataset] = rows

        by_arm: dict[str, dict[str, Mapping[str, Any]]] = {}
        for arm in PRIMARY_ARMS:
            by_arm[arm] = {str(row["query_id"]): row for row in scores[arm]["per_query"]}
            for row in scores[arm]["per_query"]:
                per_query_rows.append(
                    {
                        "dataset": dataset,
                        "arm_id": arm,
                        "query_id": row["query_id"],
                        "component_id": row["component_id"],
                        **row["metrics"],
                        "failure_reason": row["failure_reason"] or "",
                    }
                )
        query_ids = tuple(by_arm["ours_seed13"])
        components = {
            query_id: str(by_arm["ours_seed13"][query_id]["component_id"])
            for query_id in query_ids
        }
        ours_mean = {
            query_id: statistics.mean(
                float(by_arm[arm][query_id]["metrics"]["rar"])
                for arm in OURS_ARMS
            )
            for query_id in query_ids
        }
        bootstrap["datasets"][dataset] = {
            labels[arm]: paired_component_cluster_bootstrap(
                candidate=ours_mean,
                baseline={
                    query_id: float(by_arm[arm][query_id]["metrics"]["rar"])
                    for query_id in query_ids
                },
                component_ids=components,
                resamples=10_000,
                seed=13,
            )
            for arm in BASELINE_ARMS
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "table1.json"
    markdown_path = args.output_dir / "TABLE1.md"
    bootstrap_path = args.output_dir / "bootstrap_ci.json"
    csv_path = args.output_dir / "per_query_metrics.csv"
    _write_json(table_path, {"schema_version": "experiment04.table1.v1", "datasets": table})
    markdown_path.write_text(_table_markdown(table) + "\n", encoding="utf-8")
    _write_json(bootstrap_path, bootstrap)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "dataset",
                "arm_id",
                "query_id",
                "component_id",
                "ret",
                "sel",
                "ans",
                "cit",
                "rar",
                "failure_reason",
            ),
        )
        writer.writeheader()
        writer.writerows(per_query_rows)
    audit = {
        "schema_version": "experiment04.goal3_audit.v1",
        "status": "PASS",
        "datasets": manifests,
        "arms": list(PRIMARY_ARMS),
        "expected_total_outputs": 7_700,
        "observed_total_outputs": len(per_query_rows),
        "ours_seed_policy": "all_three_independent_training_seeds_mean_sample_sd_no_selection",
        "paired_rar_bootstrap": {"resamples": 10_000, "seed": 13},
        "artifacts": {
            "table1_json_sha256": _sha256(table_path),
            "table1_markdown_sha256": _sha256(markdown_path),
            "bootstrap_sha256": _sha256(bootstrap_path),
            "per_query_metrics_sha256": _sha256(csv_path),
        },
    }
    if audit["observed_total_outputs"] != audit["expected_total_outputs"]:
        audit["status"] = "FAIL"
    _write_json(args.output_dir / "goal3_audit.json", audit)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "outputs": audit["observed_total_outputs"],
                "datasets": len(manifests),
            },
            sort_keys=True,
        )
    )
    return 0 if audit["status"] == "PASS" else 2


def preflight_stage(args: argparse.Namespace) -> int:
    records, entry = _runtime_integrity(
        runtime_path=args.runtime,
        dataset=args.dataset,
        data_manifest_path=args.data_manifest,
    )
    manifest = _model_manifest(args.model_manifest)
    _verify_hash(
        args.selector_checkpoint,
        str(manifest["models"]["selector_backbone"]["runtime_checkpoint_sha256"]),
        label="selector seed13 checkpoint",
    )
    adapters = _verify_adapters(args, manifest)
    report = {
        "schema_version": "experiment04.goal3_preflight.v1",
        "dataset": args.dataset,
        "query_count": len(records),
        "ordered_ids_sha256": entry["runtime"]["ordered_ids_sha256"],
        "runtime_sha256": _sha256(args.runtime),
        "model_manifest_sha256": _sha256(args.model_manifest),
        "custom_checkpoint_hashes_verified": True,
        "custom_checkpoint_sha256": {
            "selector_seed13": _sha256(args.selector_checkpoint),
            **{
                f"grc_seed{seed}_weights": _sha256(path / "adapter_model.safetensors")
                for seed, path in adapters.items()
            },
            **{
                f"grc_seed{seed}_config": _sha256(path / "adapter_config.json")
                for seed, path in adapters.items()
            },
        },
        "spacy_version": importlib.metadata.version("spacy"),
        "spacy_model_version": importlib.metadata.version("en-core-web-sm"),
        "host": platform.node(),
        "status": "PASS",
    }
    _write_json(args.output, report)
    print(json.dumps({"status": "PASS", "dataset": args.dataset, "count": len(records)}, sort_keys=True))
    return 0


def cache_models_stage(args: argparse.Namespace) -> int:
    manifest = _model_manifest(args.model_manifest)
    models = manifest["models"]
    requested = tuple(value.strip() for value in args.models.split(",") if value.strip())
    unexpected = sorted(set(requested) - set(models))
    if not requested or unexpected:
        raise ValueError(f"invalid model cache request: {unexpected}")
    cached: dict[str, Any] = {}
    for name in requested:
        snapshot = _download_snapshot(args.cache_dir, models[name])
        cached[name] = {
            "model_id": models[name]["model_id"],
            "revision": models[name]["revision"],
            "snapshot": str(snapshot),
            "hashes_verified": True,
        }
        print(json.dumps({"stage": "cache-models", "model": name, "verified": True}, sort_keys=True))
    _write_json(
        args.output,
        {
            "schema_version": "experiment04.goal3_model_cache.v1",
            "model_manifest_sha256": _sha256(args.model_manifest),
            "models": cached,
            "status": "PASS",
        },
    )
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True, choices=tuple(EXPECTED_COUNTS))
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--data-manifest", default=DEFAULT_DATA_MANIFEST, type=Path)
    parser.add_argument("--model-manifest", default=DEFAULT_MODEL_MANIFEST, type=Path)


def _generation_common(parser: argparse.ArgumentParser) -> None:
    _common(parser)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--log-every", default=25, type=int)


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    cache_models = commands.add_parser("cache-models")
    cache_models.add_argument("--cache-dir", required=True, type=Path)
    cache_models.add_argument("--model-manifest", default=DEFAULT_MODEL_MANIFEST, type=Path)
    cache_models.add_argument("--models", required=True)
    cache_models.add_argument("--output", required=True, type=Path)

    preflight = commands.add_parser("preflight")
    _common(preflight)
    preflight.add_argument("--selector-checkpoint", required=True, type=Path)
    preflight.add_argument("--adapter13", required=True, type=Path)
    preflight.add_argument("--adapter42", required=True, type=Path)
    preflight.add_argument("--adapter73", required=True, type=Path)
    preflight.add_argument("--output", required=True, type=Path)

    prepare = commands.add_parser("prepare")
    _common(prepare)
    prepare.add_argument("--prepared", required=True, type=Path)
    prepare.add_argument("--cache-dir", required=True, type=Path)
    prepare.add_argument("--selector-checkpoint", required=True, type=Path)
    prepare.add_argument("--device", default="cuda")
    prepare.add_argument("--log-every", default=25, type=int)

    direct = commands.add_parser("generate-direct")
    _generation_common(direct)

    ours = commands.add_parser("generate-ours")
    _generation_common(ours)
    ours.add_argument("--adapter13", required=True, type=Path)
    ours.add_argument("--adapter42", required=True, type=Path)
    ours.add_argument("--adapter73", required=True, type=Path)

    freeze = commands.add_parser("freeze")
    _common(freeze)
    freeze.add_argument("--prepared", required=True, type=Path)
    freeze.add_argument("--output-dir", required=True, type=Path)
    freeze.add_argument("--attempt-id", required=True)

    score = commands.add_parser("score")
    _generation_common(score)
    score.add_argument("--sidecar", required=True, type=Path)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--dataset-dir", required=True, action="append")
    summarize.add_argument("--output-dir", required=True, type=Path)

    args = parser.parse_args()
    for name in (
        "runtime",
        "data_manifest",
        "model_manifest",
        "prepared",
        "output_dir",
        "cache_dir",
        "selector_checkpoint",
        "adapter13",
        "adapter42",
        "adapter73",
        "sidecar",
        "output",
    ):
        if hasattr(args, name):
            setattr(args, name, getattr(args, name).resolve())
    if hasattr(args, "log_every") and args.log_every <= 0:
        raise ValueError("log-every must be positive")
    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "cache-models": cache_models_stage,
        "preflight": preflight_stage,
        "prepare": prepare_stage,
        "generate-direct": generate_direct_stage,
        "generate-ours": generate_ours_stage,
        "freeze": freeze_stage,
        "score": score_stage,
        "summarize": summarize_stage,
    }
    try:
        return handlers[args.command](args)
    except Exception as error:  # noqa: BLE001 - formal logs must never echo held-out prose
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "command": args.command,
                    "error_code": _error_code("command", error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1
    finally:
        _cuda_cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
