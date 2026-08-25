#!/usr/bin/env python3
"""Experiment 04 Goal 4 frozen three-arm component ablation."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import experiment04_goal3 as g3  # noqa: E402

from evidence_rag.contracts.models import Query  # noqa: E402
from evidence_rag.contracts.validation import validate_generation  # noqa: E402
from evidence_rag.evaluation.experiment04_goal3 import (  # noqa: E402
    PreparedQuery as Goal3PreparedQuery,
)
from evidence_rag.evaluation.experiment04_goal3 import (  # noqa: E402
    citation_sentences_from_inline,
    citation_sentences_from_routings,
    maximal_whole_evidence_prefix,
    ordered_prefix_count,
    paired_component_cluster_bootstrap,
    read_jsonl,
)
from evidence_rag.evaluation.experiment04_goal4 import (  # noqa: E402
    ABLATION_ARMS,
    DIRECT_ABLATION_ARM,
    FULL_GOAL3_ARM,
    GROUNDED_ABLATION_ARMS,
    PreparedQuery,
    SystemOutput,
    append_canonical_jsonl,
    freeze_generation_manifest,
    prepare_query,
    score_frozen_arm,
    selected_evidence_set,
    system_output,
    validate_single_substitutions,
)
from evidence_rag.evaluation.sealed_runtime import ordered_id_sha256  # noqa: E402
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
from evidence_rag.selector.dual_head import load_dual_head_checkpoint  # noqa: E402
from evidence_rag.selector.nli_dual_head import load_nli_dual_head_model  # noqa: E402
from evidence_rag.selector.nli_runtime import NliRiskControlledSelector  # noqa: E402

EXPERIMENT = ROOT / "docs/full-flow/experiments/04_frozen_three_module_system_evaluation_2026-08-21"
DEFAULT_MODEL_MANIFEST = EXPERIMENT / "artifacts/goal2_model_config_manifest.json"
DEFAULT_DATA_MANIFEST = EXPERIMENT / "artifacts/goal1_data_manifest.json"
EXPECTED_COUNTS = g3.EXPECTED_COUNTS


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return g3._sha256(path)


def _progress(
    stage: str,
    dataset: str,
    count: int,
    total: int,
    *,
    arm: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "stage": stage,
        "dataset": dataset,
        "completed": count,
        "total": total,
    }
    if arm is not None:
        payload["arm"] = arm
    print(json.dumps(payload, sort_keys=True), flush=True)


def _goal3_prepared(
    path: Path,
    *,
    expected_ids: Sequence[str],
    dataset: str,
) -> list[Goal3PreparedQuery]:
    rows = [Goal3PreparedQuery.model_validate(row) for row in read_jsonl(path)]
    if [row.query_id for row in rows] != list(expected_ids):
        raise ValueError("Goal 3 preparation does not cover frozen ordered IDs")
    if any(row.dataset != dataset for row in rows):
        raise ValueError("Goal 3 prepared dataset identity differs")
    return rows


def _load_prepared(
    path: Path,
    *,
    expected_ids: Sequence[str],
    dataset: str,
    goal3_rows: Sequence[Goal3PreparedQuery],
) -> list[PreparedQuery]:
    rows = [PreparedQuery.model_validate(row) for row in read_jsonl(path)]
    if [row.query_id for row in rows] != list(expected_ids):
        raise ValueError("Goal 4 preparation does not cover frozen ordered IDs")
    if any(row.dataset != dataset for row in rows):
        raise ValueError("Goal 4 prepared dataset identity differs")
    for row, source in zip(rows, goal3_rows, strict=True):
        validate_single_substitutions(row, source)
    return rows


def _generation_paths(output_dir: Path) -> dict[str, Path]:
    return {arm: output_dir / "generations" / f"{arm}.jsonl" for arm in ABLATION_ARMS}


def _goal3_reuse_integrity(
    *,
    goal3_dir: Path,
    dataset: str,
    runtime_sha256: str,
    expected_ids: Sequence[str],
) -> dict[str, Any]:
    generation_path = goal3_dir / "generation_manifest.json"
    score_path = goal3_dir / "score_manifest.json"
    prepared_path = goal3_dir / "prepared.jsonl"
    full_generation_path = goal3_dir / "generations" / f"{FULL_GOAL3_ARM}.jsonl"
    full_score_path = goal3_dir / "scores" / f"{FULL_GOAL3_ARM}.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    score = json.loads(score_path.read_text(encoding="utf-8"))
    if (
        generation.get("schema_version") != "experiment04.generation_manifest.v1"
        or generation.get("bundle_status") != "PASS"
        or generation.get("outputs_frozen_before_scoring") is not True
        or generation.get("dataset") != dataset
        or generation.get("runtime_sha256") != runtime_sha256
    ):
        raise ValueError("Goal 3 generation bundle is not the exact frozen PASS input")
    if generation.get("ordered_ids_sha256") != ordered_id_sha256(expected_ids):
        raise ValueError("Goal 3 generation ordered IDs differ")
    if generation.get("prepared_sha256") != _sha256(prepared_path):
        raise ValueError("Goal 3 prepared hash differs from frozen generation manifest")
    full_generation = generation["arms"][FULL_GOAL3_ARM]
    if (
        full_generation.get("count") != len(expected_ids)
        or full_generation.get("ordered_ids_sha256") != ordered_id_sha256(expected_ids)
        or full_generation.get("file_sha256") != _sha256(full_generation_path)
    ):
        raise ValueError("Goal 3 Full generation artifact differs")
    if (
        score.get("schema_version") != "experiment04.dataset_score_manifest.v1"
        or score.get("status") != "PASS"
        or score.get("dataset") != dataset
        or score.get("ordered_ids_sha256") != ordered_id_sha256(expected_ids)
        or score.get("generation_manifest_sha256") != _sha256(generation_path)
        or score["arms"][FULL_GOAL3_ARM]["score_sha256"] != _sha256(full_score_path)
    ):
        raise ValueError("Goal 3 Full score artifact differs from frozen PASS bundle")
    return {
        "goal3_directory": str(goal3_dir),
        "goal3_prepared_sha256": _sha256(prepared_path),
        "goal3_generation_manifest_sha256": _sha256(generation_path),
        "goal3_score_manifest_sha256": _sha256(score_path),
        "full_generation_sha256": _sha256(full_generation_path),
        "full_score_sha256": _sha256(full_score_path),
        "full_regenerated": False,
    }


def _selector(args: argparse.Namespace, manifest: Mapping[str, Any]) -> NliRiskControlledSelector:
    identity = manifest["models"]["selector_backbone"]
    snapshot = g3._download_snapshot(args.cache_dir, identity)
    g3._verify_hash(
        args.selector_checkpoint,
        str(identity["runtime_checkpoint_sha256"]),
        label="selector seed13 checkpoint",
    )
    model = load_nli_dual_head_model(
        str(snapshot),
        revision=str(identity["revision"]),
        identity_model_id=str(identity["model_id"]),
        local_files_only=True,
        device=args.device,
    )
    load_dual_head_checkpoint(model, args.selector_checkpoint)
    model.eval()
    return NliRiskControlledSelector(
        model=model,
        safe_threshold=float(identity["safe_threshold"]),
        max_delete=int(identity["max_delete"]),
    )


def preflight_stage(args: argparse.Namespace) -> int:
    records, entry = g3._runtime_integrity(
        runtime_path=args.runtime,
        dataset=args.dataset,
        data_manifest_path=args.data_manifest,
    )
    expected_ids = [str(row["query_id"]) for row in records]
    manifest = g3._model_manifest(args.model_manifest)
    selector_identity = manifest["models"]["selector_backbone"]
    g3._verify_hash(
        args.selector_checkpoint,
        str(selector_identity["runtime_checkpoint_sha256"]),
        label="selector seed13 checkpoint",
    )
    adapter = manifest["grc_adapters"]["13"]
    g3._verify_hash(
        args.adapter13 / "adapter_model.safetensors",
        str(adapter["weights_sha256"]),
        label="GR-C seed13 adapter weights",
    )
    g3._verify_hash(
        args.adapter13 / "adapter_config.json",
        str(adapter["config_sha256"]),
        label="GR-C seed13 adapter config",
    )
    reuse = _goal3_reuse_integrity(
        goal3_dir=args.goal3_dir,
        dataset=args.dataset,
        runtime_sha256=_sha256(args.runtime),
        expected_ids=expected_ids,
    )
    _goal3_prepared(
        args.goal3_dir / "prepared.jsonl",
        expected_ids=expected_ids,
        dataset=args.dataset,
    )
    report = {
        "schema_version": "experiment04.goal4_preflight.v1",
        "dataset": args.dataset,
        "query_count": len(records),
        "ordered_ids_sha256": entry["runtime"]["ordered_ids_sha256"],
        "runtime_sha256": _sha256(args.runtime),
        "model_manifest_sha256": _sha256(args.model_manifest),
        "selector_checkpoint_sha256": _sha256(args.selector_checkpoint),
        "grc_seed13_weights_sha256": _sha256(args.adapter13 / "adapter_model.safetensors"),
        "goal3_reuse": reuse,
        "new_arms": list(ABLATION_ARMS),
        "full_generation_requested": False,
        "status": "PASS",
    }
    _write_json(args.output, report)
    _progress("preflight", args.dataset, len(records), len(records))
    return 0


def prepare_stage(args: argparse.Namespace) -> int:
    records, _entry = g3._runtime_integrity(
        runtime_path=args.runtime,
        dataset=args.dataset,
        data_manifest_path=args.data_manifest,
    )
    expected_ids = [str(row["query_id"]) for row in records]
    goal3_path = args.goal3_dir / "prepared.jsonl"
    goal3_rows = _goal3_prepared(goal3_path, expected_ids=expected_ids, dataset=args.dataset)
    completed = ordered_prefix_count(args.prepared, expected_ids, PreparedQuery.model_validate)
    if completed == len(records):
        _load_prepared(
            args.prepared,
            expected_ids=expected_ids,
            dataset=args.dataset,
            goal3_rows=goal3_rows,
        )
        _progress("prepare", args.dataset, completed, len(records))
        return 0
    selector = _selector(args, g3._model_manifest(args.model_manifest))
    by_id = {str(row["query_id"]): row for row in records}
    for index in range(completed, len(goal3_rows)):
        source = goal3_rows[index]
        prepared = prepare_query(
            source,
            question=str(by_id[source.query_id]["question"]),
            selector=selector,
        )
        validate_single_substitutions(prepared, source)
        append_canonical_jsonl(args.prepared, prepared)
        count = index + 1
        if count % args.log_every == 0 or count == len(records):
            _progress("prepare", args.dataset, count, len(records))
    return 0


def _draft_prompt(query: Query, selected: Any) -> str:
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


def _fit_generation_context(llm: GraniteLLMClient, query: Query, selected: Any) -> Any:
    if llm.config.max_input_tokens is None:
        return selected
    return maximal_whole_evidence_prefix(
        selected,
        render_prompt=lambda value: _draft_prompt(query, value),
        input_token_count=llm.input_token_count,
        max_input_tokens=llm.config.max_input_tokens,
    )


def _generation_inputs(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[PreparedQuery]]:
    records, _entry = g3._runtime_integrity(
        runtime_path=args.runtime,
        dataset=args.dataset,
        data_manifest_path=args.data_manifest,
    )
    expected_ids = [str(row["query_id"]) for row in records]
    goal3_rows = _goal3_prepared(
        args.goal3_dir / "prepared.jsonl",
        expected_ids=expected_ids,
        dataset=args.dataset,
    )
    prepared = _load_prepared(
        args.prepared,
        expected_ids=expected_ids,
        dataset=args.dataset,
        goal3_rows=goal3_rows,
    )
    return list(records), prepared


def _grounded_generator(
    *,
    client: PeftGraniteLLMClient,
    verifier_model: TrueNLIModel,
) -> VerifyAnnotateGenerator:
    draft = DraftAnswerGenerator(
        draft_generator=DraftGenerator(
            llm=NamedAdapterTextGenerator(client, "grc13"),
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
    return VerifyAnnotateGenerator(
        draft_generator=draft,
        verifier=verifier,
        abstain_when_unverified=False,
        entity_gate="observe",
        trace_enabled=True,
    )


def generate_grounded_stage(args: argparse.Namespace) -> int:
    records, prepared = _generation_inputs(args)
    expected_ids = [str(row["query_id"]) for row in records]
    paths = _generation_paths(args.output_dir)
    completed = {
        arm: ordered_prefix_count(paths[arm], expected_ids, SystemOutput.model_validate)
        for arm in GROUNDED_ABLATION_ARMS
    }
    if all(value == len(records) for value in completed.values()):
        for arm in GROUNDED_ABLATION_ARMS:
            _progress("generate-grounded", args.dataset, len(records), len(records), arm=arm)
        return 0
    manifest = g3._model_manifest(args.model_manifest)
    adapter = manifest["grc_adapters"]["13"]
    g3._verify_hash(
        args.adapter13 / "adapter_model.safetensors",
        str(adapter["weights_sha256"]),
        label="GR-C seed13 adapter weights",
    )
    g3._verify_hash(
        args.adapter13 / "adapter_config.json",
        str(adapter["config_sha256"]),
        label="GR-C seed13 adapter config",
    )
    base = g3._download_snapshot(
        args.cache_dir,
        manifest["models"]["direct_and_grounded_base"],
    )
    true_snapshot = g3._download_snapshot(
        args.cache_dir,
        manifest["models"]["true_runtime_verifier"],
    )
    client = PeftGraniteLLMClient(
        model_id=str(base),
        adapters={"grc13": str(args.adapter13)},
        config=GraniteGenerationConfig(
            max_new_tokens=256,
            temperature=0.0,
            top_p=1.0,
            max_input_tokens=2304,
        ),
        device=args.device,
        dtype=args.dtype,
    )
    generator = _grounded_generator(
        client=client,
        verifier_model=TrueNLIModel(model_id=str(true_snapshot)),
    )
    analyzer = RuleBasedQueryAnalyzer()
    by_id = {str(row["query_id"]): row for row in records}
    for arm in GROUNDED_ABLATION_ARMS:
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
                    selected_ids.index(evidence_id) + 1 for evidence_id in result.cited_evidence_ids
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
            except Exception as error:  # noqa: BLE001 - failures stay in denominator
                output = system_output(
                    prepared=item,
                    arm_id=arm,
                    selected_evidence=selected,
                    answer="",
                    citation_indices=(),
                    citation_sentences=(),
                    failure_stage="generation",
                    error_code=g3._error_code("generation", error),
                )
            append_canonical_jsonl(paths[arm], output)
            count = index + 1
            if count % args.log_every == 0 or count == len(prepared):
                _progress("generate-grounded", args.dataset, count, len(prepared), arm=arm)
    return 0


def generate_direct_stage(args: argparse.Namespace) -> int:
    records, prepared = _generation_inputs(args)
    expected_ids = [str(row["query_id"]) for row in records]
    path = _generation_paths(args.output_dir)[DIRECT_ABLATION_ARM]
    completed = ordered_prefix_count(path, expected_ids, SystemOutput.model_validate)
    if completed == len(records):
        _progress(
            "generate-direct", args.dataset, len(records), len(records), arm=DIRECT_ABLATION_ARM
        )
        return 0
    manifest = g3._model_manifest(args.model_manifest)
    snapshot = g3._download_snapshot(
        args.cache_dir,
        manifest["models"]["direct_and_grounded_base"],
    )
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
    by_id = {str(row["query_id"]): row for row in records}
    for index in range(completed, len(prepared)):
        item = prepared[index]
        query = Query(query_id=item.query_id, text=str(by_id[item.query_id]["question"]))
        selected = _fit_generation_context(
            llm,
            query,
            selected_evidence_set(item, DIRECT_ABLATION_ARM),
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
                arm_id=DIRECT_ABLATION_ARM,
                selected_evidence=selected,
                answer=result.answer,
                citation_indices=declared,
                citation_sentences=sentences,
            )
        except Exception as error:  # noqa: BLE001 - failures stay in denominator
            output = system_output(
                prepared=item,
                arm_id=DIRECT_ABLATION_ARM,
                selected_evidence=selected,
                answer="",
                citation_indices=(),
                citation_sentences=(),
                failure_stage="generation",
                error_code=g3._error_code("generation", error),
            )
        append_canonical_jsonl(path, output)
        count = index + 1
        if count % args.log_every == 0 or count == len(prepared):
            _progress(
                "generate-direct", args.dataset, count, len(prepared), arm=DIRECT_ABLATION_ARM
            )
    return 0


def freeze_stage(args: argparse.Namespace) -> int:
    records, _entry = g3._runtime_integrity(
        runtime_path=args.runtime,
        dataset=args.dataset,
        data_manifest_path=args.data_manifest,
    )
    expected_ids = [str(row["query_id"]) for row in records]
    goal3_prepared_path = args.goal3_dir / "prepared.jsonl"
    goal3_rows = _goal3_prepared(
        goal3_prepared_path,
        expected_ids=expected_ids,
        dataset=args.dataset,
    )
    prepared_rows = _load_prepared(
        args.prepared,
        expected_ids=expected_ids,
        dataset=args.dataset,
        goal3_rows=goal3_rows,
    )
    manifest = freeze_generation_manifest(
        dataset=args.dataset,
        arm_paths=_generation_paths(args.output_dir),
        expected_query_ids=expected_ids,
        runtime_sha256=_sha256(args.runtime),
        goal3_prepared_sha256=_sha256(goal3_prepared_path),
        attempt_id=args.attempt_id,
    )
    manifest["goal4_prepared_sha256"] = _sha256(args.prepared)
    manifest["model_manifest_sha256"] = _sha256(args.model_manifest)
    preflight_path = args.output_dir / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or preflight.get("dataset") != args.dataset:
        raise ValueError("Goal 4 dataset preflight is not PASS")
    manifest["preflight_sha256"] = _sha256(preflight_path)
    manifest["goal3_reuse"] = preflight["goal3_reuse"]
    manifest["implementation_sha256"] = {
        "goal4_cli": _sha256(Path(__file__).resolve()),
        "goal4_contract": _sha256(ROOT / "src/evidence_rag/evaluation/experiment04_goal4.py"),
        "goal3_contract": _sha256(ROOT / "src/evidence_rag/evaluation/experiment04_goal3.py"),
        "citation_metrics": _sha256(ROOT / "src/evidence_rag/evaluation/citation_metrics.py"),
        "dataset_slurm": _sha256(ROOT / "scripts/run_experiment04_goal4_dataset.slurm"),
    }
    manifest["selector_fail_open_queries"] = sum(
        row.arms["ablation_dense_retriever"].selection_status == "fail_open_all"
        for row in prepared_rows
    )
    manifest["single_module_substitutions_validated"] = True
    _write_json(args.output_dir / "generation_manifest.json", manifest)
    _progress("freeze", args.dataset, len(records), len(records))
    return 0 if manifest["bundle_status"] == "PASS" else 2


def _verify_frozen_generation(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "generation_manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "experiment04.goal4_generation_manifest.v1"
        or manifest.get("bundle_status") != "PASS"
        or manifest.get("outputs_frozen_before_scoring") is not True
        or manifest.get("full_regenerated") is not False
        or manifest.get("single_module_substitutions_validated") is not True
    ):
        raise ValueError("Goal 4 generation bundle is not frozen PASS before scoring")
    for arm, path in _generation_paths(output_dir).items():
        if _sha256(path) != manifest["arms"][arm]["file_sha256"]:
            raise ValueError(f"frozen Goal 4 generation hash changed: {arm}")
    return manifest


def score_stage(args: argparse.Namespace) -> int:
    generation_manifest = _verify_frozen_generation(args.output_dir)
    records, _entry = g3._runtime_integrity(
        runtime_path=args.runtime,
        dataset=args.dataset,
        data_manifest_path=args.data_manifest,
    )
    expected_ids = [str(row["query_id"]) for row in records]
    if generation_manifest["ordered_ids_sha256"] != ordered_id_sha256(expected_ids):
        raise ValueError("Goal 4 generation IDs differ from sealed runtime")
    if _sha256(args.prepared) != generation_manifest["goal4_prepared_sha256"]:
        raise ValueError("Goal 4 prepared rows changed after generation freeze")
    sidecars = g3._sidecar_integrity(
        sidecar_path=args.sidecar,
        dataset=args.dataset,
        data_manifest_path=args.data_manifest,
        expected_ids=expected_ids,
    )
    prepared_rows = read_jsonl(args.prepared)
    manifest = g3._model_manifest(args.model_manifest)
    minicheck_identity = manifest["models"]["minicheck_scorer_only"]
    snapshot = g3._download_snapshot(args.cache_dir, minicheck_identity)
    minicheck = MiniCheckNLIModel(model_id=str(snapshot))
    _tokenizer, model = minicheck._ensure_loaded()
    if args.device != "cpu":
        minicheck._model = model.to(args.device)
    entail_cache: dict[tuple[str, str], bool] = {}

    def entails(premise: str, hypothesis: str) -> bool:
        key = (premise, hypothesis)
        if key not in entail_cache:
            entail_cache[key] = (
                minicheck.classify(premise=premise, hypothesis=hypothesis) == "entailment"
            )
        return entail_cache[key]

    score_dir = args.output_dir / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    arm_entries: dict[str, Any] = {}
    for arm in ABLATION_ARMS:
        generation_path = _generation_paths(args.output_dir)[arm]
        score_path = score_dir / f"{arm}.json"
        score, judge_calls = score_frozen_arm(
            sidecar_records=sidecars,
            outputs=read_jsonl(generation_path),
            prepared_records=prepared_rows,
            arm_id=arm,
            entails=entails,
        )
        artifact = {
            "schema_version": "experiment04.goal4_arm_score.v1",
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
        "schema_version": "experiment04.goal4_dataset_score_manifest.v1",
        "dataset": args.dataset,
        "query_count": len(records),
        "ordered_ids_sha256": ordered_id_sha256(expected_ids),
        "generation_manifest_sha256": _sha256(args.output_dir / "generation_manifest.json"),
        "scorer_sidecar_sha256": _sha256(args.sidecar),
        "minicheck_config_sha256": minicheck_identity["config_sha256"],
        "generation_frozen_before_sidecar_read": True,
        "full_regenerated": False,
        "arms": arm_entries,
        "status": "PASS",
    }
    _write_json(args.output_dir / "score_manifest.json", score_manifest)
    return 0


def _goal4_scores(dataset_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = dataset_dir / "score_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "experiment04.goal4_dataset_score_manifest.v1"
        or manifest.get("status") != "PASS"
        or manifest.get("full_regenerated") is not False
        or set(manifest.get("arms", {})) != set(ABLATION_ARMS)
    ):
        raise ValueError("Goal 4 dataset score manifest is not exact PASS")
    scores: dict[str, dict[str, Any]] = {}
    for arm in ABLATION_ARMS:
        path = dataset_dir / "scores" / f"{arm}.json"
        if _sha256(path) != manifest["arms"][arm]["score_sha256"]:
            raise ValueError(f"Goal 4 score artifact hash differs: {arm}")
        scores[arm] = json.loads(path.read_text(encoding="utf-8"))["score"]
    return manifest, scores


def _goal3_full_score(dataset_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    score_manifest_path = dataset_dir / "score_manifest.json"
    generation_manifest_path = dataset_dir / "generation_manifest.json"
    score_manifest = json.loads(score_manifest_path.read_text(encoding="utf-8"))
    generation_manifest = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
    score_path = dataset_dir / "scores" / f"{FULL_GOAL3_ARM}.json"
    generation_path = dataset_dir / "generations" / f"{FULL_GOAL3_ARM}.jsonl"
    if (
        score_manifest.get("status") != "PASS"
        or generation_manifest.get("bundle_status") != "PASS"
        or score_manifest["arms"][FULL_GOAL3_ARM]["score_sha256"] != _sha256(score_path)
        or generation_manifest["arms"][FULL_GOAL3_ARM]["file_sha256"] != _sha256(generation_path)
    ):
        raise ValueError("Goal 3 Full seed13 artifact is not frozen PASS")
    return score_manifest, json.loads(score_path.read_text(encoding="utf-8"))["score"]


def _parse_dataset_dirs(items: Sequence[str]) -> dict[str, Path]:
    values = {name: Path(path).resolve() for name, path in (item.split("=", 1) for item in items)}
    if set(values) != set(EXPECTED_COUNTS):
        raise ValueError("requires exactly hotpotqa, musique-answerable, and rgb-noise")
    return values


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}"


def _table_markdown(table: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    lines = ["# Experiment 04 — Table 2", ""]
    for dataset in EXPECTED_COUNTS:
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Configuration | Ret. | Sel. | Ans. | Cit. | RAR |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in table[dataset]:
            metrics = " | ".join(
                _percent(float(row[key])) for key in ("ret", "sel", "ans", "cit", "rar")
            )
            lines.append(f"| {row['configuration']} | {metrics} |")
        lines.append("")
    lines.append(
        "Full is the frozen Goal 3 Ours seed13 artifact; each non-Full row replaces only "
        "the named module."
    )
    return "\n".join(lines)


def summarize_stage(args: argparse.Namespace) -> int:
    goal4_dirs = _parse_dataset_dirs(args.goal4_dataset_dir)
    goal3_dirs = _parse_dataset_dirs(args.full_dataset_dir)
    labels = {
        FULL_GOAL3_ARM: "Full",
        "ablation_dense_retriever": "w/ Dense Retriever",
        "ablation_top10": "w/ Top-10",
        "ablation_direct_generator": "w/ Direct Generator",
    }
    table: dict[str, list[dict[str, Any]]] = {}
    bootstrap: dict[str, Any] = {
        "schema_version": "experiment04.goal4_paired_rar_bootstrap.v1",
        "direction": "Full_minus_ablation",
        "resamples": 10_000,
        "seed": 13,
        "datasets": {},
    }
    per_query_rows: list[dict[str, Any]] = []
    manifests: dict[str, Any] = {}
    for dataset in EXPECTED_COUNTS:
        goal4_manifest, ablation_scores = _goal4_scores(goal4_dirs[dataset])
        goal3_manifest, full_score = _goal3_full_score(goal3_dirs[dataset])
        if (
            int(goal4_manifest["query_count"]) != EXPECTED_COUNTS[dataset]
            or int(goal3_manifest["query_count"]) != EXPECTED_COUNTS[dataset]
            or goal4_manifest["ordered_ids_sha256"] != goal3_manifest["ordered_ids_sha256"]
        ):
            raise ValueError(f"{dataset} Goal 3/Goal 4 frozen ID contract differs")
        scores = {FULL_GOAL3_ARM: full_score, **ablation_scores}
        table[dataset] = [
            {"configuration": labels[arm], **scores[arm]["aggregate"]}
            for arm in (FULL_GOAL3_ARM, *ABLATION_ARMS)
        ]
        by_arm = {
            arm: {str(row["query_id"]): row for row in score["per_query"]}
            for arm, score in scores.items()
        }
        query_ids = tuple(by_arm[FULL_GOAL3_ARM])
        if any(tuple(by_arm[arm]) != query_ids for arm in ABLATION_ARMS):
            raise ValueError(f"{dataset} Table 2 per-query order differs")
        components = {
            query_id: str(by_arm[FULL_GOAL3_ARM][query_id]["component_id"])
            for query_id in query_ids
        }
        bootstrap["datasets"][dataset] = {
            labels[arm]: paired_component_cluster_bootstrap(
                candidate={
                    query_id: float(by_arm[FULL_GOAL3_ARM][query_id]["metrics"]["rar"])
                    for query_id in query_ids
                },
                baseline={
                    query_id: float(by_arm[arm][query_id]["metrics"]["rar"])
                    for query_id in query_ids
                },
                component_ids=components,
                resamples=10_000,
                seed=13,
            )
            for arm in ABLATION_ARMS
        }
        for arm in (FULL_GOAL3_ARM, *ABLATION_ARMS):
            for row in scores[arm]["per_query"]:
                per_query_rows.append(
                    {
                        "dataset": dataset,
                        "arm_id": arm,
                        "query_id": row["query_id"],
                        "component_id": row["component_id"],
                        **row["metrics"],
                        "failure_reason": row["failure_reason"] or "",
                        "provenance": "reused_goal3_full" if arm == FULL_GOAL3_ARM else "new_goal4",
                    }
                )
        manifests[dataset] = {
            "goal4_directory": str(goal4_dirs[dataset]),
            "goal4_score_manifest_sha256": _sha256(goal4_dirs[dataset] / "score_manifest.json"),
            "goal4_generation_manifest_sha256": goal4_manifest["generation_manifest_sha256"],
            "goal3_full_directory": str(goal3_dirs[dataset]),
            "goal3_score_manifest_sha256": _sha256(goal3_dirs[dataset] / "score_manifest.json"),
            "goal3_full_score_sha256": goal3_manifest["arms"][FULL_GOAL3_ARM]["score_sha256"],
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "table2.json"
    markdown_path = args.output_dir / "TABLE2.md"
    bootstrap_path = args.output_dir / "bootstrap_goal4_ci.json"
    csv_path = args.output_dir / "per_query_goal4_metrics.csv"
    _write_json(table_path, {"schema_version": "experiment04.table2.v1", "datasets": table})
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
                "provenance",
            ),
        )
        writer.writeheader()
        writer.writerows(per_query_rows)
    audit = {
        "schema_version": "experiment04.goal4_audit.v1",
        "status": "PASS",
        "datasets": manifests,
        "new_arms": list(ABLATION_ARMS),
        "full_source_arm": FULL_GOAL3_ARM,
        "full_regenerated": False,
        "single_module_substitutions_validated": True,
        "expected_new_outputs": 3_300,
        "observed_new_outputs": sum(row["provenance"] == "new_goal4" for row in per_query_rows),
        "expected_reused_full_rows": 1_100,
        "observed_reused_full_rows": sum(
            row["provenance"] == "reused_goal3_full" for row in per_query_rows
        ),
        "expected_table2_rows": 4_400,
        "observed_table2_rows": len(per_query_rows),
        "paired_rar_bootstrap": {
            "direction": "Full_minus_ablation",
            "resamples": 10_000,
            "seed": 13,
        },
        "artifacts": {
            "table2_json_sha256": _sha256(table_path),
            "table2_markdown_sha256": _sha256(markdown_path),
            "bootstrap_sha256": _sha256(bootstrap_path),
            "per_query_metrics_sha256": _sha256(csv_path),
        },
    }
    if (
        audit["observed_new_outputs"] != audit["expected_new_outputs"]
        or audit["observed_reused_full_rows"] != audit["expected_reused_full_rows"]
        or audit["observed_table2_rows"] != audit["expected_table2_rows"]
    ):
        audit["status"] = "FAIL"
    _write_json(args.output_dir / "goal4_audit.json", audit)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "new_outputs": audit["observed_new_outputs"],
                "reused_full": audit["observed_reused_full_rows"],
                "datasets": len(manifests),
            },
            sort_keys=True,
        )
    )
    return 0 if audit["status"] == "PASS" else 2


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True, choices=tuple(EXPECTED_COUNTS))
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--goal3-dir", required=True, type=Path)
    parser.add_argument("--data-manifest", default=DEFAULT_DATA_MANIFEST, type=Path)
    parser.add_argument("--model-manifest", default=DEFAULT_MODEL_MANIFEST, type=Path)


def _generation_common(parser: argparse.ArgumentParser) -> None:
    _common(parser)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("float16", "bfloat16", "float32"),
    )
    parser.add_argument("--log-every", default=25, type=int)


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    _common(preflight)
    preflight.add_argument("--selector-checkpoint", required=True, type=Path)
    preflight.add_argument("--adapter13", required=True, type=Path)
    preflight.add_argument("--output", required=True, type=Path)
    prepare = commands.add_parser("prepare")
    _common(prepare)
    prepare.add_argument("--prepared", required=True, type=Path)
    prepare.add_argument("--cache-dir", required=True, type=Path)
    prepare.add_argument("--selector-checkpoint", required=True, type=Path)
    prepare.add_argument("--device", default="cuda")
    prepare.add_argument("--log-every", default=25, type=int)
    grounded = commands.add_parser("generate-grounded")
    _generation_common(grounded)
    grounded.add_argument("--adapter13", required=True, type=Path)
    direct = commands.add_parser("generate-direct")
    _generation_common(direct)
    freeze = commands.add_parser("freeze")
    _common(freeze)
    freeze.add_argument("--prepared", required=True, type=Path)
    freeze.add_argument("--output-dir", required=True, type=Path)
    freeze.add_argument("--attempt-id", required=True)
    score = commands.add_parser("score")
    _generation_common(score)
    score.add_argument("--sidecar", required=True, type=Path)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--goal4-dataset-dir", required=True, action="append")
    summarize.add_argument("--full-dataset-dir", required=True, action="append")
    summarize.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    for name in (
        "runtime",
        "goal3_dir",
        "data_manifest",
        "model_manifest",
        "prepared",
        "output_dir",
        "cache_dir",
        "selector_checkpoint",
        "adapter13",
        "sidecar",
        "output",
    ):
        if hasattr(args, name):
            setattr(args, name, getattr(args, name).resolve())
    if hasattr(args, "log_every") and args.log_every <= 0:
        raise ValueError("log-every must be positive")
    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "preflight": preflight_stage,
        "prepare": prepare_stage,
        "generate-grounded": generate_grounded_stage,
        "generate-direct": generate_direct_stage,
        "freeze": freeze_stage,
        "score": score_stage,
        "summarize": summarize_stage,
    }
    try:
        return handlers[args.command](args)
    except Exception as error:  # noqa: BLE001 - never echo held-out prose in formal logs
        name = re.sub(r"[^a-z0-9]+", "_", type(error).__name__.casefold()).strip("_")
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "stage": args.command,
                    "error_code": f"goal4_{name or 'error'}",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
