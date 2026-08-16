"""Independent MiniCheck citation scoring for the G230 Generator gate.

TRUE remains part of the frozen production routing path and is excluded from
judging.  This post-generation command reconstructs G0 sentence/citation pairs
from the archived A002/B100 traces and reads exact routing rows for GN/GC/GM.
Citation quality is compared on the paired intersection of answered tasks;
coverage is handled by the G230 answer report.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from alce_metrics import CitationReport, compute_citation_metrics
from full_flow_citation_score import (
    _mean,
    _paired_common_answered,
    _verify_example,
)
from full_flow_g230 import (
    ALLOWED_SEEDS,
    CONTEXTS,
    FULL_CONTEXT,
    STRESS_CONTEXTS,
    _baseline_results,
    _candidate_results,
    _json,
    _jsonl,
    _sha256,
    _write_json,
    _write_jsonl,
)

from evidence_rag.contracts.models import EvidenceCandidate, GenerationResult
from evidence_rag.generator.nli import MiniCheckNLIModel

CITATION_NONINFERIORITY_MARGIN = -0.02
CONFIG_ORDER = ("G0", "GN", "GC13", "GM13", "GC42", "GM42", "GC73", "GM73")


def routing_from_trace(run_row: Mapping[str, object]) -> list[dict[str, object]]:
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


def _candidate_rows(
    path: Path, expected_arms: Sequence[str]
) -> tuple[
    dict[str, dict[str, Mapping[str, object]]],
    dict[str, tuple[EvidenceCandidate, ...]],
    dict[str, str],
]:
    results, _manifest = _candidate_results(path, expected_arms)
    evidence: dict[str, tuple[EvidenceCandidate, ...]] = {}
    components: dict[str, str] = {}
    for row in _jsonl(path):
        task_id = str(row["task_id"])
        raw_evidence = row.get("evidence")
        if not isinstance(raw_evidence, Sequence):
            raise ValueError(f"candidate task {task_id} has no evidence payload")
        evidence[task_id] = tuple(
            EvidenceCandidate.model_validate(item) for item in raw_evidence
        )
        components[task_id] = str(row["component_id"])
    return results, evidence, components


def _load_all(
    *,
    a002_generations_path: Path,
    b100_generations_path: Path,
    gn_generations_path: Path,
    seed_generations: Mapping[int, Path],
) -> tuple[
    dict[str, Mapping[str, Mapping[str, object]]],
    dict[str, tuple[EvidenceCandidate, ...]],
    dict[str, str],
]:
    baseline, baseline_components = _baseline_results(
        _jsonl(a002_generations_path), _jsonl(b100_generations_path)
    )
    configs: dict[str, Mapping[str, Mapping[str, object]]] = {"G0": baseline}
    gn, canonical_evidence, components = _candidate_rows(
        gn_generations_path, ("GN",)
    )
    configs["GN"] = gn["GN"]
    if components != baseline_components:
        raise ValueError("G230 candidate component IDs differ from G0")
    for seed in ALLOWED_SEEDS:
        candidates, evidence, seed_components = _candidate_rows(
            seed_generations[seed], ("GC", "GM")
        )
        if evidence != canonical_evidence or seed_components != components:
            raise ValueError(f"G230 seed-{seed} task evidence differs from GN")
        configs[f"GC{seed}"] = candidates["GC"]
        configs[f"GM{seed}"] = candidates["GM"]
    expected_tasks = set(baseline)
    if set(canonical_evidence) != expected_tasks:
        raise ValueError("G230 citation evidence does not exactly cover G0 tasks")
    for name, rows in configs.items():
        if set(rows) != expected_tasks:
            raise ValueError(f"G230 citation config {name} has incomplete tasks")
    return configs, canonical_evidence, components


def score_rows(
    *,
    configs: Mapping[str, Mapping[str, Mapping[str, object]]],
    evidence: Mapping[str, tuple[EvidenceCandidate, ...]],
    components: Mapping[str, str],
    entails: Callable[[str, str], bool],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if tuple(configs) != CONFIG_ORDER:
        raise ValueError("G230 citation configs differ from the frozen order")
    task_ids_by_context = {
        context: sorted(
            task_id for task_id in evidence if task_id.split("::", 2)[1] == context
        )
        for context in CONTEXTS
    }
    per_case: dict[str, dict[str, dict[str, float | None]]] = {
        name: {} for name in configs
    }
    aggregate: dict[str, dict[str, object]] = {name: {} for name in configs}
    for name, rows in configs.items():
        for context, task_ids in task_ids_by_context.items():
            examples = []
            answered: dict[str, bool] = {}
            for task_id in task_ids:
                run_row = rows[task_id]
                generation = GenerationResult.model_validate(run_row["generation"])
                answered[task_id] = bool(generation.answer.strip())
                docs = {item.evidence_id: item.text for item in evidence[task_id]}
                routing = (
                    routing_from_trace(run_row)
                    if name == "G0"
                    else run_row.get("routing")
                )
                example = _verify_example(task_id, routing, docs)
                if example is not None:
                    examples.append(example)
            report: CitationReport = compute_citation_metrics(examples, entails)
            by_id = {str(item["example_id"]): item for item in report.per_example}
            for task_id in task_ids:
                item = by_id.get(task_id)
                per_case[name][task_id] = {
                    "citation_precision": (
                        float(item["citation_prec"]) if item is not None else None
                    ),
                    "citation_recall": (
                        float(item["citation_rec"]) if item is not None else None
                    ),
                }
            precision = [
                item["citation_precision"]
                for task_id in task_ids
                if (item := per_case[name][task_id])["citation_precision"] is not None
            ]
            recall = [
                item["citation_recall"]
                for task_id in task_ids
                if (item := per_case[name][task_id])["citation_recall"] is not None
            ]
            cited_precision = [
                float(item["citation_prec"])
                for item in report.per_example
                if int(item["citations"]) > 0
            ]
            aggregate[name][context] = {
                "tasks": len(task_ids),
                "answered_tasks": sum(answered.values()),
                "citation_precision_alce": _mean(cast(Sequence[float], precision)),
                "citation_precision_cited_tasks": _mean(cited_precision),
                "citation_recall_alce": _mean(cast(Sequence[float], recall)),
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
                context: {
                    metric: _paired_common_answered(
                        {
                            task_id: per_case[after_name][task_id][metric]
                            for task_id in task_ids
                        },
                        {
                            task_id: per_case[before_name][task_id][metric]
                            for task_id in task_ids
                        },
                        {task_id: components[task_id] for task_id in task_ids},
                    )
                    for metric in ("citation_precision", "citation_recall")
                }
                for context, task_ids in task_ids_by_context.items()
            }

    noninferior_by_family: dict[str, dict[int, dict[str, bool]]] = {}
    for family in ("GC", "GM"):
        noninferior_by_seed: dict[int, dict[str, bool]] = {}
        for seed in ALLOWED_SEEDS:
            full = cast(Mapping[str, Any], comparisons[f"{family}{seed}_minus_G0"])[
                FULL_CONTEXT
            ]
            noninferior_by_seed[seed] = {}
            for metric in ("citation_precision", "citation_recall"):
                item = full[metric]
                noninferior_by_seed[seed][metric] = (
                    item.get("status") == "SCORED_COMMON_ANSWERED"
                    and float(item["ci_low"]) >= CITATION_NONINFERIORITY_MARGIN
                )
        noninferior_by_family[family] = noninferior_by_seed
    pass_by_family = {
        family: all(all(metrics.values()) for metrics in seeds.values())
        for family, seeds in noninferior_by_family.items()
    }
    gate = {
        "margin": CITATION_NONINFERIORITY_MARGIN,
        "noninferior_by_family": noninferior_by_family,
        "pass_by_family": pass_by_family,
    }
    output_rows = [
        {
            "schema_version": "full-flow-g230-citation-case-v1",
            "task_id": task_id,
            "scope": task_id.split("::", 2)[0],
            "context": task_id.split("::", 2)[1],
            "query_id": task_id.rsplit("::", 1)[-1],
            "component_id": components[task_id],
            "configs": {name: per_case[name][task_id] for name in configs},
        }
        for task_id in sorted(evidence)
    ]
    return (
        {
            "schema_version": "full-flow-g230-citation-report-v1",
            "status": "COMPLETE",
            "judge": "MiniCheck-Flan-T5-Large",
            "production_verifier": "TRUE (excluded from judging)",
            "mapping": "exact sentence-citation pairs from routing or archived Generator trace",
            "contexts": [FULL_CONTEXT, *STRESS_CONTEXTS],
            "aggregate": aggregate,
            "comparisons": comparisons,
            "citation_gate": gate,
        },
        output_rows,
    )


def _markdown(summary: Mapping[str, Any]) -> str:
    def value(item: float | None) -> str:
        return "-" if item is None else f"{100 * item:.2f}%"

    aggregate = cast(Mapping[str, Mapping[str, Mapping[str, Any]]], summary["aggregate"])
    lines = [
        "# G230 Independent Citation Report",
        "",
        "Judge: MiniCheck. Production TRUE is excluded from judging.",
        "",
        "## Full Decision-Dev (TopK)",
        "",
        "| Config | Answered | Citation precision | Precision (cited tasks) | Citation recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in CONFIG_ORDER:
        item = aggregate[name][FULL_CONTEXT]
        lines.append(
            f"| {name} | {item['answered_tasks']} | "
            f"{value(item['citation_precision_alce'])} | "
            f"{value(item['citation_precision_cited_tasks'])} | "
            f"{value(item['citation_recall_alce'])} |"
        )
    lines.extend(["", "## Citation Gate", "", "```json"])
    lines.append(json.dumps(summary["citation_gate"], ensure_ascii=False, indent=2))
    lines.extend(["```", ""])
    if "final_development_gate" in summary:
        lines.extend(["## Final Development Gate", "", "```json"])
        lines.append(
            json.dumps(summary["final_development_gate"], ensure_ascii=False, indent=2)
        )
        lines.extend(["```", ""])
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a002-generations", required=True, type=Path)
    parser.add_argument("--b100-generations", required=True, type=Path)
    parser.add_argument("--gn-generations", required=True, type=Path)
    for seed in ALLOWED_SEEDS:
        parser.add_argument(f"--seed{seed}-generations", required=True, type=Path)
    parser.add_argument("--answer-report", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-rows", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    seed_generations = {
        seed: cast(Path, getattr(args, f"seed{seed}_generations"))
        for seed in ALLOWED_SEEDS
    }
    configs, evidence, components = _load_all(
        a002_generations_path=args.a002_generations,
        b100_generations_path=args.b100_generations,
        gn_generations_path=args.gn_generations,
        seed_generations=seed_generations,
    )
    judge = MiniCheckNLIModel(model_id=args.model_id)
    if args.device != "cpu":
        _tokenizer, model = judge._ensure_loaded()
        judge._model = model.to(args.device)
    cache: dict[tuple[str, str], bool] = {}

    def entails(premise: str, hypothesis: str) -> bool:
        key = (premise, hypothesis)
        if key not in cache:
            cache[key] = (
                judge.classify(premise=premise, hypothesis=hypothesis) == "entailment"
            )
        return cache[key]

    summary, rows = score_rows(
        configs=configs,
        evidence=evidence,
        components=components,
        entails=entails,
    )
    answer_report = _json(args.answer_report)
    if answer_report.get("status") != "COMPLETE_PENDING_CITATION":
        raise ValueError("G230 citation scoring requires the complete answer report")
    pre_citation = cast(Mapping[str, object], answer_report["development_gate"])
    candidate = pre_citation.get("candidate_before_citation")
    citation_by_family = cast(
        Mapping[str, bool],
        cast(Mapping[str, object], summary["citation_gate"])["pass_by_family"],
    )
    citation_pass = isinstance(candidate, str) and citation_by_family.get(candidate, False)
    final_pass = bool(pre_citation.get("pre_citation_pass")) and citation_pass
    summary["answer_report_sha256"] = _sha256(args.answer_report)
    summary["judge_device"] = args.device
    summary["unique_judge_calls"] = len(cache)
    summary["final_development_gate"] = {
        "pre_citation_pass": bool(pre_citation.get("pre_citation_pass")),
        "candidate_before_citation": candidate,
        "candidate_citation_pass": citation_pass,
        "selected_generator": candidate if final_pass else None,
        "pass": final_pass,
    }
    _write_json(args.output_json, summary)
    _write_jsonl(args.output_rows, rows)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
