"""Evaluate frozen F006 base/clean/mixed arms on the 109 F004 dev cases.

TopK and base-notes outputs are reused byte-for-byte from F001/F004.  Clean and
mixed adapters share one Granite base and one TRUE verifier in the same process.
Adapters are enabled only for the key-fact extraction call; drafting and claim
splitting explicitly use the frozen base model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from full_flow_f004 import _jsonl, _load_gold, _routing_rows, _write_json, _write_jsonl, load_cases

from evidence_rag.contracts.models import (
    GenerationResult,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
    strip_annotations,
)
from evidence_rag.evaluation.paired_metric import compare_paired
from evidence_rag.evaluation.scoring import answer_match
from evidence_rag.generator.draft import KeyFactDraftAnswerGenerator
from evidence_rag.generator.granite import (
    GraniteGenerationConfig,
    NamedAdapterTextGenerator,
    PeftGraniteLLMClient,
)
from evidence_rag.generator.key_facts import notes_manifest
from evidence_rag.generator.nli import TrueNLIModel
from evidence_rag.generator.verify_annotate import VerifyAnnotateGenerator
from evidence_rag.infrastructure.datasets import GoldCase

ARMS = ("TopK_frozen", "Base_notes_frozen", "Clean_LoRA", "Mixed_LoRA")
GENERATED_ARMS = ARMS[2:]
EXPECTED_CASES = 109


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_rows(path: Path, arm: str) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in _jsonl(path):
        if not bool(row.get("selector_changed")):
            continue
        arms = row.get("arms")
        if not isinstance(arms, Mapping) or not isinstance(arms.get(arm), Mapping):
            raise ValueError(f"frozen source has no {arm} for {row.get('query_id')}")
        output[str(row["query_id"])] = cast(Mapping[str, Any], arms[arm])
    if len(output) != EXPECTED_CASES:
        raise ValueError(f"frozen {arm} count is {len(output)}, expected {EXPECTED_CASES}")
    return output


def _run_one(
    generator: VerifyAnnotateGenerator,
    draft_generator: KeyFactDraftAnswerGenerator,
    query: Query,
    checklist: QueryChecklist,
    selected: SelectedEvidenceSet,
) -> dict[str, object]:
    started = time.perf_counter()
    generation = generator.generate_with_guidance(query, checklist, selected, None)
    return {
        "generation": generation.model_dump(mode="json"),
        "routing": _routing_rows(generator),
        "notes": json.loads(notes_manifest(draft_generator.last_notes)),
        "error": None,
        "seconds": time.perf_counter() - started,
    }


def run(
    *,
    f001_generations: Path,
    f004_generations: Path,
    selector_trace: Path,
    model_snapshot: Path,
    true_model_id: str,
    clean_adapter: Path,
    mixed_adapter: Path,
    output_dir: Path,
    limit: int | None = None,
) -> dict[str, object]:
    cases = load_cases(f001_generations, selector_trace)
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        cases = cases[:limit]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("F006 evaluation output must be absent or empty")
    topk = _frozen_rows(f001_generations, "C_topk_verify")
    base = _frozen_rows(f004_generations, "G1_notes")

    client = PeftGraniteLLMClient(
        model_id=str(model_snapshot.resolve()),
        adapters={"clean": str(clean_adapter.resolve()), "mixed": str(mixed_adapter.resolve())},
        config=GraniteGenerationConfig(max_new_tokens=256, temperature=0.0, top_p=1.0),
    )
    nli = TrueNLIModel(model_id=true_model_id)
    clean_draft = KeyFactDraftAnswerGenerator(
        client,
        note_llm=NamedAdapterTextGenerator(client, "clean"),
        guided=False,
    )
    mixed_draft = KeyFactDraftAnswerGenerator(
        client,
        note_llm=NamedAdapterTextGenerator(client, "mixed"),
        guided=False,
    )
    clean = VerifyAnnotateGenerator(
        draft_generator=clean_draft,
        nli=nli,
        entity_gate="observe",
        abstain_when_unverified=False,
    )
    mixed = VerifyAnnotateGenerator(
        draft_generator=mixed_draft,
        nli=nli,
        entity_gate="observe",
        abstain_when_unverified=False,
    )

    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        query = Query(query_id=case.query_id, text=case.question)
        checklist = QueryChecklist(query_id=case.query_id, focus=case.question, required_facts=())
        selected = SelectedEvidenceSet(query_id=case.query_id, evidence=case.selected)
        rows.append(
            {
                "schema_version": "full-flow-f006-generation-row-v1",
                "query_id": case.query_id,
                "question": case.question,
                "component_id": case.component_id,
                "topk10": [item.model_dump(mode="json") for item in case.topk10],
                "selected_evidence_ids": [item.evidence_id for item in case.selected],
                "arms": {
                    ARMS[0]: {**topk[case.query_id], "reused_from": "F001.C_topk_verify"},
                    ARMS[1]: {**base[case.query_id], "reused_from": "F004.G1_notes"},
                    ARMS[2]: _run_one(clean, clean_draft, query, checklist, selected),
                    ARMS[3]: _run_one(mixed, mixed_draft, query, checklist, selected),
                },
            }
        )
        if index % 10 == 0 or index == len(cases):
            print(
                f"[F006 eval] {index}/{len(cases)} "
                f"{(time.perf_counter()-started)/index:.2f}s/case",
                flush=True,
            )
    _write_jsonl(output_dir / "generations.jsonl", rows)
    manifest: dict[str, object] = {
        "schema_version": "full-flow-f006-eval-manifest-v1",
        "status": "COMPLETE",
        "queries": len(rows),
        "gold_loaded_at_runtime": False,
        "topk_reused_from_f001": True,
        "base_notes_reused_from_f004": True,
        "clean_mixed_same_process": True,
        "shared_granite_base": True,
        "shared_true": True,
        "adapter_scope": "key-fact extraction call only",
        "draft_and_claim_splitter_scope": "frozen Granite base with adapters disabled",
        "clean_adapter_config_sha256": _sha256(clean_adapter / "adapter_config.json"),
        "mixed_adapter_config_sha256": _sha256(mixed_adapter / "adapter_config.json"),
        "generations_sha256": _sha256(output_dir / "generations.jsonl"),
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    return manifest


def _metric(generation: object, gold: GoldCase) -> tuple[float, float]:
    result = GenerationResult.model_validate(generation)
    value = answer_match(strip_annotations(result.answer), gold.reference_answers).value
    if value is None:
        raise ValueError(f"no scorable gold for {gold.query_id}")
    return value, float(bool(result.answer.strip()))


def score(rows: Sequence[Mapping[str, Any]], gold: Mapping[str, GoldCase]) -> dict[str, object]:
    components = {str(row["query_id"]): str(row["component_id"]) for row in rows}
    values: dict[str, dict[str, dict[str, float]]] = {arm: {} for arm in ARMS}
    notes_nonempty = dict.fromkeys(ARMS, 0)
    for row in rows:
        query_id = str(row["query_id"])
        arms = row["arms"]
        if not isinstance(arms, Mapping):
            raise ValueError(f"invalid F006 arms for {query_id}")
        for arm in ARMS:
            arm_row = arms[arm]
            if not isinstance(arm_row, Mapping):
                raise ValueError(f"invalid F006 {arm} for {query_id}")
            answer_value, coverage = _metric(arm_row["generation"], gold[query_id])
            values[arm][query_id] = {"answer_match": answer_value, "coverage": coverage}
            notes_nonempty[arm] += int(bool(arm_row.get("notes")))

    aggregate = {
        arm: {
            metric: sum(item[metric] for item in values[arm].values()) / len(rows)
            for metric in ("answer_match", "coverage")
        }
        for arm in ARMS
    }
    comparisons: dict[str, object] = {}
    for label, on_arm, off_arm in (
        ("Clean_minus_Base", ARMS[2], ARMS[1]),
        ("Mixed_minus_Base", ARMS[3], ARMS[1]),
        ("Mixed_minus_Clean", ARMS[3], ARMS[2]),
        ("Mixed_minus_TopK", ARMS[3], ARMS[0]),
    ):
        comparison: dict[str, object] = {}
        for metric in ("answer_match", "coverage"):
            comparison[metric] = asdict(
                compare_paired(
                    {q: item[metric] for q, item in values[on_arm].items()},
                    {q: item[metric] for q, item in values[off_arm].items()},
                    component_ids=components,
                )
            )
        wrong_to_right = sum(
            values[off_arm][q]["answer_match"] == 0.0
            and values[on_arm][q]["answer_match"] == 1.0
            for q in components
        )
        right_to_wrong = sum(
            values[off_arm][q]["answer_match"] == 1.0
            and values[on_arm][q]["answer_match"] == 0.0
            for q in components
        )
        comparison["transitions"] = {
            "wrong_to_right": wrong_to_right,
            "right_to_wrong": right_to_wrong,
            "net": wrong_to_right - right_to_wrong,
        }
        comparisons[label] = comparison
    gate = {
        "mixed_beats_base_point": aggregate[ARMS[3]]["answer_match"]
        > aggregate[ARMS[1]]["answer_match"],
        "mixed_beats_clean_point": aggregate[ARMS[3]]["answer_match"]
        > aggregate[ARMS[2]]["answer_match"],
        "mixed_beats_topk_point": aggregate[ARMS[3]]["answer_match"]
        > aggregate[ARMS[0]]["answer_match"],
    }
    gate["pass_for_f005"] = all(gate.values())
    return {
        "schema_version": "full-flow-f006-report-v1",
        "status": "COMPLETE",
        "queries": len(rows),
        "gold_loaded_at_runtime": False,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "notes_nonempty_queries": notes_nonempty,
        "development_gate": gate,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# F006 LoRA 开发集结果",
        "",
        "| 组 | Answer match | Coverage | 有笔记题数 |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = report["aggregate"][arm]
        lines.append(
            f"| {arm} | {100*item['answer_match']:.2f}% | "
            f"{100*item['coverage']:.2f}% | {report['notes_nonempty_queries'][arm]} |"
        )
    lines.extend(["", "| 比较 | Answer delta | 95% CI | wrong→right | right→wrong |", "|---|---:|---:|---:|---:|"])
    for label, comparison in report["comparisons"].items():
        answer = comparison["answer_match"]
        transitions = comparison["transitions"]
        lines.append(
            f"| {label} | {100*answer['delta']:+.2f} pp | "
            f"[{100*answer['ci_low']:+.2f}, {100*answer['ci_high']:+.2f}] | "
            f"{transitions['wrong_to_right']} | {transitions['right_to_wrong']} |"
        )
    lines.extend(["", f"**F005 gate：** {'PASS' if report['development_gate']['pass_for_f005'] else 'FAIL'}", ""])
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--f001-generations", required=True, type=Path)
    run_parser.add_argument("--f004-generations", required=True, type=Path)
    run_parser.add_argument("--selector-trace", required=True, type=Path)
    run_parser.add_argument("--model-snapshot", required=True, type=Path)
    run_parser.add_argument("--true-model-id", required=True)
    run_parser.add_argument("--clean-adapter", required=True, type=Path)
    run_parser.add_argument("--mixed-adapter", required=True, type=Path)
    run_parser.add_argument("--output-dir", required=True, type=Path)
    run_parser.add_argument("--limit", type=int)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--generations", required=True, type=Path)
    score_parser.add_argument("--gold", required=True, type=Path)
    score_parser.add_argument("--output-json", required=True, type=Path)
    score_parser.add_argument("--output-report", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        run(
            f001_generations=args.f001_generations,
            f004_generations=args.f004_generations,
            selector_trace=args.selector_trace,
            model_snapshot=args.model_snapshot,
            true_model_id=args.true_model_id,
            clean_adapter=args.clean_adapter,
            mixed_adapter=args.mixed_adapter,
            output_dir=args.output_dir.resolve(),
            limit=args.limit,
        )
        return 0
    rows: Sequence[Mapping[str, Any]] = tuple(_jsonl(args.generations))
    wanted = {str(row["query_id"]) for row in rows}
    report = score(rows, _load_gold(args.gold, wanted))
    _write_json(args.output_json, report)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
