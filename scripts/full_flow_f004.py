"""F004 G0/G1/G2 development ablation on the 109 Selector-changed cases.

G0 is the frozen F001 D arm.  G1 and G2 run in one process with one Granite and
one TRUE instance:

    G0: selected evidence + current Verify-and-annotate (reused from F001)
    G1: G0 + question-slot key-fact notes
    G2: G1 + runtime-safe Selector signals on retained evidence

The ``run`` command has no gold argument.  Evaluation labels are accepted only
by ``score`` after all generations have been written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from evidence_rag.contracts.models import (
    EvidenceCandidate,
    GenerationResult,
    Query,
    QueryChecklist,
    RetainedEvidenceGuidance,
    SelectedEvidenceSet,
    SelectionGuidance,
    strip_annotations,
)
from evidence_rag.evaluation.paired_metric import compare_paired
from evidence_rag.evaluation.scoring import answer_match
from evidence_rag.generator.draft import KeyFactDraftAnswerGenerator
from evidence_rag.generator.granite import GraniteGenerationConfig, GraniteLLMClient
from evidence_rag.generator.key_facts import detect_question_slots, notes_manifest
from evidence_rag.generator.nli import TrueNLIModel
from evidence_rag.generator.verify_annotate import VerifyAnnotateGenerator
from evidence_rag.infrastructure.datasets import GoldCase

ARMS = ("G0_current", "G1_notes", "G2_guided_notes")
EXPECTED_CASES = 109


@dataclass(frozen=True, slots=True)
class F004Case:
    query_id: str
    question: str
    component_id: str
    topk10: tuple[EvidenceCandidate, ...]
    selected: tuple[EvidenceCandidate, ...]
    guidance: SelectionGuidance
    g0: Mapping[str, Any]


def _jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        yield value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_trace_runtime_fields(path: Path) -> dict[str, Mapping[str, Any]]:
    """Read an explicit runtime allowlist; gold-derived trace fields are ignored."""

    output: dict[str, Mapping[str, Any]] = {}
    for value in _jsonl(path):
        if value.get("dataset_kind") != "niah" or not value.get("dropped_evidence_ids"):
            continue
        query_id = str(value.get("query_id", "")).strip()
        candidates = value.get("candidates")
        selected = value.get("selected_evidence_ids")
        dropped = value.get("dropped_evidence_ids")
        if not query_id or not isinstance(candidates, list):
            raise ValueError("invalid Selector runtime trace row")
        if not isinstance(selected, list) or not isinstance(dropped, list):
            raise ValueError("invalid selected/dropped IDs in Selector trace")
        output[query_id] = {
            "query_id": query_id,
            "candidates": candidates,
            "selected_evidence_ids": selected,
            "dropped_evidence_ids": dropped,
        }
    return output


def load_cases(
    f001_generations: Path,
    selector_trace: Path,
    *,
    require_expected_count: bool = True,
) -> tuple[F004Case, ...]:
    traces = _load_trace_runtime_fields(selector_trace)
    cases: list[F004Case] = []
    for row in _jsonl(f001_generations):
        if not bool(row.get("selector_changed")):
            continue
        query_id = str(row["query_id"])
        trace = traces.get(query_id)
        if trace is None:
            raise ValueError(f"missing changed Selector trace for {query_id}")
        topk = tuple(EvidenceCandidate.model_validate(item) for item in row["topk10"])
        by_id = {item.evidence_id: item for item in topk}
        selected_ids = tuple(str(item) for item in row["selected_evidence_ids"])
        if selected_ids != tuple(str(item) for item in trace["selected_evidence_ids"]):
            raise ValueError(f"F001/Selector selected IDs differ for {query_id}")
        selected = tuple(by_id[evidence_id] for evidence_id in selected_ids)
        trace_candidates = trace["candidates"]
        if not isinstance(trace_candidates, list):
            raise ValueError(f"invalid Selector candidates for {query_id}")
        score_by_id: dict[str, Mapping[str, Any]] = {}
        for item in trace_candidates:
            if not isinstance(item, Mapping):
                raise ValueError(f"invalid Selector candidate for {query_id}")
            score_by_id[str(item["evidence_id"])] = item
        retained: list[RetainedEvidenceGuidance] = []
        for evidence in selected:
            item = score_by_id.get(evidence.evidence_id)
            if item is None or item.get("action") != "KEEP":
                raise ValueError(f"retained evidence has no KEEP trace for {query_id}")
            retained.append(
                RetainedEvidenceGuidance(
                    evidence_id=evidence.evidence_id,
                    retrieval_rank=evidence.retrieval_rank,
                    protect_signal=float(item["protect_score"]),
                    harm_signal=float(item["harm_score"]),
                    action="KEEP",
                    reason="BELOW_SAFE_THRESHOLD",
                )
            )
        dropped_ids = tuple(str(item) for item in trace["dropped_evidence_ids"])
        arms = row.get("arms")
        if not isinstance(arms, Mapping) or not isinstance(arms.get("D_selector_verify"), Mapping):
            raise ValueError(f"F001 omits D arm for {query_id}")
        cases.append(
            F004Case(
                query_id=query_id,
                question=str(row["question"]),
                component_id=str(row["component_id"]),
                topk10=topk,
                selected=selected,
                guidance=SelectionGuidance(
                    query_id=query_id,
                    selector_changed=True,
                    dropped_count=len(dropped_ids),
                    retained=tuple(retained),
                ),
                g0=cast(Mapping[str, Any], arms["D_selector_verify"]),
            )
        )
    if require_expected_count and len(cases) != EXPECTED_CASES:
        raise ValueError(
            f"F004 changed-case count differs: observed={len(cases)}, expected={EXPECTED_CASES}"
        )
    return tuple(cases)


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
    draft_generator: KeyFactDraftAnswerGenerator,
    query: Query,
    checklist: QueryChecklist,
    selected: SelectedEvidenceSet,
    guidance: SelectionGuidance | None,
) -> dict[str, object]:
    started = time.perf_counter()
    try:
        generation = generator.generate_with_guidance(
            query, checklist, selected, guidance
        )
    except Exception as error:  # noqa: BLE001 -- runtime failures are measured
        traceback.print_exc()
        return {
            "generation": GenerationResult(
                query_id=query.query_id, answer="", cited_evidence_ids=()
            ).model_dump(mode="json"),
            "routing": [],
            "notes": [],
            "error": f"{type(error).__name__}: {error}",
            "seconds": time.perf_counter() - started,
        }
    return {
        "generation": generation.model_dump(mode="json"),
        "routing": _routing_rows(generator),
        "notes": json.loads(notes_manifest(draft_generator.last_notes)),
        "error": None,
        "seconds": time.perf_counter() - started,
    }


def run_cases(
    cases: Sequence[F004Case],
    *,
    g1: VerifyAnnotateGenerator,
    g1_draft: KeyFactDraftAnswerGenerator,
    g2: VerifyAnnotateGenerator,
    g2_draft: KeyFactDraftAnswerGenerator,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        query = Query(query_id=case.query_id, text=case.question)
        checklist = QueryChecklist(query_id=case.query_id, focus=case.question, required_facts=())
        selected = SelectedEvidenceSet(query_id=case.query_id, evidence=case.selected)
        g0 = {**case.g0, "reused_from": "F001.D_selector_verify"}
        output.append(
            {
                "schema_version": "full-flow-f004-generation-row-v1",
                "query_id": case.query_id,
                "question": case.question,
                "component_id": case.component_id,
                "selector_changed": True,
                "question_slots": list(detect_question_slots(case.question)),
                "topk10": [item.model_dump(mode="json") for item in case.topk10],
                "selected_evidence_ids": [item.evidence_id for item in case.selected],
                "guidance": case.guidance.model_dump(mode="json"),
                "arms": {
                    ARMS[0]: g0,
                    ARMS[1]: _run_one(g1, g1_draft, query, checklist, selected, None),
                    ARMS[2]: _run_one(g2, g2_draft, query, checklist, selected, case.guidance),
                },
            }
        )
        if index % 10 == 0 or index == len(cases):
            print(
                f"[F004] {index}/{len(cases)}; {(time.perf_counter()-started)/index:.2f}s/case",
                flush=True,
            )
    return output


def _load_gold(path: Path, wanted: set[str]) -> dict[str, GoldCase]:
    output: dict[str, GoldCase] = {}
    for value in _jsonl(path):
        gold = GoldCase.model_validate(value)
        if gold.query_id in wanted:
            output[gold.query_id] = gold
    if set(output) != wanted:
        raise ValueError("gold file does not exactly cover F004 queries")
    return output


def score_rows(
    rows: Sequence[Mapping[str, Any]], gold: Mapping[str, GoldCase]
) -> dict[str, object]:
    metrics: dict[str, dict[str, dict[str, float]]] = {arm: {} for arm in ARMS}
    components: dict[str, str] = {}
    notes_nonempty = dict.fromkeys(ARMS, 0)
    errors = dict.fromkeys(ARMS, 0)
    for row in rows:
        query_id = str(row["query_id"])
        components[query_id] = str(row["component_id"])
        arms = row["arms"]
        if not isinstance(arms, Mapping):
            raise ValueError(f"invalid arms for {query_id}")
        for arm in ARMS:
            arm_row = arms[arm]
            if not isinstance(arm_row, Mapping):
                raise ValueError(f"invalid {arm} row for {query_id}")
            generation = GenerationResult.model_validate(arm_row["generation"])
            errors[arm] += int(bool(arm_row.get("error")))
            notes_nonempty[arm] += int(bool(arm_row.get("notes")))
            answer_score = answer_match(
                strip_annotations(generation.answer),
                gold[query_id].reference_answers,
            ).value
            if answer_score is None:
                raise ValueError(f"F004 gold has no scorable reference for {query_id}")
            metrics[arm][query_id] = {
                "answer_match": answer_score,
                "coverage": float(bool(generation.answer.strip())),
            }

    aggregates = {
        arm: {
            metric: sum(row[metric] for row in metrics[arm].values()) / len(rows)
            for metric in ("answer_match", "coverage")
        }
        for arm in ARMS
    }
    comparisons: dict[str, object] = {}
    for label, on_arm, off_arm in (
        ("G1_minus_G0", ARMS[1], ARMS[0]),
        ("G2_minus_G0", ARMS[2], ARMS[0]),
        ("G2_minus_G1", ARMS[2], ARMS[1]),
    ):
        comparisons[label] = {
            metric: asdict(
                compare_paired(
                    {query_id: values[metric] for query_id, values in metrics[on_arm].items()},
                    {query_id: values[metric] for query_id, values in metrics[off_arm].items()},
                    component_ids=components,
                )
            )
            for metric in ("answer_match", "coverage")
        }
        wrong_to_right = right_to_wrong = 0
        for query_id in components:
            before = metrics[off_arm][query_id]["answer_match"]
            after = metrics[on_arm][query_id]["answer_match"]
            wrong_to_right += int(before == 0.0 and after == 1.0)
            right_to_wrong += int(before == 1.0 and after == 0.0)
        cast(dict[str, object], comparisons[label])["transitions"] = {
            "wrong_to_right": wrong_to_right,
            "right_to_wrong": right_to_wrong,
            "net": wrong_to_right - right_to_wrong,
        }
    return {
        "schema_version": "full-flow-f004-report-v1",
        "status": "COMPLETE",
        "queries": len(rows),
        "g0_source": "frozen F001 D_selector_verify per-query output",
        "gold_loaded_at_runtime": False,
        "aggregate": aggregates,
        "comparisons": comparisons,
        "notes_nonempty_queries": notes_nonempty,
        "errors_by_arm": errors,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# F004：G0/G1/G2 关键事实笔记消融",
        "",
        "| 组 | Answer match | Coverage | 有笔记题数 | 错误 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        aggregate = report["aggregate"][arm]
        lines.append(
            f"| {arm} | {100*aggregate['answer_match']:.2f}% | "
            f"{100*aggregate['coverage']:.2f}% | {report['notes_nonempty_queries'][arm]} | "
            f"{report['errors_by_arm'][arm]} |"
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
    lines.extend(
        [
            "",
            "G0 复用 F001 的冻结 D 输出；G1/G2 在同一作业共享 Granite 和 TRUE。",
            "只有 G2 明确优于 G1，才支持 Selector 信号具有独立作用。",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--f001-generations", required=True, type=Path)
    run.add_argument("--selector-trace", required=True, type=Path)
    run.add_argument("--granite-snapshot", required=True, type=Path)
    run.add_argument("--true-model-id", required=True)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--limit", type=int)
    score = commands.add_parser("score")
    score.add_argument("--generations", required=True, type=Path)
    score.add_argument("--gold", required=True, type=Path)
    score.add_argument("--output-json", required=True, type=Path)
    score.add_argument("--output-report", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        cases = load_cases(args.f001_generations, args.selector_trace)
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError("--limit must be positive")
            cases = cases[: args.limit]
        output_dir = args.output_dir.resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError("output directory must be absent or empty")
        llm = GraniteLLMClient(
            model_id=str(args.granite_snapshot.resolve()),
            config=GraniteGenerationConfig(max_new_tokens=256, temperature=0.0, top_p=1.0),
        )
        nli = TrueNLIModel(model_id=args.true_model_id)
        g1_draft = KeyFactDraftAnswerGenerator(llm, guided=False)
        g2_draft = KeyFactDraftAnswerGenerator(llm, guided=True)
        g1 = VerifyAnnotateGenerator(
            draft_generator=g1_draft,
            nli=nli,
            entity_gate="observe",
            abstain_when_unverified=False,
        )
        g2 = VerifyAnnotateGenerator(
            draft_generator=g2_draft,
            nli=nli,
            entity_gate="observe",
            abstain_when_unverified=False,
        )
        rows = run_cases(cases, g1=g1, g1_draft=g1_draft, g2=g2, g2_draft=g2_draft)
        _write_jsonl(output_dir / "generations.jsonl", rows)
        errors: dict[str, int] = {}
        for arm in ARMS:
            count = 0
            for row in rows:
                arm_rows = cast(Mapping[str, Mapping[str, Any]], row["arms"])
                count += int(bool(arm_rows[arm].get("error")))
            errors[arm] = count
        manifest = {
            "schema_version": "full-flow-f004-run-manifest-v1",
            "status": "COMPLETE",
            "queries": len(rows),
            "gold_loaded_at_runtime": False,
            "g0_reused_from_f001": True,
            "g1_g2_same_process": True,
            "shared_granite_instance": True,
            "shared_true_instance": True,
            "errors_by_arm": errors,
            "generations_sha256": _sha256(output_dir / "generations.jsonl"),
        }
        _write_json(output_dir / "run_manifest.json", manifest)
        print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
        return int(any(errors.values()))
    score_input_rows: Sequence[Mapping[str, Any]] = tuple(_jsonl(args.generations))
    wanted = {str(row["query_id"]) for row in score_input_rows}
    report = score_rows(score_input_rows, _load_gold(args.gold, wanted))
    _write_json(args.output_json, report)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
