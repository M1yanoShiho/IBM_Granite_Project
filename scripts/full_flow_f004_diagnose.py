"""Post-generation diagnosis for F004 using the already-built F002 labels.

This command is intentionally separate from generation: gold answers and F002
diagnosis labels are loaded only after G0/G1/G2 outputs are complete.  It reports
whether notes recover empty/detail failures, whether they damage G0-correct cases,
and whether the four cases previously helped by Selector remain correct.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from full_flow_f004 import ARMS

from evidence_rag.contracts.models import GenerationResult, strip_annotations
from evidence_rag.evaluation.scoring import answer_match
from evidence_rag.infrastructure.datasets import GoldCase


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


def _match(generation: GenerationResult, gold: GoldCase) -> bool:
    score = answer_match(strip_annotations(generation.answer), gold.reference_answers).value
    if score is None:
        raise ValueError(f"no scorable gold reference for {gold.query_id}")
    return bool(score)


def diagnose_rows(
    generations: Sequence[Mapping[str, Any]],
    f002_by_id: Mapping[str, Mapping[str, Any]],
    gold_by_id: Mapping[str, GoldCase],
) -> tuple[dict[str, object], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    for row in generations:
        query_id = str(row["query_id"])
        if query_id not in f002_by_id or query_id not in gold_by_id:
            raise ValueError(f"missing F002/gold row for {query_id}")
        f002 = f002_by_id[query_id]
        verify = f002["comparisons"]["verify"]
        arms = row["arms"]
        if not isinstance(verify, Mapping) or not isinstance(arms, Mapping):
            raise ValueError(f"invalid F002/F004 row for {query_id}")
        matches: dict[str, bool] = {}
        answered: dict[str, bool] = {}
        answers: dict[str, str] = {}
        note_counts: dict[str, int] = {}
        for arm in ARMS:
            arm_row = arms[arm]
            if not isinstance(arm_row, Mapping):
                raise ValueError(f"invalid {arm} row for {query_id}")
            generation = GenerationResult.model_validate(arm_row["generation"])
            answers[arm] = generation.answer
            answered[arm] = bool(generation.answer.strip())
            matches[arm] = _match(generation, gold_by_id[query_id])
            notes = arm_row.get("notes", [])
            note_counts[arm] = len(notes) if isinstance(notes, list) else 0
        details.append(
            {
                "schema_version": "full-flow-f004-diagnosis-row-v1",
                "query_id": query_id,
                "question": str(row["question"]),
                "component_id": str(row["component_id"]),
                "reference_shape": str(f002["reference_shape"]),
                "g0_diagnosis": str(verify["primary_diagnosis"]),
                "g0_selector_transition": str(verify["transition"]),
                "question_slots": list(row.get("question_slots", [])),
                "matches": matches,
                "answered": answered,
                "note_counts": note_counts,
                "transitions": {
                    "G1_vs_G0": f"{int(matches[ARMS[0]])}->{int(matches[ARMS[1]])}",
                    "G2_vs_G0": f"{int(matches[ARMS[0]])}->{int(matches[ARMS[2]])}",
                    "G2_vs_G1": f"{int(matches[ARMS[1]])}->{int(matches[ARMS[2]])}",
                },
                "answers": answers,
            }
        )

    def grouped(key: str) -> dict[str, object]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in details:
            groups[str(row[key])].append(row)
        output: dict[str, object] = {}
        for label, group in sorted(groups.items()):
            arm_metrics: dict[str, object] = {}
            for arm in ARMS:
                arm_metrics[arm] = {
                    "answer_match": sum(bool(item["matches"][arm]) for item in group)
                    / len(group),
                    "coverage": sum(bool(item["answered"][arm]) for item in group) / len(group),
                }
            output[label] = {"queries": len(group), "arms": arm_metrics}
        return output

    transition_counts = {
        comparison: dict(
            Counter(str(row["transitions"][comparison]) for row in details)
        )
        for comparison in ("G1_vs_G0", "G2_vs_G0", "G2_vs_G1")
    }
    g0_empty = [row for row in details if not bool(row["answered"][ARMS[0]])]
    g0_correct = [row for row in details if bool(row["matches"][ARMS[0]])]
    helped = [row for row in details if row["g0_diagnosis"] == "selector_helped_generation"]
    summary: dict[str, object] = {
        "schema_version": "full-flow-f004-diagnosis-summary-v1",
        "status": "COMPLETE",
        "scope": "F004 generation complete; gold and F002 labels loaded only post-generation",
        "queries": len(details),
        "transition_counts": transition_counts,
        "g0_empty_cases": {
            "queries": len(g0_empty),
            **{
                arm: {
                    "became_nonempty": sum(
                        bool(row["answered"][arm]) for row in g0_empty
                    ),
                    "became_correct": sum(
                        bool(row["matches"][arm]) for row in g0_empty
                    ),
                }
                for arm in ARMS[1:]
            },
        },
        "g0_correct_cases": {
            "queries": len(g0_correct),
            **{
                arm: {
                    "preserved_correct": sum(
                        bool(row["matches"][arm]) for row in g0_correct
                    ),
                    "harmed": sum(
                        not bool(row["matches"][arm]) for row in g0_correct
                    ),
                }
                for arm in ARMS[1:]
            },
        },
        "previous_selector_help_cases": {
            "queries": len(helped),
            ARMS[0]: {"correct": sum(bool(row["matches"][ARMS[0]]) for row in helped)},
            ARMS[1]: {"correct": sum(bool(row["matches"][ARMS[1]]) for row in helped)},
            ARMS[2]: {"correct": sum(bool(row["matches"][ARMS[2]]) for row in helped)},
        },
        "by_g0_diagnosis": grouped("g0_diagnosis"),
        "by_reference_shape": grouped("reference_shape"),
    }
    return summary, details


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# F004 失败归因检查",
        "",
        "| 核心检查 | G1 notes | G2 guided notes |",
        "|---|---:|---:|",
    ]
    empty = summary["g0_empty_cases"]
    correct = summary["g0_correct_cases"]
    helped = summary["previous_selector_help_cases"]
    lines.extend(
        [
            f"| G0 空答案中变为非空 | {empty[ARMS[1]]['became_nonempty']}/{empty['queries']} | {empty[ARMS[2]]['became_nonempty']}/{empty['queries']} |",
            f"| G0 空答案中真正变为正确 | {empty[ARMS[1]]['became_correct']}/{empty['queries']} | {empty[ARMS[2]]['became_correct']}/{empty['queries']} |",
            f"| G0 正确题被改错 | {correct[ARMS[1]]['harmed']}/{correct['queries']} | {correct[ARMS[2]]['harmed']}/{correct['queries']} |",
            f"| 原 Selector 帮助题仍正确 | {helped[ARMS[1]]['correct']}/{helped['queries']} | {helped[ARMS[2]]['correct']}/{helped['queries']} |",
            "",
            "| 比较 | wrong→right | right→wrong | 净变化 |",
            "|---|---:|---:|---:|",
        ]
    )
    for comparison, counts in summary["transition_counts"].items():
        improved = int(counts.get("0->1", 0))
        harmed = int(counts.get("1->0", 0))
        lines.append(f"| {comparison} | {improved} | {harmed} | {improved-harmed:+d} |")
    lines.append("")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", required=True, type=Path)
    parser.add_argument("--f002-rows", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-rows", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    generations: Sequence[Mapping[str, Any]] = tuple(_jsonl(args.generations))
    wanted = {str(row["query_id"]) for row in generations}
    f002 = {
        str(row["query_id"]): row for row in _jsonl(args.f002_rows) if str(row["query_id"]) in wanted
    }
    gold = {
        case.query_id: case
        for row in _jsonl(args.gold)
        if (case := GoldCase.model_validate(row)).query_id in wanted
    }
    if set(f002) != wanted or set(gold) != wanted:
        raise ValueError("F002/gold inputs do not exactly cover F004 queries")
    summary, details = diagnose_rows(generations, f002, gold)
    _write_json(args.output_json, summary)
    _write_jsonl(args.output_rows, details)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
