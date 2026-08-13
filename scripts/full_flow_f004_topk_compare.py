"""Post-generation paired comparison of F004 G1/G2 against frozen F001 TopK C."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from full_flow_f004 import ARMS, _jsonl, _load_gold, _write_json

from evidence_rag.contracts.models import GenerationResult, strip_annotations
from evidence_rag.evaluation.paired_metric import compare_paired
from evidence_rag.evaluation.scoring import answer_match
from evidence_rag.infrastructure.datasets import GoldCase

TOPK_ARM = "C_topk_verify"


def _score(generation: object, gold: GoldCase) -> float:
    result = GenerationResult.model_validate(generation)
    value = answer_match(strip_annotations(result.answer), gold.reference_answers).value
    if value is None:
        raise ValueError(f"no scorable gold reference for {gold.query_id}")
    return value


def compare_to_topk(
    f004_rows: Sequence[Mapping[str, Any]],
    f001_by_id: Mapping[str, Mapping[str, Any]],
    gold: Mapping[str, GoldCase],
) -> dict[str, object]:
    components = {str(row["query_id"]): str(row["component_id"]) for row in f004_rows}
    output: dict[str, object] = {}
    for arm in ARMS[1:]:
        on: dict[str, float] = {}
        off: dict[str, float] = {}
        for row in f004_rows:
            query_id = str(row["query_id"])
            f004_arms = row["arms"]
            f001_arms = f001_by_id[query_id]["arms"]
            if not isinstance(f004_arms, Mapping) or not isinstance(f001_arms, Mapping):
                raise ValueError(f"invalid arm mapping for {query_id}")
            f004_arm = f004_arms[arm]
            f001_arm = f001_arms[TOPK_ARM]
            if not isinstance(f004_arm, Mapping) or not isinstance(f001_arm, Mapping):
                raise ValueError(f"invalid comparison arm for {query_id}")
            on[query_id] = _score(f004_arm["generation"], gold[query_id])
            off[query_id] = _score(f001_arm["generation"], gold[query_id])
        wrong_to_right = sum(off[q] == 0.0 and on[q] == 1.0 for q in components)
        right_to_wrong = sum(off[q] == 1.0 and on[q] == 0.0 for q in components)
        output[arm] = {
            "topk_mean": sum(off.values()) / len(off),
            "arm_mean": sum(on.values()) / len(on),
            "paired": asdict(compare_paired(on, off, component_ids=components)),
            "transitions": {
                "wrong_to_right": wrong_to_right,
                "right_to_wrong": right_to_wrong,
                "net": wrong_to_right - right_to_wrong,
            },
        }
    return {
        "schema_version": "full-flow-f004-topk-comparison-v1",
        "status": "COMPLETE",
        "scope": "same 109 F001 Selector-changed decision-dev queries",
        "topk_arm": TOPK_ARM,
        "gold_loaded_at_runtime": False,
        "comparisons": output,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# F004 与冻结 TopK 高级组比较",
        "",
        "| 新方法 | TopK C | 新方法得分 | 差值 | 95% CI | wrong→right | right→wrong |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, result in report["comparisons"].items():
        paired = result["paired"]
        transitions = result["transitions"]
        lines.append(
            f"| {arm} | {100*result['topk_mean']:.2f}% | {100*result['arm_mean']:.2f}% | "
            f"{100*paired['delta']:+.2f} pp | "
            f"[{100*paired['ci_low']:+.2f}, {100*paired['ci_high']:+.2f}] | "
            f"{transitions['wrong_to_right']} | {transitions['right_to_wrong']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f004-generations", required=True, type=Path)
    parser.add_argument("--f001-generations", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    f004_rows: Sequence[Mapping[str, Any]] = tuple(_jsonl(args.f004_generations))
    wanted = {str(row["query_id"]) for row in f004_rows}
    f001 = {
        str(row["query_id"]): row
        for row in _jsonl(args.f001_generations)
        if str(row["query_id"]) in wanted
    }
    if set(f001) != wanted:
        raise ValueError("F001 does not exactly cover F004 query IDs")
    report = compare_to_topk(f004_rows, f001, _load_gold(args.gold, wanted))
    _write_json(args.output_json, report)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
