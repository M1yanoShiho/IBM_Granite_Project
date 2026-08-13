"""Independent MiniCheck citation scoring for the three F004 verify arms.

TRUE is part of the production Generator and is therefore excluded from judging
its own citations.  All three arms use the exact sentence/citation routing trace
saved during generation.  Empty answers remain visible as coverage and receive no
invented citation score; paired quality comparisons use only queries answered by
both arms.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from alce_metrics import CitationReport, compute_citation_metrics
from full_flow_citation_score import (
    _jsonl,
    _mean,
    _paired_common_answered,
    _verify_example,
    _write_json,
    _write_jsonl,
)
from full_flow_f004 import ARMS

from evidence_rag.contracts.models import EvidenceCandidate, GenerationResult
from evidence_rag.generator.nli import MiniCheckNLIModel


def score_rows(
    rows: Sequence[Mapping[str, Any]],
    entails: Callable[[str, str], bool],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    component_ids: dict[str, str] = {}
    per_arm: dict[str, CitationReport] = {}
    per_case: dict[str, dict[str, dict[str, float | None]]] = {}
    answered: dict[str, dict[str, bool]] = {}

    for arm in ARMS:
        examples = []
        arm_answered: dict[str, bool] = {}
        for row in rows:
            query_id = str(row["query_id"])
            component_ids[query_id] = str(row["component_id"])
            candidates = tuple(
                EvidenceCandidate.model_validate(item) for item in row["topk10"]
            )
            docs = {item.evidence_id: item.text for item in candidates}
            arms = row.get("arms")
            if not isinstance(arms, Mapping) or not isinstance(arms.get(arm), Mapping):
                raise ValueError(f"query {query_id} has invalid arm: {arm}")
            arm_row = arms[arm]
            assert isinstance(arm_row, Mapping)
            generation = GenerationResult.model_validate(arm_row["generation"])
            arm_answered[query_id] = bool(generation.answer.strip())
            example = _verify_example(query_id, arm_row.get("routing"), docs)
            if example is not None:
                examples.append(example)
        report = compute_citation_metrics(examples, entails)
        per_arm[arm] = report
        answered[arm] = arm_answered
        by_id = {str(item["example_id"]): item for item in report.per_example}
        per_case[arm] = {
            query_id: {
                "citation_precision": (
                    float(item["citation_prec"]) if (item := by_id.get(query_id)) else None
                ),
                "citation_recall": (
                    float(item["citation_rec"]) if (item := by_id.get(query_id)) else None
                ),
            }
            for query_id in component_ids
        }

    aggregate: dict[str, object] = {}
    for arm in ARMS:
        precision = [
            value["citation_precision"]
            for value in per_case[arm].values()
            if value["citation_precision"] is not None
        ]
        recall = [
            value["citation_recall"]
            for value in per_case[arm].values()
            if value["citation_recall"] is not None
        ]
        cited_precision = [
            float(item["citation_prec"])
            for item in per_arm[arm].per_example
            if int(item["citations"]) > 0
        ]
        aggregate[arm] = {
            "answered_examples": sum(answered[arm].values()),
            "citation_precision_alce": _mean(precision),
            "citation_precision_cited_examples": _mean(cited_precision),
            "citation_recall_alce": _mean(recall),
        }

    comparisons: dict[str, object] = {}
    for label, on_arm, off_arm in (
        ("G1_minus_G0", ARMS[1], ARMS[0]),
        ("G2_minus_G0", ARMS[2], ARMS[0]),
        ("G2_minus_G1", ARMS[2], ARMS[1]),
    ):
        comparisons[label] = {
            metric: _paired_common_answered(
                {
                    query_id: per_case[on_arm][query_id][metric]
                    for query_id in component_ids
                },
                {
                    query_id: per_case[off_arm][query_id][metric]
                    for query_id in component_ids
                },
                component_ids,
            )
            for metric in ("citation_precision", "citation_recall")
        }

    output_rows: list[dict[str, object]] = [
        {
            "query_id": query_id,
            "component_id": component_ids[query_id],
            "arms": {arm: per_case[arm][query_id] for arm in ARMS},
        }
        for query_id in sorted(component_ids)
    ]
    summary: dict[str, object] = {
        "schema_version": "full-flow-f004-citation-report-v1",
        "status": "COMPLETE",
        "queries": len(component_ids),
        "judge": "MiniCheck-Flan-T5-Large",
        "production_verifier": "TRUE (excluded from judging)",
        "mapping": "exact sentence-citation pairs from each F004 routing trace",
        "aggregate": aggregate,
        "comparisons": comparisons,
    }
    return summary, output_rows


def _markdown(summary: Mapping[str, Any]) -> str:
    def value(item: float | None) -> str:
        return "—" if item is None else f"{100 * item:.2f}%"

    lines = [
        "# F004 独立逐句引用评分",
        "",
        "**裁判：** MiniCheck；系统内部 TRUE 不参与评分。",
        "",
        "| 组别 | Answered | Citation precision | Precision（实际有引用） | Citation recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        metric = summary["aggregate"][arm]
        lines.append(
            f"| {arm} | {metric['answered_examples']} | "
            f"{value(metric['citation_precision_alce'])} | "
            f"{value(metric['citation_precision_cited_examples'])} | "
            f"{value(metric['citation_recall_alce'])} |"
        )
    lines.extend(
        [
            "",
            "三组都使用运行时保存的精确句子—引用对应。",
            "ALCE precision 将回答中没有引用的句子计入；有引用样本 precision 另行报告。",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-rows", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows: Sequence[Mapping[str, Any]] = tuple(_jsonl(args.generations))
    judge = MiniCheckNLIModel(model_id=args.model_id)
    if args.device != "cpu":
        _tokenizer, model = judge._ensure_loaded()
        judge._model = model.to(args.device)
    entailment_cache: dict[tuple[str, str], bool] = {}

    def entails(premise: str, hypothesis: str) -> bool:
        key = (premise, hypothesis)
        if key not in entailment_cache:
            entailment_cache[key] = (
                judge.classify(premise=premise, hypothesis=hypothesis) == "entailment"
            )
        return entailment_cache[key]

    summary, per_case = score_rows(rows, entails)
    summary["judge_device"] = args.device
    summary["unique_judge_calls"] = len(entailment_cache)
    _write_json(args.output_json, summary)
    _write_jsonl(args.output_rows, per_case)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
