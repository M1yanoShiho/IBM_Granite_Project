"""Independent MiniCheck citation scoring for the three F005 confirmation arms."""

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
from full_flow_f005 import ARMS

from evidence_rag.contracts.models import EvidenceCandidate, GenerationResult
from evidence_rag.generator.nli import MiniCheckNLIModel


def score_rows(
    rows: Sequence[Mapping[str, Any]],
    entails: Callable[[str, str], bool],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    component_ids = {str(row["query_id"]): str(row["query_id"]) for row in rows}
    per_arm: dict[str, CitationReport] = {}
    per_case: dict[str, dict[str, dict[str, float | None]]] = {}
    answered: dict[str, dict[str, bool]] = {}
    for arm in ARMS:
        examples = []
        arm_answered: dict[str, bool] = {}
        for row in rows:
            query_id = str(row["query_id"])
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
            item["citation_precision"]
            for item in per_case[arm].values()
            if item["citation_precision"] is not None
        ]
        recall = [
            item["citation_recall"]
            for item in per_case[arm].values()
            if item["citation_recall"] is not None
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
        ("B_minus_A_generator_only", ARMS[1], ARMS[0]),
        ("C_minus_B_selector_only", ARMS[2], ARMS[1]),
        ("C_minus_A_full_system", ARMS[2], ARMS[0]),
    ):
        comparisons[label] = {
            metric: _paired_common_answered(
                {query_id: per_case[on_arm][query_id][metric] for query_id in component_ids},
                {query_id: per_case[off_arm][query_id][metric] for query_id in component_ids},
                component_ids,
            )
            for metric in ("citation_precision", "citation_recall")
        }
    output_rows: list[dict[str, object]] = [
        {
            "query_id": query_id,
            "arms": {arm: per_case[arm][query_id] for arm in ARMS},
        }
        for query_id in sorted(component_ids)
    ]
    return (
        {
            "schema_version": "full-flow-f005-citation-report-v1",
            "status": "COMPLETE",
            "queries": len(component_ids),
            "judge": "MiniCheck-Flan-T5-Large",
            "production_verifier": "TRUE (excluded from judging)",
            "mapping": "exact sentence-citation pairs from each routing trace",
            "aggregate": aggregate,
            "comparisons": comparisons,
        },
        output_rows,
    )


def _value(item: float | None) -> str:
    return "—" if item is None else f"{100 * item:.2f}%"


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# F005 独立逐句引用评分",
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
            f"{_value(metric['citation_precision_alce'])} | "
            f"{_value(metric['citation_precision_cited_examples'])} | "
            f"{_value(metric['citation_recall_alce'])} |"
        )
    lines.append("")
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
    cache: dict[tuple[str, str], bool] = {}

    def entails(premise: str, hypothesis: str) -> bool:
        key = (premise, hypothesis)
        if key not in cache:
            cache[key] = judge.classify(premise=premise, hypothesis=hypothesis) == "entailment"
        return cache[key]

    summary, per_case = score_rows(rows, entails)
    summary["judge_device"] = args.device
    summary["unique_judge_calls"] = len(cache)
    _write_json(args.output_json, summary)
    _write_jsonl(args.output_rows, per_case)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
