"""Independent sentence-level citation scoring for F001.

TRUE participates in the verify-and-annotate system, so it must not judge its
own citations.  This post-generation command uses MiniCheck and the ALCE metric.
Basic-generator citations are answer-level; every declared citation is therefore
assigned to every sentence (the deliberately generous baseline convention used
by the project's earlier G3 rescore).  Verify-arm sentence/citation pairs come
from the exact routing trace saved during F001.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from alce_metrics import CitationReport, ScoredExample, compute_citation_metrics

from evidence_rag.contracts.models import (
    EvidenceCandidate,
    GenerationResult,
    split_sentences,
    strip_annotations,
)
from evidence_rag.evaluation.paired_metric import compare_paired
from evidence_rag.generator.nli import MiniCheckNLIModel

ARMS = ("A_topk_basic", "B_selector_basic", "C_topk_verify", "D_selector_verify")
VERIFY_ARMS = frozenset(("C_topk_verify", "D_selector_verify"))


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


def _basic_example(
    query_id: str,
    generation: GenerationResult,
    docs: Mapping[str, str],
) -> ScoredExample | None:
    sentences = tuple(split_sentences(strip_annotations(generation.answer)))
    if not sentences:
        return None
    generous_refs = tuple(
        evidence_id for evidence_id in generation.cited_evidence_ids if evidence_id in docs
    )
    return ScoredExample(
        example_id=query_id,
        sentences=sentences,
        citations=tuple(generous_refs for _ in sentences),
        docs=docs,
    )


def _verify_example(
    query_id: str,
    routing: object,
    docs: Mapping[str, str],
) -> ScoredExample | None:
    if not isinstance(routing, list):
        raise ValueError(f"query {query_id} has invalid routing trace")
    sentences: list[str] = []
    citations: list[tuple[str, ...]] = []
    for item in routing:
        if not isinstance(item, Mapping):
            raise ValueError(f"query {query_id} has invalid routing item")
        sentence = strip_annotations(str(item.get("sentence", ""))).strip()
        if not sentence:
            continue
        citation = item.get("citation")
        refs = (str(citation),) if citation is not None and str(citation) in docs else ()
        sentences.append(sentence)
        citations.append(refs)
    if not sentences:
        return None
    return ScoredExample(query_id, tuple(sentences), tuple(citations), docs)


def build_examples(
    rows: Sequence[Mapping[str, Any]], arm: str
) -> tuple[list[ScoredExample], dict[str, str], dict[str, bool]]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    examples: list[ScoredExample] = []
    component_ids: dict[str, str] = {}
    changed: dict[str, bool] = {}
    for row in rows:
        query_id = str(row["query_id"])
        component_ids[query_id] = str(row["component_id"])
        changed[query_id] = bool(row["selector_changed"])
        candidates = tuple(EvidenceCandidate.model_validate(item) for item in row["topk10"])
        docs = {item.evidence_id: item.text for item in candidates}
        arms = row["arms"]
        if not isinstance(arms, Mapping) or not isinstance(arms.get(arm), Mapping):
            raise ValueError(f"query {query_id} has invalid arm: {arm}")
        arm_row = arms[arm]
        assert isinstance(arm_row, Mapping)
        generation = GenerationResult.model_validate(arm_row["generation"])
        example = (
            _verify_example(query_id, arm_row.get("routing"), docs)
            if arm in VERIFY_ARMS
            else _basic_example(query_id, generation, docs)
        )
        if example is not None:
            examples.append(example)
    return examples, component_ids, changed


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def score_rows(
    rows: Sequence[Mapping[str, Any]],
    entails: Callable[[str, str], bool],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    per_arm: dict[str, CitationReport] = {}
    per_case: dict[str, dict[str, dict[str, float | None]]] = {}
    component_ids: dict[str, str] = {}
    changed: dict[str, bool] = {}
    for arm in ARMS:
        examples, components, selector_changed = build_examples(rows, arm)
        component_ids.update(components)
        changed.update(selector_changed)
        report = compute_citation_metrics(examples, entails)
        per_arm[arm] = report
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

    def scope(wanted: set[str]) -> dict[str, object]:
        aggregates: dict[str, object] = {}
        for arm in ARMS:
            precision = [
                value
                for query_id in wanted
                if (value := per_case[arm][query_id]["citation_precision"]) is not None
            ]
            recall = [
                value
                for query_id in wanted
                if (value := per_case[arm][query_id]["citation_recall"]) is not None
            ]
            cited_precision = [
                item["citation_prec"]
                for item in per_arm[arm].per_example
                if str(item["example_id"]) in wanted and int(item["citations"]) > 0
            ]
            aggregates[arm] = {
                "answered_examples": len(precision),
                "citation_precision_alce": _mean(precision),
                "citation_precision_cited_examples": _mean(cited_precision),
                "citation_recall_alce": _mean(recall),
            }

        comparisons: dict[str, dict[str, object]] = {}
        components = {query_id: component_ids[query_id] for query_id in wanted}
        for label, on_arm, off_arm in (
            ("B_minus_A", ARMS[1], ARMS[0]),
            ("D_minus_C", ARMS[3], ARMS[2]),
        ):
            comparisons[label] = {}
            for metric in ("citation_precision", "citation_recall"):
                comparisons[label][metric] = asdict(
                    compare_paired(
                        {query_id: per_case[on_arm][query_id][metric] for query_id in wanted},
                        {query_id: per_case[off_arm][query_id][metric] for query_id in wanted},
                        component_ids=components,
                    )
                )
        return {
            "queries": len(wanted),
            "aggregate": aggregates,
            "selector_comparisons": comparisons,
        }

    all_ids = set(component_ids)
    changed_ids = {query_id for query_id, value in changed.items() if value}
    output_rows = [
        {
            "query_id": query_id,
            "component_id": component_ids[query_id],
            "selector_changed": changed[query_id],
            "arms": {arm: per_case[arm][query_id] for arm in ARMS},
        }
        for query_id in sorted(all_ids)
    ]
    summary: dict[str, object] = {
        "schema_version": "full-flow-f001-citation-report-v1",
        "status": "COMPLETE",
        "judge": "MiniCheck-Flan-T5-Large",
        "production_verifier": "TRUE (excluded from judging)",
        "mapping": {
            "basic_arms": "all answer-level citations assigned to every sentence (generous)",
            "verify_arms": "exact sentence-citation pairs from F001 routing trace",
        },
        "all_decision_dev": scope(all_ids),
        "selector_changed": scope(changed_ids),
    }
    return summary, output_rows


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# F001 独立逐句引用评分",
        "",
        "**裁判：** MiniCheck；系统内部 TRUE 不参与评分。",
        "",
        "| 范围 | 组别 | Answered | Citation precision | Precision (实际有引用) | Citation recall |",
        "|---|---|---:|---:|---:|---:|",
    ]

    def value(item: float | None) -> str:
        return "—" if item is None else f"{100 * item:.2f}%"

    for scope_key, scope_label in (
        ("all_decision_dev", "全部"),
        ("selector_changed", "Selector 改动题"),
    ):
        scope = summary[scope_key]
        for arm in ARMS:
            metric = scope["aggregate"][arm]
            lines.append(
                f"| {scope_label} ({scope['queries']}) | {arm} | {metric['answered_examples']} | "
                f"{value(metric['citation_precision_alce'])} | "
                f"{value(metric['citation_precision_cited_examples'])} | "
                f"{value(metric['citation_recall_alce'])} |"
            )
    lines.extend(
        [
            "",
            "基础组只有答案级引用，因此把全部引用宽松地分给每句话；高级组使用运行时保存的精确句子—引用对应。",
            "ALCE precision 将“回答了但没有任何引用”的样本记为 0；实际有引用样本的 precision 同时单独报告。",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-rows", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows = list(_jsonl(args.generations))
    judge = MiniCheckNLIModel(model_id=args.model_id)

    def entails(premise: str, hypothesis: str) -> bool:
        return judge.classify(premise=premise, hypothesis=hypothesis) == "entailment"

    summary, per_case = score_rows(rows, entails)
    _write_json(args.output_json, summary)
    _write_jsonl(args.output_rows, per_case)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
