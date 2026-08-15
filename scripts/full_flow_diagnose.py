"""F002 post-hoc diagnosis for Selector-changed F001 cases.

This command reads frozen generations and evaluation labels only after F001 has
finished.  It never participates in generation and therefore cannot leak gold
answers into the system.  The labels below are conservative diagnosis signals,
not new training targets or claims that one heuristic proves a causal mechanism.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from evidence_rag.contracts.models import EvidenceCandidate, GenerationResult, strip_annotations
from evidence_rag.evaluation.scoring import answer_match
from evidence_rag.infrastructure.datasets import GoldCase

COMPARISONS = {
    "basic": ("A_topk_basic", "B_selector_basic"),
    "verify": ("C_topk_verify", "D_selector_verify"),
}


def _jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        yield value


def _load_gold(path: Path) -> dict[str, GoldCase]:
    output: dict[str, GoldCase] = {}
    for value in _jsonl(path):
        gold = GoldCase.model_validate(value)
        if gold.query_id in output:
            raise ValueError(f"duplicate gold query: {gold.query_id}")
        output[gold.query_id] = gold
    return output


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _reference_shape(references: tuple[str, ...] | None) -> str:
    """A small, readable clue about what detail the Generator may have missed."""

    text = " ".join(references or ())
    if re.search(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", text):
        return "date_or_year"
    if re.search(r"\d", text):
        return "number"
    if any(token in text.casefold() for token in ("january", "february", "march", "april", "may ", "june", "july", "august", "september", "october", "november", "december")):
        return "date_or_year"
    if len(text.split()) >= 2:
        return "named_entity_or_phrase"
    return "short_entity"


def _generation(arm_rows: Mapping[str, Any], arm: str) -> tuple[GenerationResult, str | None]:
    value = arm_rows[arm]
    if not isinstance(value, Mapping):
        raise ValueError(f"invalid arm row: {arm}")
    return GenerationResult.model_validate(value["generation"]), value.get("error")


def diagnose_rows(
    rows: Sequence[Mapping[str, Any]], gold_by_id: Mapping[str, GoldCase]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    diagnoses: list[dict[str, object]] = []
    category_counts: dict[str, Counter[str]] = {
        name: Counter() for name in COMPARISONS
    }
    transition_counts: dict[str, Counter[str]] = {
        name: Counter() for name in COMPARISONS
    }

    for row in rows:
        if not bool(row.get("selector_changed")):
            continue
        query_id = str(row["query_id"])
        gold = gold_by_id.get(query_id)
        if gold is None:
            raise ValueError(f"missing gold query: {query_id}")
        references = gold.reference_answers
        relevant_documents = set(gold.relevant_document_ids or ())
        topk = tuple(EvidenceCandidate.model_validate(item) for item in row["topk10"])
        by_id = {item.evidence_id: item for item in topk}
        topk_documents = {item.document_id for item in topk}
        selected_ids = tuple(str(value) for value in row["selected_evidence_ids"])
        selected_documents = {by_id[value].document_id for value in selected_ids}
        removed_relevant = sorted((topk_documents - selected_documents) & relevant_documents)
        all_relevant_remain = bool(relevant_documents) and relevant_documents <= selected_documents
        arms = row["arms"]
        if not isinstance(arms, Mapping):
            raise ValueError(f"query {query_id} has invalid arms")

        per_comparison: dict[str, object] = {}
        for name, (before_arm, after_arm) in COMPARISONS.items():
            before, before_error = _generation(arms, before_arm)
            after, after_error = _generation(arms, after_arm)
            before_match = answer_match(strip_annotations(before.answer), references).value
            after_match = answer_match(strip_annotations(after.answer), references).value
            if before_match == 0.0 and after_match == 1.0:
                transition = "wrong_to_right"
                category = "selector_helped_generation"
            elif before_match == 1.0 and after_match == 0.0:
                transition = "right_to_wrong"
                category = (
                    "selector_removed_relevant_evidence"
                    if removed_relevant
                    else "generator_context_sensitivity"
                )
            elif before_match == 1.0 and after_match == 1.0:
                transition = "right_stayed_right"
                category = "correct_preserved"
            else:
                transition = "wrong_stayed_wrong"
                if after_error:
                    category = "runtime_or_parser_failure"
                elif not after.answer.strip():
                    category = "empty_output"
                elif removed_relevant:
                    category = "selector_removed_relevant_evidence"
                elif all_relevant_remain and len(relevant_documents) > 1:
                    category = "multi_document_composition_candidate"
                elif all_relevant_remain:
                    category = "answer_extraction_failure_with_evidence_present"
                elif not (selected_documents & relevant_documents):
                    category = "no_labelled_relevant_evidence_available"
                else:
                    category = "partial_evidence_or_other"
            transition_counts[name][transition] += 1
            category_counts[name][category] += 1
            per_comparison[name] = {
                "before_arm": before_arm,
                "after_arm": after_arm,
                "before_match": before_match,
                "after_match": after_match,
                "transition": transition,
                "primary_diagnosis": category,
                "before_error": before_error,
                "after_error": after_error,
                "after_answer": after.answer,
                "after_cited_evidence_ids": list(after.cited_evidence_ids),
            }

        diagnoses.append(
            {
                "schema_version": "full-flow-f002-diagnosis-row-v1",
                "query_id": query_id,
                "question": row["question"],
                "component_id": row["component_id"],
                "reference_answers": list(references or ()),
                "reference_shape": _reference_shape(references),
                "relevant_document_ids": sorted(relevant_documents),
                "selected_evidence_ids": list(selected_ids),
                "removed_relevant_document_ids": removed_relevant,
                "all_relevant_documents_remain": all_relevant_remain,
                "comparisons": per_comparison,
            }
        )

    summary: dict[str, object] = {
        "schema_version": "full-flow-f002-summary-v1",
        "status": "COMPLETE",
        "scope": "F001 selector-changed cases only; post-generation gold diagnosis",
        "queries": len(diagnoses),
        "comparisons": {
            name: {
                "transitions": dict(sorted(transition_counts[name].items())),
                "primary_diagnoses": dict(sorted(category_counts[name].items())),
            }
            for name in COMPARISONS
        },
        "interpretation_boundary": (
            "categories are conservative triage signals; answer extraction and multi-document "
            "labels identify candidates for inspection rather than proving a causal mechanism"
        ),
    }
    return diagnoses, summary


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# F002：Selector 改动题的结果归因",
        "",
        f"**状态：** `{summary['status']}`",
        "",
        f"范围：F001 中 Selector 真正改变证据的 {summary['queries']} 题。",
        "Gold 只在生成完成后用于体检，没有传给 Generator。",
    ]
    for name, label in (("basic", "基础 Generator：B 对 A"), ("verify", "高级 Generator：D 对 C")):
        result = summary["comparisons"][name]
        lines.extend(["", f"## {label}", "", "### 回答变化", "", "| 类型 | 数量 |", "|---|---:|"])
        for key, count in result["transitions"].items():
            lines.append(f"| {key} | {count} |")
        lines.extend(["", "### 首要体检信号", "", "| 类型 | 数量 |", "|---|---:|"])
        for key, count in result["primary_diagnoses"].items():
            lines.append(f"| {key} | {count} |")
    lines.extend(
        [
            "",
            "这些分类用于决定下一步查哪一类案例；其中 `candidate`/`failure_with_evidence_present`",
            "表示应人工查看的候选原因，不等于已经证明了因果。",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--output-rows", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--expected-changed", type=int, default=109)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows = list(_jsonl(args.generations))
    diagnoses, summary = diagnose_rows(rows, _load_gold(args.gold))
    if len(diagnoses) != args.expected_changed:
        raise ValueError(
            f"changed-case count differs: observed={len(diagnoses)}, "
            f"expected={args.expected_changed}"
        )
    _write_jsonl(args.output_rows, diagnoses)
    _write_json(args.output_json, summary)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
