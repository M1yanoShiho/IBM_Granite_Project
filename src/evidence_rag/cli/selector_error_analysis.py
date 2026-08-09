"""Build the frozen, diagnostic-only Selector error-analysis sample."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify Selector outcomes without retuning")
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=20)
    return parser


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    records: list[Mapping[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        records.append(value)
    return records


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"missing or invalid {label}")
    return value


def _metric(arm: Mapping[str, object], key: str) -> object:
    return arm.get(key)


def _float(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"missing or invalid {label}")
    return float(value)


def classify_errors(
    comparisons: Sequence[Mapping[str, object]],
    events: Sequence[Mapping[str, object]],
    *,
    limit: int,
) -> dict[str, object]:
    """Classify frozen outcomes; categories may overlap by design."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    event_by_query: dict[str, Mapping[str, object]] = {}
    for row in events:
        query_id = str(row.get("query_id", ""))
        if not query_id or query_id in event_by_query:
            raise ValueError("event query IDs must be non-empty and unique")
        event_by_query[query_id] = _mapping(row.get("diagnostic"), "diagnostic")

    categories: dict[str, list[dict[str, object]]] = {
        "harmful_removed_required_retained": [],
        "harmful_not_removed": [],
        "required_dropped": [],
        "backend_or_parent_failure": [],
    }
    counts = {name: 0 for name in categories}
    seen_queries: set[str] = set()
    for row in comparisons:
        query_id = str(row.get("query_id", ""))
        if not query_id or query_id in seen_queries:
            raise ValueError("comparison query IDs must be non-empty and unique")
        seen_queries.add(query_id)
        top = _mapping(row.get("top_k"), "top_k")
        mis = _mapping(row.get("reliability_mis"), "reliability_mis")
        top_recall = _float(_metric(top, "required_evidence_recall"), "TopK recall")
        mis_recall = _float(_metric(mis, "required_evidence_recall"), "MIS recall")
        event = event_by_query.get(query_id, {})
        sample = {
            "query_id": query_id,
            "top_k_selected_ids": _metric(top, "selected_ids"),
            "mis_selected_ids": _metric(mis, "selected_ids"),
            "required_document_ids": _metric(mis, "required_document_ids"),
            "harmful_document_id": _metric(mis, "harmful_document_id"),
            "top_k_required_recall": top_recall,
            "mis_required_recall": mis_recall,
            "top_k_harmful_selected": _metric(top, "harmful_selected"),
            "mis_harmful_selected": _metric(mis, "harmful_selected"),
            "diagnostic": dict(event),
        }
        matched: list[str] = []
        if (
            _metric(mis, "harmful_pool_hit") is True
            and _metric(top, "harmful_selected") is True
            and _metric(mis, "harmful_selected") is False
            and mis_recall >= top_recall
        ):
            matched.append("harmful_removed_required_retained")
        if _metric(mis, "harmful_pool_hit") is True and _metric(mis, "harmful_selected") is True:
            matched.append("harmful_not_removed")
        if mis_recall < top_recall:
            matched.append("required_dropped")
        if (
            event.get("mode") == "topk_backend_fallback"
            or _float(event.get("unresolved_parent_count", 0), "unresolved parent count") > 0
        ):
            matched.append("backend_or_parent_failure")
        for name in matched:
            counts[name] += 1
            if len(categories[name]) < limit:
                categories[name].append(sample)

    return {
        "schema_version": "1.0",
        "comparison_query_count": len(comparisons),
        "sampling_rule": "first queries in frozen comparison order",
        "maximum_examples_per_category": limit,
        "category_counts": counts,
        "examples": categories,
    }


def _write_markdown(path: Path, report: Mapping[str, object]) -> None:
    counts = _mapping(report.get("category_counts"), "category_counts")
    examples = _mapping(report.get("examples"), "examples")
    labels = {
        "harmful_removed_required_retained": "Harmful removed and required evidence retained",
        "harmful_not_removed": "Harmful evidence not removed",
        "required_dropped": "Required evidence dropped",
        "backend_or_parent_failure": "Backend or parent-mapping failure",
    }
    lines = [
        "# Selector frozen error analysis",
        "",
        "This diagnostic sample is generated after the formal run and is not used to retune it.",
        "",
    ]
    for key, label in labels.items():
        raw_examples = examples.get(key)
        if not isinstance(raw_examples, Sequence) or isinstance(raw_examples, (str, bytes)):
            raise ValueError(f"invalid examples for {key}")
        query_ids = [str(_mapping(item, f"{key} example").get("query_id")) for item in raw_examples]
        lines.extend(
            (
                f"## {label}",
                "",
                f"Total cases: {counts[key]}; archived examples: {len(query_ids)}.",
                "",
                "Query IDs: " + (", ".join(query_ids) if query_ids else "None"),
                "",
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    comparisons = _read_jsonl(arguments.comparison)
    events = _read_jsonl(arguments.events)
    report = classify_errors(comparisons, events, limit=arguments.limit)
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "error_analysis.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(arguments.output / "ERROR_ANALYSIS.md", report)
    print(json.dumps(report["category_counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
