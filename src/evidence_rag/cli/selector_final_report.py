"""Render the three frozen Selector result tables and the final module decision."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the final Selector-only report")
    parser.add_argument("--sealed-pool", required=True, type=Path)
    parser.add_argument("--sealed-summary", required=True, type=Path)
    parser.add_argument("--twowiki-pool", required=True, type=Path)
    parser.add_argument("--twowiki-summary", required=True, type=Path)
    parser.add_argument("--stability", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _read(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"missing or invalid {label}")
    return value


def _number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"missing or invalid numeric {label}")
    return float(value)


def _percent(value: object, *, na: str = "N/A") -> str:
    if value is None:
        return na
    return f"{100.0 * _number(value, 'percentage'):.2f}"


def _decimal(value: object, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{_number(value, 'decimal'):.{digits}f}"


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _pool_rows(
    sealed_pool: Mapping[str, object],
    twowiki_pool: Mapping[str, object],
) -> list[list[object]]:
    def values(pool: Mapping[str, object]) -> tuple[Mapping[str, object], Mapping[str, object]]:
        return _mapping(pool.get("candidate_pool"), "candidate_pool"), _mapping(
            pool.get("audit"), "audit"
        )

    sealed, sealed_audit = values(sealed_pool)
    twowiki, twowiki_audit = values(twowiki_pool)
    return [
        ["Queries (#)", sealed["query_count"], twowiki["query_count"]],
        ["Top-N", sealed["top_n"], twowiki["top_n"]],
        [
            "Required Recall@20 (%) ↑",
            _percent(sealed_audit.get("required_recall_at_top_n")),
            _percent(twowiki_audit.get("required_recall_at_top_n")),
        ],
        [
            "Harmful Pool-hit (%)",
            _percent(sealed_audit.get("harmful_pool_hit_rate")),
            "N/A",
        ],
        [
            "Exact-20 (%)",
            _percent(sealed.get("exact_top_n_rate")),
            _percent(twowiki.get("exact_top_n_rate")),
        ],
        [
            "Unresolved parents (#)",
            sealed_audit["unresolved_parent_count"],
            twowiki_audit["unresolved_parent_count"],
        ],
        ["Candidate pool SHA", sealed["sha256"], twowiki["sha256"]],
    ]


def _selector_rows(summary: Mapping[str, object], *, include_harm: bool) -> list[list[object]]:
    top = _mapping(summary.get("top_k"), "top_k")
    mis = _mapping(summary.get("reliability_mis"), "reliability_mis")
    rows: list[list[object]] = []
    if include_harm:
        rows.append(
            [
                "Harmful-in-context (%) ↓",
                _percent(top.get("harmful_in_context_conditional")),
                _percent(mis.get("harmful_in_context_conditional")),
            ]
        )
    rows.extend(
        [
            [
                "Supporting-document recall (%) ↑"
                if not include_harm
                else "Required-evidence recall (%) ↑",
                _percent(top.get("required_evidence_recall")),
                _percent(mis.get("required_evidence_recall")),
            ],
            [
                "Evidence precision (%) ↑",
                _percent(top.get("evidence_precision")),
                _percent(mis.get("evidence_precision")),
            ],
            [
                "Selected evidence (#)",
                _decimal(top.get("selected_evidence_count")),
                _decimal(mis.get("selected_evidence_count")),
            ],
        ]
    )
    return rows


def _paired_line(label: str, value: Mapping[str, object], *, include_p: bool) -> str:
    delta = 100.0 * _number(value.get("delta_mis_minus_top_k"), f"{label} delta")
    low = 100.0 * _number(value.get("ci_low"), f"{label} ci_low")
    high = 100.0 * _number(value.get("ci_high"), f"{label} ci_high")
    suffix = (
        f", p={_number(value.get('p_value'), f'{label} p-value'):.4f}"
        if include_p
        else ""
    )
    return f"{label}: Δ(MIS−TopK)={delta:.2f} pp, 95% CI=[{low:.2f}, {high:.2f}]{suffix}."


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    sealed_pool = _read(arguments.sealed_pool)
    sealed = _read(arguments.sealed_summary)
    twowiki_pool = _read(arguments.twowiki_pool)
    twowiki = _read(arguments.twowiki_summary)
    stability = _read(arguments.stability)
    arguments.output.mkdir(parents=True, exist_ok=True)

    pool_rows = _pool_rows(sealed_pool, twowiki_pool)
    sealed_rows = _selector_rows(sealed, include_harm=True)
    twowiki_rows = _selector_rows(twowiki, include_harm=False)
    _write_csv(arguments.output / "pool_quality_table.csv", ("Metric", "sealed600", "2Wiki"), pool_rows)
    _write_csv(
        arguments.output / "primary_selector_table.csv",
        ("Metric", "TopK", "Reliability-MIS"),
        sealed_rows,
    )
    _write_csv(
        arguments.output / "twowiki_guard_table.csv",
        ("Metric", "TopK", "Reliability-MIS"),
        twowiki_rows,
    )

    sealed_decision = _mapping(sealed.get("decision"), "sealed decision")
    twowiki_decision = _mapping(twowiki.get("decision"), "2Wiki decision")
    sealed_audit = _mapping(sealed_pool.get("audit"), "sealed pool audit")
    harm_opportunities = int(
        _number(sealed_audit.get("harmful_pool_hit_count"), "harmful_pool_hit_count")
    )
    checks = {
        "harm_opportunity_minimum_200": harm_opportunities >= 200,
        "harm_reduction": sealed_decision.get("harm_reduction_pass") is True,
        "sealed_required_recall_noninferiority": (
            sealed_decision.get("required_recall_noninferiority_pass") is True
        ),
        "twowiki_required_recall_noninferiority": (
            twowiki_decision.get("required_recall_noninferiority_pass") is True
        ),
        "stability": stability.get("pass") is True,
    }
    final_pass = all(checks.values())
    primary = {
        "schema_version": "1.0",
        "checks": checks,
        "final_selector": "reliability-mis" if final_pass else "top-k",
        "reliability_mis_pass": final_pass,
        "sealed600": sealed,
        "twowiki": twowiki,
        "stability": stability,
    }
    (arguments.output / "primary_summary.json").write_text(
        json.dumps(primary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    harm = _mapping(sealed.get("paired_harm"), "paired_harm")
    sealed_recall = _mapping(sealed.get("paired_required_recall"), "sealed recall")
    twowiki_recall = _mapping(twowiki.get("paired_required_recall"), "2Wiki recall")
    stability_arms = _mapping(stability.get("arms"), "stability arms")
    stability_mis = _mapping(stability_arms.get("reliability-mis"), "MIS stability")
    report = "\n".join(
        (
            "# Selector-only final report",
            "",
            "## Decision",
            "",
            (
                "Reliability-MIS is selected as the final Selector."
                if final_pass
                else "Reliability-MIS did not pass every frozen criterion; TopK remains the final Selector."
            ),
            "",
            "## Frozen-pool check",
            "",
            f"sealed600 harmful Pool-hit queries: {harm_opportunities}; minimum required: 200.",
            f"All pool and source-parent checks passed: {checks['harm_opportunity_minimum_200']}.",
            "",
            "## Paired results",
            "",
            _paired_line("sealed600 harmful-in-context", harm, include_p=True),
            _paired_line("sealed600 required recall", sealed_recall, include_p=False),
            _paired_line("2Wiki supporting recall", twowiki_recall, include_p=False),
            "",
            "## Stability",
            "",
            (
                "selected-ID exact agreement = "
                f"{stability_mis['exact_agreement_count']}/60 "
                f"({_percent(stability_mis['exact_agreement_rate'])}%), "
                f"Stability: {'Pass' if checks['stability'] else 'Fail'}."
            ),
            "",
            "## Frozen criteria",
            "",
            *(f"- {name}: {'Pass' if passed else 'Fail'}" for name, passed in checks.items()),
            "",
        )
    )
    (arguments.output / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"final_selector": primary["final_selector"], "checks": checks}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
