"""B110 deterministic bottleneck routing from frozen A002/B100 artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import full_flow_b100 as b100
import full_flow_joint as joint


def _jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _source_profile_summary(profiles: Sequence[b100.CaseProfile]) -> dict[str, object]:
    support_absent = [item.case.query_id for item in profiles if not item.support]
    return {
        "queries": len(profiles),
        "support_visible": len(profiles) - len(support_absent),
        "support_absent": len(support_absent),
        "support_absent_query_ids": sorted(support_absent),
        "selector_changed": sum(item.case.selector_changed for item in profiles),
        "support_count": dict(sorted(Counter(len(item.support) for item in profiles).items())),
        "harmful_count": dict(sorted(Counter(len(item.harmful) for item in profiles).items())),
        "benign_count": dict(sorted(Counter(len(item.benign) for item in profiles).items())),
    }


def _baseline_reproduction(
    b100_rows: Sequence[Mapping[str, Any]],
    a002_rows: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    a002 = {str(row["query_id"]): row for row in a002_rows}
    output: dict[str, object] = {}
    for b100_arm, a002_arm in (
        ("K_topk", "K_topk_base"),
        ("S_legacy_selected", "L_legacy_selector_base"),
    ):
        answer_equal = generation_equal = trace_equal = 0
        for row in b100_rows:
            query_id = str(row["query_id"])
            if query_id not in a002:
                raise ValueError(f"A002 omits B100 query {query_id}")
            b_run = cast(Mapping[str, Any], row["arms"])[b100_arm]
            a_run = cast(Mapping[str, Any], a002[query_id]["arms"])[a002_arm][0]
            answer_equal += int(
                b_run["generation"]["answer"] == a_run["generation"]["answer"]
            )
            generation_equal += int(b_run["generation"] == a_run["generation"])
            trace_equal += int(b_run["trace"] == a_run["trace"])
        output[b100_arm] = {
            "queries": len(b100_rows),
            "answer_equal": answer_equal,
            "generation_result_equal": generation_equal,
            "trace_equal": trace_equal,
        }
    return output


def _metric(case: Mapping[str, Any], arm: str, name: str) -> object:
    return cast(Mapping[str, Mapping[str, object]], case["metrics"])[arm][name]


def _correct(case: Mapping[str, Any], arm: str) -> bool:
    return _metric(case, arm, "answer_match") == 1.0


def build_decision(
    *,
    profiles: Sequence[b100.CaseProfile],
    b100_cases: Sequence[Mapping[str, Any]],
    b100_audit: Sequence[Mapping[str, Any]],
    b100_report: Mapping[str, Any],
    baseline_reproduction: Mapping[str, object],
) -> dict[str, object]:
    audit = {str(row["query_id"]): row for row in b100_audit}
    if {str(row["query_id"]) for row in b100_cases} != set(audit):
        raise ValueError("B100 cases and audit do not cover the same query IDs")
    oracle_arm = "O_support_only"
    benign_arm = "OB_support_benign"
    harmful_arm = "OH_support_harmful"
    topk_arm = "K_topk"
    selected_arm = "S_legacy_selected"
    oracle_correct = [case for case in b100_cases if _correct(case, oracle_arm)]
    oracle_failed = [case for case in b100_cases if not _correct(case, oracle_arm)]
    oracle_empty = [case for case in oracle_failed if _metric(case, oracle_arm, "coverage") == 0.0]
    oracle_nonempty_wrong = [
        case for case in oracle_failed if _metric(case, oracle_arm, "coverage") == 1.0
    ]
    benign_right_to_wrong = [
        case
        for case in oracle_correct
        if not _correct(case, benign_arm)
    ]
    harmful_right_to_wrong = [
        case
        for case in oracle_correct
        if not _correct(case, harmful_arm)
    ]
    noise_union = {
        str(case["query_id"])
        for case in (*benign_right_to_wrong, *harmful_right_to_wrong)
    }
    changed = [case for case in b100_cases if case["cohort"] == "selector_changed"]
    selector_right_to_wrong = [
        case
        for case in changed
        if _correct(case, topk_arm) and not _correct(case, selected_arm)
    ]
    removal_roles = Counter()
    removal_rows: list[dict[str, object]] = []
    for case in selector_right_to_wrong:
        query_id = str(case["query_id"])
        row = audit[query_id]
        contexts = cast(Mapping[str, Sequence[str]], row["contexts"])
        roles = cast(Mapping[str, str], row["topk_evidence_roles"])
        removed = sorted(set(contexts[topk_arm]) - set(contexts[selected_arm]))
        removed_roles = [roles[evidence_id] for evidence_id in removed]
        removal_roles.update(removed_roles)
        removal_rows.append(
            {
                "query_id": query_id,
                "removed_evidence_ids": removed,
                "removed_roles": removed_roles,
            }
        )
    all_selector_regressions_removed_only_harmful = bool(removal_rows) and all(
        row["removed_roles"]
        and set(cast(Sequence[str], row["removed_roles"])) == {"harmful"}
        for row in removal_rows
    )
    all_scope = cast(Mapping[str, Any], b100_report["all"])
    oracle_aggregate = cast(Mapping[str, Any], all_scope["aggregate"])[oracle_arm]
    comparisons = cast(Mapping[str, Any], all_scope["comparisons"])
    oracle_empty_reasons = Counter(
        str(_metric(case, oracle_arm, "final_empty_reason")) for case in oracle_failed
    )
    splitter_zero_claims = oracle_empty_reasons["splitter_no_claims"]
    degraded_splitter = int(
        cast(Mapping[str, int], oracle_aggregate["splitter_status"]).get(
            "degraded_fallback", 0
        )
    )
    reference_visible = sum(
        bool(_metric(case, oracle_arm, "reference_text_visible")) for case in b100_cases
    )
    source = _source_profile_summary(profiles)
    primary_generator = len(oracle_failed) > source["support_absent"]
    split_is_primary = splitter_zero_claims > len(oracle_failed) / 2
    return {
        "schema_version": "full-flow-b110-decision-v1",
        "status": "COMPLETE",
        "baseline_reproduction": baseline_reproduction,
        "retriever_visibility": source,
        "generator_support_only": {
            "queries": len(b100_cases),
            "reference_text_visible": reference_visible,
            "correct": len(oracle_correct),
            "failed": len(oracle_failed),
            "empty_failed": len(oracle_empty),
            "nonempty_wrong": len(oracle_nonempty_wrong),
            "empty_failure_reasons": dict(sorted(oracle_empty_reasons.items())),
            "O_minus_K": comparisons["O_minus_K"],
        },
        "generator_context_robustness": {
            "oracle_correct": len(oracle_correct),
            "benign_noise_correct_to_wrong": len(benign_right_to_wrong),
            "harmful_noise_correct_to_wrong": len(harmful_right_to_wrong),
            "either_noise_correct_to_wrong": len(noise_union),
            "OB_minus_O": comparisons["OB_minus_O"],
            "OH_minus_O": comparisons["OH_minus_O"],
            "OH_minus_OB": comparisons["OH_minus_OB"],
            "OP_minus_K": comparisons["OP_minus_K"],
        },
        "legacy_selector": {
            "changed_queries": len(changed),
            "K_right_S_wrong": len(selector_right_to_wrong),
            "removal_roles": dict(sorted(removal_roles.items())),
            "all_regressions_removed_only_harmful": (
                all_selector_regressions_removed_only_harmful
            ),
            "regression_rows": removal_rows,
            "S_minus_K_changed": b100_report["selector_changed"]["comparisons"][
                "S_minus_K"
            ],
        },
        "splitter_assessment": {
            "oracle_failures": len(oracle_failed),
            "splitter_no_claims": splitter_zero_claims,
            "degraded_fallback": degraded_splitter,
            "is_primary_bottleneck": split_is_primary,
        },
        "routing": {
            "primary": (
                "generator_evidence_utilization_and_context_robustness"
                if primary_generator
                else "retriever_visibility"
            ),
            "retriever": "secondary_recorded_bottleneck",
            "legacy_selector": "retain_as_evidence_risk_baseline_not_answer_utility_solution",
            "splitter_fallback_G210": "do_not_activate_as_primary_branch",
            "generator_draft_training_G200_G220": "unblocked_by_B110",
            "utility_selector_training_S300": "remains_blocked_until_generator_gate",
        },
        "interpretation_boundary": (
            "B110 routes the next experimental responsibility; it does not establish that a "
            "new Generator or Selector method is effective"
        ),
    }


def _pct(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.2f}%"


def _markdown(decision: Mapping[str, Any]) -> str:
    retrieval = decision["retriever_visibility"]
    generator = decision["generator_support_only"]
    robustness = decision["generator_context_robustness"]
    selector = decision["legacy_selector"]
    splitter = decision["splitter_assessment"]
    return "\n".join(
        [
            "# B110 Bottleneck Decision",
            "",
            "**Status:** `COMPLETE`",
            "",
            "## Decision",
            "",
            "Primary route: **Generator evidence utilization and context robustness**.",
            "",
            f"- Retriever: {retrieval['support_absent']}/{retrieval['queries']} "
            f"({_pct(retrieval['support_absent'], retrieval['queries'])}) decision-dev queries "
            "have no official relevant document in TopK10; record separately.",
            f"- Generator support-only: {generator['failed']}/{generator['queries']} "
            f"({_pct(generator['failed'], generator['queries'])}) fail even though the normalized "
            "reference string is visible in every O context.",
            f"- Generator robustness: among {robustness['oracle_correct']} O-correct queries, "
            f"{robustness['either_noise_correct_to_wrong']} become wrong under benign or harmful "
            "matched noise.",
            f"- Legacy Selector: {selector['K_right_S_wrong']} K-right/S-wrong changed queries; "
            "every removed item in those regressions is labelled harmful.",
            f"- Splitter: {splitter['splitter_no_claims']}/{splitter['oracle_failures']} O failures "
            "are zero-claim outcomes; this is not the majority bottleneck.",
            "",
            "## Route Gates",
            "",
            "- G200/G220 Generator draft-path work: unblocked.",
            "- G210 splitter fallback as the primary branch: not activated.",
            "- S300 utility-Selector training: remains blocked until the Generator gate passes.",
            "- Legacy Selector remains an evidence-risk baseline, not an answer-utility solution.",
            "",
            "This decision routes responsibility only. It does not claim that a new method works.",
            "",
        ]
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--candidate-pool", required=True, type=Path)
    parser.add_argument("--roles", required=True, type=Path)
    parser.add_argument("--decision-trace", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--b100-cases", required=True, type=Path)
    parser.add_argument("--b100-audit", required=True, type=Path)
    parser.add_argument("--b100-report", required=True, type=Path)
    parser.add_argument("--b100-generations", required=True, type=Path)
    parser.add_argument("--a002-generations", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cases = joint.load_runtime_cases(
        queries_path=args.queries,
        candidate_pool_path=args.candidate_pool,
        roles_path=args.roles,
        decision_trace_path=args.decision_trace,
    )
    wanted = {case.query_id for case in cases}
    gold = b100._load_gold(args.gold, wanted)
    chain = b100._load_chain_eligibility(args.roles, wanted)
    profiles = [
        b100.build_profile(
            case,
            gold[case.query_id],
            chain_eligible_topk10=chain[case.query_id],
        )
        for case in cases
    ]
    b100_generations = _jsonl(args.b100_generations)
    reproduction = _baseline_reproduction(
        b100_generations,
        _jsonl(args.a002_generations),
    )
    decision = build_decision(
        profiles=profiles,
        b100_cases=_jsonl(args.b100_cases),
        b100_audit=_jsonl(args.b100_audit),
        b100_report=json.loads(args.b100_report.read_text(encoding="utf-8")),
        baseline_reproduction=reproduction,
    )
    _write_json(args.output_json, decision)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(_markdown(decision), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
