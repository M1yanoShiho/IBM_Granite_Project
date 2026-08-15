from __future__ import annotations

import sys
from pathlib import Path

from evidence_rag.contracts.models import EvidenceCandidate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import full_flow_b100 as b100  # noqa: E402
import full_flow_b110_decide as b110  # noqa: E402
import full_flow_joint as joint  # noqa: E402


def _evidence(evidence_id: str, document_id: str, rank: int) -> EvidenceCandidate:
    harmful = document_id.startswith("cf::")
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=document_id,
        chunk_id=f"chunk-{evidence_id}",
        text=f"evidence {evidence_id}",
        source_uri=(
            f"synthetic://cf/{rank}/source" if harmful else f"fixture://{document_id}"
        ),
        retrieval_score=1.0 / rank,
        retrieval_rank=rank,
    )


def _profile(query_id: str, *, changed: bool, support: bool = True) -> b100.CaseProfile:
    first = _evidence("s", "support", 1) if support else _evidence("b", "benign", 1)
    harmful = _evidence("h", "cf::1::source", 2)
    topk = (first, harmful)
    selected = (first,) if changed else topk
    return b100.CaseProfile(
        case=joint.JointCase(query_id, "Who?", f"component-{query_id}", topk, selected),
        relevant_document_ids=frozenset({"support"}),
        question_type="who",
        reference_shape="named_entity_or_phrase",
        chain_eligible_topk10=True,
        support=(first,) if support else (),
        harmful=(harmful,),
        benign=() if support else (first,),
    )


def _arm(correct: bool, *, coverage: bool = True, reason: str = "nonempty") -> dict[str, object]:
    return {
        "answer_match": float(correct),
        "coverage": float(coverage),
        "final_empty_reason": reason,
        "reference_text_visible": True,
    }


def _case(query_id: str, *, changed: bool, oracle_correct: bool) -> dict[str, object]:
    metrics = {arm: _arm(True) for arm in b100.ARMS}
    if changed:
        metrics["S_legacy_selected"] = _arm(False)
        metrics["OB_support_benign"] = _arm(False)
    if not oracle_correct:
        metrics["O_support_only"] = _arm(
            False, coverage=False, reason="splitter_no_claims"
        )
    return {
        "query_id": query_id,
        "cohort": "selector_changed" if changed else "matched_unchanged",
        "metrics": metrics,
    }


def _comparison() -> dict[str, object]:
    return {"queries": 2, "paired": {"delta": 0.0}}


def test_source_summary_counts_missing_support() -> None:
    summary = b110._source_profile_summary(
        (_profile("visible", changed=False), _profile("absent", changed=False, support=False))
    )

    assert summary["support_visible"] == 1
    assert summary["support_absent"] == 1
    assert summary["support_absent_query_ids"] == ["absent"]


def test_decision_routes_generator_and_does_not_promote_splitter() -> None:
    profiles = (
        _profile("q1", changed=True),
        _profile("q2", changed=False),
        _profile("q3", changed=False),
    )
    nonempty_oracle_failure = _case("q3", changed=False, oracle_correct=False)
    nonempty_oracle_failure["metrics"]["O_support_only"] = _arm(False)
    cases = (
        _case("q1", changed=True, oracle_correct=True),
        _case("q2", changed=False, oracle_correct=False),
        nonempty_oracle_failure,
    )
    audit = (
        {
            "query_id": "q1",
            "contexts": {
                "K_topk": ["s", "h"],
                "S_legacy_selected": ["s"],
            },
            "topk_evidence_roles": {"s": "support", "h": "harmful"},
        },
        {
            "query_id": "q2",
            "contexts": {
                "K_topk": ["s", "h"],
                "S_legacy_selected": ["s", "h"],
            },
            "topk_evidence_roles": {"s": "support", "h": "harmful"},
        },
        {
            "query_id": "q3",
            "contexts": {
                "K_topk": ["s", "h"],
                "S_legacy_selected": ["s", "h"],
            },
            "topk_evidence_roles": {"s": "support", "h": "harmful"},
        },
    )
    comparisons = {label: _comparison() for label, _, _ in b100.PAIR_SPECS}
    report = {
        "all": {
            "aggregate": {
                "O_support_only": {
                    "splitter_status": {"structured": 2, "degraded_fallback": 0}
                }
            },
            "comparisons": comparisons,
        },
        "selector_changed": {"comparisons": {"S_minus_K": _comparison()}},
    }

    decision = b110.build_decision(
        profiles=profiles,
        b100_cases=cases,
        b100_audit=audit,
        b100_report=report,
        baseline_reproduction={},
    )

    assert decision["routing"]["primary"] == (
        "generator_evidence_utilization_and_context_robustness"
    )
    assert decision["generator_context_robustness"][
        "benign_noise_correct_to_wrong"
    ] == 1
    assert decision["legacy_selector"]["all_regressions_removed_only_harmful"]
    assert not decision["splitter_assessment"]["is_primary_bottleneck"]


def test_baseline_reproduction_compares_primary_a002_run() -> None:
    generation = {"answer": "answer", "cited_evidence_ids": ["e"]}
    trace = {"final_answer": "answer"}
    b100_row = {
        "query_id": "q",
        "arms": {
            "K_topk": {"generation": generation, "trace": trace},
            "S_legacy_selected": {"generation": generation, "trace": trace},
        },
    }
    a002_row = {
        "query_id": "q",
        "arms": {
            "K_topk_base": [{"generation": generation, "trace": trace}],
            "L_legacy_selector_base": [{"generation": generation, "trace": trace}],
        },
    }

    result = b110._baseline_reproduction((b100_row,), (a002_row,))

    assert result["K_topk"]["trace_equal"] == 1
    assert result["S_legacy_selected"]["generation_result_equal"] == 1
