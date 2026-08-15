from __future__ import annotations

import sys
from pathlib import Path

import pytest

from evidence_rag.contracts.models import (
    EvidenceCandidate,
    GenerationResult,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
)
from evidence_rag.generator.trace import GeneratorTrace, unavailable_draft_trace
from evidence_rag.infrastructure.datasets import GoldCase

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import full_flow_b100 as b100  # noqa: E402
import full_flow_joint as joint  # noqa: E402


def _candidate(index: int, role: str) -> EvidenceCandidate:
    if role == "support":
        document_id = f"support-{index}"
        source_uri = f"fixture://support/{index}"
    elif role == "harmful":
        document_id = f"cf::{index}::source-{index}"
        source_uri = f"synthetic://cf/{index}/source-{index}"
    else:
        document_id = f"benign-{index}"
        source_uri = f"fixture://benign/{index}"
    return EvidenceCandidate(
        evidence_id=f"e{index}",
        document_id=document_id,
        chunk_id=f"c{index}",
        text=f"{role} evidence {index}",
        source_uri=source_uri,
        retrieval_score=float(11 - index),
        retrieval_rank=index,
    )


def _profile(
    query_id: str,
    *,
    changed: bool,
    roles: tuple[str, ...] = (
        "benign",
        "support",
        "harmful",
        "support",
        "benign",
        "harmful",
        "benign",
        "benign",
        "benign",
        "benign",
    ),
    question: str = "Who wrote the fixture?",
    reference: str = "Ada Lovelace",
) -> b100.CaseProfile:
    topk = tuple(_candidate(index, role) for index, role in enumerate(roles, 1))
    selected = topk[:-1] if changed else topk
    case = joint.JointCase(query_id, question, f"component-{query_id}", topk, selected)
    relevant = tuple(item.document_id for item, role in zip(topk, roles, strict=True) if role == "support")
    return b100.build_profile(
        case,
        GoldCase(
            query_id=query_id,
            relevant_document_ids=relevant,
            reference_answers=(reference,),
        ),
        chain_eligible_topk10=True,
    )


class FakeTracedGenerator:
    def __init__(self) -> None:
        self.last_trace: GeneratorTrace | None = None

    def generate(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> GenerationResult:
        del checklist
        ids = tuple(item.evidence_id for item in selected.evidence)
        answer = "Ada Lovelace" if any("support" in item.text for item in selected.evidence) else ""
        result = GenerationResult(
            query_id=query.query_id,
            answer=answer,
            cited_evidence_ids=(ids[0],) if answer and ids else (),
        )
        self.last_trace = GeneratorTrace(
            query_id=query.query_id,
            selected_evidence_ids=ids,
            draft=unavailable_draft_trace(answer),
            claims=(),
            final_answer=answer,
            final_cited_evidence_ids=result.cited_evidence_ids,
            final_empty_reason="empty_draft" if not answer else "",
        )
        return result


def test_context_matrix_balances_noise_and_changes_only_position() -> None:
    profile = _profile("changed", changed=True)
    contexts = b100.build_contexts(profile)

    assert [item.evidence_id for item in contexts["O_support_only"]] == ["e2", "e4"]
    assert [item.evidence_id for item in contexts["OB_support_benign"]] == [
        "e2",
        "e4",
        "e1",
        "e5",
    ]
    assert [item.evidence_id for item in contexts["OH_support_harmful"]] == [
        "e2",
        "e4",
        "e3",
        "e6",
    ]
    assert len(contexts["OB_support_benign"]) == len(contexts["OH_support_harmful"])
    assert {item.evidence_id for item in contexts["OP_support_last"]} == {
        item.evidence_id for item in contexts["K_topk"]
    }
    assert [item.evidence_id for item in contexts["OP_support_last"]][-2:] == ["e2", "e4"]


def test_matching_is_deterministic_unique_and_prefers_same_structure() -> None:
    changed = (
        _profile("c1", changed=True),
        _profile(
            "c2",
            changed=True,
            roles=("support",) + ("benign",) * 8 + ("harmful",),
            question="When was the fixture built?",
            reference="in 2009",
        ),
    )
    controls = (
        _profile("u1", changed=False),
        _profile(
            "u2",
            changed=False,
            roles=("support",) + ("benign",) * 8 + ("harmful",),
            question="When did this happen?",
            reference="in 2010",
        ),
        _profile("u3", changed=False, question="Where is the fixture?"),
    )

    first = b100.match_unchanged(changed, controls)
    second = b100.match_unchanged(changed, controls)

    assert first == second
    assert [(left.case.query_id, right.case.query_id) for left, right, _ in first] == [
        ("c1", "u1"),
        ("c2", "u2"),
    ]
    assert len({right.case.query_id for _, right, _ in first}) == 2


def test_prepare_runtime_rows_strip_gold_fields() -> None:
    runtime, audit, matching = b100.prepare_rows(
        [_profile("c1", changed=True), _profile("u1", changed=False)]
    )

    assert len(runtime) == len(audit) == 2
    assert matching["pairs"] == 1
    serialized = str(runtime)
    assert "reference_answers" not in serialized
    assert "relevant_document_ids" not in serialized
    assert "topk_evidence_roles" not in serialized
    assert "relevant_document_ids" in str(audit)


def test_arm_order_is_a_deterministic_permutation() -> None:
    assert b100.arm_order("q1") == b100.arm_order("q1")
    assert set(b100.arm_order("q1")) == set(b100.ARMS)
    assert len(b100.arm_order("q1")) == len(b100.ARMS)


def test_run_contexts_records_all_six_traces() -> None:
    runtime, _, _ = b100.prepare_rows(
        [_profile("c1", changed=True), _profile("u1", changed=False)]
    )
    rows = b100.run_contexts(runtime, generator=FakeTracedGenerator())

    assert len(rows) == 2
    assert set(rows[0]["arms"]) == set(b100.ARMS)
    assert all(rows[0]["arms"][arm]["trace"] is not None for arm in b100.ARMS)


def test_score_reports_oracle_and_noise_metrics() -> None:
    profiles = [_profile("c1", changed=True), _profile("u1", changed=False)]
    runtime, audit, _ = b100.prepare_rows(profiles)
    generated = b100.run_contexts(runtime, generator=FakeTracedGenerator())
    gold = {
        profile.case.query_id: GoldCase(
            query_id=profile.case.query_id,
            relevant_document_ids=tuple(profile.relevant_document_ids),
            reference_answers=("Ada Lovelace",),
        )
        for profile in profiles
    }

    cases, report = b100.score_rows(generated, runtime, audit, gold)

    assert len(cases) == 2
    assert report["all"]["aggregate"]["O_support_only"]["answer_match"] == 1.0
    assert report["all"]["comparisons"]["OH_minus_OB"]["queries"] == 2
    assert report["errors_by_arm"] == dict.fromkeys(b100.ARMS, 0)
    assert report["single_support"]["queries"] == 0


def test_runtime_parser_does_not_accept_gold() -> None:
    with pytest.raises(SystemExit):
        b100._parser().parse_args(
            [
                "run",
                "--contexts",
                "contexts.jsonl",
                "--granite-snapshot",
                "granite",
                "--true-snapshot",
                "true",
                "--output-dir",
                "out",
                "--gold",
                "gold.jsonl",
            ]
        )


def test_runtime_loader_rejects_a_gold_field(tmp_path: Path) -> None:
    path = tmp_path / "runtime.jsonl"
    path.write_text(
        '{"schema_version":"full-flow-b100-runtime-context-v1","query_id":"q",'
        '"question":"q?","component_id":"c","selector_changed":true,'
        '"matched_pair_id":"q","cohort":"selector_changed","contexts":{},'
        '"reference_answers":["leak"]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden"):
        b100._runtime_contexts(path)
