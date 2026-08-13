from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from evidence_rag.contracts.models import (
    EvidenceCandidate,
    RetainedEvidenceGuidance,
    SelectedEvidenceSet,
    SelectionGuidance,
)
from evidence_rag.selector.guidance import build_selection_guidance
from evidence_rag.selector.models import (
    SelectorAction,
    SelectorCandidateDecision,
    SelectorDecisionReason,
    SelectorDecisionTrace,
)


def _evidence(index: int) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=f"e{index}",
        document_id=f"d{index}",
        chunk_id=f"c{index}",
        text=f"evidence text {index}",
        source_uri=f"memory://{index}",
        retrieval_score=float(11 - index),
        retrieval_rank=index,
    )


def _decision(index: int, *, dropped: bool = False) -> SelectorCandidateDecision:
    return SelectorCandidateDecision(
        evidence_id=f"e{index}",
        retrieval_rank=index,
        protect_score=0.1,
        harm_score=0.9,
        safe_score=0.9,
        action=SelectorAction.DROP_HARM if dropped else SelectorAction.ABSTAIN_KEEP,
        reason=(
            SelectorDecisionReason.SAFE_SCORE_THRESHOLD_MET
            if dropped
            else SelectorDecisionReason.MAX_DELETE_REACHED
        ),
    )


def _trace() -> SelectorDecisionTrace:
    decisions = tuple(_decision(index, dropped=index == 10) for index in range(1, 11))
    return SelectorDecisionTrace(
        query_id="q1",
        policy_enabled=True,
        max_delete=1,
        safe_threshold=0.8,
        dependency_ready=True,
        baseline_evidence_ids=tuple(f"e{index}" for index in range(1, 11)),
        selected_evidence_ids=tuple(f"e{index}" for index in range(1, 10)),
        dropped_evidence_ids=("e10",),
        decisions=decisions,
    )


def _selected() -> SelectedEvidenceSet:
    return SelectedEvidenceSet(
        query_id="q1", evidence=tuple(_evidence(index) for index in range(1, 10))
    )


def test_guidance_copies_only_retained_runtime_selector_signals() -> None:
    guidance = build_selection_guidance(_trace(), _selected())
    payload = guidance.model_dump(mode="json")
    serialized = json.dumps(payload)

    assert guidance.selector_changed is True
    assert guidance.dropped_count == 1
    assert tuple(item.evidence_id for item in guidance.retained) == tuple(
        f"e{index}" for index in range(1, 10)
    )
    assert "evidence text" not in serialized
    assert "e10" not in serialized
    assert set(payload) == {
        "schema_version",
        "query_id",
        "selector_changed",
        "dropped_count",
        "retained",
    }
    assert set(payload["retained"][0]) == {
        "schema_version",
        "evidence_id",
        "retrieval_rank",
        "protect_signal",
        "harm_signal",
        "action",
        "reason",
    }


@pytest.mark.parametrize(
    "forbidden",
    ("gold_answer", "reference_answers", "required_document_ids", "gold_chain"),
)
def test_guidance_contract_rejects_evaluation_only_fields(forbidden: str) -> None:
    payload = {
        "query_id": "q1",
        "selector_changed": False,
        "dropped_count": 0,
        "retained": [],
        forbidden: "must not enter runtime",
    }

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SelectionGuidance.model_validate(payload)


def test_retained_item_rejects_drop_action() -> None:
    with pytest.raises(ValidationError):
        RetainedEvidenceGuidance(
            evidence_id="e1",
            retrieval_rank=1,
            protect_signal=0.1,
            harm_signal=0.9,
            action="DROP_HARM",
            reason="SAFE_SCORE_THRESHOLD_MET",
        )


def test_builder_rejects_selected_evidence_mismatch() -> None:
    selected = _selected().model_copy(update={"evidence": _selected().evidence[:-1]})

    with pytest.raises(ValueError, match="selected evidence IDs differ"):
        build_selection_guidance(_trace(), selected)
