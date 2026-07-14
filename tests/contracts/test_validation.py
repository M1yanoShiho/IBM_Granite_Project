import pytest

from evidence_rag.contracts.models import (
    CandidateSet,
    EvidenceCandidate,
    GenerationResult,
    SelectionItem,
    SelectionResult,
)
from evidence_rag.contracts.validation import resolve_selection, validate_generation


def candidate(evidence_id: str, text: str) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id="doc-1",
        chunk_id=f"chunk-{evidence_id}",
        text=text,
        source_uri="fixture://doc-1",
        retrieval_score=1.0,
        retrieval_rank=1,
    )


def test_pipeline_resolves_original_retriever_text() -> None:
    candidates = CandidateSet(
        query_id="q-1",
        candidates=(candidate("ev-1", "original retriever text"),),
    )
    selection = SelectionResult(
        query_id="q-1",
        items=(SelectionItem(evidence_id="ev-1", selection_score=0.9, selection_rank=1),),
    )
    selected = resolve_selection(candidates, selection)
    assert selected.evidence[0].text == "original retriever text"


def test_unknown_selected_id_is_rejected() -> None:
    candidates = CandidateSet(
        query_id="q-1",
        candidates=(candidate("ev-1", "text"),),
    )
    selection = SelectionResult(
        query_id="q-1",
        items=(SelectionItem(evidence_id="forged", selection_score=0.9, selection_rank=1),),
    )
    with pytest.raises(ValueError, match="unknown evidence"):
        resolve_selection(candidates, selection)


def test_generator_cannot_cite_unselected_id() -> None:
    candidates = CandidateSet(
        query_id="q-1",
        candidates=(candidate("ev-1", "text"),),
    )
    selection = SelectionResult(
        query_id="q-1",
        items=(SelectionItem(evidence_id="ev-1", selection_score=0.9, selection_rank=1),),
    )
    selected = resolve_selection(candidates, selection)
    result = GenerationResult(
        query_id="q-1",
        answer="Unsupported.",
        cited_evidence_ids=("ev-2",),
    )
    with pytest.raises(ValueError, match="unselected evidence"):
        validate_generation(selected, result)
