import pytest
from pydantic import ValidationError

from evidence_rag.contracts.models import (
    CandidateSet,
    EvidenceCandidate,
    GenerationResult,
    PipelineRun,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
    SelectionItem,
    SelectionResult,
    count_sentences,
)


def candidate(evidence_id: str = "ev-1") -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id="doc-1",
        chunk_id=f"chunk-{evidence_id}",
        text="Revenue increased.",
        source_uri="fixture://doc-1",
        retrieval_score=1.0,
        retrieval_rank=1,
    )


def checklist(query_id: str = "q-1") -> QueryChecklist:
    return QueryChecklist(
        query_id=query_id,
        focus="question",
        required_facts=("question",),
    )


def test_candidate_ids_must_be_unique() -> None:
    with pytest.raises(ValidationError):
        CandidateSet(query_id="q-1", candidates=(candidate(), candidate()))


def test_selector_result_contains_no_evidence_text_or_status() -> None:
    item = SelectionItem(evidence_id="ev-1", selection_score=0.9, selection_rank=1)
    assert not hasattr(item, "text")
    assert not hasattr(item, "status")


def test_selected_ids_must_be_unique() -> None:
    item = SelectionItem(evidence_id="ev-1", selection_score=0.9, selection_rank=1)
    with pytest.raises(ValidationError):
        SelectionResult(query_id="q-1", items=(item, item))


def test_nonempty_answer_requires_citation() -> None:
    with pytest.raises(ValidationError):
        GenerationResult(query_id="q-1", answer="Revenue increased.", cited_evidence_ids=())


def test_pipeline_run_rejects_stage_query_mismatch() -> None:
    evidence = candidate()
    with pytest.raises(ValidationError, match="query IDs"):
        PipelineRun(
            query=Query(query_id="q-1", text="question"),
            checklist=checklist(),
            top_k=1,
            max_selected=1,
            candidates=CandidateSet(query_id="q-other", candidates=(evidence,)),
            selection=SelectionResult(
                query_id="q-1",
                items=(
                    SelectionItem(
                        evidence_id=evidence.evidence_id,
                        selection_score=1.0,
                        selection_rank=1,
                    ),
                ),
            ),
            selected=SelectedEvidenceSet(query_id="q-1", evidence=(evidence,)),
            generation=GenerationResult(
                query_id="q-1",
                answer="Revenue increased.",
                cited_evidence_ids=(evidence.evidence_id,),
            ),
        )


def test_pipeline_run_rejects_selection_and_selected_evidence_mismatch() -> None:
    evidence = candidate()
    with pytest.raises(ValidationError, match="selected evidence"):
        PipelineRun(
            query=Query(query_id="q-1", text="question"),
            checklist=checklist(),
            top_k=1,
            max_selected=1,
            candidates=CandidateSet(query_id="q-1", candidates=(evidence,)),
            selection=SelectionResult(
                query_id="q-1",
                items=(
                    SelectionItem(
                        evidence_id=evidence.evidence_id,
                        selection_score=1.0,
                        selection_rank=1,
                    ),
                ),
            ),
            selected=SelectedEvidenceSet(query_id="q-1", evidence=()),
            generation=GenerationResult(
                query_id="q-1",
                answer="",
                cited_evidence_ids=(),
            ),
        )


def test_uncited_answer_is_allowed_only_when_every_sentence_is_labelled() -> None:
    """The invariant is swapped, not dropped. It used to guarantee "every answer is
    grounded", which forced a generator with nothing verified to abstain wholesale
    and discard its annotations. It now guarantees "every ungrounded sentence is
    labelled" -- a different, and for this method more honest, promise."""
    GenerationResult(
        query_id="q-1",
        answer="Costs fell. [unverified] Revenue rose. [unverified]",
        cited_evidence_ids=(),
    )

    with pytest.raises(ValidationError, match="mark every sentence"):
        GenerationResult(
            query_id="q-1",
            answer="Costs fell. Revenue rose. [unverified]",
            cited_evidence_ids=(),
        )


def test_abstention_and_all_unverified_remain_distinguishable() -> None:
    abstained = GenerationResult(query_id="q-1", answer="", cited_evidence_ids=())
    answered = GenerationResult(
        query_id="q-1", answer="Costs fell. [unverified]", cited_evidence_ids=()
    )

    assert abstained.answer == ""
    assert answered.answer != ""
    with pytest.raises(ValidationError, match="empty answer cannot contain citations"):
        GenerationResult(query_id="q-1", answer="", cited_evidence_ids=("ev-1",))


def test_count_sentences_ignores_the_annotation_label() -> None:
    assert count_sentences("Costs fell. [unverified] Revenue rose. [unverified]") == 2
    assert count_sentences("") == 0
