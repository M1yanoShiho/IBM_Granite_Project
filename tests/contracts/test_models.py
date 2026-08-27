import pytest
from pydantic import ValidationError

from evidence_rag.contracts.models import (
    CandidateSet,
    EvidenceCandidate,
    GenerationResult,
    PipelineRun,
    Query,
    QueryChecklist,
    RetrieverProvenance,
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


def a_provenance(name: str = "bm25") -> RetrieverProvenance:
    return RetrieverProvenance(
        name=name, implementation_version=f"{name}-v1", parameters_sha256="a" * 64
    )


def test_a_candidate_set_can_name_the_retriever_that_produced_it() -> None:
    """M0 §4 freezes the retriever because the Graph 2.0 claim is conditional on a fixed
    candidate pool. Nothing downstream could tell two pools apart while the pool itself said
    nothing about its producer, so the producer now travels inside the artefact."""
    candidates = CandidateSet(
        query_id="q-1", candidates=(candidate(),), retriever=a_provenance()
    )
    assert candidates.retriever is not None
    assert candidates.retriever.name == "bm25"
    assert candidates.model_dump(mode="json")["retriever"]["name"] == "bm25"


def test_a_candidate_set_without_a_retriever_serializes_exactly_as_it_did_before() -> None:
    """The field is omitted, not written as null. Every frozen artefact in this repository was
    hashed before the field existed; a `"retriever":null` key would change those bytes and turn
    a provenance fix into a mass re-freeze of unrelated evidence."""
    payload = CandidateSet(query_id="q-1", candidates=(candidate(),)).model_dump(mode="json")
    assert "retriever" not in payload


def test_a_retriever_provenance_is_rejected_without_a_parameter_digest() -> None:
    """Name alone is not identity: `bm25 k1=1.5 b=0.75` and `bm25 k1=0.9 b=0.4` are the same
    name over different pools, and §4's freeze is a statement about the pool."""
    with pytest.raises(ValidationError):
        RetrieverProvenance(name="bm25", implementation_version="bm25-v1", parameters_sha256="")


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
