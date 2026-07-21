import pytest

from evidence_rag.contracts.models import (
    EvidenceCandidate,
    QueryChecklist,
    SelectedEvidenceSet,
)
from evidence_rag.generator.models import (
    Claim,
    ClaimSpan,
    ClaimVerification,
    DraftAnswer,
    RequiredFactCoverage,
    VerificationReport,
    validate_fact_coverage,
    validate_supporting_evidence,
    validate_verification_report,
)


def claim(claim_id: str, *, start: int = 0, end: int = 5, faithful: bool = True) -> Claim:
    return Claim(
        claim_id=claim_id,
        text=f"claim {claim_id}",
        span=ClaimSpan(start=start, end=end),
        faithful_to_answer=faithful,
    )


def verification(
    claim_id: str,
    *,
    status: str = "supported",
    evidence_ids: tuple[str, ...] = ("ev-1",),
    entity_consistent: bool = True,
    contradicted: bool = False,
) -> ClaimVerification:
    return ClaimVerification(
        claim_id=claim_id,
        status=status,
        supporting_evidence_ids=evidence_ids,
        entity_consistent=entity_consistent,
        contradicted=contradicted,
    )


def evidence(evidence_id: str) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id="doc-1",
        chunk_id=f"chunk-{evidence_id}",
        text="evidence text",
        source_uri="fixture://doc-1",
        retrieval_score=1.0,
        retrieval_rank=1,
    )


# --- model-level invariants -------------------------------------------------


def test_claim_span_rejects_non_positive_length() -> None:
    with pytest.raises(ValueError, match="end must be greater than start"):
        ClaimSpan(start=5, end=5)


def test_draft_answer_rejects_claims_beyond_answer_length() -> None:
    with pytest.raises(ValueError, match="span exceeds answer_text length"):
        DraftAnswer(
            query_id="q-1",
            answer_text="short",
            claims=(claim("c1", start=0, end=999),),
        )


def test_draft_answer_empty_answer_cannot_have_claims() -> None:
    with pytest.raises(ValueError, match="empty answer_text cannot have claims"):
        DraftAnswer(query_id="q-1", answer_text="", claims=(claim("c1"),))


def test_draft_answer_rejects_duplicate_claim_ids() -> None:
    with pytest.raises(ValueError, match="claim IDs must be unique"):
        DraftAnswer(
            query_id="q-1",
            answer_text="answer text here",
            claims=(claim("c1"), claim("c1")),
        )


def test_supported_claim_requires_evidence() -> None:
    with pytest.raises(ValueError, match="must have supporting evidence"):
        verification("c1", status="supported", evidence_ids=())


def test_supported_claim_must_be_entity_consistent() -> None:
    with pytest.raises(ValueError, match="must be entity-consistent"):
        verification("c1", status="supported", entity_consistent=False)


def test_unsupported_claim_cannot_have_supporting_evidence() -> None:
    with pytest.raises(ValueError, match="cannot have supporting evidence"):
        verification("c1", status="unsupported", evidence_ids=("ev-1",))


def test_claim_verification_rejects_duplicate_evidence_ids() -> None:
    with pytest.raises(ValueError, match="supporting evidence IDs must be unique"):
        verification("c1", evidence_ids=("ev-1", "ev-1"))


def test_required_fact_covered_cannot_carry_gap_question() -> None:
    with pytest.raises(ValueError, match="covered fact cannot carry a gap question"):
        RequiredFactCoverage(required_fact="deal value", covered=True, gap_question="how much?")


def test_required_fact_uncovered_requires_gap_question() -> None:
    with pytest.raises(ValueError, match="uncovered fact requires a gap question"):
        RequiredFactCoverage(required_fact="deal value", covered=False)


# --- validate_verification_report -------------------------------------------


def test_verification_report_covers_only_faithful_claims() -> None:
    draft = DraftAnswer(
        query_id="q-1",
        answer_text="Pfizer acquired Seagen. It was headquartered in London.",
        claims=(
            claim("c1", start=0, end=22, faithful=True),
            claim("c2", start=23, end=55, faithful=False),
        ),
    )
    report = VerificationReport(
        query_id="q-1",
        claims=(verification("c1"),),
        fact_coverage=(),
    )
    validate_verification_report(draft, report)


def test_verification_report_rejects_query_id_mismatch() -> None:
    draft = DraftAnswer(query_id="q-1", answer_text="answer", claims=(claim("c1"),))
    report = VerificationReport(query_id="q-2", claims=(verification("c1"),), fact_coverage=())
    with pytest.raises(ValueError, match="query IDs differ"):
        validate_verification_report(draft, report)


def test_verification_report_rejects_verifying_unfaithful_claim() -> None:
    draft = DraftAnswer(
        query_id="q-1",
        answer_text="answer text here",
        claims=(claim("c1", faithful=True), claim("c2", start=6, end=16, faithful=False)),
    )
    report = VerificationReport(
        query_id="q-1",
        claims=(verification("c1"), verification("c2")),
        fact_coverage=(),
    )
    with pytest.raises(ValueError, match="verifiable claims"):
        validate_verification_report(draft, report)


def test_verification_report_rejects_missing_faithful_claim() -> None:
    draft = DraftAnswer(query_id="q-1", answer_text="answer", claims=(claim("c1", faithful=True),))
    report = VerificationReport(query_id="q-1", claims=(), fact_coverage=())
    with pytest.raises(ValueError, match="verifiable claims"):
        validate_verification_report(draft, report)


# --- validate_supporting_evidence -------------------------------------------


def test_supporting_evidence_within_selected_set_passes() -> None:
    selected = SelectedEvidenceSet(query_id="q-1", evidence=(evidence("ev-1"), evidence("ev-2")))
    report = VerificationReport(
        query_id="q-1",
        claims=(verification("c1", evidence_ids=("ev-1", "ev-2")),),
        fact_coverage=(),
    )
    validate_supporting_evidence(selected, report)


def test_supporting_evidence_outside_selected_set_is_rejected() -> None:
    selected = SelectedEvidenceSet(query_id="q-1", evidence=(evidence("ev-1"),))
    report = VerificationReport(
        query_id="q-1",
        claims=(verification("c1", evidence_ids=("ev-forged",)),),
        fact_coverage=(),
    )
    with pytest.raises(ValueError, match="cites unselected evidence"):
        validate_supporting_evidence(selected, report)


# --- validate_fact_coverage -------------------------------------------------


def test_fact_coverage_matches_checklist() -> None:
    checklist = QueryChecklist(
        query_id="q-1",
        focus="deal",
        required_facts=("deal value", "deal date"),
    )
    report = VerificationReport(
        query_id="q-1",
        claims=(),
        fact_coverage=(
            RequiredFactCoverage(required_fact="deal value", covered=True),
            RequiredFactCoverage(required_fact="deal date", covered=False, gap_question="when?"),
        ),
    )
    validate_fact_coverage(checklist, report)


def test_fact_coverage_missing_required_fact_is_rejected() -> None:
    checklist = QueryChecklist(
        query_id="q-1",
        focus="deal",
        required_facts=("deal value", "deal date"),
    )
    report = VerificationReport(
        query_id="q-1",
        claims=(),
        fact_coverage=(RequiredFactCoverage(required_fact="deal value", covered=True),),
    )
    with pytest.raises(ValueError, match="does not match checklist required_facts"):
        validate_fact_coverage(checklist, report)


def test_fact_coverage_extra_fact_is_rejected() -> None:
    checklist = QueryChecklist(query_id="q-1", focus="deal", required_facts=("deal value",))
    report = VerificationReport(
        query_id="q-1",
        claims=(),
        fact_coverage=(
            RequiredFactCoverage(required_fact="deal value", covered=True),
            RequiredFactCoverage(required_fact="unexpected", covered=True),
        ),
    )
    with pytest.raises(ValueError, match="does not match checklist required_facts"):
        validate_fact_coverage(checklist, report)
