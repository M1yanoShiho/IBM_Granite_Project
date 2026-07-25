import pytest

from evidence_rag.contracts.models import EvidenceCandidate, SelectedEvidenceSet
from evidence_rag.generator.models import (
    Claim,
    ClaimSpan,
    ClaimVerification,
    DraftAnswer,
    VerificationReport,
)
from evidence_rag.generator.repair import AnswerRepairer


def evidence(evidence_id: str) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=f"doc-{evidence_id}",
        chunk_id=f"chunk-{evidence_id}",
        text=f"Evidence for {evidence_id}.",
        source_uri=f"fixture://{evidence_id}",
        retrieval_score=1.0,
        retrieval_rank=1,
    )


def test_repair_preserves_fully_supported_answer_and_verified_citations() -> None:
    draft = DraftAnswer(
        query_id="q-1",
        answer_text="Revenue rose. Profit stayed stable.",
        claims=(
            Claim(
                claim_id="claim-1",
                text="Revenue rose.",
                span=ClaimSpan(start=0, end=13),
                faithful_to_answer=True,
            ),
            Claim(
                claim_id="claim-2",
                text="Profit stayed stable.",
                span=ClaimSpan(start=14, end=35),
                faithful_to_answer=True,
            ),
        ),
    )
    report = VerificationReport(
        query_id="q-1",
        claims=(
            ClaimVerification(
                claim_id="claim-1",
                status="supported",
                supporting_evidence_ids=("ev-1",),
                entity_consistent=True,
                contradicted=False,
            ),
            ClaimVerification(
                claim_id="claim-2",
                status="supported",
                supporting_evidence_ids=("ev-2",),
                entity_consistent=True,
                contradicted=False,
            ),
        ),
        fact_coverage=(),
    )
    selected = SelectedEvidenceSet(
        query_id="q-1",
        evidence=(evidence("ev-1"), evidence("ev-2")),
    )

    result = AnswerRepairer().repair(draft, report, selected)

    assert result.answer == draft.answer_text
    assert result.cited_evidence_ids == ("ev-1", "ev-2")


def test_repair_does_not_preserve_text_outside_verified_claim_spans() -> None:
    draft = DraftAnswer(
        query_id="q-1",
        answer_text="Revenue rose. Unsplit forecast doubled.",
        claims=(
            Claim(
                claim_id="claim-1",
                text="Revenue rose.",
                span=ClaimSpan(start=0, end=13),
                faithful_to_answer=True,
            ),
        ),
    )
    report = VerificationReport(
        query_id="q-1",
        claims=(
            ClaimVerification(
                claim_id="claim-1",
                status="supported",
                supporting_evidence_ids=("ev-1",),
                entity_consistent=True,
                contradicted=False,
            ),
        ),
        fact_coverage=(),
    )

    result = AnswerRepairer().repair(
        draft,
        report,
        SelectedEvidenceSet(query_id="q-1", evidence=(evidence("ev-1"),)),
    )

    assert result.answer == "Revenue rose."
    assert result.cited_evidence_ids == ("ev-1",)


def test_repair_removes_unsupported_claim_and_its_citation() -> None:
    draft = DraftAnswer(
        query_id="q-1",
        answer_text="Revenue rose. Profit stayed stable.",
        claims=(
            Claim(
                claim_id="claim-1",
                text="Revenue rose.",
                span=ClaimSpan(start=0, end=13),
                faithful_to_answer=True,
            ),
            Claim(
                claim_id="claim-2",
                text="Profit stayed stable.",
                span=ClaimSpan(start=14, end=35),
                faithful_to_answer=True,
            ),
        ),
    )
    report = VerificationReport(
        query_id="q-1",
        claims=(
            ClaimVerification(
                claim_id="claim-1",
                status="supported",
                supporting_evidence_ids=("ev-1",),
                entity_consistent=True,
                contradicted=False,
            ),
            ClaimVerification(
                claim_id="claim-2",
                status="unsupported",
                supporting_evidence_ids=(),
                entity_consistent=True,
                contradicted=False,
            ),
        ),
        fact_coverage=(),
    )
    selected = SelectedEvidenceSet(
        query_id="q-1",
        evidence=(evidence("ev-1"), evidence("ev-2")),
    )

    result = AnswerRepairer().repair(draft, report, selected)

    assert result.answer == "Revenue rose."
    assert result.cited_evidence_ids == ("ev-1",)


def test_repair_removes_unfaithful_claim_without_b_verdict() -> None:
    draft = DraftAnswer(
        query_id="q-1",
        answer_text="Revenue may rise. Profit stayed stable.",
        claims=(
            Claim(
                claim_id="claim-1",
                text="Revenue will rise.",
                span=ClaimSpan(start=0, end=17),
                faithful_to_answer=False,
            ),
            Claim(
                claim_id="claim-2",
                text="Profit stayed stable.",
                span=ClaimSpan(start=18, end=39),
                faithful_to_answer=True,
            ),
        ),
    )
    report = VerificationReport(
        query_id="q-1",
        claims=(
            ClaimVerification(
                claim_id="claim-2",
                status="supported",
                supporting_evidence_ids=("ev-2",),
                entity_consistent=True,
                contradicted=False,
            ),
        ),
        fact_coverage=(),
    )
    selected = SelectedEvidenceSet(
        query_id="q-1",
        evidence=(evidence("ev-2"),),
    )

    result = AnswerRepairer().repair(draft, report, selected)

    assert result.answer == "Profit stayed stable."
    assert result.cited_evidence_ids == ("ev-2",)


def test_repair_refuses_when_only_claim_has_conflicting_evidence() -> None:
    draft = DraftAnswer(
        query_id="q-1",
        answer_text="Revenue rose.",
        claims=(
            Claim(
                claim_id="claim-1",
                text="Revenue rose.",
                span=ClaimSpan(start=0, end=13),
                faithful_to_answer=True,
            ),
        ),
    )
    report = VerificationReport(
        query_id="q-1",
        claims=(
            ClaimVerification(
                claim_id="claim-1",
                status="supported",
                supporting_evidence_ids=("ev-1",),
                entity_consistent=True,
                contradicted=True,
            ),
        ),
        fact_coverage=(),
    )
    selected = SelectedEvidenceSet(
        query_id="q-1",
        evidence=(evidence("ev-1"),),
    )

    result = AnswerRepairer().repair(draft, report, selected)

    assert result.answer == ""
    assert result.cited_evidence_ids == ()


def test_repair_refuses_when_only_claim_has_entity_mismatch() -> None:
    draft = DraftAnswer(
        query_id="q-1",
        answer_text="Pfizer revenue rose.",
        claims=(
            Claim(
                claim_id="claim-1",
                text="Pfizer revenue rose.",
                span=ClaimSpan(start=0, end=20),
                faithful_to_answer=True,
            ),
        ),
    )
    report = VerificationReport(
        query_id="q-1",
        claims=(
            ClaimVerification(
                claim_id="claim-1",
                status="unsupported",
                supporting_evidence_ids=(),
                entity_consistent=False,
                contradicted=False,
            ),
        ),
        fact_coverage=(),
    )
    selected = SelectedEvidenceSet(
        query_id="q-1",
        evidence=(evidence("ev-1"),),
    )

    result = AnswerRepairer().repair(draft, report, selected)

    assert result.answer == ""
    assert result.cited_evidence_ids == ()


def test_repair_rejects_selected_evidence_for_another_query() -> None:
    draft = DraftAnswer(query_id="q-1", answer_text="", claims=())
    report = VerificationReport(query_id="q-1", claims=(), fact_coverage=())
    selected = SelectedEvidenceSet(query_id="q-other", evidence=())

    with pytest.raises(ValueError, match="query IDs differ"):
        AnswerRepairer().repair(draft, report, selected)


def test_repair_deduplicates_citations_in_claim_order() -> None:
    draft = DraftAnswer(
        query_id="q-1",
        answer_text="Revenue rose. Profit rose.",
        claims=(
            Claim(
                claim_id="claim-1",
                text="Revenue rose.",
                span=ClaimSpan(start=0, end=13),
                faithful_to_answer=True,
            ),
            Claim(
                claim_id="claim-2",
                text="Profit rose.",
                span=ClaimSpan(start=14, end=26),
                faithful_to_answer=True,
            ),
        ),
    )
    report = VerificationReport(
        query_id="q-1",
        claims=(
            ClaimVerification(
                claim_id="claim-1",
                status="supported",
                supporting_evidence_ids=("ev-1", "ev-2"),
                entity_consistent=True,
                contradicted=False,
            ),
            ClaimVerification(
                claim_id="claim-2",
                status="supported",
                supporting_evidence_ids=("ev-2",),
                entity_consistent=True,
                contradicted=False,
            ),
        ),
        fact_coverage=(),
    )
    selected = SelectedEvidenceSet(
        query_id="q-1",
        evidence=(evidence("ev-1"), evidence("ev-2")),
    )

    result = AnswerRepairer().repair(draft, report, selected)

    assert result.cited_evidence_ids == ("ev-1", "ev-2")


def test_repair_rejects_report_missing_a_faithful_claim() -> None:
    draft = DraftAnswer(
        query_id="q-1",
        answer_text="Revenue rose.",
        claims=(
            Claim(
                claim_id="claim-1",
                text="Revenue rose.",
                span=ClaimSpan(start=0, end=13),
                faithful_to_answer=True,
            ),
        ),
    )
    report = VerificationReport(query_id="q-1", claims=(), fact_coverage=())
    selected = SelectedEvidenceSet(query_id="q-1", evidence=())

    with pytest.raises(ValueError, match="verifiable claims"):
        AnswerRepairer().repair(draft, report, selected)


def test_repair_rejects_report_citing_unselected_evidence() -> None:
    draft = DraftAnswer(
        query_id="q-1",
        answer_text="Revenue rose.",
        claims=(
            Claim(
                claim_id="claim-1",
                text="Revenue rose.",
                span=ClaimSpan(start=0, end=13),
                faithful_to_answer=True,
            ),
        ),
    )
    report = VerificationReport(
        query_id="q-1",
        claims=(
            ClaimVerification(
                claim_id="claim-1",
                status="supported",
                supporting_evidence_ids=("ev-unknown",),
                entity_consistent=True,
                contradicted=False,
            ),
        ),
        fact_coverage=(),
    )
    selected = SelectedEvidenceSet(
        query_id="q-1",
        evidence=(evidence("ev-1"),),
    )

    with pytest.raises(ValueError, match="unselected evidence"):
        AnswerRepairer().repair(draft, report, selected)


def test_repair_refuses_nonempty_draft_with_no_verifiable_claims() -> None:
    draft = DraftAnswer(query_id="q-1", answer_text="General summary.", claims=())
    report = VerificationReport(query_id="q-1", claims=(), fact_coverage=())
    selected = SelectedEvidenceSet(query_id="q-1", evidence=())

    result = AnswerRepairer().repair(draft, report, selected)

    assert result.answer == ""
    assert result.cited_evidence_ids == ()


@pytest.mark.parametrize(
    ("trusted_ids", "expected"),
    (
        ({"claim-2", "claim-3"}, "Beta fell. Gamma stayed."),
        ({"claim-1", "claim-3"}, "Alpha rose. Gamma stayed."),
        ({"claim-1", "claim-2"}, "Alpha rose. Beta fell."),
        ({"claim-3"}, "Gamma stayed."),
    ),
)
def test_repair_assembles_remaining_claims_after_boundary_deletions(
    trusted_ids: set[str],
    expected: str,
) -> None:
    answer = "Alpha rose. Beta fell. Gamma stayed."
    claims = (
        Claim(
            claim_id="claim-1",
            text="Alpha rose.",
            span=ClaimSpan(start=0, end=11),
            faithful_to_answer=True,
        ),
        Claim(
            claim_id="claim-2",
            text="Beta fell.",
            span=ClaimSpan(start=12, end=22),
            faithful_to_answer=True,
        ),
        Claim(
            claim_id="claim-3",
            text="Gamma stayed.",
            span=ClaimSpan(start=23, end=36),
            faithful_to_answer=True,
        ),
    )
    verdicts = tuple(
        ClaimVerification(
            claim_id=item.claim_id,
            status="supported" if item.claim_id in trusted_ids else "unsupported",
            supporting_evidence_ids=(f"ev-{index}",)
            if item.claim_id in trusted_ids
            else (),
            entity_consistent=True,
            contradicted=False,
        )
        for index, item in enumerate(claims, start=1)
    )
    selected = SelectedEvidenceSet(
        query_id="q-1",
        evidence=tuple(evidence(f"ev-{index}") for index in range(1, 4)),
    )

    result = AnswerRepairer().repair(
        DraftAnswer(query_id="q-1", answer_text=answer, claims=claims),
        VerificationReport(query_id="q-1", claims=verdicts, fact_coverage=()),
        selected,
    )

    assert result.answer == expected


def test_repair_normalizes_whitespace_between_and_within_retained_spans() -> None:
    draft = DraftAnswer(
        query_id="q-1",
        answer_text="Revenue\nrose. Unsupported.",
        claims=(
            Claim(
                claim_id="claim-1",
                text="Revenue rose.",
                span=ClaimSpan(start=0, end=13),
                faithful_to_answer=True,
            ),
            Claim(
                claim_id="claim-2",
                text="Unsupported.",
                span=ClaimSpan(start=14, end=26),
                faithful_to_answer=True,
            ),
        ),
    )
    report = VerificationReport(
        query_id="q-1",
        claims=(
            ClaimVerification(
                claim_id="claim-1",
                status="supported",
                supporting_evidence_ids=("ev-1",),
                entity_consistent=True,
                contradicted=False,
            ),
            ClaimVerification(
                claim_id="claim-2",
                status="unsupported",
                supporting_evidence_ids=(),
                entity_consistent=True,
                contradicted=False,
            ),
        ),
        fact_coverage=(),
    )

    result = AnswerRepairer().repair(
        draft,
        report,
        SelectedEvidenceSet(query_id="q-1", evidence=(evidence("ev-1"),)),
    )

    assert result.answer == "Revenue rose."


def test_repair_removes_leading_connector_left_by_deleted_claim() -> None:
    draft = DraftAnswer(
        query_id="q-1",
        answer_text="Revenue fell. However, profit rose.",
        claims=(
            Claim(
                claim_id="claim-1",
                text="Revenue fell.",
                span=ClaimSpan(start=0, end=13),
                faithful_to_answer=True,
            ),
            Claim(
                claim_id="claim-2",
                text="Profit rose.",
                span=ClaimSpan(start=14, end=35),
                faithful_to_answer=True,
            ),
        ),
    )
    report = VerificationReport(
        query_id="q-1",
        claims=(
            ClaimVerification(
                claim_id="claim-1",
                status="unsupported",
                supporting_evidence_ids=(),
                entity_consistent=True,
                contradicted=False,
            ),
            ClaimVerification(
                claim_id="claim-2",
                status="supported",
                supporting_evidence_ids=("ev-2",),
                entity_consistent=True,
                contradicted=False,
            ),
        ),
        fact_coverage=(),
    )

    result = AnswerRepairer().repair(
        draft,
        report,
        SelectedEvidenceSet(query_id="q-1", evidence=(evidence("ev-2"),)),
    )

    assert result.answer == "Profit rose."


def test_repair_keeps_connector_when_previous_claim_remains() -> None:
    draft = DraftAnswer(
        query_id="q-1",
        answer_text="Revenue fell. However, profit rose. Forecast failed.",
        claims=(
            Claim(
                claim_id="claim-1",
                text="Revenue fell.",
                span=ClaimSpan(start=0, end=13),
                faithful_to_answer=True,
            ),
            Claim(
                claim_id="claim-2",
                text="Profit rose.",
                span=ClaimSpan(start=14, end=35),
                faithful_to_answer=True,
            ),
            Claim(
                claim_id="claim-3",
                text="Forecast failed.",
                span=ClaimSpan(start=36, end=52),
                faithful_to_answer=True,
            ),
        ),
    )
    report = VerificationReport(
        query_id="q-1",
        claims=(
            ClaimVerification(
                claim_id="claim-1",
                status="supported",
                supporting_evidence_ids=("ev-1",),
                entity_consistent=True,
                contradicted=False,
            ),
            ClaimVerification(
                claim_id="claim-2",
                status="supported",
                supporting_evidence_ids=("ev-2",),
                entity_consistent=True,
                contradicted=False,
            ),
            ClaimVerification(
                claim_id="claim-3",
                status="unsupported",
                supporting_evidence_ids=(),
                entity_consistent=True,
                contradicted=False,
            ),
        ),
        fact_coverage=(),
    )

    result = AnswerRepairer().repair(
        draft,
        report,
        SelectedEvidenceSet(
            query_id="q-1",
            evidence=(evidence("ev-1"), evidence("ev-2")),
        ),
    )

    assert result.answer == "Revenue fell. However, profit rose."
