import pytest

from evidence_rag.contracts.models import (
    EvidenceCandidate,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
)
from evidence_rag.generator.evidence_recheck import EvidenceRecheckResult
from evidence_rag.generator.models import (
    Claim,
    ClaimSpan,
    ClaimVerification,
    DraftAnswer,
    RequiredFactCoverage,
    VerificationReport,
)
from evidence_rag.generator.verified import VerifiedGenerator


def evidence(evidence_id: str, text: str) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=f"doc-{evidence_id}",
        chunk_id=f"chunk-{evidence_id}",
        text=text,
        source_uri=f"fixture://{evidence_id}",
        retrieval_score=1.0,
        retrieval_rank=1,
    )


class FixedDraftGenerator:
    def __init__(self, draft: DraftAnswer) -> None:
        self.draft = draft

    def generate(self, query: Query, selected: SelectedEvidenceSet) -> DraftAnswer:
        return self.draft


class FixedVerifier:
    def __init__(self, report: VerificationReport) -> None:
        self.report = report

    def verify(
        self,
        draft: DraftAnswer,
        selected: SelectedEvidenceSet,
        checklist: QueryChecklist,
    ) -> VerificationReport:
        return self.report


class FixedRechecker:
    def __init__(self, result: EvidenceRecheckResult) -> None:
        self.result = result
        self.calls = 0

    def recheck(
        self,
        coverage: RequiredFactCoverage,
        selected: SelectedEvidenceSet,
    ) -> EvidenceRecheckResult:
        self.calls += 1
        return self.result


def test_verified_generator_keeps_supported_answer_and_adds_found_gap() -> None:
    draft = DraftAnswer(
        query_id="q-1",
        answer_text="Revenue rose 8%.",
        claims=(
            Claim(
                claim_id="claim-1",
                text="Revenue rose 8%.",
                span=ClaimSpan(start=0, end=16),
                faithful_to_answer=True,
            ),
        ),
    )
    coverage = RequiredFactCoverage(
        required_fact="profit change",
        covered=False,
        gap_question="How did profit change?",
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
        fact_coverage=(coverage,),
    )
    rechecker = FixedRechecker(
        EvidenceRecheckResult(
            required_fact="profit change",
            found=True,
            answer_fragment="Profit remained stable.",
            evidence_ids=("ev-2",),
        )
    )
    generator = VerifiedGenerator(
        draft_generator=FixedDraftGenerator(draft),
        verifier=FixedVerifier(report),
        evidence_rechecker=rechecker,
    )
    selected = SelectedEvidenceSet(
        query_id="q-1",
        evidence=(
            evidence("ev-1", "Revenue rose 8%."),
            evidence("ev-2", "Profit remained stable."),
        ),
    )

    result = generator.generate(
        Query(query_id="q-1", text="How did the company perform?"),
        QueryChecklist(
            query_id="q-1",
            focus="company performance",
            required_facts=("profit change",),
        ),
        selected,
    )

    assert result.answer == "Revenue rose 8%. Profit remained stable."
    assert result.cited_evidence_ids == ("ev-1", "ev-2")
    assert rechecker.calls == 1


def test_verified_generator_refuses_when_repair_and_recheck_find_nothing() -> None:
    draft = DraftAnswer(query_id="q-1", answer_text="", claims=())
    coverage = RequiredFactCoverage(
        required_fact="profit change",
        covered=False,
        gap_question="How did profit change?",
    )
    rechecker = FixedRechecker(
        EvidenceRecheckResult(
            required_fact="profit change",
            found=False,
            answer_fragment="",
            evidence_ids=(),
        )
    )
    generator = VerifiedGenerator(
        draft_generator=FixedDraftGenerator(draft),
        verifier=FixedVerifier(
            VerificationReport(
                query_id="q-1",
                claims=(),
                fact_coverage=(coverage,),
            )
        ),
        evidence_rechecker=rechecker,
    )

    result = generator.generate(
        Query(query_id="q-1", text="How did profit change?"),
        QueryChecklist(
            query_id="q-1",
            focus="profit",
            required_facts=("profit change",),
        ),
        SelectedEvidenceSet(
            query_id="q-1",
            evidence=(evidence("ev-1", "No profit figure is reported."),),
        ),
    )

    assert result.answer == ""
    assert result.cited_evidence_ids == ()


def test_verified_generator_rejects_recheck_citing_unselected_evidence() -> None:
    draft = DraftAnswer(query_id="q-1", answer_text="", claims=())
    coverage = RequiredFactCoverage(
        required_fact="profit change",
        covered=False,
        gap_question="How did profit change?",
    )
    generator = VerifiedGenerator(
        draft_generator=FixedDraftGenerator(draft),
        verifier=FixedVerifier(
            VerificationReport(
                query_id="q-1",
                claims=(),
                fact_coverage=(coverage,),
            )
        ),
        evidence_rechecker=FixedRechecker(
            EvidenceRecheckResult(
                required_fact="profit change",
                found=True,
                answer_fragment="Profit remained stable.",
                evidence_ids=("ev-not-selected",),
            )
        ),
    )

    with pytest.raises(ValueError, match="unselected evidence"):
        generator.generate(
            Query(query_id="q-1", text="How did profit change?"),
            QueryChecklist(
                query_id="q-1",
                focus="profit",
                required_facts=("profit change",),
            ),
            SelectedEvidenceSet(
                query_id="q-1",
                evidence=(evidence("ev-1", "Profit remained stable."),),
            ),
        )


def test_verified_generator_skips_recheck_for_covered_fact() -> None:
    draft = DraftAnswer(query_id="q-1", answer_text="", claims=())
    rechecker = FixedRechecker(
        EvidenceRecheckResult(
            required_fact="profit change",
            found=True,
            answer_fragment="This must not be added.",
            evidence_ids=("ev-1",),
        )
    )
    generator = VerifiedGenerator(
        draft_generator=FixedDraftGenerator(draft),
        verifier=FixedVerifier(
            VerificationReport(
                query_id="q-1",
                claims=(),
                fact_coverage=(
                    RequiredFactCoverage(
                        required_fact="profit change",
                        covered=True,
                    ),
                ),
            )
        ),
        evidence_rechecker=rechecker,
    )

    result = generator.generate(
        Query(query_id="q-1", text="How did profit change?"),
        QueryChecklist(
            query_id="q-1",
            focus="profit",
            required_facts=("profit change",),
        ),
        SelectedEvidenceSet(
            query_id="q-1",
            evidence=(evidence("ev-1", "Profit remained stable."),),
        ),
    )

    assert result.answer == ""
    assert rechecker.calls == 0
