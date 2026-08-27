import pytest

from evidence_rag.contracts.models import (
    EvidenceCandidate,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
)
from evidence_rag.generator.draft import DraftAnswerGenerator
from evidence_rag.generator.evidence_recheck import EvidenceRecheckResult
from evidence_rag.generator.models import (
    Claim,
    ClaimSpan,
    ClaimVerification,
    DraftAnswer,
    RequiredFactCoverage,
    VerificationReport,
)
from evidence_rag.generator.verified import (
    VerifiedGenerator,
    format_unconfirmed_disclosure,
    is_unconfirmed_disclosure,
)


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

    def generate(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> DraftAnswer:
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
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> EvidenceRecheckResult:
        self.calls += 1
        return self.result


class SequenceLLM:
    def __init__(self, *responses: str) -> None:
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return next(self.responses)


def test_verified_generator_runs_real_a1_a2_and_a3_components_together() -> None:
    llm = SequenceLLM(
        "Revenue rose 8%.",
        '{"claims":[{"source_text":"Revenue rose 8%.",'
        '"text":"Revenue rose by 8%."}]}',
        '{"results":[{"claim_id":"claim-1","faithful":true}]}',
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
        fact_coverage=(
            RequiredFactCoverage(required_fact="revenue change", covered=True),
        ),
    )
    generator = VerifiedGenerator(
        draft_generator=DraftAnswerGenerator(llm=llm),
        verifier=FixedVerifier(report),
        evidence_rechecker=FixedRechecker(
            EvidenceRecheckResult(
                required_fact="unused",
                found=False,
                answer_fragment="",
                evidence_ids=(),
            )
        ),
    )
    selected = SelectedEvidenceSet(
        query_id="q-1",
        evidence=(evidence("ev-1", "Revenue rose 8%."),),
    )

    result = generator.generate(
        Query(query_id="q-1", text="How did revenue change?"),
        QueryChecklist(
            query_id="q-1",
            focus="2024 performance",
            required_facts=("revenue change",),
            constraints=("exclude forecasts",),
        ),
        selected,
    )

    assert result.answer == "Revenue rose 8%."
    assert result.cited_evidence_ids == ("ev-1",)
    # the draft prompt no longer carries the checklist: completeness is retired as
    # a runtime mechanism and comprehensiveness is asked for directly
    assert "Cover every part of the question" in llm.prompts[0]


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


def test_verified_generator_does_not_append_duplicate_recheck_fragment() -> None:
    draft = DraftAnswer(
        query_id="q-1",
        answer_text="Profit remained stable.",
        claims=(
            Claim(
                claim_id="claim-1",
                text="Profit remained stable.",
                span=ClaimSpan(start=0, end=23),
                faithful_to_answer=True,
            ),
        ),
    )
    generator = VerifiedGenerator(
        draft_generator=FixedDraftGenerator(draft),
        verifier=FixedVerifier(
            VerificationReport(
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
                fact_coverage=(
                    RequiredFactCoverage(
                        required_fact="profit change",
                        covered=False,
                        gap_question="How did profit change?",
                    ),
                ),
            )
        ),
        evidence_rechecker=FixedRechecker(
            EvidenceRecheckResult(
                required_fact="profit change",
                found=True,
                answer_fragment="  Profit remained stable.  ",
                evidence_ids=("ev-1",),
            )
        ),
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

    assert result.answer == "Profit remained stable."
    assert result.cited_evidence_ids == ("ev-1",)


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


def test_verified_generator_answers_partially_and_discloses_the_missing_fact() -> None:
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
    generator = VerifiedGenerator(
        draft_generator=FixedDraftGenerator(draft),
        verifier=FixedVerifier(
            VerificationReport(
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
        ),
        evidence_rechecker=FixedRechecker(
            EvidenceRecheckResult(
                required_fact="profit change",
                found=False,
                answer_fragment="",
                evidence_ids=(),
            )
        ),
    )

    result = generator.generate(
        Query(query_id="q-1", text="How did the company perform?"),
        QueryChecklist(
            query_id="q-1",
            focus="performance",
            required_facts=("profit change",),
        ),
        SelectedEvidenceSet(
            query_id="q-1",
            evidence=(evidence("ev-1", "Revenue rose 8%."),),
        ),
    )

    # partial answering: the confirmed claim is kept, the unconfirmable required
    # fact is disclosed rather than costing the whole answer
    assert result.answer.startswith("Revenue rose 8%.")
    assert is_unconfirmed_disclosure(result.answer.split("Revenue rose 8%.")[1].strip())
    assert "profit change" in result.answer
    assert result.cited_evidence_ids == ("ev-1",)


def test_verified_generator_abstains_when_nothing_at_all_is_confirmable() -> None:
    """The disclosure never stands alone: it carries no citation, and a non-empty
    answer without citations would violate GenerationResult."""
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
    generator = VerifiedGenerator(
        draft_generator=FixedDraftGenerator(draft),
        verifier=FixedVerifier(
            VerificationReport(
                query_id="q-1",
                claims=(
                    ClaimVerification(
                        claim_id="claim-1",
                        status="unsupported",  # nothing survives repair
                        supporting_evidence_ids=(),
                        entity_consistent=True,
                        contradicted=False,
                    ),
                ),
                fact_coverage=(
                    RequiredFactCoverage(
                        required_fact="profit change",
                        covered=False,
                        gap_question="How did profit change?",
                    ),
                ),
            )
        ),
        evidence_rechecker=FixedRechecker(
            EvidenceRecheckResult(
                required_fact="profit change",
                found=False,
                answer_fragment="",
                evidence_ids=(),
            )
        ),
    )

    result = generator.generate(
        Query(query_id="q-1", text="How did the company perform?"),
        QueryChecklist(query_id="q-1", focus="performance", required_facts=("profit change",)),
        SelectedEvidenceSet(query_id="q-1", evidence=(evidence("ev-1", "Revenue rose 8%."),)),
    )

    assert result.answer == ""
    assert result.cited_evidence_ids == ()


def test_malformed_recheck_is_downgraded_to_unconfirmed_not_fatal() -> None:
    """A malformed recheck response cost 5.5% of G3 queries entirely, and those
    failures skewed toward harder cases. Under partial answering it becomes
    'this fact stayed unconfirmed' and is reported through the sink."""

    class ExplodingRechecker:
        def recheck(self, coverage, checklist, selected):  # type: ignore[no-untyped-def]
            raise ValueError("recheck output must contain a boolean 'found'")

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
    seen: list[tuple[str, Exception]] = []
    generator = VerifiedGenerator(
        draft_generator=FixedDraftGenerator(draft),
        verifier=FixedVerifier(
            VerificationReport(
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
                fact_coverage=(
                    RequiredFactCoverage(
                        required_fact="profit change",
                        covered=False,
                        gap_question="How did profit change?",
                    ),
                ),
            )
        ),
        evidence_rechecker=ExplodingRechecker(),
        on_recheck_error=lambda fact, exc: seen.append((fact, exc)),
    )

    result = generator.generate(
        Query(query_id="q-1", text="How did the company perform?"),
        QueryChecklist(query_id="q-1", focus="performance", required_facts=("profit change",)),
        SelectedEvidenceSet(query_id="q-1", evidence=(evidence("ev-1", "Revenue rose 8%."),)),
    )

    assert result.answer.startswith("Revenue rose 8%.")
    assert "profit change" in result.answer  # disclosed, not silently dropped
    assert result.cited_evidence_ids == ("ev-1",)
    assert [fact for fact, _ in seen] == ["profit change"]


def test_disclosure_helpers_round_trip_and_are_recognisable() -> None:
    disclosure = format_unconfirmed_disclosure(("profit change", "headcount."))

    assert disclosure == (
        "Not confirmed from the provided documents: profit change; headcount."
    )
    assert is_unconfirmed_disclosure(disclosure)
    assert not is_unconfirmed_disclosure("Revenue rose 8%.")
    assert format_unconfirmed_disclosure(()) == ""


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
