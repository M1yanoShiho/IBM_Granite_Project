import json

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
from evidence_rag.generator.nli import NLILabel
from evidence_rag.generator.verifier import Verifier

ANSWER = "Acme's revenue was $1.2B in 2024."
SUPPORT_TEXT = "Acme Inc. reported revenue of 1,200 million dollars in 2024."


class FakeNLI:
    def __init__(self, labels: dict[str, NLILabel], default: NLILabel = "neutral") -> None:
        self.labels = labels
        self.default = default

    def classify(self, premise: str, hypothesis: str) -> NLILabel:
        return self.labels.get(premise, self.default)


class FakeLLM:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)

    def generate(self, prompt: str) -> str:
        return self.responses.pop(0)


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


def claim(claim_id: str, text: str, *, faithful_to_answer: bool = True) -> Claim:
    return Claim(
        claim_id=claim_id,
        text=text,
        span=ClaimSpan(start=0, end=len(ANSWER)),
        faithful_to_answer=faithful_to_answer,
    )


def checklist(*required_facts: str) -> QueryChecklist:
    return QueryChecklist(query_id="q-1", focus="revenue", required_facts=required_facts)


def draft(*claims: Claim, answer_text: str = ANSWER) -> DraftAnswer:
    return DraftAnswer(query_id="q-1", answer_text=answer_text, claims=claims)


def selected(*items: EvidenceCandidate) -> SelectedEvidenceSet:
    return SelectedEvidenceSet(query_id="q-1", evidence=items)


def test_report_passes_every_cross_check() -> None:
    support = evidence("ev-1", SUPPORT_TEXT)
    evidence_set = selected(support)
    draft_answer = draft(
        claim("claim-1", ANSWER),
        claim("claim-2", "Acme was founded by Globex.", faithful_to_answer=False),
    )
    task = checklist("global revenue 2024", "UK revenue 2024")
    verifier = Verifier(
        FakeNLI({SUPPORT_TEXT: "entailment"}),
        llm=FakeLLM(
            json.dumps({"covered": True}),
            json.dumps({"covered": False, "gap_question": "What was UK revenue in 2024?"}),
        ),
    )

    report = verifier.verify(draft_answer, evidence_set, task)

    # The three helpers run inside verify(); asserting them here documents the
    # contract B owes A rather than trusting the call site.
    validate_verification_report(draft_answer, report)
    validate_supporting_evidence(evidence_set, report)
    validate_fact_coverage(task, report)

    assert [item.claim_id for item in report.claims] == ["claim-1"]
    assert report.claims[0].status == "supported"
    assert report.claims[0].supporting_evidence_ids == ("ev-1",)
    assert [item.covered for item in report.fact_coverage] == [True, False]


def test_declined_answer_still_reports_coverage_for_every_required_fact() -> None:
    task = checklist("global revenue 2024", "UK revenue 2024")
    empty_draft = draft(answer_text="")
    verifier = Verifier(FakeNLI({}), llm=FakeLLM())

    report = verifier.verify(empty_draft, selected(evidence("ev-1", SUPPORT_TEXT)), task)

    assert report.claims == ()
    assert [item.covered for item in report.fact_coverage] == [False, False]
    validate_verification_report(empty_draft, report)
    validate_fact_coverage(task, report)


def test_entity_swapped_evidence_yields_an_unsupported_claim_end_to_end() -> None:
    swapped = evidence("ev-1", "Globex reported revenue of $1.2B in 2024.")
    verifier = Verifier(
        FakeNLI({swapped.text: "entailment"}),
        llm=FakeLLM(json.dumps({"covered": True})),
    )

    report = verifier.verify(
        draft(claim("claim-1", ANSWER)),
        selected(swapped),
        checklist("global revenue 2024"),
    )

    assert report.claims[0].status == "unsupported"
    assert report.claims[0].entity_consistent is False


@pytest.mark.parametrize("field", ("selected", "checklist"))
def test_query_id_mismatch_is_rejected(field: str) -> None:
    evidence_set = SelectedEvidenceSet(
        query_id="q-other" if field == "selected" else "q-1",
        evidence=(evidence("ev-1", SUPPORT_TEXT),),
    )
    task = QueryChecklist(
        query_id="q-other" if field == "checklist" else "q-1",
        focus="revenue",
        required_facts=("global revenue 2024",),
    )
    verifier = Verifier(FakeNLI({}), llm=FakeLLM(json.dumps({"covered": True})))

    with pytest.raises(ValueError, match="query IDs differ"):
        verifier.verify(draft(claim("claim-1", ANSWER)), evidence_set, task)


def test_a_verdict_citing_unselected_evidence_is_caught() -> None:
    report = VerificationReport(
        query_id="q-1",
        claims=(
            ClaimVerification(
                claim_id="claim-1",
                status="supported",
                supporting_evidence_ids=("ev-not-selected",),
                entity_consistent=True,
                contradicted=False,
            ),
        ),
        fact_coverage=(RequiredFactCoverage(required_fact="global revenue 2024", covered=True),),
    )

    with pytest.raises(ValueError, match="cites unselected evidence"):
        validate_supporting_evidence(selected(evidence("ev-1", SUPPORT_TEXT)), report)
