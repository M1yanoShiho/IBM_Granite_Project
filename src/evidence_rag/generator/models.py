"""Internal Generator interfaces: DraftAnswer / VerificationReport.

These types are the frozen handoff between the two Generator sub-roles in
docs/generator/generator-Development Plan.md (section 5):

    A (generation & orchestration)  -- produces DraftAnswer
    B (verification & attribution)  -- consumes DraftAnswer, produces VerificationReport
    A                                -- consumes VerificationReport to patch/assemble
                                        the final GenerationResult

They are NOT part of the cross-module contracts in `evidence_rag.contracts` and
must never leak outside this package -- `GenerationResult` stays the only object
Selector/Pipeline/other modules see (plan section 2, "范围边界").
"""

from typing import Literal

from pydantic import Field, model_validator

from evidence_rag.contracts.models import FrozenModel, NonEmpty, QueryChecklist, SelectedEvidenceSet

AttributionStatus = Literal["supported", "unsupported"]


class ClaimSpan(FrozenModel):
    """Character offsets ``[start, end)`` into ``DraftAnswer.answer_text`` that a
    claim was extracted from.

    Needed because ``Claim.text`` is rewritten to be atomic and self-contained
    (coreference resolved), so it is not necessarily a literal substring of the
    answer anymore. A3 uses the span, not the text, to locate what to delete or
    patch during "删假/补漏" assembly.
    """

    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def _end_after_start(self) -> "ClaimSpan":
        if self.end <= self.start:
            raise ValueError("claim span end must be greater than start")
        return self


class Claim(FrozenModel):
    """One atomic, self-contained statement extracted from a DraftAnswer (A2)."""

    schema_version: Literal["1.0"] = "1.0"
    claim_id: NonEmpty
    text: NonEmpty
    span: ClaimSpan
    faithful_to_answer: bool
    """A2's own self-check: does ``text`` still mean what the source span said,
    after being rewritten for atomicity/self-containment? False marks a
    splitter hallucination -- surfaced here rather than silently trusted, so it
    can be excluded (or flagged) before B spends an NLI pass verifying it.
    """


class DraftAnswer(FrozenModel):
    """A's output: an initial answer plus the claims extracted from it."""

    schema_version: Literal["1.0"] = "1.0"
    query_id: NonEmpty
    answer_text: str
    """Empty string represents the decline-to-answer case (plan A5); an empty
    ``answer_text`` must carry no claims, mirroring the ``GenerationResult``
    contract's empty-answer/no-citations rule."""
    claims: tuple[Claim, ...]

    @model_validator(mode="after")
    def _claims_are_consistent(self) -> "DraftAnswer":
        if not self.answer_text.strip() and self.claims:
            raise ValueError("empty answer_text cannot have claims")
        ids = tuple(claim.claim_id for claim in self.claims)
        if len(ids) != len(set(ids)):
            raise ValueError("claim IDs must be unique")
        length = len(self.answer_text)
        for claim in self.claims:
            if claim.span.end > length:
                raise ValueError(f"claim {claim.claim_id!r} span exceeds answer_text length")
        return self


class ClaimVerification(FrozenModel):
    """B's per-claim verdict (B2 attribution + B3 entity check).

    ``claim_id`` must reference a ``Claim.claim_id`` from the ``DraftAnswer``
    this report was produced against -- see ``validate_verification_report``.
    """

    schema_version: Literal["1.0"] = "1.0"
    claim_id: NonEmpty
    status: AttributionStatus
    supporting_evidence_ids: tuple[NonEmpty, ...]
    """Evidence IDs where NLI judged entailment AND the entity check (plan
    section 4) passed. Non-empty iff ``status == "supported"``."""
    entity_consistent: bool
    """False when the best entailing evidence still had a mismatched entity --
    this is what downgraded an otherwise-entailed claim to "unsupported" (plan
    section 4, step 4). Stays True when nothing entailed at all, since there is
    no entity match to have failed."""
    contradicted: bool
    """True if at least one evidence directly contradicts the claim (NLI
    contradiction), recorded independently of ``status`` so contradiction cases
    can be reported/analysed separately (plan section 6, "contradiction 捕获数")."""

    @model_validator(mode="after")
    def _support_matches_status(self) -> "ClaimVerification":
        ids = self.supporting_evidence_ids
        if len(ids) != len(set(ids)):
            raise ValueError("supporting evidence IDs must be unique")
        if self.status == "supported":
            if not ids:
                raise ValueError("supported claim must have supporting evidence")
            if not self.entity_consistent:
                raise ValueError("supported claim must be entity-consistent")
        elif ids:
            raise ValueError("unsupported claim cannot have supporting evidence")
        return self


class RequiredFactCoverage(FrozenModel):
    """B4's completeness verdict for one ``QueryChecklist.required_facts`` entry."""

    schema_version: Literal["1.0"] = "1.0"
    required_fact: NonEmpty
    covered: bool
    gap_question: NonEmpty | None = None
    """Set iff ``covered`` is False: a concrete sub-question A4 uses to re-search
    the already-selected evidence (plan section 3 step 5) -- never a new
    retrieval call."""

    @model_validator(mode="after")
    def _gap_question_matches_coverage(self) -> "RequiredFactCoverage":
        if self.covered and self.gap_question is not None:
            raise ValueError("covered fact cannot carry a gap question")
        if not self.covered and self.gap_question is None:
            raise ValueError("uncovered fact requires a gap question")
        return self


class VerificationReport(FrozenModel):
    """B's output: per-claim attribution verdicts + per-required-fact coverage."""

    schema_version: Literal["1.0"] = "1.0"
    query_id: NonEmpty
    claims: tuple[ClaimVerification, ...]
    fact_coverage: tuple[RequiredFactCoverage, ...]

    @model_validator(mode="after")
    def _unique_keys(self) -> "VerificationReport":
        claim_ids = tuple(item.claim_id for item in self.claims)
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim verifications must have unique claim IDs")
        facts = tuple(item.required_fact for item in self.fact_coverage)
        if len(facts) != len(set(facts)):
            raise ValueError("fact coverage entries must have unique required_fact values")
        return self


def validate_verification_report(draft: DraftAnswer, report: VerificationReport) -> None:
    """Cross-check that ``report`` was produced against exactly this ``draft``:
    same query, one verdict per claim, no more and no less.

    Intended for the A+B integration test (plan section 5, "集成测试").
    """
    if draft.query_id != report.query_id:
        raise ValueError("draft answer and verification report query IDs differ")
    draft_ids = {claim.claim_id for claim in draft.claims}
    report_ids = {item.claim_id for item in report.claims}
    if draft_ids != report_ids:
        raise ValueError("verification report does not cover exactly the draft's claims")


def validate_supporting_evidence(selected: SelectedEvidenceSet, report: VerificationReport) -> None:
    """Cross-check that every evidence ID a claim cites actually belongs to the
    already-selected evidence -- verification must stay inside
    ``SelectedEvidenceSet`` and never reference anything the Generator wasn't
    given (docs/README.md: "Generator 不应该自己重新检索新证据")."""
    allowed = {item.evidence_id for item in selected.evidence}
    for claim in report.claims:
        unknown = set(claim.supporting_evidence_ids) - allowed
        if unknown:
            raise ValueError(f"claim {claim.claim_id!r} cites unselected evidence: {sorted(unknown)}")


def validate_fact_coverage(checklist: QueryChecklist, report: VerificationReport) -> None:
    """Cross-check that ``report.fact_coverage`` covers exactly the checklist's
    ``required_facts`` -- same set, one entry each."""
    expected = set(checklist.required_facts)
    actual = {item.required_fact for item in report.fact_coverage}
    if expected != actual:
        raise ValueError("fact coverage does not match checklist required_facts")
