"""ABLATION-ONLY. Not on the live path.

Part of the retired generate -> verify -> patch design (roles A/B), reached only
through ``verified.VerifiedGenerator``, which exists solely as the ``verify-only``
arm -- the published delete-filter baseline the main method is measured against.
Deleting it would make that comparison irreproducible. The live path is
``verify_annotate.VerifyAnnotateGenerator``; do not extend this code. See
``docs/generator/design-review.md``.
"""

from collections.abc import Callable, Sequence
from typing import Protocol

from pydantic import ValidationError

from evidence_rag.contracts.models import (
    GenerationResult,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
)
from evidence_rag.generator.draft import DraftAnswerGenerator
from evidence_rag.generator.evidence_recheck import (
    EvidenceRechecker,
    EvidenceRecheckResult,
)
from evidence_rag.generator.granite import GraniteLLMClient, TextGenerator
from evidence_rag.generator.models import (
    DraftAnswer,
    RequiredFactCoverage,
    VerificationReport,
)
from evidence_rag.generator.nli import NLIModel, build_nli_model
from evidence_rag.generator.repair import AnswerRepairer
from evidence_rag.generator.verifier import Verifier

UNCONFIRMED_DISCLOSURE_PREFIX = "Not confirmed from the provided documents:"
"""Opening of the sentence that discloses required facts the evidence could not
confirm. Kept as a constant because citation scoring has to recognise -- and
drop -- exactly this sentence."""


def format_unconfirmed_disclosure(required_facts: Sequence[str]) -> str:
    """One sentence naming the required facts that stayed unconfirmed.

    Answering the confirmable part while stating plainly what could not be
    confirmed is the specified behaviour; refusing the whole question because one
    required fact is missing was an implementation deviation.
    """
    joined = "; ".join(fact.strip().rstrip(".") for fact in required_facts if fact.strip())
    return f"{UNCONFIRMED_DISCLOSURE_PREFIX} {joined}." if joined else ""


def is_unconfirmed_disclosure(sentence: str) -> bool:
    """True for the disclosure sentence produced above.

    **Citation scorers must drop it.** It is meta-text about the answer, carries
    no citation by construction, and scoring it would count a guaranteed-unsupported
    sentence against recall -- re-introducing the very answer-length dilution that
    sentence-level scoring exists to remove.
    """
    return sentence.strip().startswith(UNCONFIRMED_DISCLOSURE_PREFIX)


def strip_unconfirmed_disclosure(answer: str) -> str:
    """The answer with its disclosure sentence removed.

    **Correctness scoring (STR-EM) must read this, not the raw answer.** The
    disclosure *names the unconfirmed fact*, so whenever that wording shares
    tokens with a gold short answer it produces a spurious exact-match -- and only
    in the partial-answering arm, which is precisely where a spurious gain would
    flatter the change. The disclosure is always the trailing part of the answer.
    """
    index = answer.find(UNCONFIRMED_DISCLOSURE_PREFIX)
    return answer[:index].strip() if index >= 0 else answer


class DraftProducer(Protocol):
    def generate(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> DraftAnswer: ...


class VerificationProvider(Protocol):
    def verify(
        self,
        draft: DraftAnswer,
        selected: SelectedEvidenceSet,
        checklist: QueryChecklist,
    ) -> VerificationReport: ...


class RecheckProvider(Protocol):
    def recheck(
        self,
        coverage: RequiredFactCoverage,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> EvidenceRecheckResult: ...


class RepairProvider(Protocol):
    def repair(
        self,
        draft: DraftAnswer,
        report: VerificationReport,
        selected: SelectedEvidenceSet,
    ) -> GenerationResult: ...


class VerifiedGenerator:
    """Run one draft -> verify -> repair -> evidence-recheck pass.

    Answers **partially**: whatever the evidence confirms is answered, and any
    required fact that stayed unconfirmed is disclosed explicitly in a trailing
    sentence (``format_unconfirmed_disclosure``). The chain abstains entirely only
    when nothing at all could be confirmed. Refusing the whole question because a
    single required fact was missing collapsed coverage from 0.552 to 0.331 in G3
    while ~96% of those abstentions were triggered by genuine gaps that recheck
    simply could not fill.
    """

    def __init__(
        self,
        draft_generator: DraftProducer | None = None,
        verifier: VerificationProvider | None = None,
        evidence_rechecker: RecheckProvider | None = None,
        repairer: RepairProvider | None = None,
        *,
        llm: TextGenerator | None = None,
        nli: NLIModel | None = None,
        on_recheck_error: Callable[[str, Exception], None] | None = None,
    ) -> None:
        shared_llm = llm
        if draft_generator is None or verifier is None or evidence_rechecker is None:
            shared_llm = shared_llm or GraniteLLMClient()
        self.draft_generator = draft_generator or DraftAnswerGenerator(llm=shared_llm)
        self.verifier = verifier or Verifier(
            nli or build_nli_model(),
            llm=shared_llm,
        )
        self.evidence_rechecker = evidence_rechecker or EvidenceRechecker(llm=shared_llm)
        self.repairer = repairer or AnswerRepairer()
        self.on_recheck_error = on_recheck_error
        """Observability sink for degraded rechecks -- they are downgraded to
        'unconfirmed', never silently dropped."""

    @staticmethod
    def _normalize_answer_part(text: str) -> str:
        normalized = " ".join(text.split())
        if normalized and normalized[-1] not in ".!?":
            normalized = f"{normalized}."
        return normalized

    def generate(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> GenerationResult:
        if query.query_id != checklist.query_id or query.query_id != selected.query_id:
            raise ValueError("query, checklist, and selected evidence query IDs differ")

        draft = self.draft_generator.generate(query, checklist, selected)
        report = self.verifier.verify(draft, selected, checklist)
        repaired = self.repairer.repair(draft, report, selected)

        repaired_answer = self._normalize_answer_part(repaired.answer)
        fragments: list[str] = []
        seen_fragments = {repaired_answer.casefold()} if repaired_answer else set()
        citations = list(repaired.cited_evidence_ids)
        seen = set(citations)
        allowed_evidence_ids = {item.evidence_id for item in selected.evidence}
        unconfirmed: list[str] = []
        for coverage in report.fact_coverage:
            if coverage.covered:
                continue
            try:
                result = self.evidence_rechecker.recheck(coverage, checklist, selected)
            except (ValueError, ValidationError) as exc:
                # A malformed recheck response is a runtime condition, not a bug:
                # the LLM was asked for JSON and did not comply. Under partial
                # answering the honest reading is "this fact stayed unconfirmed",
                # so it joins the disclosure instead of destroying the whole query.
                # Aborting instead cost 5.5% of G3 queries, and those failures were
                # not random -- they skewed toward harder, more paraphrased cases.
                unconfirmed.append(coverage.required_fact)
                if self.on_recheck_error is not None:
                    self.on_recheck_error(coverage.required_fact, exc)
                continue
            if result.required_fact != coverage.required_fact:
                raise ValueError("recheck result does not match the requested fact")
            unknown = set(result.evidence_ids) - allowed_evidence_ids
            if unknown:
                raise ValueError(
                    f"recheck cites unselected evidence: {sorted(unknown)}"
                )
            if not result.found:
                unconfirmed.append(coverage.required_fact)
                continue
            fragment = self._normalize_answer_part(result.answer_fragment)
            if fragment.casefold() not in seen_fragments:
                seen_fragments.add(fragment.casefold())
                fragments.append(fragment)
            for evidence_id in result.evidence_ids:
                if evidence_id not in seen:
                    seen.add(evidence_id)
                    citations.append(evidence_id)

        parts = [part for part in (repaired_answer, *fragments) if part]
        if not parts:
            # Nothing was confirmable, so there is nothing honest to say. The
            # disclosure cannot stand alone either: it carries no citation, and a
            # non-empty answer without citations violates GenerationResult.
            return GenerationResult(
                query_id=query.query_id,
                answer="",
                cited_evidence_ids=(),
            )
        disclosure = format_unconfirmed_disclosure(unconfirmed)
        if disclosure:
            parts.append(disclosure)
        return GenerationResult(
            query_id=query.query_id,
            answer=" ".join(parts),
            cited_evidence_ids=tuple(citations),
        )
