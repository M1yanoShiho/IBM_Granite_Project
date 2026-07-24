from typing import Protocol

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
from evidence_rag.generator.nli import DebertaNLIModel, NLIModel
from evidence_rag.generator.repair import AnswerRepairer
from evidence_rag.generator.verifier import Verifier


class DraftProducer(Protocol):
    def generate(self, query: Query, selected: SelectedEvidenceSet) -> DraftAnswer: ...


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
    """Run one draft -> verify -> repair -> evidence-recheck pass."""

    def __init__(
        self,
        draft_generator: DraftProducer | None = None,
        verifier: VerificationProvider | None = None,
        evidence_rechecker: RecheckProvider | None = None,
        repairer: RepairProvider | None = None,
        *,
        llm: TextGenerator | None = None,
        nli: NLIModel | None = None,
    ) -> None:
        shared_llm = llm
        if draft_generator is None or verifier is None or evidence_rechecker is None:
            shared_llm = shared_llm or GraniteLLMClient()
        self.draft_generator = draft_generator or DraftAnswerGenerator(llm=shared_llm)
        self.verifier = verifier or Verifier(
            nli or DebertaNLIModel(),
            llm=shared_llm,
        )
        self.evidence_rechecker = evidence_rechecker or EvidenceRechecker(llm=shared_llm)
        self.repairer = repairer or AnswerRepairer()

    def generate(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> GenerationResult:
        if query.query_id != checklist.query_id or query.query_id != selected.query_id:
            raise ValueError("query, checklist, and selected evidence query IDs differ")

        draft = self.draft_generator.generate(query, selected)
        report = self.verifier.verify(draft, selected, checklist)
        repaired = self.repairer.repair(draft, report, selected)

        fragments: list[str] = []
        citations = list(repaired.cited_evidence_ids)
        seen = set(citations)
        allowed_evidence_ids = {item.evidence_id for item in selected.evidence}
        for coverage in report.fact_coverage:
            if coverage.covered:
                continue
            result = self.evidence_rechecker.recheck(coverage, selected)
            if result.required_fact != coverage.required_fact:
                raise ValueError("recheck result does not match the requested fact")
            unknown = set(result.evidence_ids) - allowed_evidence_ids
            if unknown:
                raise ValueError(
                    f"recheck cites unselected evidence: {sorted(unknown)}"
                )
            if not result.found:
                continue
            fragments.append(result.answer_fragment)
            for evidence_id in result.evidence_ids:
                if evidence_id not in seen:
                    seen.add(evidence_id)
                    citations.append(evidence_id)

        parts = [part for part in (repaired.answer.strip(), *fragments) if part]
        return GenerationResult(
            query_id=query.query_id,
            answer=" ".join(parts),
            cited_evidence_ids=tuple(citations),
        )
