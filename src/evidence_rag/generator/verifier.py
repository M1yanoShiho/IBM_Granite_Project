"""ABLATION-ONLY. Not on the live path.

Part of the retired generate -> verify -> patch design (roles A/B), reached only
through ``verified.VerifiedGenerator``, which exists solely as the ``verify-only``
arm -- the published delete-filter baseline the main method is measured against.
Deleting it would make that comparison irreproducible. The live path is
``verify_annotate.VerifyAnnotateGenerator``; do not extend this code. See
``docs/generator/design-review.md``.

B5 -- assemble the VerificationReport that A consumes to patch the draft.

Single pass, verification only: no patching (A3/A4) and no iteration (plan
section 9, "只跑一轮"). The report is validated against its own inputs before it
leaves this module, so a bug here surfaces as a failed cross-check rather than
as a silently wrong edit to the answer downstream.
"""

from collections.abc import Callable

from evidence_rag.contracts.models import QueryChecklist, SelectedEvidenceSet
from evidence_rag.generator.attribution import Attributor, EntityMismatchRecord
from evidence_rag.generator.completeness import CompletenessChecker
from evidence_rag.generator.entity_check import EntityChecker
from evidence_rag.generator.granite import TextGenerator
from evidence_rag.generator.models import (
    DraftAnswer,
    VerificationReport,
    validate_fact_coverage,
    validate_supporting_evidence,
    validate_verification_report,
)
from evidence_rag.generator.nli import NLIModel


class Verifier:
    """B's entry point: ``DraftAnswer`` + ``SelectedEvidenceSet`` + ``QueryChecklist``
    -> ``VerificationReport``."""

    def __init__(
        self,
        nli: NLIModel,
        entity_checker: EntityChecker | None = None,
        llm: TextGenerator | None = None,
        completeness_checker: CompletenessChecker | None = None,
        on_entity_mismatch: Callable[[EntityMismatchRecord], None] | None = None,
    ) -> None:
        self.attributor = Attributor(
            nli,
            entity_checker=entity_checker,
            on_entity_mismatch=on_entity_mismatch,
        )
        self.completeness_checker = completeness_checker or CompletenessChecker(llm=llm)

    def verify(
        self,
        draft: DraftAnswer,
        selected: SelectedEvidenceSet,
        checklist: QueryChecklist,
    ) -> VerificationReport:
        if draft.query_id != selected.query_id:
            raise ValueError("draft answer and selected evidence query IDs differ")
        if draft.query_id != checklist.query_id:
            raise ValueError("draft answer and checklist query IDs differ")

        report = VerificationReport(
            query_id=draft.query_id,
            claims=self.attributor.verify_claims(draft.claims, selected),
            fact_coverage=self.completeness_checker.check(draft.answer_text, checklist),
        )

        validate_verification_report(draft, report)
        validate_supporting_evidence(selected, report)
        validate_fact_coverage(checklist, report)
        return report
