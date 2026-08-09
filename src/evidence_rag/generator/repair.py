"""ABLATION-ONLY. Not on the live path.

Part of the retired generate -> verify -> patch design (roles A/B), reached only
through ``verified.VerifiedGenerator``, which exists solely as the ``verify-only``
arm -- the published delete-filter baseline the main method is measured against.
Deleting it would make that comparison irreproducible. The live path is
``verify_annotate.VerifyAnnotateGenerator``; do not extend this code. See
``docs/generator/design-review.md``.
"""

import re

from evidence_rag.contracts.models import GenerationResult, SelectedEvidenceSet
from evidence_rag.generator.models import (
    ClaimVerification,
    DraftAnswer,
    VerificationReport,
    validate_supporting_evidence,
    validate_verification_report,
)

LEADING_CONNECTOR = re.compile(
    r"^(?:and|but|however|therefore|moreover|also|meanwhile|instead)\s*,?\s+",
    re.IGNORECASE,
)


class AnswerRepairer:
    """Remove untrusted draft claims and emit citations verified by B."""

    def repair(
        self,
        draft: DraftAnswer,
        report: VerificationReport,
        selected: SelectedEvidenceSet,
    ) -> GenerationResult:
        if draft.query_id != selected.query_id:
            raise ValueError("draft answer and selected evidence query IDs differ")
        validate_verification_report(draft, report)
        validate_supporting_evidence(selected, report)

        verifications = {item.claim_id: item for item in report.claims}
        trusted_claims = tuple(
            claim
            for claim in draft.claims
            if claim.faithful_to_answer
            and self._is_trusted(verifications[claim.claim_id])
        )
        trusted_ids = {claim.claim_id for claim in trusted_claims}

        citations: list[str] = []
        seen: set[str] = set()
        for claim in draft.claims:
            if claim.claim_id not in trusted_ids:
                continue
            for evidence_id in verifications[claim.claim_id].supporting_evidence_ids:
                if evidence_id not in seen:
                    seen.add(evidence_id)
                    citations.append(evidence_id)

        answer = self._assemble_retained_spans(draft, trusted_ids)
        return GenerationResult(
            query_id=draft.query_id,
            answer=answer,
            cited_evidence_ids=tuple(citations),
        )

    @staticmethod
    def _is_trusted(verification: ClaimVerification) -> bool:
        return (
            verification.status == "supported"
            and verification.entity_consistent
            and not verification.contradicted
        )

    @staticmethod
    def _normalize_fragment(fragment: str) -> str:
        return " ".join(fragment.split())

    @classmethod
    def _assemble_retained_spans(
        cls,
        draft: DraftAnswer,
        trusted_ids: set[str],
    ) -> str:
        fragments: list[str] = []
        for index, claim in enumerate(draft.claims):
            if claim.claim_id not in trusted_ids:
                continue
            fragment = cls._normalize_fragment(
                draft.answer_text[claim.span.start : claim.span.end]
            )
            previous_was_removed = (
                index > 0 and draft.claims[index - 1].claim_id not in trusted_ids
            )
            if previous_was_removed:
                fragment = LEADING_CONNECTOR.sub("", fragment, count=1)
                if fragment:
                    fragment = fragment[0].upper() + fragment[1:]
            fragments.append(fragment)
        return " ".join(fragments)
