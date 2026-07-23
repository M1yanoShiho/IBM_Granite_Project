"""B2 -- per-claim attribution: which selected evidence actually supports a claim.

Every claim is scored against *every* selected evidence with no early exit
(plan section 3, "第一版全跑不剪枝"): stopping at the first entailment would
throw away the contradiction signal that a later evidence carries, and that
signal is one of the module's headline metrics.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from evidence_rag.contracts.models import SelectedEvidenceSet
from evidence_rag.generator.entity_check import (
    EntityChecker,
    EntityConsistencyChecker,
    EntityMismatch,
)
from evidence_rag.generator.models import Claim, ClaimVerification
from evidence_rag.generator.nli import NLIModel


@dataclass(frozen=True)
class EntityMismatchRecord:
    """One claim x evidence pair that NLI entailed but the entity check rejected.

    ``VerificationReport`` only keeps the boolean, so this is the channel for the
    "反事实干扰捕获" case study in plan section 6.
    """

    claim_id: str
    evidence_id: str
    mismatches: tuple[EntityMismatch, ...]


class Attributor:
    def __init__(
        self,
        nli: NLIModel,
        entity_checker: EntityChecker | None = None,
        on_entity_mismatch: Callable[[EntityMismatchRecord], None] | None = None,
    ) -> None:
        self.nli = nli
        self.entity_checker = entity_checker or EntityConsistencyChecker()
        self.on_entity_mismatch = on_entity_mismatch

    def verify_claim(self, claim: Claim, selected: SelectedEvidenceSet) -> ClaimVerification:
        supporting: list[str] = []
        seen: set[str] = set()
        contradicted = False
        entity_mismatch = False

        for item in selected.evidence:
            label = self.nli.classify(premise=item.text, hypothesis=claim.text)
            if label == "contradiction":
                contradicted = True
                continue
            if label != "entailment":
                continue
            consistency = self.entity_checker.check(claim.text, item.text)
            if consistency.consistent:
                if item.evidence_id not in seen:
                    seen.add(item.evidence_id)
                    supporting.append(item.evidence_id)
                continue
            entity_mismatch = True
            if self.on_entity_mismatch is not None:
                self.on_entity_mismatch(
                    EntityMismatchRecord(
                        claim_id=claim.claim_id,
                        evidence_id=item.evidence_id,
                        mismatches=consistency.mismatches,
                    )
                )

        if supporting:
            # ``entity_consistent`` reports *why* a claim ended up unsupported, so
            # a claim that some other evidence supports cleanly stays consistent
            # even if a different evidence had a swapped entity -- which is also
            # what ClaimVerification's own validator requires. The per-pair
            # mismatches are not lost: they went to ``on_entity_mismatch``.
            return ClaimVerification(
                claim_id=claim.claim_id,
                status="supported",
                supporting_evidence_ids=tuple(supporting),
                entity_consistent=True,
                contradicted=contradicted,
            )
        return ClaimVerification(
            claim_id=claim.claim_id,
            status="unsupported",
            supporting_evidence_ids=(),
            entity_consistent=not entity_mismatch,
            contradicted=contradicted,
        )

    def verify_claims(
        self,
        claims: Iterable[Claim],
        selected: SelectedEvidenceSet,
    ) -> tuple[ClaimVerification, ...]:
        """Verify the faithful claims only.

        A claim A2 flagged ``faithful_to_answer=False`` is a splitter
        hallucination, not an answer error: verifying it would produce a verdict
        that could wrongly drive A3 to edit an answer span that was never wrong.
        ``validate_verification_report`` enforces this same exclusion.
        """
        return tuple(
            self.verify_claim(claim, selected) for claim in claims if claim.faithful_to_answer
        )
