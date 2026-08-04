"""Verify-and-annotate: keep unverified content, labelled, instead of deleting it.

The published method is a filter -- generate, verify, delete what fails -- and
filters trade recall for precision by construction. Measured: citation precision
0.762 against baseline's 0.602, but coverage 0.552 against 0.932 and correctness
0.189 against 0.273.

TRUE's calibrated recall is 0.747, so roughly one genuinely supported claim in
four is missed. Under a delete policy those become *destroyed correct content*,
which is the main driver of the correctness gap. Annotating them instead converts
that damage from lost content into a lower citation-recall figure, which is a
strictly better trade: the reader still gets the content and can see it is
unverified.

Three-way routing per claim:

    entailed and entity-consistent  -> keep, attach the VERIFIED citation
    entailed, and the evidence carries a COMPETING value in the same role -> drop
    anything else                   -> keep, annotated unverified, no citation

The third row is the substance of the redesign. It covers genuine no-evidence
cases and verifier misses alike, and deliberately does not try to tell them
apart -- ``contradicted`` is unavailable under a binary verifier backend, so an
entity conflict is the only concrete contradiction signal there is, and it alone
triggers a drop.

Blind adjudication of the first run's drops put the false-veto rate at **0.700**,
with a further 0.100 that should have been annotated, so two things narrowed the
destructive path: only a *genuine* conflict drops a claim (an entity the evidence
never mentions is an absence, and absences are annotated), and proper nouns are
detected with ``SpacyEntityExtractor`` rather than by capitalisation, which is
what produced vetoes on ``name:some`` and ``name:season``.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from evidence_rag.contracts.models import (
    GenerationResult,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
)
from evidence_rag.generator.draft import DraftAnswerGenerator
from evidence_rag.generator.entity_check import (
    EntityChecker,
    EntityConsistencyChecker,
    SpacyEntityExtractor,
)
from evidence_rag.generator.granite import GraniteLLMClient, TextGenerator
from evidence_rag.generator.models import Claim, DraftAnswer
from evidence_rag.generator.nli import NLIModel, build_nli_model

UNVERIFIED_MARKER = "[unverified]"
"""Suffix marking a kept-but-unverified sentence.

Kept as a constant with the predicate below so the marker text can never drift
between generator and scorer -- the same discipline already applied to the
unconfirmed-facts disclosure."""

CITATION_RE = re.compile(r"\[(\d+)\]")
SENTENCE_END = re.compile(r"[.!?]")


def is_unverified_annotation(sentence: str) -> bool:
    """True for a sentence the generator kept without being able to verify it.

    Scorers import this rather than re-spelling the marker. Annotated sentences
    carry no citation by construction, so under ALCE they score zero citation
    recall -- that is the intended, visible cost of this design and must not be
    hidden."""
    return UNVERIFIED_MARKER in sentence


def strip_unverified_marker(text: str) -> str:
    """The text without annotation markers, for correctness scoring and display."""
    return " ".join(text.replace(UNVERIFIED_MARKER, " ").split())


def declared_indices(answer_text: str, claim: Claim) -> tuple[int, ...]:
    """The 1-based citation indices the model declared for this claim.

    Read from the claim's span through the end of the sentence containing it,
    because a model puts its citation at the end of the sentence while the
    splitter's span often stops at the claim's own last word.
    """
    stop = claim.span.end
    # Only reach forward when the span stops mid-sentence. The splitter now
    # anchors paraphrased claims to whole sentences, terminator included, and
    # extending past that would swallow the NEXT sentence's citations and credit
    # them to this claim -- which would corrupt the declared-citation survival
    # rate, one of the numbers this design is reported on.
    if not answer_text[claim.span.start : claim.span.end].rstrip().endswith((".", "!", "?")):
        match = SENTENCE_END.search(answer_text, claim.span.end)
        stop = match.end() if match else len(answer_text)
    seen: list[int] = []
    for raw in CITATION_RE.findall(answer_text[claim.span.start : stop]):
        index = int(raw)
        if index not in seen:
            seen.append(index)
    return tuple(seen)


@dataclass
class ClaimRouting:
    """What happened to one claim, and why."""

    claim_id: str
    outcome: str  # "verified" | "dropped_entity_conflict" | "unverified"
    citation: str | None = None
    declared_indices: tuple[int, ...] = ()
    declared_verified: bool = False
    """the declared citation itself carried the support"""
    rescued_by_scan: bool = False
    """the declared citation failed but the full scan found support anyway"""
    claim_text: str = ""
    sentence: str = ""
    """the sentence as it was written into the answer, so scoring can attach the
    verified citation to exactly that sentence instead of approximating"""
    conflict_evidence_id: str | None = None
    conflict_detail: tuple[str, ...] = ()
    """which entities clashed, for auditing the one path that destroys content"""


@dataclass
class RoutingStats:
    claims: int = 0
    verified: int = 0
    unverified: int = 0
    dropped_entity_conflict: int = 0
    declared_total: int = 0
    """claims that carried a declared citation at all"""
    declared_verified: int = 0
    """of those, ones whose declared citation actually supported the claim"""
    rescued_by_scan: int = 0
    routings: list[ClaimRouting] = field(default_factory=list)


class CitationRoutedVerifier:
    """Verify a claim against its declared citation first, then fall back.

    The fallback scan is **mandatory**, not an optimisation toggle: models
    frequently get the content right and the citation index wrong, and without the
    scan a mislabelled reference would delete a true statement for no reason.
    Routing only changes the *order* of comparisons, never a verdict.
    """

    def __init__(self, nli: NLIModel, entity_checker: EntityChecker | None = None) -> None:
        self.nli = nli
        # spaCy NER for the proper-noun side: the capitalisation heuristic in the
        # rule-based extractor treats sentence-initial common nouns as names, which
        # the audit found to be the dominant false-veto mechanism.
        self.entity_checker = entity_checker or EntityConsistencyChecker(SpacyEntityExtractor())

    def _supports(
        self, evidence_text: str, claim_text: str
    ) -> tuple[bool, bool, bool, tuple[str, ...]]:
        """(entailed, entity_consistent, genuine_conflict, mismatch detail).

        ``genuine_conflict`` is the *destructive* signal and is deliberately
        narrower than "inconsistent": it requires the evidence to actually carry a
        competing value in the same role. A claim entity the evidence simply never
        mentions is an absence, not a contradiction, and under annotate-not-delete
        an absence must not destroy content.
        """
        if self.nli.classify(premise=evidence_text, hypothesis=claim_text) != "entailment":
            return False, False, False, ()
        consistency = self.entity_checker.check(claim_text, evidence_text)
        mismatches = tuple(getattr(consistency, "mismatches", ()))
        detail = tuple(
            f"{m.entity_type}:{m.normalized}"
            + (f"!={'/'.join(m.evidence_values)}" if m.evidence_values else " (absent)")
            for m in mismatches
        )
        genuine = any(m.evidence_values for m in mismatches)
        return True, consistency.consistent, genuine, detail

    def route(
        self,
        claim: Claim,
        answer_text: str,
        selected: SelectedEvidenceSet,
    ) -> ClaimRouting:
        evidence = list(selected.evidence)
        by_index = {position: item for position, item in enumerate(evidence, start=1)}
        declared = declared_indices(answer_text, claim)

        for index in declared:
            item = by_index.get(index)
            if item is None:
                continue
            entailed, consistent, _genuine, _detail = self._supports(item.text, claim.text)
            if entailed and consistent:
                return ClaimRouting(
                    claim_id=claim.claim_id,
                    outcome="verified",
                    citation=item.evidence_id,
                    declared_indices=declared,
                    declared_verified=True,
                    claim_text=claim.text,
                )

        # mandatory fallback: the declared citation may simply be mislabelled
        conflict_id: str | None = None
        conflict_detail: tuple[str, ...] = ()
        for item in evidence:
            entailed, consistent, genuine, detail = self._supports(item.text, claim.text)
            if entailed and consistent:
                return ClaimRouting(
                    claim_id=claim.claim_id,
                    outcome="verified",
                    citation=item.evidence_id,
                    declared_indices=declared,
                    rescued_by_scan=bool(declared),
                    claim_text=claim.text,
                )
            # Only a GENUINE conflict earns destructive power. An entity the
            # evidence never mentions is an absence, and absences are annotated.
            if entailed and genuine and conflict_id is None:
                conflict_id = item.evidence_id
                conflict_detail = detail
        return ClaimRouting(
            claim_id=claim.claim_id,
            outcome="dropped_entity_conflict" if conflict_id else "unverified",
            declared_indices=declared,
            claim_text=claim.text,
            conflict_evidence_id=conflict_id,
            conflict_detail=conflict_detail,
        )


class VerifyAnnotateGenerator:
    """Draft, then verify every claim and keep what cannot be verified, labelled.

    The contract is unchanged -- ``generate(query, checklist, selected)`` -- and
    the checklist is simply unused in this path: completeness is retired as a
    runtime mechanism, and changing the signature would need cross-team agreement
    while buying nothing.
    """

    def __init__(
        self,
        draft_generator: DraftAnswerGenerator | None = None,
        verifier: CitationRoutedVerifier | None = None,
        *,
        llm: TextGenerator | None = None,
        nli: NLIModel | None = None,
    ) -> None:
        shared_llm = llm
        if draft_generator is None:
            shared_llm = shared_llm or GraniteLLMClient()
        self.draft_generator = draft_generator or DraftAnswerGenerator(llm=shared_llm)
        self.verifier = verifier or CitationRoutedVerifier(nli or build_nli_model())
        self.stats = RoutingStats()
        self.last_routings: list[ClaimRouting] = []
        """Routing for the most recent query, carrying each kept sentence and the
        citation actually verified for it. Scoring reads this so citation precision
        rests on the real per-sentence mapping rather than on an approximation."""

    @staticmethod
    def _sentence(text: str) -> str:
        normalized = " ".join(text.split())
        # Removing an inline citation leaves the punctuation stranded
        # ("Revenue rose 8% [1]." -> "Revenue rose 8% ."), so close that gap
        # before deciding whether the sentence needs a terminator.
        normalized = re.sub(r"\s+([.!?,;:])", r"\1", normalized)
        if normalized and normalized[-1] not in ".!?":
            normalized = f"{normalized}."
        return normalized

    def _claim_text(self, draft: DraftAnswer, claim: Claim) -> str:
        """The claim as it appeared in the answer, with declared markers removed --
        the declared citation was a routing hint and must not survive into the
        output, where only the verified citation belongs."""
        span = draft.answer_text[claim.span.start : claim.span.end]
        return self._sentence(CITATION_RE.sub(" ", span))

    def generate(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> GenerationResult:
        if query.query_id != checklist.query_id or query.query_id != selected.query_id:
            raise ValueError("query, checklist, and selected evidence query IDs differ")

        draft = self.draft_generator.generate(query, checklist, selected)
        parts: list[str] = []
        citations: list[str] = []
        seen: set[str] = set()
        verified_any = False
        self.last_routings = []

        for claim in draft.claims:
            if not claim.faithful_to_answer:
                continue
            routing = self.verifier.route(claim, draft.answer_text, selected)
            self.last_routings.append(routing)
            self._record(routing)
            if routing.outcome == "dropped_entity_conflict":
                continue
            sentence = self._claim_text(draft, claim)
            if not sentence:
                continue
            if routing.outcome == "verified" and routing.citation is not None:
                verified_any = True
                routing.sentence = sentence
                parts.append(sentence)
                if routing.citation not in seen:
                    seen.add(routing.citation)
                    citations.append(routing.citation)
            else:
                routing.sentence = f"{sentence} {UNVERIFIED_MARKER}"
                parts.append(routing.sentence)

        # An answer of nothing but unverified annotations would carry no citation
        # and violate GenerationResult. Abstaining preserves the contract and is
        # the honest outcome anyway.
        if not verified_any:
            return GenerationResult(
                query_id=query.query_id, answer="", cited_evidence_ids=()
            )
        return GenerationResult(
            query_id=query.query_id,
            answer=" ".join(parts),
            cited_evidence_ids=tuple(citations),
        )

    def _record(self, routing: ClaimRouting) -> None:
        self.stats.claims += 1
        self.stats.routings.append(routing)
        if routing.declared_indices:
            self.stats.declared_total += 1
            if routing.declared_verified:
                self.stats.declared_verified += 1
            if routing.rescued_by_scan:
                self.stats.rescued_by_scan += 1
        if routing.outcome == "verified":
            self.stats.verified += 1
        elif routing.outcome == "unverified":
            self.stats.unverified += 1
        else:
            self.stats.dropped_entity_conflict += 1


def annotation_rate(sentences: Sequence[str]) -> float:
    """Share of kept sentences carrying the unverified marker."""
    if not sentences:
        return 0.0
    return sum(1 for s in sentences if is_unverified_annotation(s)) / len(sentences)
