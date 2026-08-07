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

Routing per claim, with the entity gate disengaged (the audited default):

    entailed      -> keep, attach the VERIFIED citation
    not entailed  -> keep, annotated unverified, no citation

**There is no drop path.** Nothing the Generator produces is destroyed; the
contract's guarantee -- every ungrounded sentence is labelled -- now holds with no
exception carved out of it.

With the gate engaged (``entity_gate=True``) a third row exists: entailed, but
the evidence carries a competing value in the same role -> drop. That was the
default through G6 and is kept runnable as the control arm. It was disengaged
because two blind adjudications, on disjoint populations with entirely different
trigger composition, both returned a false-veto rate of 14/20 = 0.700; wrong
destruction 0.850, CI [0.640, 0.948]. The spaCy switch removed every
``name:some``-class trigger and moved the error rate not at all, so the failure
was never about which spans were extracted. The three correctly-dropped items
settle it: an incomplete claim, a quantifier-scope question and an
approved-versus-implemented distinction -- none an entity conflict, so even the
0.150 correct rate is coincidental.

The gate is *not* deleted, and its verdict is still computed and logged on every
claim in every arm. It measurably helps against adversarial entity substitution
(G1 counterfactual slice: 0.963 verifier-alone -> 1.000 with the layer engaged),
so this is a threat-model choice -- a defence that is expensive on benign data --
rather than a component that failed.

Conflict is deliberately NOT routed to a "the evidence contradicts this" label.
Roughly 70% of those labels would be wrong, and telling a reader the evidence
conflicts with a claim the evidence actually supports asserts something false
about the evidence; deleting at least asserts nothing. A signal that unreliable
must not drive a user-visible label.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from evidence_rag.contracts.models import (
    UNVERIFIED_ANNOTATION,
    GenerationResult,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
    is_unverified_annotation,
    strip_unverified_annotation,
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

UNVERIFIED_MARKER = UNVERIFIED_ANNOTATION
"""Re-exported from ``contracts``: the label is now part of the GenerationResult
guarantee ("every ungrounded sentence is labelled"), so the contract owns it and
generator, scorer and validator cannot drift apart."""

CITATION_RE = re.compile(r"\[(\d+)\]")
SENTENCE_END = re.compile(r"[.!?]")


# Re-exported so existing importers (scorers) keep working; the definitions live
# in contracts because the contract is stated in terms of them.
strip_unverified_marker = strip_unverified_annotation


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
    gated_outcome: str = ""
    gated_citation: str | None = None
    """What the entity gate WOULD have decided, recorded whether or not the gate is
    driving routing. With the gate engaged these equal ``outcome``/``citation`` by
    construction; with it disengaged they are the observe-only log, which turns
    "the gate would have destroyed ~51 claims" from an extrapolation into a count."""


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
    gate_would_drop: int = 0
    """claims the entity gate would have destroyed, counted in every arm"""
    gate_would_drop_now_cited: int = 0
    """of those, ones that instead received a citation -- the direct measurement"""
    gate_would_drop_now_annotated: int = 0
    gate_changed_citation: int = 0
    """claims cited either way, but the gate would have picked other evidence"""
    routings: list[ClaimRouting] = field(default_factory=list)


class CitationRoutedVerifier:
    """Verify a claim against its declared citation first, then fall back.

    The fallback scan is **mandatory**, not an optimisation toggle: models
    frequently get the content right and the citation index wrong, and without the
    scan a mislabelled reference would delete a true statement for no reason.
    Routing only changes the *order* of comparisons, never a verdict.
    """

    def __init__(
        self,
        nli: NLIModel,
        entity_checker: EntityChecker | None = None,
        *,
        entity_gate: bool = True,
    ) -> None:
        self.nli = nli
        # spaCy NER for the proper-noun side: the capitalisation heuristic in the
        # rule-based extractor treats sentence-initial common nouns as names, which
        # the audit found to be the dominant false-veto mechanism.
        self.entity_checker = entity_checker or EntityConsistencyChecker(SpacyEntityExtractor())
        self.entity_gate = entity_gate
        """Whether entity consistency may *change routing*. The check runs either
        way; with the gate off its verdict is recorded and ignored.

        Off is the audited setting. Two blind adjudications on disjoint populations
        with entirely different trigger composition both returned a false-veto rate
        of 14/20 = 0.700 -- the spaCy switch removed every ``name:some``-class
        trigger and moved the error rate not at all, so the failure was never about
        which spans were extracted. Wrong destruction is 0.850, CI [0.640, 0.948].
        The layer is not worthless: on the G1 adversarial entity-substitution slice
        it took the verifier from 0.963 to 1.000. It is a defence against
        adversarial substitution that is expensive on benign data, and the gate
        setting is where that trade-off is made."""

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
        """Route one claim, computing BOTH the entailment-only verdict and the
        verdict the entity gate would have reached.

        The declared citation is tried first and the full scan is mandatory, in
        both modes: models frequently get the content right and the citation index
        wrong, and without the scan a mislabelled reference would delete or
        un-cite a true statement for no reason.
        """
        evidence = list(selected.evidence)
        by_index = {position: item for position, item in enumerate(evidence, start=1)}
        declared = declared_indices(answer_text, claim)

        entailed_id: str | None = None  # entailment alone -- the ungated citation
        entailed_declared = False
        gated_id: str | None = None  # entailed AND entity-consistent
        gated_declared = False
        conflict_id: str | None = None
        conflict_detail: tuple[str, ...] = ()

        declared_items = [by_index[i] for i in declared if i in by_index]
        # The declared prefix is scanned first and then the full evidence set; an
        # item appearing in both is examined twice, which the memo-free NLI call
        # makes cheap enough and which keeps this behaviourally identical to the
        # two-pass version it replaces.
        for from_declared, item in [(True, i) for i in declared_items] + [
            (False, i) for i in evidence
        ]:
            entailed, consistent, genuine, detail = self._supports(item.text, claim.text)
            if not entailed:
                continue
            if entailed_id is None:
                entailed_id, entailed_declared = item.evidence_id, from_declared
            if consistent:
                gated_id, gated_declared = item.evidence_id, from_declared
                break  # both verdicts are now settled
            if genuine and conflict_id is None and not from_declared:
                # Only a GENUINE conflict earns destructive power. An entity the
                # evidence never mentions is an absence, and absences are annotated.
                # Restricted to the full-evidence pass so that which evidence gets
                # named as the conflict is unchanged from the two-pass version --
                # the control arm has to stay comparable to G6 exactly.
                conflict_id, conflict_detail = item.evidence_id, detail

        gated_outcome = (
            "verified"
            if gated_id
            else "dropped_entity_conflict"
            if conflict_id
            else "unverified"
        )
        if self.entity_gate:
            citation, from_declared = gated_id, gated_declared
            outcome = gated_outcome
        else:
            citation, from_declared = entailed_id, entailed_declared
            outcome = "verified" if entailed_id else "unverified"
        return ClaimRouting(
            claim_id=claim.claim_id,
            outcome=outcome,
            citation=citation,
            declared_indices=declared,
            declared_verified=bool(citation) and from_declared,
            rescued_by_scan=bool(citation) and not from_declared and bool(declared),
            claim_text=claim.text,
            conflict_evidence_id=conflict_id,
            conflict_detail=conflict_detail,
            gated_outcome=gated_outcome,
            gated_citation=gated_id,
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
        abstain_when_unverified: bool = False,
        entity_gate: bool = True,
    ) -> None:
        self.abstain_when_unverified = abstain_when_unverified
        """The pre-lift behaviour, kept so the capped arm can be run as the
        ablation that isolates what the contract change bought."""
        shared_llm = llm
        if draft_generator is None:
            shared_llm = shared_llm or GraniteLLMClient()
        self.draft_generator = draft_generator or DraftAnswerGenerator(llm=shared_llm)
        self.verifier = verifier or CitationRoutedVerifier(
            nli or build_nli_model(), entity_gate=entity_gate
        )
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

        # Abstain only when there is genuinely nothing to say. An answer made
        # entirely of annotations used to be impossible -- the contract demanded a
        # citation -- which forced a wholesale abstention that threw the
        # annotations away, capping this policy at 4.8% of kept sentences. With
        # the invariant swapped, that answer is now legal and is emitted.
        if not parts or (self.abstain_when_unverified and not verified_any):
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
        # Observe-only accounting. Counted in every arm, so the gated arms report
        # the same quantity as a self-check and the ungated arm reports what the
        # gate would have cost -- measured, not extrapolated.
        if routing.gated_outcome == "dropped_entity_conflict":
            self.stats.gate_would_drop += 1
            if routing.outcome == "verified":
                self.stats.gate_would_drop_now_cited += 1
            elif routing.outcome == "unverified":
                self.stats.gate_would_drop_now_annotated += 1
        elif (
            routing.outcome == "verified"
            and routing.gated_outcome == "verified"
            and routing.citation != routing.gated_citation
        ):
            self.stats.gate_changed_citation += 1


def annotation_rate(sentences: Sequence[str]) -> float:
    """Share of kept sentences carrying the unverified marker."""
    if not sentences:
        return 0.0
    return sum(1 for s in sentences if is_unverified_annotation(s)) / len(sentences)
