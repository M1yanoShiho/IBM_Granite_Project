"""Verify-and-annotate routing: keep unverified content labelled, drop only conflicts."""

import pytest

from evidence_rag.contracts.models import (
    EvidenceCandidate,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
)
from evidence_rag.generator.models import Claim, ClaimSpan, DraftAnswer
from evidence_rag.generator.verify_annotate import (
    UNVERIFIED_MARKER,
    CitationRoutedVerifier,
    VerifyAnnotateGenerator,
    declared_indices,
    is_unverified_annotation,
    strip_unverified_marker,
)


def evidence(evidence_id: str, text: str, rank: int = 1) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=f"doc-{evidence_id}",
        chunk_id=f"chunk-{evidence_id}",
        text=text,
        source_uri=f"fixture://{evidence_id}",
        retrieval_score=1.0,
        retrieval_rank=rank,
    )


class ScriptedNLI:
    """Entails when the evidence text contains the trigger for that claim."""

    def __init__(self, entailing: set[tuple[str, str]]) -> None:
        self.entailing = entailing
        self.calls: list[tuple[str, str]] = []

    def classify(self, premise: str, hypothesis: str) -> str:
        self.calls.append((premise, hypothesis))
        return "entailment" if (premise, hypothesis) in self.entailing else "neutral"


class _Mismatch:
    """Mirrors EntityMismatch: evidence_values non-empty means the evidence carried
    a competing value in the same role; empty means the entity is simply absent."""

    def __init__(self, evidence_values: tuple[str, ...]) -> None:
        self.entity_type = "name"
        self.normalized = "claim-entity"
        self.evidence_values = evidence_values


class StubEntityChecker:
    def __init__(
        self,
        inconsistent: set[str] | None = None,
        absent: set[str] | None = None,
    ) -> None:
        self.inconsistent = inconsistent or set()  # genuine conflict
        self.absent = absent or set()  # entity missing, nothing competing

    def check(self, claim_text: str, evidence_text: str):  # type: ignore[no-untyped-def]
        conflicting = evidence_text in self.inconsistent
        missing = evidence_text in self.absent

        class _Result:
            consistent = not (conflicting or missing)
            mismatches = (
                (_Mismatch(("rival-value",)),)
                if conflicting
                else (_Mismatch(()),)
                if missing
                else ()
            )

        return _Result()


def _claim(claim_id: str, text: str, start: int, end: int, faithful: bool = True) -> Claim:
    return Claim(
        claim_id=claim_id,
        text=text,
        span=ClaimSpan(start=start, end=end),
        faithful_to_answer=faithful,
    )


# --- declared citation parsing ---------------------------------------------


def test_declared_indices_reads_to_the_end_of_the_sentence() -> None:
    """The splitter's span usually stops at the claim's last word while the model
    puts its citation after it, so parsing must run on to the sentence end."""
    answer = "Revenue rose 8% [2][3]. Costs fell [1]."
    claim = _claim("claim-1", "Revenue rose 8%.", 0, 15)

    assert declared_indices(answer, claim) == (2, 3)


def test_declared_indices_is_empty_when_the_model_cited_nothing() -> None:
    assert declared_indices("Revenue rose 8%.", _claim("c", "x", 0, 15)) == ()


# --- three-way routing ------------------------------------------------------


def test_declared_citation_that_verifies_short_circuits_the_scan() -> None:
    selected = SelectedEvidenceSet(
        query_id="q",
        evidence=(
            evidence("ev-1", "Costs fell.", 1),
            evidence("ev-2", "Revenue rose 8%.", 2),
        ),
    )
    nli = ScriptedNLI({("Revenue rose 8%.", "Revenue rose 8%.")})
    verifier = CitationRoutedVerifier(nli, StubEntityChecker())

    routing = verifier.route(
        _claim("claim-1", "Revenue rose 8%.", 0, 15), "Revenue rose 8% [2].", selected
    )

    assert routing.outcome == "verified"
    assert routing.citation == "ev-2"
    assert routing.declared_verified is True
    # ev-1 was never consulted: routing changes the order, not the verdict
    assert len(nli.calls) == 1


def test_fallback_scan_rescues_a_claim_whose_declared_citation_is_wrong() -> None:
    """Models get the content right and the index wrong often enough that without
    the fallback a mislabelled reference would delete a true statement."""
    selected = SelectedEvidenceSet(
        query_id="q",
        evidence=(
            evidence("ev-1", "Revenue rose 8%.", 1),
            evidence("ev-2", "Unrelated.", 2),
        ),
    )
    nli = ScriptedNLI({("Revenue rose 8%.", "Revenue rose 8%.")})
    verifier = CitationRoutedVerifier(nli, StubEntityChecker())

    routing = verifier.route(
        _claim("claim-1", "Revenue rose 8%.", 0, 15), "Revenue rose 8% [2].", selected
    )

    assert routing.outcome == "verified"
    assert routing.citation == "ev-1"  # the VERIFIED citation, not the declared one
    assert routing.rescued_by_scan is True


def test_unsupported_claim_is_annotated_not_deleted() -> None:
    selected = SelectedEvidenceSet(query_id="q", evidence=(evidence("ev-1", "Unrelated."),))
    verifier = CitationRoutedVerifier(ScriptedNLI(set()), StubEntityChecker())

    routing = verifier.route(_claim("claim-1", "Revenue rose 8%.", 0, 15), "Revenue rose 8%.", selected)

    assert routing.outcome == "unverified"
    assert routing.citation is None


def test_entity_conflict_is_the_only_thing_that_drops_a_claim() -> None:
    selected = SelectedEvidenceSet(query_id="q", evidence=(evidence("ev-1", "Globex rose 8%."),))
    nli = ScriptedNLI({("Globex rose 8%.", "Acme rose 8%.")})
    verifier = CitationRoutedVerifier(nli, StubEntityChecker(inconsistent={"Globex rose 8%."}))

    routing = verifier.route(_claim("claim-1", "Acme rose 8%.", 0, 13), "Acme rose 8%.", selected)

    assert routing.outcome == "dropped_entity_conflict"


def test_absent_entity_is_annotated_not_dropped() -> None:
    """An entity the evidence never mentions is an absence, not a contradiction.
    Under annotate-not-delete only a genuine conflict may destroy content -- the
    audit put the false-veto rate on the old rule at 0.700."""
    selected = SelectedEvidenceSet(query_id="q", evidence=(evidence("ev-1", "Something else."),))
    nli = ScriptedNLI({("Something else.", "Acme rose 8%.")})
    verifier = CitationRoutedVerifier(nli, StubEntityChecker(absent={"Something else."}))

    routing = verifier.route(_claim("claim-1", "Acme rose 8%.", 0, 13), "Acme rose 8%.", selected)

    assert routing.outcome == "unverified"


def test_clean_support_elsewhere_beats_a_conflict() -> None:
    selected = SelectedEvidenceSet(
        query_id="q",
        evidence=(evidence("ev-1", "Globex rose 8%.", 1), evidence("ev-2", "Acme rose 8%.", 2)),
    )
    nli = ScriptedNLI(
        {("Globex rose 8%.", "Acme rose 8%."), ("Acme rose 8%.", "Acme rose 8%.")}
    )
    verifier = CitationRoutedVerifier(nli, StubEntityChecker(inconsistent={"Globex rose 8%."}))

    routing = verifier.route(_claim("claim-1", "Acme rose 8%.", 0, 13), "Acme rose 8%.", selected)

    assert routing.outcome == "verified"
    assert routing.citation == "ev-2"


# --- generator assembly -----------------------------------------------------


class FixedDraft:
    def __init__(self, draft: DraftAnswer) -> None:
        self.draft = draft

    def generate(self, query, checklist, selected):  # type: ignore[no-untyped-def]
        return self.draft


def _run(draft: DraftAnswer, selected: SelectedEvidenceSet, nli: ScriptedNLI, entity=None):  # type: ignore[no-untyped-def]
    generator = VerifyAnnotateGenerator(
        draft_generator=FixedDraft(draft),
        verifier=CitationRoutedVerifier(nli, entity or StubEntityChecker()),
    )
    result = generator.generate(
        Query(query_id="q", text="what happened?"),
        QueryChecklist(query_id="q", focus="f", required_facts=()),
        selected,
    )
    return result, generator.stats


def test_answer_keeps_verified_and_annotated_claims_together() -> None:
    answer = "Revenue rose 8% [1]. Costs fell sharply [2]."
    draft = DraftAnswer(
        query_id="q",
        answer_text=answer,
        claims=(
            _claim("claim-1", "Revenue rose 8%.", 0, 20),
            _claim("claim-2", "Costs fell sharply.", 21, 44),
        ),
    )
    selected = SelectedEvidenceSet(
        query_id="q",
        evidence=(evidence("ev-1", "Revenue rose 8%.", 1), evidence("ev-2", "Unrelated.", 2)),
    )
    nli = ScriptedNLI({("Revenue rose 8%.", "Revenue rose 8%.")})

    result, stats = _run(draft, selected, nli)

    assert result.cited_evidence_ids == ("ev-1",)
    assert "Revenue rose 8%." in result.answer
    assert UNVERIFIED_MARKER in result.answer  # the unsupported claim was KEPT
    assert stats.verified == 1 and stats.unverified == 1
    # the declared marker never survives into the output
    assert "[1]" not in result.answer and "[2]" not in result.answer


def test_routing_records_the_exact_sentence_to_citation_mapping() -> None:
    """Citation precision is the headline metric, so it must rest on the real
    per-sentence mapping rather than on the flat citation list, which is only
    exactly recoverable in ~70% of answers."""
    answer = "Revenue rose 8% [1]. Costs fell sharply [2]."
    draft = DraftAnswer(
        query_id="q",
        answer_text=answer,
        claims=(
            _claim("claim-1", "Revenue rose 8%.", 0, 20),
            _claim("claim-2", "Costs fell sharply.", 21, 44),
        ),
    )
    selected = SelectedEvidenceSet(
        query_id="q",
        evidence=(evidence("ev-1", "Revenue rose 8%.", 1), evidence("ev-2", "Unrelated.", 2)),
    )
    generator = VerifyAnnotateGenerator(
        draft_generator=FixedDraft(draft),
        verifier=CitationRoutedVerifier(
            ScriptedNLI({("Revenue rose 8%.", "Revenue rose 8%.")}), StubEntityChecker()
        ),
    )
    generator.generate(
        Query(query_id="q", text="what?"),
        QueryChecklist(query_id="q", focus="f", required_facts=()),
        selected,
    )

    mapping = {r.sentence: r.citation for r in generator.last_routings}
    assert mapping["Revenue rose 8%."] == "ev-1"
    assert mapping[f"Costs fell sharply. {UNVERIFIED_MARKER}"] is None


def test_entity_conflict_records_the_offending_evidence_for_audit() -> None:
    """Entity conflict is now the only path that destroys content, so the drop has
    to be auditable: which passage caused it, and which entities clashed."""
    selected = SelectedEvidenceSet(query_id="q", evidence=(evidence("ev-1", "Globex rose 8%."),))
    nli = ScriptedNLI({("Globex rose 8%.", "Acme rose 8%.")})
    verifier = CitationRoutedVerifier(nli, StubEntityChecker(inconsistent={"Globex rose 8%."}))

    routing = verifier.route(_claim("claim-1", "Acme rose 8%.", 0, 13), "Acme rose 8%.", selected)

    assert routing.outcome == "dropped_entity_conflict"
    assert routing.conflict_evidence_id == "ev-1"
    assert routing.claim_text == "Acme rose 8%."


def test_zero_verified_claims_abstains_to_preserve_the_contract() -> None:
    draft = DraftAnswer(
        query_id="q",
        answer_text="Revenue rose 8% [1].",
        claims=(_claim("claim-1", "Revenue rose 8%.", 0, 20),),
    )
    selected = SelectedEvidenceSet(query_id="q", evidence=(evidence("ev-1", "Unrelated."),))

    result, _ = _run(draft, selected, ScriptedNLI(set()))

    assert result.answer == ""
    assert result.cited_evidence_ids == ()


def test_unfaithful_claims_are_never_routed() -> None:
    draft = DraftAnswer(
        query_id="q",
        answer_text="Revenue rose 8% [1].",
        claims=(_claim("claim-1", "Revenue rose 8%.", 0, 20, faithful=False),),
    )
    selected = SelectedEvidenceSet(query_id="q", evidence=(evidence("ev-1", "Revenue rose 8%."),))

    result, stats = _run(draft, selected, ScriptedNLI(set()))

    assert stats.claims == 0
    assert result.answer == ""


# --- marker helpers ---------------------------------------------------------


@pytest.mark.parametrize(
    ("sentence", "expected"),
    (("Costs fell. [unverified]", True), ("Costs fell.", False)),
)
def test_annotation_predicate(sentence: str, expected: bool) -> None:
    assert is_unverified_annotation(sentence) is expected


def test_strip_marker_leaves_clean_prose() -> None:
    assert strip_unverified_marker("Costs fell. [unverified]") == "Costs fell."
