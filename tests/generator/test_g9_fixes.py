"""G9's behaviour-neutral changes, pinned as behaviour-neutral.

Five of G9's six changes must not move a number. Each is checked here by
comparing the two configurations directly rather than by trusting the argument
that they are equivalent.
"""

import sys
from pathlib import Path

import pytest

from evidence_rag.composition import build_generator
from evidence_rag.contracts.models import (
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
    sentence_spans,
    split_sentences,
)
from evidence_rag.generator.claim_splitter import _sentence_spans
from evidence_rag.generator.models import DraftAnswer
from evidence_rag.generator.verify_annotate import (
    CitationRoutedVerifier,
    VerifyAnnotateGenerator,
)
from evidence_rag.infrastructure.config import ModuleConfig

from .test_verify_annotate import (
    FixedDraft,
    ScriptedNLI,
    StubEntityChecker,
    _claim,
    evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


# --- Task 5: one sentence rule, not three ----------------------------------


def test_the_claim_splitter_uses_the_contract_sentence_rule() -> None:
    """It carried its own with 10 abbreviations against the contract's ~50."""
    assert _sentence_spans is sentence_spans


@pytest.mark.parametrize(
    "text",
    [
        "Acme Inc. in Ohio filed a report.",
        "The eruption of Mount St. Helens was a VEI 5 event.",
        "Patrick S. Castagne wrote the anthem.",
        "The team won on Jan. 11, 1970.",
        "Francis Ouimet won the 1913 U.S. Open.",
    ],
)
def test_splitter_and_validator_agree_on_abbreviations(text: str) -> None:
    """The parity the G8 test established for the scorer, extended to the splitter."""
    assert len(_sentence_spans(text)) == 1
    assert len(split_sentences(text)) == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('He coined "Manifest Destiny." It means expansion.', 2),
        ("She said \"stop.\" He left.", 2),
        ("A list (see below.) Then more.", 2),
        ('They played "The Parent Trap."', 1),
    ],
)
def test_a_closing_quote_does_not_swallow_the_next_sentence(text: str, expected: int) -> None:
    """Caught by measuring the splitter change rather than reasoning about it.

    Requiring whitespace immediately after the terminator meant `."` never split,
    silently merging two real sentences -- the more damaging direction, since
    `GenerationResult` counts sentences to decide whether every uncited one is
    labelled, so an under-count lets an unlabelled sentence through.
    """
    assert len(split_sentences(text)) == expected
    assert len(_sentence_spans(text)) == expected


def test_sentence_spans_offsets_recover_the_sentences() -> None:
    text = "Mount St. Helens erupted. Rome fell. Carthage did too."
    spans = _sentence_spans(text)
    assert [text[a:b] for a, b in spans] == split_sentences(text)
    assert len(spans) == 3


# --- Task 3: the memo is a de-duplicator, not a behaviour change -----------


def _routing_case():
    selected = SelectedEvidenceSet(
        query_id="q",
        evidence=(evidence("ev-1", "Globex rose 8%.", 1), evidence("ev-2", "Acme rose 8%.", 2)),
    )
    entailing = {("Globex rose 8%.", "Acme rose 8%."), ("Acme rose 8%.", "Acme rose 8%.")}
    claim = _claim("claim-1", "Acme rose 8%.", 0, 13)
    return selected, entailing, claim


def test_memo_on_and_off_give_identical_verdicts() -> None:
    selected, entailing, claim = _routing_case()
    answer = "Acme rose 8% [1]."

    on = CitationRoutedVerifier(
        ScriptedNLI(entailing), StubEntityChecker(), entity_gate="gate", memoise=True
    ).route(claim, answer, selected)
    off = CitationRoutedVerifier(
        ScriptedNLI(entailing), StubEntityChecker(), entity_gate="gate", memoise=False
    ).route(claim, answer, selected)

    assert (on.outcome, on.citation, on.gated_outcome, on.gated_citation) == (
        off.outcome,
        off.citation,
        off.gated_outcome,
        off.gated_citation,
    )


def test_the_memo_actually_removes_the_duplicate_call() -> None:
    """The declared prefix is re-reached in the full pass; that is the 19.7%.

    The duplicate only happens when the declared item does NOT settle the scan --
    otherwise `route` breaks before reaching the full pass. So the fixture makes
    the declared evidence entailed but entity-inconsistent, which is exactly the
    case the G8 measurement counted.
    """
    selected, entailing, claim = _routing_case()
    answer = "Acme rose 8% [1]."  # declares ev-1, which conflicts
    entity = StubEntityChecker(inconsistent={"Globex rose 8%."})

    counted_off = ScriptedNLI(entailing)
    CitationRoutedVerifier(
        counted_off, entity, entity_gate="gate", memoise=False
    ).route(claim, answer, selected)
    counted_on = ScriptedNLI(entailing)
    CitationRoutedVerifier(counted_on, entity, entity_gate="gate", memoise=True).route(
        claim, answer, selected
    )

    assert len(counted_off.calls) > len(set(counted_off.calls))  # the duplicate is real
    assert len(counted_on.calls) < len(counted_off.calls)
    assert len(set(counted_on.calls)) == len(counted_on.calls)  # no pair judged twice


# --- Task 4: the gate is a three-way choice --------------------------------


def _generate(gate) -> tuple[str, tuple[str, ...], object]:  # type: ignore[no-untyped-def]
    draft = DraftAnswer(
        query_id="q",
        answer_text="Acme rose 8% [1].",
        claims=(_claim("claim-1", "Acme rose 8%.", 0, 17),),
    )
    selected = SelectedEvidenceSet(query_id="q", evidence=(evidence("ev-1", "Globex rose 8%."),))
    generator = VerifyAnnotateGenerator(
        draft_generator=FixedDraft(draft),
        verifier=CitationRoutedVerifier(
            ScriptedNLI({("Globex rose 8%.", "Acme rose 8%.")}),
            StubEntityChecker(inconsistent={"Globex rose 8%."}),
            entity_gate=gate,
        ),
    )
    result = generator.generate(
        Query(query_id="q", text="what?"),
        QueryChecklist(query_id="q", focus="f", required_facts=()),
        selected,
    )
    return result.answer, result.cited_evidence_ids, generator.stats


def test_gate_destroys_observe_flags_off_is_silent() -> None:
    gated_answer, gated_citations, gated_stats = _generate("gate")
    obs_answer, obs_citations, obs_stats = _generate("observe")
    off_answer, off_citations, off_stats = _generate("off")

    # gate: the claim is destroyed
    assert gated_answer == "" and gated_citations == ()
    assert gated_stats.dropped_entity_conflict == 1

    # observe: cited, flagged, nothing destroyed
    assert obs_citations == ("ev-1",) and "[may warrant review]" in obs_answer
    assert obs_stats.dropped_entity_conflict == 0 and obs_stats.review_flagged == 1

    # off: cited, NOT flagged, nothing destroyed
    assert off_citations == ("ev-1",) and "[may warrant review]" not in off_answer
    assert off_stats.dropped_entity_conflict == 0 and off_stats.review_flagged == 0


def test_off_does_not_record_a_gate_verdict_it_never_computed() -> None:
    """"Not checked" must not be logged as "checked and clean", or the
    observe-only counters would count an `off` run as one where the gate agreed."""
    _, _, off_stats = _generate("off")
    assert off_stats.gate_would_drop == 0
    assert all(r.gated_outcome == "" for r in off_stats.routings)


def test_the_superseded_boolean_keeps_its_old_meaning() -> None:
    """`False` used to mean "gate off but still checking" -- that is `observe`,
    not `off`. Mapping it to `off` would silently stop the review flag."""
    assert CitationRoutedVerifier(ScriptedNLI(set()), entity_gate=True).entity_gate == "gate"
    assert CitationRoutedVerifier(ScriptedNLI(set()), entity_gate=False).entity_gate == "observe"


def test_an_unknown_gate_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="gate/observe/off"):
        CitationRoutedVerifier(ScriptedNLI(set()), entity_gate="disabled")  # type: ignore[arg-type]


# --- Task 2: the method is reachable from the factory ----------------------


class _NoopLLM:
    def generate(self, prompt: str) -> str:
        return ""


def _build(parameters: dict) -> VerifyAnnotateGenerator:
    generator = build_generator(
        ModuleConfig(name="verify-annotate", parameters=parameters),
        llm=_NoopLLM(),
        nli=ScriptedNLI(set()),
    )
    assert isinstance(generator, VerifyAnnotateGenerator)
    return generator


def test_build_generator_can_build_the_main_method() -> None:
    generator = _build({})
    assert generator.verifier.entity_gate == "observe"
    assert generator.abstain_when_unverified is False


def test_build_generator_passes_the_gate_mode_through() -> None:
    assert _build({"entity_gate": "gate"}).verifier.entity_gate == "gate"


def test_build_generator_rejects_unknown_parameters() -> None:
    with pytest.raises(ValueError, match="unknown generator parameters"):
        build_generator(
            ModuleConfig(name="verify-annotate", parameters={"threshold": 0.7}),
            llm=_NoopLLM(),
            nli=ScriptedNLI(set()),
        )
