"""The review flag: cited, hedged, and invisible to every metric.

The entity layer was measured at 0.678 cited-precision on the samples it flags
against 0.888 elsewhere -- an error rate of 0.322 against 0.112. That lift is
what makes it a usable screening signal, and screening is all it is used for: a
flagged claim keeps its citation and gains a label that asserts low confidence
about itself rather than anything about the evidence.
"""

import sys
from pathlib import Path

from evidence_rag.contracts.models import (
    REVIEW_ANNOTATION,
    UNVERIFIED_ANNOTATION,
    GenerationResult,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
    is_review_flagged,
    is_unverified_annotation,
    strip_annotations,
)
from evidence_rag.generator.models import DraftAnswer
from evidence_rag.generator.verify_annotate import (
    CitationRoutedVerifier,
    VerifyAnnotateGenerator,
)

from .test_verify_annotate import (  # reuse the fixtures the routing tests use
    FixedDraft,
    ScriptedNLI,
    StubEntityChecker,
    _claim,
    evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _flagging_generator() -> tuple[VerifyAnnotateGenerator, SelectedEvidenceSet]:
    """A claim entailed by evidence the entity checker calls a genuine conflict:
    cited under the ungated policy, destroyed under the gate."""
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
            entity_gate=False,
        ),
    )
    return generator, selected


def _run(generator: VerifyAnnotateGenerator, selected: SelectedEvidenceSet) -> GenerationResult:
    return generator.generate(
        Query(query_id="q", text="what?"),
        QueryChecklist(query_id="q", focus="f", required_facts=()),
        selected,
    )


def test_a_flagged_sentence_keeps_its_citation() -> None:
    generator, selected = _flagging_generator()
    result = _run(generator, selected)

    assert REVIEW_ANNOTATION in result.answer
    assert result.cited_evidence_ids == ("ev-1",)
    assert generator.stats.review_flagged == 1


def test_the_flag_is_not_the_unverified_label() -> None:
    """They mean different things and nothing may conflate them: `[unverified]`
    is 'no supporting evidence found, no citation', the flag is 'supported and
    cited, but a secondary check disagreed'."""
    generator, selected = _flagging_generator()
    answer = _run(generator, selected).answer

    assert is_review_flagged(answer)
    assert not is_unverified_annotation(answer)
    assert UNVERIFIED_ANNOTATION not in answer


def test_the_flag_does_not_change_routing() -> None:
    generator, selected = _flagging_generator()
    _run(generator, selected)
    routing = generator.last_routings[0]

    assert routing.outcome == "verified"
    assert routing.citation == "ev-1"
    assert routing.review_flagged is True
    assert generator.stats.dropped_entity_conflict == 0


def test_the_gated_arms_never_flag() -> None:
    """With the gate engaged a would-drop claim is dropped, so it can never also
    be flagged -- the ablation arms stay exactly what they were."""
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
            entity_gate=True,
        ),
    )

    result = _run(generator, selected)

    assert generator.stats.review_flagged == 0
    assert REVIEW_ANNOTATION not in result.answer
    assert generator.stats.dropped_entity_conflict == 1


def test_the_label_is_stripped_before_anything_scores_it() -> None:
    text = f"Acme rose 8%. {REVIEW_ANNOTATION}"
    assert strip_annotations(text) == "Acme rose 8%."


def test_the_scorer_treats_a_flagged_sentence_as_an_ordinary_cited_one() -> None:
    """Metrics-neutrality, checked where it could actually break: the scorer must
    re-attach the label to its sentence, strip it, and still find the citation."""
    from g5_score import _match_key, annotated_sent_split

    flagged = f"Acme rose 8%. {REVIEW_ANNOTATION}"
    plain = "Acme rose 8%."

    # the label rejoins its own sentence rather than becoming a sentence of its own
    assert annotated_sent_split(flagged) == [flagged]
    # so the sentence count -- ALCE's recall denominator -- is unchanged
    assert len(annotated_sent_split(flagged)) == len(annotated_sent_split(plain))
    # and the routing lookup that attaches the citation still matches
    assert _match_key(flagged) == _match_key(plain)
    # and it is never mistaken for the uncited label, which would zero its citation
    assert not is_unverified_annotation(annotated_sent_split(flagged)[0])


def test_flag_and_unverified_can_coexist_in_one_answer() -> None:
    answer = f"Alpha holds. {REVIEW_ANNOTATION} Beta is unclear. {UNVERIFIED_ANNOTATION}"
    result = GenerationResult(query_id="q", answer=answer, cited_evidence_ids=("ev-1",))

    assert is_review_flagged(result.answer)
    assert is_unverified_annotation(result.answer)
