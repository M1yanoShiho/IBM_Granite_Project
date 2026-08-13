from __future__ import annotations

import pytest

from evidence_rag.contracts.models import (
    EvidenceCandidate,
    Query,
    QueryChecklist,
    RetainedEvidenceGuidance,
    SelectedEvidenceSet,
    SelectionGuidance,
)
from evidence_rag.generator.draft import KeyFactDraftAnswerGenerator
from evidence_rag.generator.key_facts import (
    KeyFactNoteExtractor,
    detect_question_slots,
)


class ScriptedLLM:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


class NoSplit:
    def split(self, answer_text: str) -> tuple[object, ...]:
        del answer_text
        return ()


def _evidence(index: int, text: str) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=f"e{index}",
        document_id=f"d{index}",
        chunk_id=f"c{index}",
        text=text,
        source_uri=f"memory://{index}",
        retrieval_score=float(3 - index),
        retrieval_rank=index,
    )


def _selected() -> SelectedEvidenceSet:
    return SelectedEvidenceSet(
        query_id="q1",
        evidence=(
            _evidence(1, "The album was released on 1 March 1973."),
            _evidence(2, "A different edition appeared later."),
        ),
    )


def _guidance() -> SelectionGuidance:
    return SelectionGuidance(
        query_id="q1",
        selector_changed=True,
        dropped_count=1,
        retained=(
            RetainedEvidenceGuidance(
                evidence_id="e1",
                retrieval_rank=1,
                protect_signal=0.9,
                harm_signal=0.1,
                action="KEEP",
                reason="BELOW_SAFE_THRESHOLD",
            ),
            RetainedEvidenceGuidance(
                evidence_id="e2",
                retrieval_rank=2,
                protect_signal=0.2,
                harm_signal=0.9,
                action="ABSTAIN_KEEP",
                reason="MAX_DELETE_REACHED",
            ),
        ),
    )


@pytest.mark.parametrize(
    ("question", "expected"),
    (
        ("When was the album released?", ("DATE_OR_YEAR",)),
        ("Where is the smallest bone?", ("LOCATION",)),
        ("Who is the environmental minister?", ("PERSON",)),
        ("How many volumes are there?", ("NUMBER",)),
        ("Which team won?", ("NAME_OR_TITLE",)),
        ("Describe the four demands.", ()),
    ),
)
def test_slot_detector_is_finite_and_question_only(
    question: str, expected: tuple[str, ...]
) -> None:
    assert detect_question_slots(question) == expected


def test_notes_only_extracts_complete_value_and_maps_evidence_index() -> None:
    llm = ScriptedLLM(
        '{"facts":[{"slot":"DATE_OR_YEAR","value":"1 March 1973","evidence":[1]}]}'
    )
    extractor = KeyFactNoteExtractor(llm, guided=False)

    notes = extractor.extract(
        Query(query_id="q1", text="When was the album released?"), _selected()
    )

    assert notes[0].value == "1 March 1973"
    assert notes[0].evidence_ids == ("e1",)
    assert "selector:" not in llm.prompts[0]


def test_guided_notes_show_runtime_selector_signals_without_dropped_text() -> None:
    llm = ScriptedLLM(
        '{"facts":[{"slot":"DATE_OR_YEAR","value":"1 March 1973","evidence":[1]}]}'
    )
    extractor = KeyFactNoteExtractor(llm, guided=True)

    extractor.extract(
        Query(query_id="q1", text="When was the album released?"),
        _selected(),
        _guidance(),
    )

    prompt = llm.prompts[0]
    assert "selector:protect-leaning" in prompt
    assert "selector:retained-by-conservative-cap" in prompt
    assert "uncalibrated-protect:0.9" in prompt
    assert "dropped" not in prompt.casefold()


def test_guided_notes_require_matching_guidance() -> None:
    extractor = KeyFactNoteExtractor(ScriptedLLM("unused"), guided=True)

    with pytest.raises(ValueError, match="require SelectionGuidance"):
        extractor.extract(
            Query(query_id="q1", text="When was it released?"), _selected(), None
        )


def test_malformed_or_unsupported_notes_fall_back_to_no_notes() -> None:
    malformed = KeyFactNoteExtractor(ScriptedLLM("not json"))
    unsupported = KeyFactNoteExtractor(
        ScriptedLLM('{"facts":[{"slot":"PERSON","value":"Alice","evidence":[1]}]}')
    )
    query = Query(query_id="q1", text="When was it released?")

    assert malformed.extract(query, _selected()) == ()
    assert unsupported.extract(query, _selected()) == ()


def test_notes_first_draft_uses_note_then_generates_answer() -> None:
    llm = ScriptedLLM(
        '{"facts":[{"slot":"DATE_OR_YEAR","value":"1 March 1973","evidence":[1]}]}',
        "The album was released on 1 March 1973 [1].",
    )
    generator = KeyFactDraftAnswerGenerator(
        llm,
        guided=False,
        claim_splitter=NoSplit(),  # type: ignore[arg-type]
    )
    checklist = QueryChecklist(query_id="q1", focus="album release", required_facts=())

    draft = generator.generate(
        Query(query_id="q1", text="When was the album released?"),
        checklist,
        _selected(),
    )

    assert draft.answer_text == "The album was released on 1 March 1973 [1]."
    assert "Key-fact notes:" in llm.prompts[1]
    assert "DATE_OR_YEAR: 1 March 1973 [1]" in llm.prompts[1]


def test_no_detectable_slot_falls_back_to_current_draft_prompt() -> None:
    llm = ScriptedLLM("The demands included civil rights [1].")
    generator = KeyFactDraftAnswerGenerator(
        llm,
        guided=False,
        claim_splitter=NoSplit(),  # type: ignore[arg-type]
    )
    checklist = QueryChecklist(query_id="q1", focus="demands", required_facts=())

    generator.generate(
        Query(query_id="q1", text="Describe the demands."), checklist, _selected()
    )

    assert len(llm.prompts) == 1
    assert "Cover every part of the question" in llm.prompts[0]


def test_note_llm_can_be_separate_from_frozen_draft_llm() -> None:
    note_llm = ScriptedLLM(
        '{"facts":[{"slot":"DATE_OR_YEAR","value":"1 March 1973","evidence":[1]}]}'
    )
    draft_llm = ScriptedLLM("The album was released on 1 March 1973 [1].")
    generator = KeyFactDraftAnswerGenerator(
        draft_llm,
        guided=False,
        claim_splitter=NoSplit(),  # type: ignore[arg-type]
        note_llm=note_llm,
    )
    checklist = QueryChecklist(query_id="q1", focus="album release", required_facts=())

    draft = generator.generate(
        Query(query_id="q1", text="When was the album released?"),
        checklist,
        _selected(),
    )

    assert len(note_llm.prompts) == 1
    assert len(draft_llm.prompts) == 1
    assert "Key-fact notes:" in draft_llm.prompts[0]
    assert draft.answer_text == "The album was released on 1 March 1973 [1]."
