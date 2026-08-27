from typing import Protocol

from evidence_rag.contracts.models import (
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
    SelectionGuidance,
)
from evidence_rag.generator.claim_splitter import ClaimSplitter
from evidence_rag.generator.granite import GraniteLLMClient, TextGenerator
from evidence_rag.generator.key_facts import (
    GUIDED_DRAFT_PROMPT,
    KeyFactNote,
    KeyFactNoteExtractor,
    render_key_fact_notes,
)
from evidence_rag.generator.models import Claim, DraftAnswer
from evidence_rag.generator.trace import (
    DraftMode,
    DraftStageTrace,
    DraftTextTrace,
    SplitterTrace,
)

DRAFT_PROMPT = (
    "Answer the question using only the evidence below.\n"
    "Cover every part of the question the evidence supports. If the question has "
    "more than one reasonable reading, address each reading you can.\n"
    "Write one independently verifiable factual claim per sentence. Keep every "
    "sentence self-contained and do not combine separate facts into one sentence.\n"
    "Name the relevant entities, dates, quantities, and conditions explicitly. "
    "Avoid unresolved pronouns such as it, they, this, or that.\n"
    "Do not describe the answer, the evidence, or the sources. Do not include "
    "unsupported intermediate reasoning.\n"
    "End every factual sentence with the bracketed number(s) of the evidence that "
    "supports it, for example: The sky is blue [1][3]. Use only the evidence "
    "numbers shown below.\n\n"
    "If the evidence does not contain the answer, say: I don't know.\n\n"
    "Evidence:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)
"""Comprehensiveness is a *soft target* in this prompt, not a runtime mechanism.

Three checklist constructions failed to drive completeness at runtime -- a
checklist built before retrieval cannot know which requirements the retrieved
evidence can satisfy -- so the completeness/recheck loop is retired and coverage
of the question is asked for here instead, then measured in evaluation by
qa_pairs STR-EM.

The inline citations this asks for are **routing hints, not results**: every one
is verified downstream, and the share that survives verification is itself a
reported number, because how far model-declared citations can be trusted is the
baseline claim this project exists to test."""

UNKNOWN_ANSWERS = {
    "",
    "unknown",
    "i don't know",
    "i do not know",
    "not in the evidence",
    "not contained in the evidence",
}


class DraftTextGenerator(Protocol):
    def generate_answer(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> str: ...


class ClaimsSplitter(Protocol):
    def split(self, answer_text: str) -> tuple[Claim, ...]: ...


class DraftAnswerProducer(Protocol):
    """Structural contract shared by the ordinary and notes-first draft paths."""

    def generate(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> DraftAnswer: ...


class DraftGenerator:
    """Generate the unverified answer that A2 will split into claims."""

    def __init__(
        self,
        llm: TextGenerator | None = None,
        prompt_template: str = DRAFT_PROMPT,
        *,
        trace_enabled: bool = False,
    ) -> None:
        self.llm = llm or GraniteLLMClient()
        self.prompt_template = prompt_template
        self.trace_enabled = trace_enabled
        self.last_trace: DraftTextTrace | None = None

    def generate_answer(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> str:
        self.last_trace = None
        if query.query_id != checklist.query_id or query.query_id != selected.query_id:
            raise ValueError("query, checklist, and selected evidence query IDs differ")
        if not selected.evidence:
            if self.trace_enabled:
                self.last_trace = DraftTextTrace(
                    raw_output="",
                    raw_output_available=True,
                    empty_reason="no_selected_evidence",
                )
            return ""
        context = "\n".join(
            f"[{index}] ({item.evidence_id}) {item.text}"
            for index, item in enumerate(selected.evidence, start=1)
        )
        prompt = self.prompt_template.format(
            context=context,
            question=query.text,
            focus=checklist.focus,
            required_facts="; ".join(checklist.required_facts) or "none",
            constraints="; ".join(checklist.constraints) or "none",
        )
        raw_output = self.llm.generate(prompt)
        answer = raw_output.strip()
        normalized = answer.lower().strip(".!?\"' ")
        result = "" if normalized in UNKNOWN_ANSWERS else answer
        if self.trace_enabled:
            self.last_trace = DraftTextTrace(
                raw_output=raw_output,
                raw_output_available=True,
                empty_reason="model_decline_or_empty" if not result else "",
            )
        return result


class DraftAnswerGenerator:
    """Compose A1 draft generation and A2 claim splitting into A's handoff."""

    def __init__(
        self,
        draft_generator: DraftTextGenerator | None = None,
        claim_splitter: ClaimsSplitter | None = None,
        llm: TextGenerator | None = None,
        *,
        trace_enabled: bool = False,
    ) -> None:
        shared_llm = llm
        if draft_generator is None or claim_splitter is None:
            shared_llm = shared_llm or GraniteLLMClient()
        self.trace_enabled = trace_enabled
        self.draft_generator = draft_generator or DraftGenerator(
            llm=shared_llm, trace_enabled=trace_enabled
        )
        self.claim_splitter = claim_splitter or ClaimSplitter(
            llm=shared_llm, trace_enabled=trace_enabled
        )
        self.last_trace: DraftStageTrace | None = None

    def generate(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> DraftAnswer:
        self.last_trace = None
        if query.query_id != checklist.query_id or query.query_id != selected.query_id:
            raise ValueError("query, checklist, and selected evidence query IDs differ")
        answer_text = self.draft_generator.generate_answer(query, checklist, selected)
        claims = self.claim_splitter.split(answer_text)
        draft = DraftAnswer(
            query_id=query.query_id,
            answer_text=answer_text,
            claims=claims,
        )
        if self.trace_enabled:
            text_trace = getattr(self.draft_generator, "last_trace", None)
            if not isinstance(text_trace, DraftTextTrace):
                text_trace = DraftTextTrace(
                    raw_output=answer_text,
                    raw_output_available=False,
                    empty_reason="unknown" if not answer_text.strip() else "",
                )
            splitter_trace = getattr(self.claim_splitter, "last_trace", None)
            if not isinstance(splitter_trace, SplitterTrace):
                splitter_trace = SplitterTrace(
                    status="unavailable", output_claim_count=len(claims)
                )
            self.last_trace = DraftStageTrace(
                mode="ordinary",
                raw_draft_text=text_trace.raw_output,
                normalized_draft_text=answer_text,
                raw_output_available=text_trace.raw_output_available,
                draft_empty_reason=text_trace.empty_reason,
                splitter=splitter_trace,
            )
        return draft


class KeyFactDraftAnswerGenerator:
    """Optional notes-first draft path used only by the F003B/F004 experiment.

    ``guided=False`` is G1 (question + selected evidence only). ``guided=True``
    is G2 (same notes call plus the runtime-safe Selector sidecar).  Both feed the
    unchanged claim splitter and downstream verifier.
    """

    def __init__(
        self,
        llm: TextGenerator,
        *,
        guided: bool = False,
        claim_splitter: ClaimsSplitter | None = None,
        note_llm: TextGenerator | None = None,
        trace_enabled: bool = False,
    ) -> None:
        self.llm = llm
        self.guided = guided
        # F006 may adapt only the extraction call while keeping drafting and
        # claim splitting on the frozen base model. Existing F003/F004 callers
        # omit ``note_llm`` and preserve the original single-client behaviour.
        self.note_extractor = KeyFactNoteExtractor(note_llm or llm, guided=guided)
        self.trace_enabled = trace_enabled
        self.claim_splitter = claim_splitter or ClaimSplitter(
            llm=llm, trace_enabled=trace_enabled
        )
        self.last_notes: tuple[KeyFactNote, ...] = ()
        self.last_trace: DraftStageTrace | None = None

    def generate(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
        guidance: SelectionGuidance | None = None,
    ) -> DraftAnswer:
        self.last_trace = None
        if query.query_id != checklist.query_id or query.query_id != selected.query_id:
            raise ValueError("query, checklist, and selected evidence query IDs differ")
        self.last_notes = self.note_extractor.extract(query, selected, guidance)
        if not self.last_notes:
            fallback = DraftGenerator(llm=self.llm, trace_enabled=self.trace_enabled)
            answer_text = fallback.generate_answer(query, checklist, selected)
            text_trace = fallback.last_trace
            mode: DraftMode = "key_fact_fallback"
        else:
            context = "\n".join(
                f"[{index}] ({item.evidence_id}) {item.text}"
                for index, item in enumerate(selected.evidence, start=1)
            )
            raw_output = self.llm.generate(
                GUIDED_DRAFT_PROMPT.format(
                    key_fact_notes=render_key_fact_notes(self.last_notes, selected),
                    context=context,
                    question=query.text,
                )
            )
            answer_text = raw_output.strip()
            if answer_text.lower().strip(".!?\"' ") in UNKNOWN_ANSWERS:
                answer_text = ""
            text_trace = DraftTextTrace(
                raw_output=raw_output,
                raw_output_available=True,
                empty_reason="model_decline_or_empty" if not answer_text else "",
            )
            mode = "key_fact_notes"
        claims = self.claim_splitter.split(answer_text)
        draft = DraftAnswer(
            query_id=query.query_id,
            answer_text=answer_text,
            claims=claims,
        )
        if self.trace_enabled:
            if not isinstance(text_trace, DraftTextTrace):
                text_trace = DraftTextTrace(
                    raw_output=answer_text,
                    raw_output_available=False,
                    empty_reason="unknown" if not answer_text.strip() else "",
                )
            splitter_trace = getattr(self.claim_splitter, "last_trace", None)
            if not isinstance(splitter_trace, SplitterTrace):
                splitter_trace = SplitterTrace(
                    status="unavailable", output_claim_count=len(claims)
                )
            self.last_trace = DraftStageTrace(
                mode=mode,
                raw_draft_text=text_trace.raw_output,
                normalized_draft_text=answer_text,
                raw_output_available=text_trace.raw_output_available,
                draft_empty_reason=text_trace.empty_reason,
                key_fact_note_count=len(self.last_notes),
                splitter=splitter_trace,
            )
        return draft

    def generate_with_guidance(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
        guidance: SelectionGuidance | None,
    ) -> DraftAnswer:
        return self.generate(query, checklist, selected, guidance)
