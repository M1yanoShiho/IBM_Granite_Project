from typing import Protocol

from evidence_rag.contracts.models import Query, QueryChecklist, SelectedEvidenceSet
from evidence_rag.generator.claim_splitter import ClaimSplitter
from evidence_rag.generator.granite import GraniteLLMClient, TextGenerator
from evidence_rag.generator.models import Claim, DraftAnswer

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


class DraftGenerator:
    """Generate the unverified answer that A2 will split into claims."""

    def __init__(
        self,
        llm: TextGenerator | None = None,
        prompt_template: str = DRAFT_PROMPT,
    ) -> None:
        self.llm = llm or GraniteLLMClient()
        self.prompt_template = prompt_template

    def generate_answer(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> str:
        if query.query_id != checklist.query_id or query.query_id != selected.query_id:
            raise ValueError("query, checklist, and selected evidence query IDs differ")
        if not selected.evidence:
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
        answer = self.llm.generate(prompt).strip()
        normalized = answer.lower().strip(".!?\"' ")
        return "" if normalized in UNKNOWN_ANSWERS else answer


class DraftAnswerGenerator:
    """Compose A1 draft generation and A2 claim splitting into A's handoff."""

    def __init__(
        self,
        draft_generator: DraftTextGenerator | None = None,
        claim_splitter: ClaimsSplitter | None = None,
        llm: TextGenerator | None = None,
    ) -> None:
        shared_llm = llm
        if draft_generator is None or claim_splitter is None:
            shared_llm = shared_llm or GraniteLLMClient()
        self.draft_generator = draft_generator or DraftGenerator(llm=shared_llm)
        self.claim_splitter = claim_splitter or ClaimSplitter(llm=shared_llm)

    def generate(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> DraftAnswer:
        if query.query_id != checklist.query_id or query.query_id != selected.query_id:
            raise ValueError("query, checklist, and selected evidence query IDs differ")
        answer_text = self.draft_generator.generate_answer(query, checklist, selected)
        claims = self.claim_splitter.split(answer_text)
        return DraftAnswer(
            query_id=query.query_id,
            answer_text=answer_text,
            claims=claims,
        )
