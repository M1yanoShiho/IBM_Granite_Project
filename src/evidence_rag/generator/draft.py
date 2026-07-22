from typing import Protocol

from evidence_rag.contracts.models import Query, SelectedEvidenceSet
from evidence_rag.generator.claim_splitter import ClaimSplitter
from evidence_rag.generator.granite import GraniteLLMClient, TextGenerator
from evidence_rag.generator.models import Claim, DraftAnswer

DRAFT_PROMPT = (
    "Answer the question using only the evidence below.\n"
    "If the evidence does not contain the answer, say: I don't know.\n\n"
    "Evidence:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

UNKNOWN_ANSWERS = {
    "",
    "unknown",
    "i don't know",
    "i do not know",
    "not in the evidence",
    "not contained in the evidence",
}


class DraftTextGenerator(Protocol):
    def generate_answer(self, query: Query, selected: SelectedEvidenceSet) -> str: ...


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

    def generate_answer(self, query: Query, selected: SelectedEvidenceSet) -> str:
        if query.query_id != selected.query_id:
            raise ValueError("query and selected evidence query IDs differ")
        if not selected.evidence:
            return ""
        context = "\n".join(
            f"[{index}] ({item.evidence_id}) {item.text}"
            for index, item in enumerate(selected.evidence, start=1)
        )
        prompt = self.prompt_template.format(context=context, question=query.text)
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

    def generate(self, query: Query, selected: SelectedEvidenceSet) -> DraftAnswer:
        if query.query_id != selected.query_id:
            raise ValueError("query and selected evidence query IDs differ")
        answer_text = self.draft_generator.generate_answer(query, selected)
        claims = self.claim_splitter.split(answer_text)
        return DraftAnswer(
            query_id=query.query_id,
            answer_text=answer_text,
            claims=claims,
        )
