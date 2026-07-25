import json
from typing import Any

from pydantic import model_validator

from evidence_rag.contracts.models import (
    FrozenModel,
    NonEmpty,
    QueryChecklist,
    SelectedEvidenceSet,
)
from evidence_rag.generator.granite import GraniteLLMClient, TextGenerator
from evidence_rag.generator.models import RequiredFactCoverage

RECHECK_PROMPT = (
    "Answer the gap question using only the numbered evidence below. "
    "Return JSON only as "
    '{{"found":true,"answer_fragment":"...","evidence_indices":[1]}} '
    "or "
    '{{"found":false,"answer_fragment":"","evidence_indices":[]}}.\n\n'
    "Evidence:\n{context}\n\n"
    "Focus: {focus}\n"
    "Constraints: {constraints}\n"
    "Gap question: {gap_question}"
)


class EvidenceRecheckResult(FrozenModel):
    required_fact: NonEmpty
    found: bool
    answer_fragment: str
    evidence_ids: tuple[NonEmpty, ...]

    @model_validator(mode="after")
    def _content_matches_found(self) -> "EvidenceRecheckResult":
        if self.found:
            if not self.answer_fragment.strip():
                raise ValueError("found result requires a nonempty answer fragment")
            if not self.evidence_ids:
                raise ValueError("found result requires supporting evidence")
        elif self.answer_fragment.strip() or self.evidence_ids:
            raise ValueError("not-found result cannot contain an answer or evidence")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("recheck evidence IDs must be unique")
        return self


class EvidenceRechecker:
    """Use a gap question to re-read selected evidence without new retrieval."""

    def __init__(
        self,
        llm: TextGenerator | None = None,
        prompt_template: str = RECHECK_PROMPT,
        max_fragment_chars: int = 1000,
    ) -> None:
        if max_fragment_chars < 1:
            raise ValueError("max_fragment_chars must be positive")
        self.llm = llm or GraniteLLMClient()
        self.prompt_template = prompt_template
        self.max_fragment_chars = max_fragment_chars

    def recheck(
        self,
        coverage: RequiredFactCoverage,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> EvidenceRecheckResult:
        if coverage.covered:
            raise ValueError("cannot recheck a fact already covered")
        if checklist.query_id != selected.query_id:
            raise ValueError("checklist and selected evidence query IDs differ")
        if not selected.evidence:
            return EvidenceRecheckResult(
                required_fact=coverage.required_fact,
                found=False,
                answer_fragment="",
                evidence_ids=(),
            )
        context = "\n".join(
            f"[{index}] ({item.evidence_id}) {item.text}"
            for index, item in enumerate(selected.evidence, start=1)
        )
        prompt = self.prompt_template.format(
            context=context,
            focus=checklist.focus,
            constraints="; ".join(checklist.constraints) or "none",
            gap_question=coverage.gap_question,
        )
        data = self._load_json(self.llm.generate(prompt))
        found = data.get("found")
        if not isinstance(found, bool):
            raise ValueError("recheck output must contain a boolean 'found'")
        raw_fragment = data.get("answer_fragment", "")
        if not isinstance(raw_fragment, str):
            raise ValueError("recheck answer_fragment must be a string")
        if len(raw_fragment.strip()) > self.max_fragment_chars:
            raise ValueError("recheck answer_fragment is too long")
        indices = data.get("evidence_indices", [])
        if not isinstance(indices, list):
            raise ValueError("recheck evidence_indices must be an array")

        evidence_ids: list[str] = []
        seen: set[str] = set()
        for raw_index in indices:
            if not isinstance(raw_index, int) or isinstance(raw_index, bool):
                raise ValueError("recheck evidence indices must be integers")
            if not 1 <= raw_index <= len(selected.evidence):
                raise ValueError(f"recheck evidence index out of range: {raw_index}")
            evidence_id = selected.evidence[raw_index - 1].evidence_id
            if evidence_id not in seen:
                seen.add(evidence_id)
                evidence_ids.append(evidence_id)

        return EvidenceRecheckResult(
            required_fact=coverage.required_fact,
            found=found,
            answer_fragment=raw_fragment.strip(),
            evidence_ids=tuple(evidence_ids),
        )

    @staticmethod
    def _load_json(raw: str) -> dict[str, Any]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM output must be valid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("LLM output must be a JSON object")
        return data
