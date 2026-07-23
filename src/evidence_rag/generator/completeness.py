"""B4 -- completeness: does the draft answer cover every required fact?

The other half of the checklist-driven loop (plan section 3, step 4). Where B2
deletes what is unsupported, this finds what is *missing* and hands A4 a
concrete sub-question to re-read the already-selected evidence with -- never a
new retrieval.
"""

import json
from typing import Any

from evidence_rag.contracts.models import QueryChecklist
from evidence_rag.generator.granite import GraniteLLMClient, TextGenerator
from evidence_rag.generator.models import RequiredFactCoverage

COVERAGE_PROMPT = (
    "Decide whether the answer already states the required fact.\n"
    "If it does not, write one specific question that asks for exactly the missing fact.\n"
    "Return JSON only in this shape: "
    '{{"covered": false, "gap_question": "..."}} or {{"covered": true}}.\n\n'
    "Required fact: {required_fact}\n"
    "Constraints: {constraints}\n\n"
    "Answer:\n{answer}"
)


def fallback_gap_question(required_fact: str) -> str:
    """Used when the answer is empty (no LLM call is worth making) or when the
    model claimed a gap but gave no usable question. ``RequiredFactCoverage``
    requires an uncovered fact to carry one, so there must always be a value."""
    return f"What do the sources say about {required_fact}?"


class CompletenessChecker:
    """One LLM call per required fact.

    Deliberately not batched: a 3B model returns far more reliable JSON for a
    single yes/no-plus-question judgement than for a whole array, and a malformed
    verdict then only costs one fact instead of the entire report.
    """

    def __init__(
        self,
        llm: TextGenerator | None = None,
        prompt_template: str = COVERAGE_PROMPT,
    ) -> None:
        self.llm = llm or GraniteLLMClient()
        self.prompt_template = prompt_template

    def check(
        self,
        answer_text: str,
        checklist: QueryChecklist,
    ) -> tuple[RequiredFactCoverage, ...]:
        # QueryChecklist does not enforce unique required_facts but
        # VerificationReport does, so collapse repeats before judging them.
        required_facts = tuple(dict.fromkeys(checklist.required_facts))
        if not answer_text.strip():
            # A5 declined to answer: nothing can be covered, and asking the model
            # to confirm that for every fact is a wasted call.
            return tuple(
                RequiredFactCoverage(
                    required_fact=fact,
                    covered=False,
                    gap_question=fallback_gap_question(fact),
                )
                for fact in required_facts
            )

        constraints = "; ".join(checklist.constraints) or "none"
        coverage: list[RequiredFactCoverage] = []
        for fact in required_facts:
            prompt = self.prompt_template.format(
                required_fact=fact,
                constraints=constraints,
                answer=answer_text,
            )
            data = _load_json(self.llm.generate(prompt))
            covered = data.get("covered")
            if not isinstance(covered, bool):
                raise ValueError("coverage output must contain a boolean 'covered'")
            gap_question: str | None = None
            if not covered:
                raw_question = data.get("gap_question")
                gap_question = (
                    raw_question.strip()
                    if isinstance(raw_question, str) and raw_question.strip()
                    else fallback_gap_question(fact)
                )
            coverage.append(
                RequiredFactCoverage(
                    required_fact=fact,
                    covered=covered,
                    gap_question=gap_question,
                )
            )
        return tuple(coverage)


def _load_json(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM output must be valid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("LLM output must be a JSON object")
    return data
