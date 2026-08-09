"""ABLATION-ONLY. Not on the live path.

Part of the retired generate -> verify -> patch design (roles A/B), reached only
through ``verified.VerifiedGenerator``, which exists solely as the ``verify-only``
arm -- the published delete-filter baseline the main method is measured against.
Deleting it would make that comparison irreproducible. The live path is
``verify_annotate.VerifyAnnotateGenerator``; do not extend this code. See
``docs/generator/design-review.md``.

B4 -- completeness: does the draft answer cover every required fact?

The other half of the checklist-driven loop (plan section 3, step 4). Where B2
deletes what is unsupported, this finds what is *missing* and hands A4 a
concrete sub-question to re-read the already-selected evidence with -- never a
new retrieval.
"""

from evidence_rag.contracts.models import QueryChecklist
from evidence_rag.generator.granite import GraniteLLMClient, TextGenerator
from evidence_rag.generator.json_parsing import parse_json_object
from evidence_rag.generator.models import RequiredFactCoverage

COVERAGE_PROMPT = (
    "Decide whether the answer satisfies the requirement below.\n"
    "If it does not, write one specific question that asks for exactly what is missing.\n"
    "Return JSON only in this shape: "
    '{{"covered": false, "gap_question": "..."}} or {{"covered": true}}.\n\n'
    "Requirement: {required_fact}\n"
    "Constraints: {constraints}\n\n"
    "Answer:\n{answer}"
)
"""Phrased against *requirements* because that is what the checklist now carries.

``GraniteQueryAnalyzer`` emits coverage obligations ("The answer must specify the
men's international record holder") rather than assertions, so asking whether the
answer "states this fact" would be asking about the wrong kind of object. The
placeholder name stays ``required_fact`` so no call site changes."""


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


def _load_json(raw: str) -> dict[str, object]:
    # tolerate the trailing prose a real Granite model appends after the JSON
    return parse_json_object(raw)
