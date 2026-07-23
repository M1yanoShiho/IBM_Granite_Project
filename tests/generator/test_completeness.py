import json

import pytest

from evidence_rag.contracts.models import QueryChecklist
from evidence_rag.generator.completeness import CompletenessChecker, fallback_gap_question

ANSWER = "Acme's global revenue was $1.2B in 2024."


class FakeLLM:
    """Returns one canned JSON verdict per call, in order."""

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


def checklist(*required_facts: str, constraints: tuple[str, ...] = ()) -> QueryChecklist:
    return QueryChecklist(
        query_id="q-1",
        focus="revenue",
        required_facts=required_facts,
        constraints=constraints,
    )


def test_covered_fact_carries_no_gap_question() -> None:
    checker = CompletenessChecker(llm=FakeLLM(json.dumps({"covered": True})))

    coverage = checker.check(ANSWER, checklist("global revenue 2024"))

    assert len(coverage) == 1
    assert coverage[0].covered is True
    assert coverage[0].gap_question is None


def test_uncovered_fact_carries_the_models_gap_question() -> None:
    llm = FakeLLM(
        json.dumps({"covered": True}),
        json.dumps(
            {"covered": False, "gap_question": "What was the company's UK revenue in 2024?"}
        ),
    )
    checker = CompletenessChecker(llm=llm)

    coverage = checker.check(ANSWER, checklist("global revenue 2024", "UK revenue 2024"))

    assert [item.covered for item in coverage] == [True, False]
    assert coverage[1].gap_question == "What was the company's UK revenue in 2024?"


def test_constraints_reach_the_prompt() -> None:
    llm = FakeLLM(json.dumps({"covered": True}))
    checker = CompletenessChecker(llm=llm)

    checker.check(ANSWER, checklist("global revenue 2024", constraints=("in USD",)))

    assert "in USD" in llm.prompts[0]


def test_empty_answer_marks_every_fact_uncovered_without_calling_the_model() -> None:
    llm = FakeLLM()
    checker = CompletenessChecker(llm=llm)

    coverage = checker.check("   ", checklist("global revenue 2024", "UK revenue 2024"))

    assert [item.covered for item in coverage] == [False, False]
    assert all(item.gap_question for item in coverage)
    assert llm.prompts == []


def test_missing_gap_question_falls_back_so_the_verdict_stays_constructible() -> None:
    checker = CompletenessChecker(llm=FakeLLM(json.dumps({"covered": False})))

    coverage = checker.check(ANSWER, checklist("UK revenue 2024"))

    assert coverage[0].gap_question == fallback_gap_question("UK revenue 2024")


def test_repeated_required_facts_are_judged_once() -> None:
    llm = FakeLLM(json.dumps({"covered": True}))
    checker = CompletenessChecker(llm=llm)

    coverage = checker.check(ANSWER, checklist("global revenue 2024", "global revenue 2024"))

    assert len(coverage) == 1
    assert len(llm.prompts) == 1


@pytest.mark.parametrize("response", ("not json", json.dumps(["covered"]), json.dumps({})))
def test_unusable_model_output_is_rejected(response: str) -> None:
    checker = CompletenessChecker(llm=FakeLLM(response))

    with pytest.raises(ValueError):
        checker.check(ANSWER, checklist("global revenue 2024"))
