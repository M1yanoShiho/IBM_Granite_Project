"""Checklist construction: the rule-based floor fix and the LLM analyzer.

Two previous checklist constructions failed in ways automatic metrics could not
see -- one measured whether the draft restated background provenance notes, the
other emitted keyword bags. These tests pin the properties that made those
failures possible.
"""

from evidence_rag.contracts.models import Query
from evidence_rag.query_analysis import (
    GraniteQueryAnalyzer,
    RuleBasedQueryAnalyzer,
)


def _query(text: str) -> Query:
    return Query(query_id="q-1", text=text)


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


# --- Route A floor fix ------------------------------------------------------


def test_rule_based_emits_no_requirement_when_the_metrics_vocabulary_misses() -> None:
    """The old fallback used `focus` -- a bag of question tokens -- as a required
    fact, which asked the completeness checker a malformed question on every
    open-domain query. Emitting nothing is the correct floor."""
    checklist = RuleBasedQueryAnalyzer().analyze(
        _query("Where can adipose tissue be found in the body?")
    )

    assert checklist.required_facts == ()
    assert checklist.focus  # focus is still populated; it is a topic, not a fact


def test_rule_based_still_extracts_in_domain_metrics() -> None:
    checklist = RuleBasedQueryAnalyzer().analyze(
        _query("What was the revenue growth in 2024?")
    )

    assert set(checklist.required_facts) == {"revenue", "growth"}
    assert "year:2024" in checklist.constraints


# --- Route B LLM analyzer ---------------------------------------------------


def test_llm_analyzer_sees_only_the_question_text() -> None:
    llm = FakeLLM('{"requirements": []}')
    question = "Who has the highest goals in world football?"

    GraniteQueryAnalyzer(llm).analyze(_query(question))

    assert len(llm.prompts) == 1
    assert question in llm.prompts[0]


def test_llm_analyzer_returns_requirements_deduped_and_capped() -> None:
    llm = FakeLLM(
        '{"requirements": ["The answer must specify the men\'s record holder.",'
        ' "The answer must specify the men\'s record holder.",'
        ' "The answer must specify the women\'s record holder."]}'
    )

    checklist = GraniteQueryAnalyzer(llm, max_requirements=2).analyze(
        _query("Who has the highest goals in world football?")
    )

    assert checklist.required_facts == (
        "The answer must specify the men's record holder.",
        "The answer must specify the women's record holder.",
    )


def test_llm_analyzer_allows_an_empty_checklist() -> None:
    """An unambiguous question carries no completeness obligation. Forcing at
    least one item per query is what made the rule-based fallback always fire."""
    checklist = GraniteQueryAnalyzer(FakeLLM('{"requirements": []}')).analyze(
        _query("Who wrote Hamlet?")
    )

    assert checklist.required_facts == ()


def test_llm_analyzer_tolerates_fenced_and_trailing_prose() -> None:
    llm = FakeLLM(
        '```json\n{"requirements": ["The answer must specify the year."]}\n```\n'
        "I listed one requirement because the question is about timing."
    )

    checklist = GraniteQueryAnalyzer(llm).analyze(_query("When did it open?"))

    assert checklist.required_facts == ("The answer must specify the year.",)


def test_llm_analyzer_degrades_to_empty_on_unusable_output() -> None:
    """An unusable response means 'no known obligation', which is a legal state.
    Raising would drop the query and bias the sample, which is how the recheck
    JSON crash cost 5.5% of a previous run."""
    for response in ("not json at all", '{"requirements": "not a list"}', "{}"):
        checklist = GraniteQueryAnalyzer(FakeLLM(response)).analyze(_query("Who wrote Hamlet?"))
        assert checklist.required_facts == ()


def test_llm_analyzer_still_reports_focus_and_constraints() -> None:
    checklist = GraniteQueryAnalyzer(FakeLLM('{"requirements": []}')).analyze(
        _query("What happened to Boeing in 2019?")
    )

    assert checklist.focus
    assert "year:2019" in checklist.constraints
    assert "entity:Boeing" in checklist.constraints
