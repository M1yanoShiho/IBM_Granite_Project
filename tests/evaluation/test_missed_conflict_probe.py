from evidence_rag.evaluation.missed_conflict_probe import (
    PROMPTS,
    STAGE_A_PROMPT,
    STAGE_B_PROMPT,
    PairOutcome,
    classify_pair,
    summarize_prompt,
)


def test_prompts_include_baseline_and_targeted() -> None:
    assert "baseline" in PROMPTS
    assert {"verbatim", "attribute"} <= set(PROMPTS)
    for template in PROMPTS.values():
        assert "{question}" in template and "{passage}" in template


def test_decoupled_stage_prompts_have_placeholders() -> None:
    assert "{question}" in STAGE_A_PROMPT
    # Stage B must carry the question too: the target type alone ("a person's name") cannot
    # disambiguate which entity in the passage is being asked for.
    assert "{target}" in STAGE_B_PROMPT
    assert "{question}" in STAGE_B_PROMPT
    assert "{passage}" in STAGE_B_PROMPT


def test_lenient_matcher_credits_verbose_answer() -> None:
    from evidence_rag.selector.answer_equivalence import lenient_equivalent

    # A wordier but correct answer: exact scoring calls it wrong, lenient credits it.
    exact = classify_pair(
        "The answer is Kennedy", "Nixon", gold_value="Kennedy", replacement_value="Nixon"
    )
    assert exact.needle_gold is False
    lenient = classify_pair(
        "The answer is Kennedy",
        "Nixon",
        gold_value="Kennedy",
        replacement_value="Nixon",
        equivalent=lenient_equivalent,
    )
    assert lenient.needle_gold is True
    assert lenient.missed_conflict is False


def test_classify_twins_collapsed() -> None:
    o = classify_pair("Kennedy", "Kennedy", gold_value="Kennedy", replacement_value="Nixon")
    assert o.missed_conflict is True
    assert o.needle_gold is True
    assert o.cf_replacement is False


def test_classify_twins_separated() -> None:
    o = classify_pair("Kennedy", "Nixon", gold_value="Kennedy", replacement_value="Nixon")
    assert o.missed_conflict is False
    assert o.needle_gold is True
    assert o.cf_replacement is True


def test_classify_collapsed_on_shared_wrong_entity() -> None:
    # both twins yield the same NON-swapped entity -> gate blind, neither value recovered
    o = classify_pair("Paris", "Paris", gold_value="Kennedy", replacement_value="Nixon")
    assert o.missed_conflict is True
    assert o.needle_gold is False
    assert o.cf_replacement is False


def test_classify_needle_none_is_not_missed_conflict() -> None:
    o = classify_pair("NONE", "Nixon", gold_value="Kennedy", replacement_value="Nixon")
    assert o.missed_conflict is False
    assert o.needle_gold is False
    assert o.cf_replacement is True


def test_summarize_rates() -> None:
    outcomes = [
        PairOutcome(missed_conflict=True, needle_gold=True, cf_replacement=False),
        PairOutcome(missed_conflict=False, needle_gold=True, cf_replacement=True),
        PairOutcome(missed_conflict=True, needle_gold=False, cf_replacement=False),
        PairOutcome(missed_conflict=False, needle_gold=True, cf_replacement=True),
    ]
    s = summarize_prompt("baseline", outcomes)
    assert s.prompt == "baseline"
    assert s.n == 4
    assert s.missed_conflict == 2
    assert s.missed_conflict_rate == 0.5
    assert s.needle_gold_rate == 0.75
    assert s.cf_replacement_rate == 0.5
