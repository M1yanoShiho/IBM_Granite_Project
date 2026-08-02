import pytest

from evidence_rag.generator.claim_splitter import ClaimSplitter


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return next(self.responses)


def test_claim_splitter_extracts_and_verifies_one_claim() -> None:
    llm = FakeLLM(
        [
            '{"claims":[{"source_text":"Revenue rose 8%.",'
            '"text":"Revenue rose by 8%."}]}',
            '{"results":[{"claim_id":"claim-1","faithful":true}]}',
        ]
    )

    claims = ClaimSplitter(llm=llm).split("Revenue rose 8%.")

    assert len(claims) == 1
    assert claims[0].claim_id == "claim-1"
    assert claims[0].text == "Revenue rose by 8%."
    assert claims[0].span.start == 0
    assert claims[0].span.end == 16
    assert claims[0].faithful_to_answer is True


def test_claim_splitter_returns_no_claims_for_empty_answer() -> None:
    llm = FakeLLM([])

    claims = ClaimSplitter(llm=llm).split("  ")

    assert claims == ()
    assert llm.prompts == []


def test_claim_splitter_makes_multiple_claims_self_contained() -> None:
    answer = "Pfizer raised revenue. It launched Product X."
    llm = FakeLLM(
        [
            '{"claims":['
            '{"source_text":"Pfizer raised revenue.","text":"Pfizer raised revenue."},'
            '{"source_text":"It launched Product X.",'
            '"text":"Pfizer launched Product X."}]}',
            '{"results":['
            '{"claim_id":"claim-1","faithful":true},'
            '{"claim_id":"claim-2","faithful":true}]}',
        ]
    )

    claims = ClaimSplitter(llm=llm).split(answer)

    assert tuple(item.claim_id for item in claims) == ("claim-1", "claim-2")
    assert claims[1].text == "Pfizer launched Product X."
    assert answer[claims[1].span.start : claims[1].span.end] == "It launched Product X."


def test_claim_splitter_keeps_claim_that_fails_faithfulness_check() -> None:
    llm = FakeLLM(
        [
            '{"claims":[{"source_text":"Revenue may rise.",'
            '"text":"Revenue will rise."}]}',
            '{"results":[{"claim_id":"claim-1","faithful":false}]}',
        ]
    )

    claims = ClaimSplitter(llm=llm).split("Revenue may rise.")

    assert len(claims) == 1
    assert claims[0].faithful_to_answer is False


def test_claim_splitter_anchors_paraphrased_source_text_to_a_sentence() -> None:
    # real Granite often paraphrases source_text instead of quoting it verbatim;
    # the claim anchors to the best-overlapping answer sentence rather than raising
    answer = "Pfizer's revenue increased by eight percent this year."
    llm = FakeLLM(
        [
            '{"claims":[{"source_text":"Revenue rose 8%.",'
            '"text":"Revenue rose by 8%."}]}',
            '{"results":[{"claim_id":"claim-1","faithful":true}]}',
        ]
    )

    claims = ClaimSplitter(llm=llm).split(answer)

    assert len(claims) == 1
    assert answer[claims[0].span.start : claims[0].span.end] == answer


def test_claim_splitter_skips_unlocatable_claim() -> None:
    # a source_text that overlaps no answer sentence is dropped, not fatal, and
    # no faithfulness call is made because nothing was located
    llm = FakeLLM(
        [
            '{"claims":[{"source_text":"Bananas are a yellow fruit.",'
            '"text":"Bananas are yellow."}]}',
        ]
    )

    claims = ClaimSplitter(llm=llm).split("Revenue rose.")

    assert claims == ()
    assert len(llm.prompts) == 1


def test_claim_splitter_rejects_malformed_json_output() -> None:
    llm = FakeLLM(["not JSON"])

    with pytest.raises(ValueError, match="valid JSON"):
        ClaimSplitter(llm=llm).split("Revenue rose.")


def test_claim_splitter_requires_one_faithfulness_result_per_claim() -> None:
    llm = FakeLLM(
        [
            '{"claims":[{"source_text":"Revenue rose.",'
            '"text":"Revenue rose."}]}',
            '{"results":[]}',
        ]
    )

    with pytest.raises(ValueError, match="exactly the split claims"):
        ClaimSplitter(llm=llm).split("Revenue rose.")


def test_claim_splitter_skips_faithfulness_check_when_no_claims_are_found() -> None:
    llm = FakeLLM(['{"claims":[]}'])

    claims = ClaimSplitter(llm=llm).split("This is a heading.")

    assert claims == ()
    assert len(llm.prompts) == 1


def test_claim_splitter_rejects_json_without_claims_array() -> None:
    llm = FakeLLM(['{"answer":"Revenue rose."}'])

    with pytest.raises(ValueError, match="claims array"):
        ClaimSplitter(llm=llm).split("Revenue rose.")


def test_claim_splitter_rejects_claim_without_required_text_fields() -> None:
    llm = FakeLLM(['{"claims":[{"text":"Revenue rose."}]}'])

    with pytest.raises(ValueError, match="source_text and text"):
        ClaimSplitter(llm=llm).split("Revenue rose.")


def test_claim_splitter_drops_meta_narrative_claims() -> None:
    # "the information is sourced from..." talks about the answer, not the world
    answer = "The festival ended on November 16. The information is sourced from the 2015 event details."
    llm = FakeLLM(
        [
            '{"claims":['
            '{"source_text":"The festival ended on November 16.","text":"The festival ended on November 16."},'
            '{"source_text":"The information is sourced from the 2015 event details.",'
            '"text":"The information is sourced from the 2015 event details."}]}',
            '{"results":[{"claim_id":"claim-1","faithful":true}]}',
        ]
    )

    claims = ClaimSplitter(llm=llm).split(answer)

    assert [c.text for c in claims] == ["The festival ended on November 16."]
    # the faithfulness call only ever saw the surviving claim
    assert "sourced from" not in llm.prompts[1]


def test_claim_splitter_drops_claims_with_an_unresolved_subject() -> None:
    # "This film ..." is not self-contained: nothing downstream knows which film
    answer = "The movie is titled Sunshine. This film aired on NBC in 1973."
    llm = FakeLLM(
        [
            '{"claims":['
            '{"source_text":"The movie is titled Sunshine.","text":"The movie is titled Sunshine."},'
            '{"source_text":"This film aired on NBC in 1973.","text":"This film aired on NBC in 1973."}]}',
            '{"results":[{"claim_id":"claim-1","faithful":true}]}',
        ]
    )

    claims = ClaimSplitter(llm=llm).split(answer)

    assert [c.text for c in claims] == ["The movie is titled Sunshine."]


def test_claim_splitter_drops_a_restatement_subsumed_by_a_more_specific_claim() -> None:
    """The shape actually observed in the G3 data: overlapping claim spans that
    duplicated text in the repaired answer ("West Germany won the World Cup in
    1954 West Germany won the World Cup in 1954 and again in 1974.")."""
    answer = "West Germany won the World Cup in 1954 and again in 1974."
    llm = FakeLLM(
        [
            '{"claims":['
            '{"source_text":"West Germany won the World Cup in 1954","text":"West Germany won the World Cup in 1954."},'
            '{"source_text":"West Germany won the World Cup in 1954 and again in 1974.",'
            '"text":"West Germany won the World Cup in 1954 and again in 1974."}]}',
            '{"results":[{"claim_id":"claim-2","faithful":true}]}',
        ]
    )

    claims = ClaimSplitter(llm=llm).split(answer)

    assert len(claims) == 1
    assert "1974" in claims[0].text  # the more specific claim is the one kept


def test_claim_splitter_keeps_distinct_claims_about_the_same_subject() -> None:
    # guard against the redundancy rule being too eager
    answer = "Sunshine aired on NBC in 1973. Sunshine starred Cliff DeYoung and Cristina Raines."
    llm = FakeLLM(
        [
            '{"claims":['
            '{"source_text":"Sunshine aired on NBC in 1973.","text":"Sunshine aired on NBC in 1973."},'
            '{"source_text":"Sunshine starred Cliff DeYoung and Cristina Raines.",'
            '"text":"Sunshine starred Cliff DeYoung and Cristina Raines."}]}',
            '{"results":[{"claim_id":"claim-1","faithful":true},{"claim_id":"claim-2","faithful":true}]}',
        ]
    )

    claims = ClaimSplitter(llm=llm).split(answer)

    assert len(claims) == 2


def test_claim_splitter_rejects_faithfulness_output_without_results_array() -> None:
    llm = FakeLLM(
        [
            '{"claims":[{"source_text":"Revenue rose.",'
            '"text":"Revenue rose."}]}',
            '{"faithful":true}',
        ]
    )

    with pytest.raises(ValueError, match="results array"):
        ClaimSplitter(llm=llm).split("Revenue rose.")
