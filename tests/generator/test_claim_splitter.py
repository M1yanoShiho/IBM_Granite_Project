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


def test_claim_splitter_rejects_source_text_not_found_in_answer() -> None:
    llm = FakeLLM(
        [
            '{"claims":[{"source_text":"Revenue fell.",'
            '"text":"Revenue fell."}]}',
        ]
    )

    with pytest.raises(ValueError, match="source_text is not in answer_text"):
        ClaimSplitter(llm=llm).split("Revenue rose.")


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
