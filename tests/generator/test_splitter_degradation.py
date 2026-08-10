"""The splitter's failure floor.

Post-freeze, above `generator-frozen-g9-qampari-2026-08-10`. It changes no
reported number: this path only runs where the frozen code raised.

The defect was never that parsing fails -- it is that a parse failure was FATAL.
The splitter sits upstream of every verification arm, so one malformed response
removed the query from all of them while the baseline kept it. On QAMPARI that
ran at 18/400 = 4.5% against ~0.5% on calibration, and the baseline answered 17
of those 18 -- which is what made the coverage comparison untestable.
"""

import pytest

from evidence_rag.generator.claim_splitter import ClaimSplitter


class BrokenLLM:
    """Emits something the structured parser cannot use."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return self.payload


MALFORMED = (
    "not json at all",
    '{"claims": "not a list"}',
    '{"claims": [{"source_text": "", "text": ""}]}',
    '{"wrong_key": []}',
    "",
)


@pytest.mark.parametrize("payload", MALFORMED)
def test_a_malformed_response_no_longer_loses_the_query(payload: str) -> None:
    answer = "Revenue rose 8%. Costs fell sharply."
    claims = ClaimSplitter(llm=BrokenLLM(payload)).split(answer)

    assert claims, f"query lost on payload {payload!r}"
    assert [c.text for c in claims] == ["Revenue rose 8%.", "Costs fell sharply."]


@pytest.mark.parametrize("payload", MALFORMED)
def test_strict_mode_still_raises(payload: str) -> None:
    """The frozen behaviour stays reachable and stays tested."""
    with pytest.raises((ValueError, KeyError, TypeError)):
        ClaimSplitter(llm=BrokenLLM(payload), degrade_on_failure=False).split(
            "Revenue rose 8%."
        )


def test_degraded_claims_are_marked_as_such() -> None:
    """A sentence-level claim is a weaker unit than a real split -- one carrying
    two facts needs both supported to verify -- so it must not pass for a normal
    one."""
    claims = ClaimSplitter(llm=BrokenLLM("not json")).split("Alpha holds. Beta holds.")

    assert all(c.degraded for c in claims)
    assert all(c.faithful_to_answer for c in claims)  # it IS the answer, verbatim


def test_a_successful_split_is_not_marked_degraded() -> None:
    class GoodLLM:
        def generate(self, prompt: str) -> str:
            if prompt.lstrip().startswith("Check whether"):
                return '{"results":[{"claim_id":"claim-1","faithful":true}]}'
            return '{"claims":[{"source_text":"Revenue rose 8%.","text":"Revenue rose 8%."}]}'

    claims = ClaimSplitter(llm=GoodLLM()).split("Revenue rose 8%.")

    assert claims and not any(c.degraded for c in claims)


def test_degraded_spans_point_at_the_real_answer_text() -> None:
    """Spans are what the assembler and the citation router use, so a degraded
    claim still has to be locatable in the draft."""
    answer = "Mount St. Helens erupted in 1980. Rome fell earlier."
    claims = ClaimSplitter(llm=BrokenLLM("not json")).split(answer)

    for claim in claims:
        assert answer[claim.span.start : claim.span.end] == claim.text
    # the shared abbreviation rule applies here too -- "St." is not a boundary
    assert len(claims) == 2


def test_an_empty_answer_still_yields_nothing() -> None:
    """Degrading must not invent claims where there was no answer."""
    assert ClaimSplitter(llm=BrokenLLM("not json")).split("   ") == ()
