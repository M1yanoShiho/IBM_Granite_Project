import pytest
from pydantic import ValidationError

from evidence_rag.generator.models import Claim, ClaimSpan, DraftAnswer


def claim(
    claim_id: str = "claim-1",
    *,
    start: int = 0,
    end: int = 16,
    faithful_to_answer: bool = True,
) -> Claim:
    return Claim(
        claim_id=claim_id,
        text="Revenue increased.",
        span=ClaimSpan(start=start, end=end),
        faithful_to_answer=faithful_to_answer,
    )


def test_draft_answer_keeps_claim_that_failed_faithfulness_check() -> None:
    unfaithful_claim = claim(faithful_to_answer=False)

    draft = DraftAnswer(
        query_id="q-1",
        answer_text="Revenue rose 8%.",
        claims=(unfaithful_claim,),
    )

    assert draft.claims == (unfaithful_claim,)
    assert draft.claims[0].faithful_to_answer is False


@pytest.mark.parametrize(("start", "end"), ((1, 1), (2, 1)))
def test_claim_span_requires_end_after_start(start: int, end: int) -> None:
    with pytest.raises(ValidationError, match="end must be greater than start"):
        ClaimSpan(start=start, end=end)


def test_empty_draft_answer_cannot_have_claims() -> None:
    with pytest.raises(ValidationError, match="empty answer_text cannot have claims"):
        DraftAnswer(query_id="q-1", answer_text="  ", claims=(claim(),))


def test_draft_answer_requires_unique_claim_ids() -> None:
    with pytest.raises(ValidationError, match="claim IDs must be unique"):
        DraftAnswer(
            query_id="q-1",
            answer_text="Revenue rose 8%.",
            claims=(claim(), claim()),
        )


def test_draft_answer_rejects_claim_span_beyond_answer_text() -> None:
    with pytest.raises(ValidationError, match="span exceeds answer_text length"):
        DraftAnswer(
            query_id="q-1",
            answer_text="Revenue rose 8%.",
            claims=(claim(end=17),),
        )
