import pytest

from evidence_rag.contracts.models import (
    EvidenceCandidate,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
)
from evidence_rag.generator.draft import DraftAnswerGenerator
from evidence_rag.generator.models import Claim, ClaimSpan


class FakeDraftTextGenerator:
    def __init__(self, answer_text: str) -> None:
        self.answer_text = answer_text

    def generate_answer(
        self,
        query: Query,
        checklist: QueryChecklist,
        selected: SelectedEvidenceSet,
    ) -> str:
        return self.answer_text


class FakeClaimSplitter:
    def __init__(self, claims: tuple[Claim, ...]) -> None:
        self.claims = claims
        self.answers: list[str] = []

    def split(self, answer_text: str) -> tuple[Claim, ...]:
        self.answers.append(answer_text)
        return self.claims


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)

    def generate(self, prompt: str) -> str:
        return next(self.responses)


def selected_evidence(query_id: str = "q-1") -> SelectedEvidenceSet:
    evidence = EvidenceCandidate(
        evidence_id="ev-1",
        document_id="doc-1",
        chunk_id="chunk-1",
        text="Revenue rose 8%.",
        source_uri="fixture://ev-1",
        retrieval_score=1.0,
        retrieval_rank=1,
    )
    return SelectedEvidenceSet(query_id=query_id, evidence=(evidence,))


def checklist(query_id: str = "q-1") -> QueryChecklist:
    return QueryChecklist(query_id=query_id, focus="performance", required_facts=())


def test_draft_answer_generator_composes_text_and_claims() -> None:
    claim = Claim(
        claim_id="claim-1",
        text="Revenue rose by 8%.",
        span=ClaimSpan(start=0, end=16),
        faithful_to_answer=True,
    )
    splitter = FakeClaimSplitter((claim,))
    generator = DraftAnswerGenerator(
        draft_generator=FakeDraftTextGenerator("Revenue rose 8%."),
        claim_splitter=splitter,
    )

    draft = generator.generate(
        Query(query_id="q-1", text="What changed?"),
        checklist(),
        selected_evidence(),
    )

    assert draft.query_id == "q-1"
    assert draft.answer_text == "Revenue rose 8%."
    assert draft.claims == (claim,)
    assert splitter.answers == ["Revenue rose 8%."]


def test_draft_answer_generator_rejects_query_id_mismatch() -> None:
    generator = DraftAnswerGenerator(
        draft_generator=FakeDraftTextGenerator("unused"),
        claim_splitter=FakeClaimSplitter(()),
    )

    with pytest.raises(ValueError, match="query IDs differ"):
        generator.generate(
            Query(query_id="q-1", text="What changed?"),
            checklist(),
            selected_evidence(query_id="q-other"),
        )


def test_draft_answer_generator_returns_empty_draft_when_a1_declines() -> None:
    splitter = FakeClaimSplitter(())
    generator = DraftAnswerGenerator(
        draft_generator=FakeDraftTextGenerator(""),
        claim_splitter=splitter,
    )

    draft = generator.generate(
        Query(query_id="q-1", text="What changed?"),
        checklist(),
        selected_evidence(),
    )

    assert draft.answer_text == ""
    assert draft.claims == ()
    assert splitter.answers == [""]


def test_draft_answer_generator_preserves_unfaithful_claim_for_b() -> None:
    claim = Claim(
        claim_id="claim-1",
        text="Revenue will rise.",
        span=ClaimSpan(start=0, end=17),
        faithful_to_answer=False,
    )
    generator = DraftAnswerGenerator(
        draft_generator=FakeDraftTextGenerator("Revenue may rise."),
        claim_splitter=FakeClaimSplitter((claim,)),
    )

    draft = generator.generate(
        Query(query_id="q-1", text="What may happen?"),
        checklist(),
        selected_evidence(),
    )

    assert draft.claims == (claim,)
    assert draft.claims[0].faithful_to_answer is False


def test_draft_answer_generator_runs_a1_and_a2_with_one_shared_llm() -> None:
    llm = FakeLLM(
        [
            "Revenue rose 8%.",
            '{"claims":[{"source_text":"Revenue rose 8%.",'
            '"text":"Revenue rose by 8%."}]}',
            '{"results":[{"claim_id":"claim-1","faithful":true}]}',
        ]
    )

    draft = DraftAnswerGenerator(llm=llm).generate(
        Query(query_id="q-1", text="What changed?"),
        checklist(),
        selected_evidence(),
    )

    assert draft.answer_text == "Revenue rose 8%."
    assert tuple(item.text for item in draft.claims) == ("Revenue rose by 8%.",)
