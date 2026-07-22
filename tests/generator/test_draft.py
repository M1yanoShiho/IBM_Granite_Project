import pytest

from evidence_rag.contracts.models import EvidenceCandidate, Query, SelectedEvidenceSet
from evidence_rag.generator.draft import DraftGenerator


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def evidence(evidence_id: str, text: str) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=f"doc-{evidence_id}",
        chunk_id=f"chunk-{evidence_id}",
        text=text,
        source_uri=f"fixture://{evidence_id}",
        retrieval_score=1.0,
        retrieval_rank=1,
    )


def test_draft_generator_answers_using_selected_evidence() -> None:
    llm = FakeLLM("Revenue increased by ten percent.")
    generator = DraftGenerator(llm=llm)
    selected = SelectedEvidenceSet(
        query_id="q-1",
        evidence=(evidence("ev-1", "Revenue increased by ten percent."),),
    )

    answer = generator.generate_answer(
        Query(query_id="q-1", text="What changed?"),
        selected,
    )

    assert answer == "Revenue increased by ten percent."
    assert "What changed?" in llm.prompts[0]
    assert "[1] (ev-1) Revenue increased by ten percent." in llm.prompts[0]


def test_draft_generator_rejects_query_id_mismatch() -> None:
    generator = DraftGenerator(llm=FakeLLM("unused"))
    selected = SelectedEvidenceSet(query_id="q-other", evidence=())

    with pytest.raises(ValueError, match="query IDs differ"):
        generator.generate_answer(Query(query_id="q-1", text="What changed?"), selected)


def test_draft_generator_returns_empty_without_selected_evidence() -> None:
    llm = FakeLLM("must not be used")
    generator = DraftGenerator(llm=llm)

    answer = generator.generate_answer(
        Query(query_id="q-1", text="What changed?"),
        SelectedEvidenceSet(query_id="q-1", evidence=()),
    )

    assert answer == ""
    assert llm.prompts == []


def test_draft_generator_normalizes_model_decline_to_empty_answer() -> None:
    generator = DraftGenerator(llm=FakeLLM("I don't know."))
    selected = SelectedEvidenceSet(
        query_id="q-1",
        evidence=(evidence("ev-1", "Profit was stable."),),
    )

    answer = generator.generate_answer(
        Query(query_id="q-1", text="What was the revenue?"),
        selected,
    )

    assert answer == ""


def test_draft_generator_includes_every_selected_evidence_in_rank_order() -> None:
    llm = FakeLLM("Combined answer.")
    generator = DraftGenerator(llm=llm)
    selected = SelectedEvidenceSet(
        query_id="q-1",
        evidence=(
            evidence("ev-1", "First fact."),
            evidence("ev-2", "Second fact."),
        ),
    )

    generator.generate_answer(Query(query_id="q-1", text="Summarize."), selected)

    prompt = llm.prompts[0]
    assert prompt.index("[1] (ev-1) First fact.") < prompt.index("[2] (ev-2) Second fact.")
