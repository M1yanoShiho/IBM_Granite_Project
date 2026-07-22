from evidence_rag.contracts.models import (
    EvidenceCandidate,
    Query,
    QueryChecklist,
    SelectedEvidenceSet,
)
from evidence_rag.generator.granite import GraniteGenerator, parse_citation_output


def checklist(query_id: str) -> QueryChecklist:
    return QueryChecklist(query_id=query_id, focus="what changed", required_facts=("change",))


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


def test_parse_citation_output_extracts_answer_and_indices() -> None:
    answer, indices = parse_citation_output(
        "Answer: Revenue increased by ten percent.\nEvidence: [2], [1], [2]"
    )

    assert answer == "Revenue increased by ten percent."
    assert indices == (2, 1)


def test_granite_generator_maps_citation_numbers_to_evidence_ids() -> None:
    llm = FakeLLM("Answer: Revenue increased by ten percent.\nEvidence: [2]")
    generator = GraniteGenerator(llm=llm)
    selected = SelectedEvidenceSet(
        query_id="q",
        evidence=(
            evidence("ev-a", "Profit was stable."),
            evidence("ev-b", "Revenue increased by ten percent."),
        ),
    )

    result = generator.generate(Query(query_id="q", text="What changed?"), checklist("q"), selected)

    assert result.answer == "Revenue increased by ten percent."
    assert result.cited_evidence_ids == ("ev-b",)
    assert "[2] (ev-b) Revenue increased by ten percent." in llm.prompts[0]


def test_granite_generator_returns_empty_when_model_declines() -> None:
    generator = GraniteGenerator(llm=FakeLLM("Answer: I don't know\nEvidence: []"))
    selected = SelectedEvidenceSet(
        query_id="q",
        evidence=(evidence("ev-a", "Profit was stable."),),
    )

    result = generator.generate(Query(query_id="q", text="What changed?"), checklist("q"), selected)

    assert result.answer == ""
    assert result.cited_evidence_ids == ()
