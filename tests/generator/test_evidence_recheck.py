import pytest

from evidence_rag.contracts.models import EvidenceCandidate, SelectedEvidenceSet
from evidence_rag.generator.evidence_recheck import EvidenceRechecker
from evidence_rag.generator.models import RequiredFactCoverage


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


def uncovered_fact() -> RequiredFactCoverage:
    return RequiredFactCoverage(
        required_fact="2024 revenue growth",
        covered=False,
        gap_question="What was Pfizer's revenue growth in 2024?",
    )


def test_rechecker_finds_missing_fact_and_maps_evidence_numbers_to_ids() -> None:
    llm = FakeLLM(
        '{"found":true,"answer_fragment":"Revenue grew by 8% in 2024.",'
        '"evidence_indices":[2]}'
    )
    selected = SelectedEvidenceSet(
        query_id="q-1",
        evidence=(
            evidence("ev-1", "Profit stayed stable."),
            evidence("ev-2", "Pfizer revenue grew by 8% in 2024."),
        ),
    )

    result = EvidenceRechecker(llm=llm).recheck(uncovered_fact(), selected)

    assert result.required_fact == "2024 revenue growth"
    assert result.found is True
    assert result.answer_fragment == "Revenue grew by 8% in 2024."
    assert result.evidence_ids == ("ev-2",)
    assert "What was Pfizer's revenue growth in 2024?" in llm.prompts[0]
    assert "[2] (ev-2) Pfizer revenue grew by 8% in 2024." in llm.prompts[0]


def test_rechecker_returns_not_found_without_selected_evidence_or_llm_call() -> None:
    llm = FakeLLM("must not be used")

    result = EvidenceRechecker(llm=llm).recheck(
        uncovered_fact(),
        SelectedEvidenceSet(query_id="q-1", evidence=()),
    )

    assert result.found is False
    assert result.answer_fragment == ""
    assert result.evidence_ids == ()
    assert llm.prompts == []


def test_rechecker_returns_not_found_when_selected_evidence_lacks_the_fact() -> None:
    llm = FakeLLM(
        '{"found":false,"answer_fragment":"","evidence_indices":[]}'
    )
    selected = SelectedEvidenceSet(
        query_id="q-1",
        evidence=(evidence("ev-1", "Profit stayed stable."),),
    )

    result = EvidenceRechecker(llm=llm).recheck(uncovered_fact(), selected)

    assert result.found is False
    assert result.answer_fragment == ""
    assert result.evidence_ids == ()


def test_rechecker_rejects_fact_already_marked_covered() -> None:
    coverage = RequiredFactCoverage(
        required_fact="2024 revenue growth",
        covered=True,
    )
    selected = SelectedEvidenceSet(
        query_id="q-1",
        evidence=(evidence("ev-1", "Revenue grew."),),
    )

    with pytest.raises(ValueError, match="already covered"):
        EvidenceRechecker(llm=FakeLLM("unused")).recheck(coverage, selected)


def test_rechecker_rejects_malformed_json() -> None:
    selected = SelectedEvidenceSet(
        query_id="q-1",
        evidence=(evidence("ev-1", "Revenue grew."),),
    )

    with pytest.raises(ValueError, match="valid JSON"):
        EvidenceRechecker(llm=FakeLLM("not JSON")).recheck(
            uncovered_fact(),
            selected,
        )


def test_rechecker_rejects_evidence_index_outside_selected_set() -> None:
    llm = FakeLLM(
        '{"found":true,"answer_fragment":"Revenue grew.",'
        '"evidence_indices":[2]}'
    )
    selected = SelectedEvidenceSet(
        query_id="q-1",
        evidence=(evidence("ev-1", "Revenue grew."),),
    )

    with pytest.raises(ValueError, match="out of range"):
        EvidenceRechecker(llm=llm).recheck(uncovered_fact(), selected)


def test_rechecker_rejects_found_answer_without_supporting_evidence() -> None:
    llm = FakeLLM(
        '{"found":true,"answer_fragment":"Revenue grew.",'
        '"evidence_indices":[]}'
    )
    selected = SelectedEvidenceSet(
        query_id="q-1",
        evidence=(evidence("ev-1", "Revenue grew."),),
    )

    with pytest.raises(ValueError, match="requires supporting evidence"):
        EvidenceRechecker(llm=llm).recheck(uncovered_fact(), selected)


def test_rechecker_deduplicates_evidence_ids_in_model_order() -> None:
    llm = FakeLLM(
        '{"found":true,"answer_fragment":"Revenue grew by 8%.",'
        '"evidence_indices":[2,1,2]}'
    )
    selected = SelectedEvidenceSet(
        query_id="q-1",
        evidence=(
            evidence("ev-1", "Revenue grew."),
            evidence("ev-2", "Growth was 8%."),
        ),
    )

    result = EvidenceRechecker(llm=llm).recheck(uncovered_fact(), selected)

    assert result.evidence_ids == ("ev-2", "ev-1")
