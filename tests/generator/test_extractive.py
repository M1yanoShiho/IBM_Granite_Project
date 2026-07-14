from evidence_rag.contracts.models import (
    EvidenceCandidate,
    Query,
    SelectedEvidenceSet,
)
from evidence_rag.generator.extractive import ExtractiveGenerator


def test_generator_uses_selected_content_and_ids() -> None:
    evidence = EvidenceCandidate(
        evidence_id="ev-1",
        document_id="doc-1",
        chunk_id="chunk-1",
        text="Revenue increased by ten percent.",
        source_uri="fixture://doc-1",
        retrieval_score=1.0,
        retrieval_rank=1,
    )
    result = ExtractiveGenerator().generate(
        Query(query_id="q-1", text="What changed?"),
        SelectedEvidenceSet(query_id="q-1", evidence=(evidence,)),
    )
    assert "Revenue increased" in result.answer
    assert result.cited_evidence_ids == ("ev-1",)


def test_empty_selection_returns_empty_output() -> None:
    result = ExtractiveGenerator().generate(
        Query(query_id="q-2", text="What changed?"),
        SelectedEvidenceSet(query_id="q-2", evidence=()),
    )
    assert result.answer == ""
    assert result.cited_evidence_ids == ()
