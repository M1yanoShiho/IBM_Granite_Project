from evidence_rag.composition import build_baseline
from evidence_rag.contracts.models import Document, Query


def test_pipeline_returns_answer_and_original_evidence_id() -> None:
    pipeline = build_baseline(
        (
            Document(
                document_id="doc-1",
                text="Revenue increased by ten percent.",
                source_uri="fixture://doc-1",
            ),
        )
    )
    result = pipeline.run(
        Query(query_id="q-1", text="revenue increase"),
        top_k=3,
        max_selected=2,
    )
    assert result.answer
    assert result.cited_evidence_ids[0].startswith("ev-")


def test_pipeline_returns_empty_output_when_nothing_is_retrieved() -> None:
    pipeline = build_baseline(
        (
            Document(
                document_id="doc-1",
                text="Revenue increased.",
                source_uri="fixture://doc-1",
            ),
        )
    )
    result = pipeline.run(
        Query(query_id="q-2", text="volcano"),
        top_k=3,
        max_selected=2,
    )
    assert result.answer == ""
