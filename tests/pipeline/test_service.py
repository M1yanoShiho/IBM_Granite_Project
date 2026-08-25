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


def test_run_with_trace_exposes_every_module_output() -> None:
    pipeline = build_baseline(
        (
            Document(
                document_id="doc-1",
                text="Revenue increased by ten percent.",
                source_uri="fixture://doc-1",
            ),
        )
    )
    query = Query(query_id="q-trace", text="revenue increase")

    trace = pipeline.run_with_trace(query, top_k=3, max_selected=2)

    assert trace.query == query
    assert trace.checklist.query_id == query.query_id
    assert trace.checklist.focus
    assert trace.checklist.required_facts
    assert trace.candidates.candidates
    assert trace.selection.items
    assert trace.selected.evidence
    assert trace.generation == pipeline.run(query, top_k=3, max_selected=2)
    assert trace.top_k == 3
    assert trace.max_selected == 2


def test_checklist_is_generated_per_query() -> None:
    pipeline = build_baseline(
        (
            Document(
                document_id="doc-1",
                text="IBM revenue increased in 2023.",
                source_uri="fixture://doc-1",
            ),
        )
    )

    trace = pipeline.run_with_trace(
        Query(query_id="q-checklist", text="What was IBM revenue in 2023?"),
        top_k=3,
        max_selected=2,
    )

    assert "revenue" in tuple(item.lower() for item in trace.checklist.required_facts)
    assert "year:2023" in trace.checklist.constraints
    assert "entity:IBM" in trace.checklist.constraints
