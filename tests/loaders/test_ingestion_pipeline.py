"""End-to-end guard: multimodal loader output flows through the unchanged pipeline."""

from evidence_rag.composition import build_baseline
from evidence_rag.contracts.models import Document, Query, SourceMetadata
from evidence_rag.retriever.chunking import PrechunkedChunker


def test_multimodal_documents_flow_through_unchanged_baseline() -> None:
    documents = (
        Document(
            document_id="report::p3",
            text="Revenue increased by ten percent in the fourth quarter.",
            source_uri="/data/report.pdf#page=3",
            metadata=SourceMetadata(
                source_type="pdf", file_name="report.pdf", page_number=3
            ),
        ),
        Document(
            document_id="chart",
            text="A bar chart showing profit trending upward across four quarters.",
            source_uri="/data/chart.png",
            metadata=SourceMetadata(
                source_type="image", file_name="chart.png", image_path="/data/chart.png"
            ),
        ),
        Document(
            document_id="notes",
            text="Headcount stayed flat throughout the year.",
            source_uri="/data/notes.txt",
            metadata=SourceMetadata(source_type="txt", file_name="notes.txt"),
        ),
    )

    pipeline = build_baseline(documents)
    run = pipeline.run_with_trace(
        Query(query_id="q", text="revenue fourth quarter"), top_k=3, max_selected=2
    )

    # PipelineRun construction re-validates every stage contract; reaching here
    # means the multimodal documents were indistinguishable from plain text.
    top = run.candidates.candidates[0]
    assert top.document_id == "report::p3"
    assert top.source_uri.endswith("report.pdf#page=3")
    # Structured provenance survives all the way into the selected evidence.
    assert top.metadata is not None
    assert top.metadata.source_type == "pdf"
    assert top.metadata.page_number == 3
    selected_metadata = run.selected.evidence[0].metadata
    assert selected_metadata is not None
    assert selected_metadata.page_number == 3
    assert run.generation.answer


def test_prechunked_corpus_is_never_resplit_by_the_pipeline() -> None:
    table_text = "| quarter | revenue |\n" + "\n".join(
        f"| Q{i} | {100 + i} million |" for i in range(1, 60)
    )
    documents = (
        Document(
            document_id="report.pdf::c1",
            text=table_text,
            source_uri="/data/report.pdf#page=2",
            metadata=SourceMetadata(
                source_type="pdf", file_name="report.pdf", page_number=2
            ),
        ),
        Document(
            document_id="notes.txt",
            text="Revenue commentary for the year.",
            source_uri="/data/notes.txt",
            metadata=SourceMetadata(source_type="txt", file_name="notes.txt"),
        ),
    )

    pipeline = build_baseline(documents, chunker=PrechunkedChunker())

    # One chunk per document: the 170+ word table was not split.
    retriever = pipeline.retriever
    assert [chunk.document_id for chunk in retriever.chunks] == [  # type: ignore[attr-defined]
        "report.pdf::c1",
        "notes.txt",
    ]
    run = pipeline.run_with_trace(Query(query_id="q", text="Q7 revenue"), top_k=2, max_selected=1)
    top = run.candidates.candidates[0]
    assert top.document_id == "report.pdf::c1"
    assert top.text == table_text
    assert top.source_uri.endswith("#page=2")
