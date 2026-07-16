from evidence_rag.contracts.models import Document, SourceMetadata
from evidence_rag.retriever.chunking import PrechunkedChunker, WordChunker


def _long_document() -> Document:
    return Document(
        document_id="report.pdf::c1",
        text=" ".join(f"cell{i}" for i in range(300)),
        source_uri="/data/report.pdf#page=2",
        metadata=SourceMetadata(source_type="pdf", file_name="report.pdf", page_number=2),
    )


def test_prechunked_chunker_never_splits_long_documents() -> None:
    document = _long_document()

    chunks = PrechunkedChunker().chunk(document)

    assert len(chunks) == 1
    assert chunks[0].text == document.text
    assert chunks[0].metadata == document.metadata
    assert chunks[0].source_uri == document.source_uri
    assert chunks[0].chunk_id.startswith("chunk-")
    assert chunks[0].evidence_id.startswith("ev-")
    # Sanity: the default WordChunker would have split this document.
    assert len(WordChunker().chunk(document)) > 1


def test_prechunked_chunker_skips_whitespace_only_documents() -> None:
    document = Document(document_id="blank", text="   \n  ", source_uri="fixture://blank")

    assert PrechunkedChunker().chunk(document) == ()


def test_prechunked_chunk_ids_are_deterministic() -> None:
    document = _long_document()

    first = PrechunkedChunker().chunk(document)
    second = PrechunkedChunker().chunk(document)

    assert first == second
