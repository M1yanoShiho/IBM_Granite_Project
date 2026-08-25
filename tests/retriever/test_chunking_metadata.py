from collections.abc import Sequence

from evidence_rag.contracts.models import Document, Query, SourceMetadata
from evidence_rag.retriever.bm25 import BM25Retriever
from evidence_rag.retriever.chunking import WordChunker
from evidence_rag.retriever.granite import GraniteDenseRetriever


def test_chunker_copies_document_metadata_to_every_chunk() -> None:
    metadata = SourceMetadata(source_type="pdf", file_name="report.pdf", page_number=3)
    document = Document(
        document_id="report::p3",
        text=" ".join(f"word{i}" for i in range(250)),
        source_uri="/data/report.pdf#page=3",
        metadata=metadata,
    )

    chunks = WordChunker(chunk_size=100, overlap=10).chunk(document)

    assert len(chunks) > 1
    assert all(chunk.metadata == metadata for chunk in chunks)


def test_chunker_keeps_metadata_none_for_plain_documents() -> None:
    document = Document(
        document_id="plain",
        text="Plain text without provenance metadata.",
        source_uri="fixture://plain",
    )

    chunks = WordChunker().chunk(document)

    assert chunks
    assert all(chunk.metadata is None for chunk in chunks)


class _OnesEmbedder:
    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0,) for _ in texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        return (1.0,)


def _corpus() -> tuple[Document, Document]:
    return (
        Document(
            document_id="report.pdf::c1",
            text="Revenue increased by ten percent.",
            source_uri="/data/report.pdf#page=3",
            metadata=SourceMetadata(source_type="pdf", file_name="report.pdf", page_number=3),
        ),
        Document(
            document_id="plain",
            text="Revenue commentary without provenance.",
            source_uri="fixture://plain",
        ),
    )


def test_bm25_candidates_carry_source_metadata() -> None:
    retriever = BM25Retriever(_corpus())

    result = retriever.retrieve(Query(query_id="q", text="revenue"), top_k=2)

    by_id = {candidate.document_id: candidate for candidate in result.candidates}
    metadata = by_id["report.pdf::c1"].metadata
    assert metadata is not None
    assert metadata.source_type == "pdf"
    assert metadata.page_number == 3
    assert by_id["plain"].metadata is None


def test_dense_candidates_carry_source_metadata() -> None:
    retriever = GraniteDenseRetriever(_corpus(), embedder=_OnesEmbedder())

    result = retriever.retrieve(Query(query_id="q", text="revenue"), top_k=2)

    by_id = {candidate.document_id: candidate for candidate in result.candidates}
    metadata = by_id["report.pdf::c1"].metadata
    assert metadata is not None
    assert metadata.page_number == 3
    assert by_id["plain"].metadata is None
