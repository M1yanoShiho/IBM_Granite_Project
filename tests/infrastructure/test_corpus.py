import pytest

from evidence_rag.contracts.models import Document
from evidence_rag.infrastructure.corpus import Chunk, CorpusBuilder, WordChunker
from evidence_rag.retriever.chunking import Chunk as RetrieverChunk
from evidence_rag.retriever.chunking import WordChunker as RetrieverWordChunker


def documents() -> tuple[Document, ...]:
    return (
        Document(
            document_id="annual-report",
            text="Revenue increased by ten percent. Operating cost remained stable.",
            source_uri="fixture://annual-report",
        ),
    )


def test_repeated_builds_are_equal_with_stable_signatures_and_ids() -> None:
    builder = CorpusBuilder(WordChunker(chunk_size=4, overlap=1))

    first = builder.build(documents(), dataset_signature="dataset-signature")
    second = builder.build(documents(), dataset_signature="dataset-signature")

    assert first == second
    assert first.manifest.corpus_signature == second.manifest.corpus_signature
    assert tuple(chunk.chunk_id for chunk in first.chunks) == tuple(
        chunk.chunk_id for chunk in second.chunks
    )


def test_changed_chunk_size_changes_corpus_signature() -> None:
    first = CorpusBuilder(WordChunker(chunk_size=4, overlap=1)).build(
        documents(), dataset_signature="dataset-signature"
    )
    second = CorpusBuilder(WordChunker(chunk_size=5, overlap=1)).build(
        documents(), dataset_signature="dataset-signature"
    )

    assert first.manifest.corpus_signature != second.manifest.corpus_signature


def test_duplicate_documents_fail_clearly() -> None:
    with pytest.raises(ValueError, match="duplicate document ID: annual-report"):
        CorpusBuilder().build(documents() * 2, dataset_signature="dataset-signature")


class DuplicateIdChunker(WordChunker):
    def chunk(self, document: Document) -> tuple[Chunk, ...]:
        return (
            Chunk(
                document_id=document.document_id,
                chunk_id="chunk-shared",
                evidence_id="ev-shared",
                text=document.text,
                source_uri=document.source_uri,
            ),
        )


def test_duplicate_chunk_and_evidence_ids_fail_clearly() -> None:
    other_document = Document(
        document_id="other-report",
        text="Other text.",
        source_uri="fixture://other-report",
    )

    with pytest.raises(ValueError, match="duplicate chunk ID: chunk-shared"):
        CorpusBuilder(DuplicateIdChunker()).build(
            documents() + (other_document,), dataset_signature="dataset-signature"
        )


def test_retriever_imports_reexport_canonical_corpus_types() -> None:
    assert RetrieverChunk is Chunk
    assert RetrieverWordChunker is WordChunker
