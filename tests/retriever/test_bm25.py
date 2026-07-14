from evidence_rag.contracts.models import Document, Query
from evidence_rag.retriever.bm25 import BM25Retriever
from evidence_rag.retriever.chunking import WordChunker


def documents() -> tuple[Document, ...]:
    return (
        Document(
            document_id="annual-report",
            text="Revenue increased by ten percent. Operating cost remained stable.",
            source_uri="fixture://annual-report",
        ),
        Document(
            document_id="policy",
            text="The company adopted a new travel policy.",
            source_uri="fixture://policy",
        ),
    )


def test_ids_are_stable_and_have_different_meanings() -> None:
    chunker = WordChunker(chunk_size=5, overlap=1)
    first = chunker.chunk(documents()[0])
    second = chunker.chunk(documents()[0])
    assert first == second
    assert first[0].document_id != first[0].chunk_id
    assert first[0].chunk_id != first[0].evidence_id


def test_bm25_returns_ranked_candidates() -> None:
    retriever = BM25Retriever(documents(), WordChunker(chunk_size=20, overlap=0))
    result = retriever.retrieve(Query(query_id="q-1", text="revenue increase"), top_k=2)
    assert result.candidates[0].document_id == "annual-report"
    assert result.candidates[0].retrieval_rank == 1


def test_no_matching_terms_returns_empty_candidates() -> None:
    retriever = BM25Retriever(documents(), WordChunker(chunk_size=20, overlap=0))
    result = retriever.retrieve(Query(query_id="q-2", text="volcano"), top_k=2)
    assert result.candidates == ()
