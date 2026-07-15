import math

import pytest

from evidence_rag.contracts.models import Document, Query
from evidence_rag.infrastructure.corpus import CorpusBuilder, WordChunker
from evidence_rag.retriever.bm25 import BM25Retriever


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


def test_from_corpus_uses_snapshot_chunk_ids_directly() -> None:
    snapshot = CorpusBuilder(WordChunker(chunk_size=3, overlap=0)).build(
        documents(), dataset_signature="dataset-signature"
    )

    retriever = BM25Retriever.from_corpus(snapshot)
    result = retriever.retrieve(Query(query_id="q-3", text="revenue policy"), top_k=10)

    assert tuple(chunk.chunk_id for chunk in retriever.chunks) == tuple(
        chunk.chunk_id for chunk in snapshot.chunks
    )
    assert {candidate.chunk_id for candidate in result.candidates} <= {
        chunk.chunk_id for chunk in snapshot.chunks
    }


def test_from_corpus_does_not_invoke_a_chunker_again() -> None:
    class CountingChunker(WordChunker):
        calls = 0

        def chunk(self, document: Document):  # type: ignore[no-untyped-def]
            type(self).calls += 1
            return super().chunk(document)

    snapshot = CorpusBuilder(CountingChunker(chunk_size=3, overlap=0)).build(
        documents(), dataset_signature="dataset-signature"
    )
    calls_after_build = CountingChunker.calls

    BM25Retriever.from_corpus(snapshot)

    assert CountingChunker.calls == calls_after_build


@pytest.mark.parametrize(
    ("k1", "b", "parameter"),
    (
        (-0.01, 0.75, "k1"),
        (math.inf, 0.75, "k1"),
        (math.nan, 0.75, "k1"),
        (1.5, -0.01, "b"),
        (1.5, 1.01, "b"),
        (1.5, math.inf, "b"),
        (1.5, math.nan, "b"),
    ),
)
@pytest.mark.parametrize("constructor", ("documents", "corpus"))
def test_bm25_parameters_are_validated_at_direct_construction_boundaries(
    k1: float,
    b: float,
    parameter: str,
    constructor: str,
) -> None:
    with pytest.raises(ValueError, match=parameter):
        if constructor == "documents":
            BM25Retriever(documents(), k1=k1, b=b)
        else:
            snapshot = CorpusBuilder().build(documents(), dataset_signature="dataset-signature")
            BM25Retriever.from_corpus(snapshot, k1=k1, b=b)
