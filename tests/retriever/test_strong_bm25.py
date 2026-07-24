import math

import pytest

from evidence_rag.contracts.models import Document, Query
from evidence_rag.infrastructure.corpus import CorpusBuilder, WordChunker
from evidence_rag.retriever.strong_bm25 import (
    ENGLISH_STOPWORDS,
    StrongBM25Retriever,
    build_analyzer,
)


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


def test_strong_bm25_uses_anserini_defaults() -> None:
    retriever = StrongBM25Retriever(documents(), WordChunker(chunk_size=20, overlap=0))
    assert (retriever.k1, retriever.b) == (0.9, 0.4)


def test_strong_bm25_returns_ranked_candidates() -> None:
    retriever = StrongBM25Retriever(documents(), WordChunker(chunk_size=20, overlap=0))
    result = retriever.retrieve(Query(query_id="q-1", text="revenue increase"), top_k=2)
    assert result.candidates[0].document_id == "annual-report"
    assert result.candidates[0].retrieval_rank == 1


def test_no_matching_terms_returns_empty_candidates() -> None:
    retriever = StrongBM25Retriever(documents(), WordChunker(chunk_size=20, overlap=0))
    result = retriever.retrieve(Query(query_id="q-2", text="volcano"), top_k=2)
    assert result.candidates == ()


def test_stopwords_are_filtered_so_they_never_match() -> None:
    # "the" is a stopword; a query of only stopwords scores nothing even though the
    # word literally appears in the "policy" document.
    retriever = StrongBM25Retriever(documents(), WordChunker(chunk_size=20, overlap=0))
    result = retriever.retrieve(Query(query_id="q-3", text="the by a"), top_k=5)
    assert result.candidates == ()


def test_analyzer_drops_stopwords_and_lowercases() -> None:
    analyze = build_analyzer()
    tokens = analyze("The Company Adopted A Travel Policy")
    assert "the" not in tokens
    assert "a" not in tokens
    assert tokens == ("company", "adopted", "travel", "policy")


def test_retrieval_is_case_insensitive() -> None:
    retriever = StrongBM25Retriever(documents(), WordChunker(chunk_size=20, overlap=0))
    lower = retriever.retrieve(Query(query_id="q-4", text="revenue"), top_k=2)
    upper = retriever.retrieve(Query(query_id="q-4", text="REVENUE"), top_k=2)
    assert [c.evidence_id for c in lower.candidates] == [c.evidence_id for c in upper.candidates]
    assert lower.candidates[0].document_id == "annual-report"


def test_disabling_stopwords_keeps_them() -> None:
    analyze = build_analyzer(stopwords=())
    assert analyze("the by a") == ("the", "by", "a")


def test_custom_stopwords_override_default() -> None:
    analyze = build_analyzer(stopwords={"revenue"})
    assert "revenue" not in analyze("Revenue increased")
    assert "the" in analyze("the revenue")


def test_from_corpus_uses_snapshot_chunk_ids_directly() -> None:
    snapshot = CorpusBuilder(WordChunker(chunk_size=3, overlap=0)).build(
        documents(), dataset_signature="dataset-signature"
    )

    retriever = StrongBM25Retriever.from_corpus(snapshot)
    result = retriever.retrieve(Query(query_id="q-5", text="revenue policy"), top_k=10)

    assert isinstance(retriever, StrongBM25Retriever)
    assert (retriever.k1, retriever.b) == (0.9, 0.4)
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

    StrongBM25Retriever.from_corpus(snapshot)

    assert CountingChunker.calls == calls_after_build


@pytest.mark.parametrize(
    ("k1", "b", "parameter"),
    (
        (-0.01, 0.4, "k1"),
        (math.inf, 0.4, "k1"),
        (0.9, -0.01, "b"),
        (0.9, 1.01, "b"),
        (0.9, math.nan, "b"),
    ),
)
def test_parameters_are_validated(k1: float, b: float, parameter: str) -> None:
    with pytest.raises(ValueError, match=parameter):
        StrongBM25Retriever(documents(), k1=k1, b=b)


def test_default_stopword_set_is_nonempty_and_lowercase() -> None:
    assert ENGLISH_STOPWORDS
    assert all(word == word.lower() for word in ENGLISH_STOPWORDS)
