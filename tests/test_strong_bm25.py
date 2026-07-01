"""Tests for the tuned, analyzed BM25 (the fair lexical baseline).

``StrongBM25Retriever`` addresses the critique that the naive ``BM25Retriever``
(default k1/b, no stopword removal, no stemming) sits below Anserini/Lucene BM25
and so overstates the neural gap. It keeps the same ``Retriever`` contract as
``BM25Retriever`` (drop-in for ``_build_component`` / ``tune_alpha`` / fusion).
"""

from __future__ import annotations

import pytest

from src.retrieval.base import RetrievedChunk, Retriever
from src.retrieval.strong_bm25 import StrongBM25Retriever


def test_satisfies_retriever_contract() -> None:
    retriever = StrongBM25Retriever(corpus=["granite retrieval"], doc_ids=["d1"], top_k=1)

    assert isinstance(retriever, Retriever)


def test_returns_ranked_retrieved_chunks() -> None:
    retriever = StrongBM25Retriever(
        corpus=[
            "ibm granite retrieval improves enterprise search",
            "banana cake recipe with sugar and butter",
            "scifact benchmark contains scientific claims",
        ],
        doc_ids=["doc-granite", "doc-recipe", "doc-scifact"],
        top_k=2,
    )

    results = retriever.retrieve("granite enterprise retrieval")

    assert len(results) == 2
    assert all(isinstance(item, RetrievedChunk) for item in results)
    assert results[0].doc_id == "doc-granite"
    assert results[0].score >= results[1].score


def test_query_stopwords_do_not_drive_ranking() -> None:
    # A document made only of stopwords must not out-rank a real content match:
    # the analyzer drops "the/of/and/to/in" so ranking is driven by "granite".
    # (>=4 docs so "granite" is a minority term with positive BM25 idf.)
    retriever = StrongBM25Retriever(
        corpus=[
            "the of and to in",
            "granite dense retrieval",
            "banana cake recipe sugar",
            "scientific claims benchmark corpus",
        ],
        doc_ids=["stopwords", "content", "recipe", "science"],
        top_k=1,
    )

    results = retriever.retrieve("the granite")

    assert results[0].doc_id == "content"


def test_stopword_only_query_returns_empty() -> None:
    retriever = StrongBM25Retriever(
        corpus=["granite retrieval"], doc_ids=["d1"], top_k=5
    )

    assert retriever.retrieve("the of and to") == []


def test_injected_stemmer_enables_morphological_match() -> None:
    # No stemmer is bundled (dependency-light); an injected one must be applied to
    # BOTH corpus and query so a plural query term matches a singular doc term.
    naive_stem = lambda token: token[:-1] if token.endswith("s") else token

    retriever = StrongBM25Retriever(
        corpus=["feline cat", "canine dog", "avian bird", "equine horse"],
        doc_ids=["cat-doc", "dog-doc", "bird-doc", "horse-doc"],
        top_k=1,
        stemmer=naive_stem,
    )

    results = retriever.retrieve("cats")

    assert results[0].doc_id == "cat-doc"
    assert results[0].score > 0.0


def test_tuned_parameters_default_to_anserini_beir_values() -> None:
    retriever = StrongBM25Retriever(corpus=["a b c"], doc_ids=["d1"], top_k=1)

    assert retriever.k1 == 0.9
    assert retriever.b == 0.4


def test_rejects_mismatched_corpus_and_doc_ids() -> None:
    with pytest.raises(ValueError, match="corpus and doc_ids"):
        StrongBM25Retriever(corpus=["one document"], doc_ids=["d1", "d2"], top_k=1)


def test_handles_empty_corpus() -> None:
    retriever = StrongBM25Retriever(corpus=[], doc_ids=[], top_k=5)

    assert retriever.retrieve("anything") == []
