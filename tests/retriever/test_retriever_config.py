from pathlib import Path

import pytest

from evidence_rag.composition import build_retriever, prepare_retriever_index
from evidence_rag.contracts.models import Document, Query
from evidence_rag.infrastructure.config import ModuleConfig
from evidence_rag.infrastructure.corpus import CorpusBuilder, CorpusSnapshot, WordChunker
from evidence_rag.retriever.bm25 import BM25Retriever
from evidence_rag.retriever.hybrid import ConvexHybridRetriever, HybridRetriever
from evidence_rag.retriever.indexing import read_index_manifest
from evidence_rag.retriever.strong_bm25 import StrongBM25Retriever

QUERY = Query(query_id="q-1", text="revenue policy")


def corpus() -> CorpusSnapshot:
    documents = (
        Document(
            document_id="annual-report",
            text="Revenue increased while operating costs remained stable.",
            source_uri="fixture://annual-report",
        ),
        Document(
            document_id="policy",
            text="The company adopted a new travel policy.",
            source_uri="fixture://policy",
        ),
    )
    return CorpusBuilder(WordChunker(chunk_size=20, overlap=0)).build(
        documents, dataset_signature="dataset-signature"
    )


def config(name: str, **parameters: object) -> ModuleConfig:
    return ModuleConfig(name=name, parameters=parameters)


def test_build_bm25_and_strong_bm25_from_config() -> None:
    snapshot = corpus()
    assert isinstance(build_retriever(config("bm25"), snapshot), BM25Retriever)
    strong = build_retriever(config("strong-bm25"), snapshot)
    assert isinstance(strong, StrongBM25Retriever)
    assert (strong.k1, strong.b) == (0.9, 0.4)


def test_build_hybrid_rrf_from_nested_config() -> None:
    snapshot = corpus()
    retriever = build_retriever(
        config(
            "hybrid",
            fusion="rrf",
            retrievers=[{"name": "bm25"}, {"name": "strong-bm25"}],
        ),
        snapshot,
    )
    assert isinstance(retriever, HybridRetriever)
    result = retriever.retrieve(QUERY, top_k=5)
    assert result.query_id == "q-1"
    assert tuple(c.retrieval_rank for c in result.candidates) == tuple(
        range(1, len(result.candidates) + 1)
    )


def test_build_hybrid_convex_from_nested_config() -> None:
    snapshot = corpus()
    retriever = build_retriever(
        config(
            "hybrid",
            fusion="convex",
            alpha=0.3,
            retrievers=[{"name": "bm25"}, {"name": "strong-bm25"}],
        ),
        snapshot,
    )
    assert isinstance(retriever, ConvexHybridRetriever)


def test_unknown_retriever_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown retriever: mystery"):
        build_retriever(config("mystery"), corpus())


def test_unknown_parameter_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown retriever parameter: nonsense"):
        build_retriever(config("bm25", nonsense=1), corpus())


def test_convex_requires_exactly_two_arms() -> None:
    with pytest.raises(ValueError, match="exactly two retrievers"):
        build_retriever(
            config("hybrid", fusion="convex", retrievers=[{"name": "bm25"}]),
            corpus(),
        )


def test_prepare_persists_strong_bm25_implementation(tmp_path: Path) -> None:
    snapshot = corpus()
    manifest = prepare_retriever_index(config("strong-bm25"), snapshot, tmp_path)
    assert manifest.implementation == "strong-bm25"
    assert manifest.implementation_version == "strong-bm25-v1"
    assert read_index_manifest(tmp_path).parameters == {"k1": 0.9, "b": 0.4}


def test_persisted_index_round_trips_for_hybrid(tmp_path: Path) -> None:
    snapshot = corpus()
    hybrid_config = config(
        "hybrid",
        fusion="rrf",
        retrievers=[{"name": "bm25"}, {"name": "strong-bm25"}],
    )
    prepare_retriever_index(hybrid_config, snapshot, tmp_path)

    loaded = build_retriever(hybrid_config, snapshot, index_directory=tmp_path)
    in_memory = build_retriever(hybrid_config, snapshot)
    assert loaded.retrieve(QUERY, top_k=5) == in_memory.retrieve(QUERY, top_k=5)


def test_persisted_load_rejects_parameter_drift(tmp_path: Path) -> None:
    snapshot = corpus()
    prepare_retriever_index(config("bm25", k1=1.5, b=0.75), snapshot, tmp_path)

    with pytest.raises(ValueError, match="parameters mismatch"):
        build_retriever(config("bm25", k1=1.2, b=0.75), snapshot, index_directory=tmp_path)


def test_manifest_implementation_matches_registry(tmp_path: Path) -> None:
    snapshot = corpus()
    manifest = prepare_retriever_index(config("bm25"), snapshot, tmp_path)
    assert manifest.implementation == "bm25"
    assert manifest.implementation_version == "bm25-v1"
