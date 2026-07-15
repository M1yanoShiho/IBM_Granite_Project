from collections.abc import Iterable
from pathlib import Path

from evidence_rag.contracts.models import Document
from evidence_rag.contracts.protocols import Generator, Retriever, Selector
from evidence_rag.generator.extractive import ExtractiveGenerator
from evidence_rag.generator.granite import GraniteGenerator, GraniteLLMClient, TextGenerator
from evidence_rag.infrastructure.config import ExperimentConfig, ModuleConfig
from evidence_rag.infrastructure.corpus import CorpusSnapshot
from evidence_rag.pipeline.service import EvidenceRAGPipeline
from evidence_rag.retriever.bm25 import BM25Retriever, validate_bm25_parameters
from evidence_rag.retriever.granite import (
    GraniteDenseRetriever,
    Query2DocRetriever,
    TextEmbedder,
)
from evidence_rag.retriever.indexing import (
    MANIFEST_FILENAME,
    SNAPSHOT_FILENAME,
    BM25IndexPlugin,
    IndexManifest,
    read_index_manifest,
)
from evidence_rag.selector.corroboration import CorroborationSelector
from evidence_rag.selector.top_k import TopKSelector


def _reject_parameters(config: ModuleConfig, module_kind: str) -> None:
    if config.parameters:
        names = ", ".join(sorted(config.parameters))
        raise ValueError(f"{module_kind} {config.name!r} does not accept parameters: {names}")


def _bm25_parameters(config: ModuleConfig) -> tuple[float, float]:
    allowed = {"k1", "b"}
    unknown = sorted(set(config.parameters) - allowed)
    if unknown:
        raise ValueError(f"unknown retriever parameter: {unknown[0]}")

    try:
        return validate_bm25_parameters(
            config.parameters.get("k1", 1.5),
            config.parameters.get("b", 0.75),
        )
    except ValueError as error:
        raise ValueError(f"invalid retriever parameters: {error}") from error


def prepare_retriever_index(
    config: ModuleConfig,
    corpus: CorpusSnapshot,
    directory: Path,
) -> IndexManifest:
    if config.name != "bm25":
        raise ValueError(f"unknown retriever: {config.name}")
    k1, b = _bm25_parameters(config)
    plugin = BM25IndexPlugin()
    directory = Path(directory)
    if (directory / MANIFEST_FILENAME).exists() or (directory / SNAPSHOT_FILENAME).exists():
        plugin.load(
            directory,
            expected_corpus_signature=corpus.manifest.corpus_signature,
            expected_k1=k1,
            expected_b=b,
        )
    else:
        plugin.build(corpus, directory, k1=k1, b=b)
    return read_index_manifest(directory)


def build_retriever(
    config: ModuleConfig,
    corpus: CorpusSnapshot,
    *,
    index_directory: Path | None = None,
) -> Retriever:
    if config.name != "bm25":
        raise ValueError(f"unknown retriever: {config.name}")
    k1, b = _bm25_parameters(config)
    if index_directory is not None:
        return BM25IndexPlugin().load(
            index_directory,
            expected_corpus_signature=corpus.manifest.corpus_signature,
            expected_k1=k1,
            expected_b=b,
        )
    return BM25Retriever.from_corpus(corpus, k1=k1, b=b)


def build_selector(config: ModuleConfig) -> Selector:
    if config.name != "top-k":
        raise ValueError(f"unknown selector: {config.name}")
    _reject_parameters(config, "selector")
    return TopKSelector()


def build_generator(config: ModuleConfig) -> Generator:
    if config.name != "extractive":
        raise ValueError(f"unknown generator: {config.name}")
    _reject_parameters(config, "generator")
    return ExtractiveGenerator()


def build_baseline_from_corpus(
    corpus: CorpusSnapshot,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> EvidenceRAGPipeline:
    return EvidenceRAGPipeline(
        retriever=BM25Retriever.from_corpus(corpus, k1=k1, b=b),
        selector=TopKSelector(),
        generator=ExtractiveGenerator(),
    )


def build_pipeline_from_config(
    config: ExperimentConfig,
    corpus: CorpusSnapshot,
    *,
    index_directory: Path | None = None,
) -> EvidenceRAGPipeline:
    return EvidenceRAGPipeline(
        retriever=build_retriever(
            config.retriever,
            corpus,
            index_directory=index_directory,
        ),
        selector=build_selector(config.selector),
        generator=build_generator(config.generator),
    )


def build_baseline(documents: Iterable[Document]) -> EvidenceRAGPipeline:
    return EvidenceRAGPipeline(
        retriever=BM25Retriever(documents),
        selector=TopKSelector(),
        generator=ExtractiveGenerator(),
    )


def build_granite_baseline(
    documents: Iterable[Document],
    *,
    embedder: TextEmbedder | None = None,
    llm: TextGenerator | None = None,
) -> EvidenceRAGPipeline:
    return EvidenceRAGPipeline(
        retriever=GraniteDenseRetriever(documents, embedder=embedder),
        selector=TopKSelector(),
        generator=GraniteGenerator(llm=llm),
    )


def build_q2d_granite_baseline(
    documents: Iterable[Document],
    *,
    embedder: TextEmbedder | None = None,
    llm: TextGenerator | None = None,
) -> EvidenceRAGPipeline:
    shared_llm = llm or GraniteLLMClient()
    dense_retriever = GraniteDenseRetriever(documents, embedder=embedder)
    return EvidenceRAGPipeline(
        retriever=Query2DocRetriever(dense_retriever, shared_llm),
        selector=TopKSelector(),
        generator=GraniteGenerator(llm=shared_llm),
    )


def build_q2d_corroboration_granite(
    documents: Iterable[Document],
    *,
    embedder: TextEmbedder | None = None,
    llm: TextGenerator | None = None,
    alpha: float = 0.6,
) -> EvidenceRAGPipeline:
    shared_llm = llm or GraniteLLMClient()
    dense_retriever = GraniteDenseRetriever(documents, embedder=embedder)
    return EvidenceRAGPipeline(
        retriever=Query2DocRetriever(dense_retriever, shared_llm),
        selector=CorroborationSelector(shared_llm, alpha=alpha),
        generator=GraniteGenerator(llm=shared_llm),
    )
