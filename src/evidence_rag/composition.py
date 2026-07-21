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
from evidence_rag.retriever.chunking import Chunker
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
from evidence_rag.selector.gated import GatedCorroborationSelector, GatedCoverageSelector
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


def _float_parameter(name: str, value: object, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid selector parameter {name}: expected a number")
    number = float(value)
    if not low <= number <= high:
        raise ValueError(f"invalid selector parameter {name}: must be in [{low}, {high}]")
    return number


def _int_parameter(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"invalid selector parameter {name}: expected an integer")
    if value < minimum:
        raise ValueError(f"invalid selector parameter {name}: must be >= {minimum}")
    return value


def _selector_parameters(
    config: ModuleConfig,
    allowed: frozenset[str],
) -> dict[str, float | int]:
    unknown = sorted(set(config.parameters) - allowed)
    if unknown:
        raise ValueError(f"unknown selector parameter: {unknown[0]}")
    parameters: dict[str, float | int] = {}
    if "alpha" in config.parameters:
        parameters["alpha"] = _float_parameter("alpha", config.parameters["alpha"], 0.0, 1.0)
    if "margin" in config.parameters:
        parameters["margin"] = _int_parameter("margin", config.parameters["margin"], 1)
    if "support_cap" in config.parameters:
        parameters["support_cap"] = _int_parameter(
            "support_cap", config.parameters["support_cap"], 0
        )
    if "top_n" in config.parameters:
        parameters["top_n"] = _int_parameter("top_n", config.parameters["top_n"], 1)
    return parameters


def build_selector(config: ModuleConfig, *, llm: TextGenerator | None = None) -> Selector:
    if config.name == "top-k":
        _reject_parameters(config, "selector")
        return TopKSelector()
    if config.name == "corroboration":
        parameters = _selector_parameters(config, frozenset({"alpha", "top_n"}))
        client = llm if llm is not None else GraniteLLMClient()
        return CorroborationSelector(
            client,
            alpha=float(parameters.get("alpha", 0.6)),
            top_n=int(parameters.get("top_n", 20)),
        )
    if config.name in {"gated-corroboration", "gated-coverage-corroboration"}:
        parameters = _selector_parameters(
            config, frozenset({"alpha", "margin", "support_cap", "top_n"})
        )
        client = llm if llm is not None else GraniteLLMClient()
        selector_class = (
            GatedCoverageSelector
            if config.name == "gated-coverage-corroboration"
            else GatedCorroborationSelector
        )
        return selector_class(
            client,
            alpha=float(parameters.get("alpha", 0.6)),
            margin=int(parameters.get("margin", 2)),
            support_cap=int(parameters.get("support_cap", 1)),
            top_n=int(parameters.get("top_n", 20)),
        )
    raise ValueError(f"unknown selector: {config.name}")


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


def build_baseline(
    documents: Iterable[Document],
    *,
    chunker: Chunker | None = None,
) -> EvidenceRAGPipeline:
    return EvidenceRAGPipeline(
        retriever=BM25Retriever(documents, chunker=chunker),
        selector=TopKSelector(),
        generator=ExtractiveGenerator(),
    )


def build_granite_baseline(
    documents: Iterable[Document],
    *,
    embedder: TextEmbedder | None = None,
    llm: TextGenerator | None = None,
    chunker: Chunker | None = None,
) -> EvidenceRAGPipeline:
    return EvidenceRAGPipeline(
        retriever=GraniteDenseRetriever(documents, embedder=embedder, chunker=chunker),
        selector=TopKSelector(),
        generator=GraniteGenerator(llm=llm),
    )


def build_q2d_granite_baseline(
    documents: Iterable[Document],
    *,
    embedder: TextEmbedder | None = None,
    llm: TextGenerator | None = None,
    chunker: Chunker | None = None,
) -> EvidenceRAGPipeline:
    shared_llm = llm or GraniteLLMClient()
    dense_retriever = GraniteDenseRetriever(documents, embedder=embedder, chunker=chunker)
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
    chunker: Chunker | None = None,
) -> EvidenceRAGPipeline:
    shared_llm = llm or GraniteLLMClient()
    dense_retriever = GraniteDenseRetriever(documents, embedder=embedder, chunker=chunker)
    return EvidenceRAGPipeline(
        retriever=Query2DocRetriever(dense_retriever, shared_llm),
        selector=CorroborationSelector(shared_llm, alpha=alpha),
        generator=GraniteGenerator(llm=shared_llm),
    )
