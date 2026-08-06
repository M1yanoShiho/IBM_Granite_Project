import hashlib
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from evidence_rag.contracts.models import Document
from evidence_rag.contracts.protocols import Generator, Retriever, Selector
from evidence_rag.generator.extractive import ExtractiveGenerator
from evidence_rag.generator.granite import GraniteGenerator, GraniteLLMClient, TextGenerator
from evidence_rag.infrastructure.config import ExperimentConfig, ModuleConfig
from evidence_rag.infrastructure.corpus import CorpusSnapshot
from evidence_rag.materializer.source_parent import read_parent_index
from evidence_rag.pipeline.service import EvidenceRAGPipeline
from evidence_rag.retriever.bm25 import BM25Retriever, validate_bm25_parameters
from evidence_rag.retriever.chunking import Chunker
from evidence_rag.retriever.fusion import DEFAULT_RRF_K
from evidence_rag.retriever.granite import (
    DEFAULT_GRANITE_EMBEDDING_MODEL_ID,
    DecomposingRetriever,
    GraniteDenseRetriever,
    GraniteEmbedder,
    HyDERetriever,
    Query2DocRetriever,
    TextEmbedder,
)
from evidence_rag.retriever.hybrid import ConvexHybridRetriever, HybridRetriever
from evidence_rag.retriever.indexing import (
    MANIFEST_FILENAME,
    SNAPSHOT_FILENAME,
    IndexManifest,
    load_index,
    read_index_manifest,
    write_index,
)
from evidence_rag.retriever.strong_bm25 import StrongBM25Retriever
from evidence_rag.selector.corroboration import CorroborationSelector
from evidence_rag.selector.gated import GatedCorroborationSelector, GatedCoverageSelector
from evidence_rag.selector.top_k import TopKSelector


def _reject_parameters(config: ModuleConfig, module_kind: str) -> None:
    if config.parameters:
        names = ", ".join(sorted(config.parameters))
        raise ValueError(f"{module_kind} {config.name!r} does not accept parameters: {names}")


# The retriever registry: each name maps to an implementation version. Parameter
# normalisation (below) turns a ModuleConfig into the canonical parameter dict that
# is persisted in the index manifest and drives construction; wrappers/hybrid nest
# a fully-normalised base config so a single signature covers the whole tree.
_RETRIEVER_VERSIONS: dict[str, str] = {
    "bm25": "bm25-v1",
    "strong-bm25": "strong-bm25-v1",
    "granite-dense": "granite-dense-v1",
    "hybrid": "hybrid-v1",
    "query2doc": "query2doc-v1",
    "hyde": "hyde-v1",
    "decompose": "decompose-v1",
}


def _implementation_version(name: str) -> str:
    try:
        return _RETRIEVER_VERSIONS[name]
    except KeyError:
        raise ValueError(f"unknown retriever: {name}") from None


def retriever_implementation_version(name: str) -> str:
    """Public lookup of a retriever's index implementation version (for provenance)."""

    return _implementation_version(name)


def _reject_unknown(parameters: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise ValueError(f"unknown retriever parameter: {unknown[0]}")


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"retriever parameter {name!r} must be an integer")
    if value <= 0:
        raise ValueError(f"retriever parameter {name!r} must be positive")
    return value


def _optional_positive_int(name: str, value: object) -> int | None:
    return None if value is None else _positive_int(name, value)


def _flag(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"retriever parameter {name!r} must be a boolean")
    return value


def _positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"retriever parameter {name!r} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"retriever parameter {name!r} must be positive")
    return number


def _unit_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"retriever parameter {name!r} must be a number")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"retriever parameter {name!r} must be in [0, 1]")
    return number


def _nested_config(raw: object, field: str) -> ModuleConfig:
    if not isinstance(raw, Mapping) or "name" not in raw:
        raise ValueError(f"retriever parameter {field!r} must name a base retriever")
    return ModuleConfig(name=str(raw["name"]), parameters=dict(raw.get("parameters", {})))


def _normalise_nested(raw: object, field: str) -> dict[str, Any]:
    name, version, parameters = _normalise_retriever(_nested_config(raw, field))
    return {"name": name, "implementation_version": version, "parameters": parameters}


def _sparse_parameters(name: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    _reject_unknown(parameters, {"k1", "b"})
    default_k1, default_b = (1.5, 0.75) if name == "bm25" else (0.9, 0.4)
    try:
        k1, b = validate_bm25_parameters(
            parameters.get("k1", default_k1),
            parameters.get("b", default_b),
        )
    except ValueError as error:
        raise ValueError(f"invalid retriever parameters: {error}") from error
    return {"k1": k1, "b": b}


def _dense_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    _reject_unknown(parameters, {"embedder_model_id", "query_prefix", "document_prefix"})
    model_id = (
        parameters.get("embedder_model_id")
        or os.getenv("GRANITE_EMBEDDING_MODEL_ID")
        or DEFAULT_GRANITE_EMBEDDING_MODEL_ID
    )
    return {
        "embedder_model_id": str(model_id),
        "query_prefix": str(parameters.get("query_prefix", "")),
        "document_prefix": str(parameters.get("document_prefix", "")),
    }


def _hybrid_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    _reject_unknown(parameters, {"fusion", "retrievers", "k", "alpha", "pool_size"})
    fusion = parameters.get("fusion", "rrf")
    if fusion not in {"rrf", "convex"}:
        raise ValueError("hybrid parameter 'fusion' must be 'rrf' or 'convex'")
    raw_arms = parameters.get("retrievers")
    if not isinstance(raw_arms, Sequence) or isinstance(raw_arms, (str, bytes)):
        raise ValueError("hybrid parameter 'retrievers' must be a list of retriever configs")
    arms = [_normalise_nested(arm, "retrievers") for arm in raw_arms]
    result: dict[str, Any] = {
        "fusion": fusion,
        "retrievers": arms,
        "pool_size": _optional_positive_int("pool_size", parameters.get("pool_size")),
    }
    if fusion == "rrf":
        if not arms:
            raise ValueError("hybrid 'rrf' requires at least one retriever")
        result["k"] = _positive_int("k", parameters.get("k", DEFAULT_RRF_K))
    else:
        if len(arms) != 2:
            raise ValueError("hybrid 'convex' requires exactly two retrievers (sparse, dense)")
        result["alpha"] = _unit_float("alpha", parameters.get("alpha", 0.5))
    return result


def _wrapper_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    _reject_unknown(parameters, {"base"})
    if "base" not in parameters:
        raise ValueError("retriever wrapper requires a 'base' retriever config")
    return {"base": _normalise_nested(parameters["base"], "base")}


def _decompose_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    _reject_unknown(
        parameters,
        {"base", "k", "pool_size", "include_original", "original_weight", "fusion"},
    )
    if "base" not in parameters:
        raise ValueError("retriever wrapper requires a 'base' retriever config")
    normalised = {
        "base": _normalise_nested(parameters["base"], "base"),
        "k": _positive_int("k", parameters.get("k", DEFAULT_RRF_K)),
        "pool_size": _optional_positive_int("pool_size", parameters.get("pool_size")),
    }
    # The three keys below are recorded only when set away from their default:
    # `parameters` is bound into the index signature, so emitting them unconditionally
    # would change every existing decompose index's expected parameters and reject
    # caches written before these options existed.
    include_original = _flag("include_original", parameters.get("include_original", False))
    if include_original:
        normalised["include_original"] = True
    original_weight = _positive_float(
        "original_weight", parameters.get("original_weight", 1.0)
    )
    if original_weight != 1.0:
        if not include_original:
            raise ValueError(
                "decompose parameter 'original_weight' requires 'include_original'"
            )
        normalised["original_weight"] = original_weight
    fusion = parameters.get("fusion", "rrf")
    if fusion not in {"rrf", "best-rank"}:
        raise ValueError("decompose parameter 'fusion' must be 'rrf' or 'best-rank'")
    if fusion != "rrf":
        normalised["fusion"] = fusion
    return normalised


def _normalise_retriever(config: ModuleConfig) -> tuple[str, str, dict[str, Any]]:
    name = config.name
    version = _implementation_version(name)
    if name in {"bm25", "strong-bm25"}:
        parameters = _sparse_parameters(name, config.parameters)
    elif name == "granite-dense":
        parameters = _dense_parameters(config.parameters)
    elif name == "hybrid":
        parameters = _hybrid_parameters(config.parameters)
    elif name in {"query2doc", "hyde"}:
        parameters = _wrapper_parameters(config.parameters)
    else:  # decompose (only remaining registered name)
        parameters = _decompose_parameters(config.parameters)
    return name, version, parameters


def _construct_retriever(
    name: str,
    parameters: Mapping[str, Any],
    corpus: CorpusSnapshot,
) -> Retriever:
    if name == "bm25":
        return BM25Retriever.from_corpus(corpus, k1=parameters["k1"], b=parameters["b"])
    if name == "strong-bm25":
        return StrongBM25Retriever.from_corpus(corpus, k1=parameters["k1"], b=parameters["b"])
    if name == "granite-dense":
        embedder = GraniteEmbedder(
            model_id=parameters["embedder_model_id"],
            query_prefix=parameters["query_prefix"],
            document_prefix=parameters["document_prefix"],
        )
        return GraniteDenseRetriever.from_corpus(corpus, embedder=embedder)
    if name == "hybrid":
        arms = [
            _construct_retriever(arm["name"], arm["parameters"], corpus)
            for arm in parameters["retrievers"]
        ]
        if parameters["fusion"] == "rrf":
            return HybridRetriever(arms, k=parameters["k"], pool_size=parameters["pool_size"])
        return ConvexHybridRetriever(
            arms[0],
            arms[1],
            alpha=parameters["alpha"],
            pool_size=parameters["pool_size"],
        )
    base_config = parameters["base"]
    base = _construct_retriever(base_config["name"], base_config["parameters"], corpus)
    llm = GraniteLLMClient()
    if name == "query2doc":
        return Query2DocRetriever(base, llm)
    if name == "hyde":
        return HyDERetriever(base, llm)
    return DecomposingRetriever(
        base,
        llm,
        k=parameters["k"],
        pool_size=parameters["pool_size"],
        include_original=bool(parameters.get("include_original", False)),
        original_weight=float(parameters.get("original_weight", 1.0)),
        fusion=str(parameters.get("fusion", "rrf")),
    )


def prepare_retriever_index(
    config: ModuleConfig,
    corpus: CorpusSnapshot,
    directory: Path,
) -> IndexManifest:
    name, version, parameters = _normalise_retriever(config)
    directory = Path(directory)
    if (directory / MANIFEST_FILENAME).exists() or (directory / SNAPSHOT_FILENAME).exists():
        load_index(
            directory,
            expected_corpus_signature=corpus.manifest.corpus_signature,
            expected_implementation=name,
            expected_implementation_version=version,
            expected_parameters=parameters,
        )
    else:
        write_index(
            corpus,
            directory,
            implementation=name,
            implementation_version=version,
            parameters=parameters,
        )
    return read_index_manifest(directory)


def build_retriever(
    config: ModuleConfig,
    corpus: CorpusSnapshot,
    *,
    index_directory: Path | None = None,
) -> Retriever:
    name, version, parameters = _normalise_retriever(config)
    if index_directory is not None:
        snapshot = load_index(
            index_directory,
            expected_corpus_signature=corpus.manifest.corpus_signature,
            expected_implementation=name,
            expected_implementation_version=version,
            expected_parameters=parameters,
        )
        return _construct_retriever(name, parameters, snapshot)
    return _construct_retriever(name, parameters, corpus)


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


SOURCE_PARENT_INDEX_ENV = "SOURCE_PARENT_INDEX"


def _source_parent_index_path() -> Path:
    """Resolve the SAME_SOURCE sidecar path, failing loudly when it is absent.

    A silent fallback to the document unit would run a whole experiment on the old vote counting
    and no metric would reveal it.
    """
    raw_path = os.environ.get(SOURCE_PARENT_INDEX_ENV)
    if not raw_path:
        raise ValueError(
            f"support_unit='parent' needs the {SOURCE_PARENT_INDEX_ENV} environment variable "
            "pointing at a source_parent.jsonl sidecar; build one with "
            "'python -m evidence_rag.cli.build_source_parent'"
        )
    return Path(raw_path)


def source_parent_provenance(config: ModuleConfig) -> dict[str, str]:
    """Identity of the sidecar a `support_unit="parent"` run actually loaded.

    The sidecar comes from the environment, not the config, so without this the archived
    provenance cannot distinguish a run against a stale sidecar from one against a regenerated
    one — the config would read `support_unit = "parent"` in both cases. Empty for every other
    selector, so callers can merge it unconditionally.
    """
    if config.parameters.get("support_unit") != "parent":
        return {}
    path = _source_parent_index_path()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"source_parent_index": str(path), "source_parent_sha256": digest}


def _load_parent_index() -> Mapping[str, str]:
    return read_parent_index(_source_parent_index_path()).parent_by_document


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
            config,
            frozenset({"alpha", "margin", "support_cap", "top_n", "equivalence", "support_unit"}),
        )
        client = llm if llm is not None else GraniteLLMClient()
        selector_class = (
            GatedCoverageSelector
            if config.name == "gated-coverage-corroboration"
            else GatedCorroborationSelector
        )
        raw_equivalence = config.parameters.get("equivalence", "exact")
        if raw_equivalence not in ("exact", "lenient"):
            raise ValueError(f"invalid selector parameter equivalence: {raw_equivalence!r}")
        equivalence: Literal["exact", "lenient"] = (
            "lenient" if raw_equivalence == "lenient" else "exact"
        )
        raw_support_unit = config.parameters.get("support_unit", "document")
        if raw_support_unit not in ("document", "parent"):
            raise ValueError(f"invalid selector parameter support_unit: {raw_support_unit!r}")
        parent_by_document = (
            _load_parent_index() if raw_support_unit == "parent" else None
        )
        return selector_class(
            client,
            alpha=float(parameters.get("alpha", 0.6)),
            margin=int(parameters.get("margin", 2)),
            support_cap=int(parameters.get("support_cap", 1)),
            top_n=int(parameters.get("top_n", 20)),
            equivalence=equivalence,
            parent_by_document=parent_by_document,
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
