import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from evidence_rag.contracts.models import Document, RetrieverProvenance
from evidence_rag.contracts.protocols import Generator, Retriever, Selector
from evidence_rag.generator.claim_splitter import ClaimSplitter
from evidence_rag.generator.draft import DraftAnswerGenerator, DraftGenerator
from evidence_rag.generator.extractive import ExtractiveGenerator
from evidence_rag.generator.granite import (
    GraniteGenerationConfig,
    GraniteGenerator,
    GraniteLLMClient,
    NamedAdapterTextGenerator,
    PeftGraniteLLMClient,
    TextGenerator,
)
from evidence_rag.generator.nli import NLIModel, TrueNLIModel
from evidence_rag.generator.verify_annotate import (
    CitationRoutedVerifier,
    VerifyAnnotateGenerator,
)
from evidence_rag.infrastructure.config import ExperimentConfig, ModuleConfig
from evidence_rag.infrastructure.corpus import CorpusSnapshot
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
from evidence_rag.selector.dual_head import load_dual_head_checkpoint
from evidence_rag.selector.nli_dual_head import load_nli_dual_head_model
from evidence_rag.selector.nli_runtime import NliRiskControlledSelector
from evidence_rag.selector.top_k import TopKSelector

_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _reject_parameters(config: ModuleConfig, module_kind: str) -> None:
    if config.parameters:
        names = ", ".join(sorted(config.parameters))
        raise ValueError(f"{module_kind} {config.name!r} does not accept parameters: {names}")


def _verify_file_sha256(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    observed = digest.hexdigest()
    if observed != expected:
        raise ValueError(f"{label} SHA-256 differs: expected {expected}, observed {observed}")


def _expand_runtime_value(value: object, *, label: str) -> str:
    """Resolve explicit ``${NAME}`` references without storing HPC paths in Git."""

    raw = str(value)
    names = {match.group(1) for match in _ENV_REFERENCE.finditer(raw)}
    missing = sorted(name for name in names if name not in os.environ)
    if missing:
        raise ValueError(f"{label} requires environment variable(s): {', '.join(missing)}")
    return _ENV_REFERENCE.sub(lambda match: os.environ[match.group(1)], raw)


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
    *,
    embedder: TextEmbedder | None = None,
) -> Retriever:
    if name == "bm25":
        return BM25Retriever.from_corpus(corpus, k1=parameters["k1"], b=parameters["b"])
    if name == "strong-bm25":
        return StrongBM25Retriever.from_corpus(corpus, k1=parameters["k1"], b=parameters["b"])
    if name == "granite-dense":
        active_embedder = embedder or GraniteEmbedder(
            model_id=parameters["embedder_model_id"],
            query_prefix=parameters["query_prefix"],
            document_prefix=parameters["document_prefix"],
        )
        return GraniteDenseRetriever.from_corpus(corpus, embedder=active_embedder)
    if name == "hybrid":
        arms = [
            _construct_retriever(
                arm["name"],
                arm["parameters"],
                corpus,
                embedder=embedder,
            )
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
    base = _construct_retriever(
        base_config["name"],
        base_config["parameters"],
        corpus,
        embedder=embedder,
    )
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


def retriever_provenance(manifest: IndexManifest) -> RetrieverProvenance:
    """The producer identity a candidate pool is stamped with (M0 §4).

    Derived from the INDEX MANIFEST rather than from the experiment config, because the index
    manifest is the artefact `build_retriever` validated the retriever against — `load_index`
    raises unless implementation, version and every normalised parameter match. Reading the
    config instead would record what the operator asked for; this records what was actually
    loaded, and those are the same thing only when nothing went wrong.

    Parameters collapse to a digest so the identity is fixed-width and exact. Their canonical
    form is already the one `_index_signature` hashes, so two runs whose pools differ because a
    parameter moved get two different digests without anyone having to enumerate which
    parameters matter to which retriever.
    """

    return RetrieverProvenance(
        name=manifest.implementation,
        implementation_version=manifest.implementation_version,
        parameters_sha256=hashlib.sha256(
            json.dumps(
                manifest.parameters,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    )


def build_retriever(
    config: ModuleConfig,
    corpus: CorpusSnapshot,
    *,
    index_directory: Path | None = None,
    embedder: TextEmbedder | None = None,
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
        return _construct_retriever(name, parameters, snapshot, embedder=embedder)
    return _construct_retriever(name, parameters, corpus, embedder=embedder)


def build_selector(config: ModuleConfig, *, model: Any | None = None) -> Selector:
    if config.name == "top-k":
        _reject_parameters(config, "selector")
        return TopKSelector()
    if config.name == "nli-risk-controlled":
        parameters = dict(config.parameters)
        unknown = sorted(
            set(parameters)
            - {
                "model_snapshot",
                "model_id",
                "revision",
                "checkpoint_path",
                "checkpoint_sha256",
                "safe_threshold",
                "max_delete",
                "local_files_only",
                "device",
            }
        )
        if unknown:
            raise ValueError(f"unknown selector parameters: {unknown}")
        if model is None:
            required = {
                "model_snapshot",
                "model_id",
                "revision",
                "checkpoint_path",
                "checkpoint_sha256",
            }
            missing = sorted(required - set(parameters))
            if missing:
                raise ValueError(f"nli-risk-controlled selector is missing parameters: {missing}")
            checkpoint = Path(
                _expand_runtime_value(
                    parameters["checkpoint_path"],
                    label="Selector checkpoint",
                )
            )
            _verify_file_sha256(
                checkpoint,
                str(parameters["checkpoint_sha256"]),
                label="Selector checkpoint",
            )
            device = str(parameters.get("device", "auto"))
            model_snapshot = _expand_runtime_value(
                parameters["model_snapshot"],
                label="Selector model snapshot",
            )
            model = load_nli_dual_head_model(
                model_snapshot,
                revision=str(parameters["revision"]),
                identity_model_id=str(parameters["model_id"]),
                local_files_only=bool(parameters.get("local_files_only", True)),
                device=device,
            )
            load_dual_head_checkpoint(model, checkpoint)
            model.eval()
        return NliRiskControlledSelector(
            model=model,
            safe_threshold=float(parameters.get("safe_threshold", 0.9212157130241394)),
            max_delete=int(parameters.get("max_delete", 2)),
        )
    raise ValueError(f"unknown selector: {config.name}")


def build_generator(
    config: ModuleConfig,
    *,
    llm: TextGenerator | None = None,
    nli: NLIModel | None = None,
    grc_client: Any | None = None,
    entity_checker: Any | None = None,
) -> Generator:
    """Build the configured Generator.

    ``llm`` and ``nli`` are injectable because ``GraniteLLMClient`` loads its
    weights in ``__init__``. A caller that only wants to check the wiring should
    not have to load a production model. See ``docs/generator/design-review.md``.
    """
    if config.name == "extractive":
        _reject_parameters(config, "generator")
        return ExtractiveGenerator()
    if config.name == "granite":
        _reject_parameters(config, "generator")
        return GraniteGenerator(llm=llm)
    if config.name == "verify-annotate":
        # The main method. Until G9 this factory could only build the 29-line
        # `extractive` placeholder, so the config-driven CLI could not run the
        # method this project is about -- it existed only inside the experiment
        # script. Behaviour is unchanged: the script path constructs the same
        # object with the same defaults.
        parameters = dict(config.parameters)
        gate = parameters.pop("entity_gate", "observe")
        abstain = bool(parameters.pop("abstain_when_unverified", False))
        if parameters:
            raise ValueError(f"unknown generator parameters: {sorted(parameters)}")
        return VerifyAnnotateGenerator(
            llm=llm,
            nli=nli,
            entity_gate=gate,
            abstain_when_unverified=abstain,
        )
    if config.name == "grounded-grc":
        parameters = dict(config.parameters)
        adapter_name = str(parameters.pop("adapter_name", "grc"))
        gate = parameters.pop("entity_gate", "observe")
        model_snapshot = parameters.pop("model_snapshot", None)
        model_config_sha256 = parameters.pop("model_config_sha256", None)
        adapter_path_raw = parameters.pop("adapter_path", None)
        adapter_weights_sha256 = parameters.pop("adapter_weights_sha256", None)
        adapter_config_sha256 = parameters.pop("adapter_config_sha256", None)
        true_snapshot = parameters.pop("true_snapshot", None)
        true_config_sha256 = parameters.pop("true_config_sha256", None)
        device = str(parameters.pop("device", "auto"))
        max_new_tokens = int(parameters.pop("max_new_tokens", 256))
        max_input_tokens_raw = parameters.pop("max_input_tokens", None)
        max_input_tokens = (
            int(max_input_tokens_raw) if max_input_tokens_raw is not None else None
        )
        temperature = float(parameters.pop("temperature", 0.0))
        top_p = float(parameters.pop("top_p", 1.0))
        if parameters:
            raise ValueError(f"unknown generator parameters: {sorted(parameters)}")
        if grc_client is None:
            required = {
                "model_snapshot": model_snapshot,
                "model_config_sha256": model_config_sha256,
                "adapter_path": adapter_path_raw,
                "adapter_weights_sha256": adapter_weights_sha256,
                "adapter_config_sha256": adapter_config_sha256,
            }
            missing = sorted(name for name, value in required.items() if value is None)
            if missing:
                raise ValueError(f"grounded-grc generator is missing parameters: {missing}")
            model_path = Path(
                _expand_runtime_value(model_snapshot, label="Granite model snapshot")
            )
            adapter_path = Path(
                _expand_runtime_value(adapter_path_raw, label="GR-C adapter")
            )
            _verify_file_sha256(
                model_path / "config.json",
                str(model_config_sha256),
                label="Granite model config",
            )
            _verify_file_sha256(
                adapter_path / "adapter_model.safetensors",
                str(adapter_weights_sha256),
                label="GR-C adapter weights",
            )
            _verify_file_sha256(
                adapter_path / "adapter_config.json",
                str(adapter_config_sha256),
                label="GR-C adapter config",
            )
            grc_client = PeftGraniteLLMClient(
                model_id=str(model_path),
                adapters={adapter_name: str(adapter_path)},
                config=GraniteGenerationConfig(
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    max_input_tokens=max_input_tokens,
                ),
                device=device,
            )
        if nli is None:
            if true_snapshot is None or true_config_sha256 is None:
                raise ValueError(
                    "grounded-grc generator requires true_snapshot and true_config_sha256"
                )
            true_path = Path(
                _expand_runtime_value(true_snapshot, label="TRUE model snapshot")
            )
            _verify_file_sha256(
                true_path / "config.json",
                str(true_config_sha256),
                label="TRUE verifier config",
            )
            nli = TrueNLIModel(model_id=str(true_path))
        draft = DraftAnswerGenerator(
            draft_generator=DraftGenerator(
                llm=NamedAdapterTextGenerator(grc_client, adapter_name),
                trace_enabled=True,
            ),
            claim_splitter=ClaimSplitter(llm=grc_client, trace_enabled=True),
            trace_enabled=True,
        )
        verifier = (
            CitationRoutedVerifier(nli, entity_checker, entity_gate=gate)
            if entity_checker is not None
            else None
        )
        return VerifyAnnotateGenerator(
            draft_generator=draft,
            verifier=verifier,
            nli=nli,
            entity_gate=gate,
            abstain_when_unverified=False,
            trace_enabled=True,
        )
    raise ValueError(f"unknown generator: {config.name}")


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
    embedder: TextEmbedder | None = None,
    selector_model: Any | None = None,
    llm: TextGenerator | None = None,
    nli: NLIModel | None = None,
    grc_client: Any | None = None,
    entity_checker: Any | None = None,
) -> EvidenceRAGPipeline:
    return EvidenceRAGPipeline(
        retriever=build_retriever(
            config.retriever,
            corpus,
            index_directory=index_directory,
            embedder=embedder,
        ),
        selector=build_selector(config.selector, model=selector_model),
        generator=build_generator(
            config.generator,
            llm=llm,
            nli=nli,
            grc_client=grc_client,
            entity_checker=entity_checker,
        ),
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
