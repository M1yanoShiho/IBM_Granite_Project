"""Primary evaluation entry point: system vs. baselines on standard benchmarks.

Orchestrates the headline experiment of the project:

1. Load a standard benchmark (corpus, queries, qrels) via ``eval.benchmarks``.
2. Index the corpus once (``src.ingestion.indexer``).
3. Run each retriever over all queries:
   - the **Granite dense retriever** (the delivered system), and
   - the **BM25 baseline** (and optionally a sentence-transformers dense baseline).
4. Score every run with ``eval.ir_metrics`` (precision/recall/nDCG/MRR).
5. Save a comparison table for the report.

Run from the project root once the retrievers are wired:

    python -m eval.run_benchmark

The full pipeline is implemented and unit-tested: BM25 and the dense retrievers
(``granite_dense`` / ``st_dense``, over P5's FAISS index) all run end-to-end.
Constructing a dense retriever downloads its embedding model on first use — see
:func:`_build_retrievers`.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from eval.benchmarks.loader import BenchmarkData, load_benchmark
from eval.ir_metrics import Run, evaluate_run, per_query_scores
from src.ingestion.chunker import Chunk, chunk_document
from src.ingestion.indexer import FaissIndex, VectorIndexer
from src.retrieval.base import Retriever
from src.retrieval.bm25_baseline import BM25Retriever
from src.retrieval.strong_bm25 import StrongBM25Retriever
from src.retrieval.embedder import Embedder
from src.retrieval.hybrid import ConvexHybridRetriever, HybridRetriever
from src.retrieval.reranker import (
    DEFAULT_RERANKER_MODEL_ID,
    LLMListwiseReranker,
    Reranker,
    TwoStageRetriever,
)
from src.retrieval.retriever import DenseRetriever
from src.retrieval.sparse_index import SparseIndex
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.splade_encoder import SpladeEncoder


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark evaluation run.

    Attributes
    ----------
    dataset:
        Benchmark name to evaluate on (e.g. ``"scifact"``, ``"nq"``).
    k_values:
        Cut-offs at which to report precision/recall/nDCG.
    retrievers:
        Which retrievers to compare (system + baselines): ``"bm25"`` or any name
        in :data:`DENSE_SPECS` (``granite_dense``, ``st_dense``, and the modern
        same-class baselines ``gte_dense`` / ``e5_dense`` / ``bge_dense``).
    results_path:
        Where to write the comparison table (CSV).
    index_cache_dir:
        If set, dense retrievers cache their FAISS index here (keyed by dataset,
        retriever name, and the chunk/model ablation params) so reruns skip
        re-embedding. ``None`` = no caching.
    chunk_size, chunk_overlap:
        chunk_size:
            Token window size for splitting documents into chunks. SciFact abstracts
            are short (≈150 words), so 512 yields ~1 chunk/doc; chunk ablations are
            more informative on long-document corpora (e.g. NQ).
        chunk_overlap:
            Number of tokens shared between consecutive chunks (sliding window).
        Passed to :func:`~src.ingestion.chunker.chunk_document` when building the
        dense retrievers' index — the chunking ablation knobs. Default 512/50.
    pooling:
        How chunk scores collapse to a doc score in :func:`build_run`
        (``"max"`` or ``"mean"``). An ablation knob; defaults to contract-3 max.
    chunk_unit:
        Unit for ``chunk_size``/``chunk_overlap`` when building the dense
        retrievers' index: ``"word"`` (default, whitespace split) or ``"token"``
        (the embedding model's own sub-word tokens). Token mode caps the chunk
        size at the model's ``max_seq_length`` so long documents are split rather
        than silently truncated at encode time; it leaves short-document corpora
        (e.g. SciFact, ~1 chunk/doc) effectively unchanged. Opt-in so existing
        word-mode results stay reproducible.
    dense_fanout:
        How many chunks the dense retrievers fetch per query, as a multiple of
        ``max(k_values)``. The retriever ranks *chunks* but the metrics score
        *docs* (chunks are max/mean-pooled to docs in :func:`build_run`), so on
        multi-chunk documents the top-``max(k)`` chunks can collapse to fewer
        than ``max(k)`` distinct docs and understate recall@k. Over-fetching
        chunks, then trimming the pooled run to the top ``max(k)`` docs
        (:func:`build_run`), guarantees enough distinct docs while keeping the
        comparison symmetric with BM25 (which returns ``max(k)`` docs directly).
        Default 10; raise it if a corpus has many chunks per document. Has no
        effect on single-chunk corpora like SciFact.
    embedding_model_id:
        Optional override for the dense retrievers' embedding model (e.g. to
        compare Granite variants); ``None`` uses each backend's default. Sweep one
        dense retriever at a time, since it applies to whichever dense retrievers
        are in the run.
    append:
        Ablation/sweep mode. When ``True``, :func:`run` tags every row with this
        config (dataset, chunk size/overlap, pooling, model) and appends to
        ``results_path`` instead of overwriting — so a loop over configs builds
        one master CSV. ``False`` keeps the original single-run schema.
    per_query_out:
        If set, :func:`run` also writes a wide per-query CSV here (``qid`` + one
        column per retriever, scored by ``per_query_metric``) alongside the
        aggregate table. This is the input to :mod:`eval.significance` (paired
        significance testing + failure analysis); ``None`` = don't write it.
    per_query_metric:
        The metric used for ``per_query_out`` (default ``"ndcg@10"``); any single
        ranx metric name.
    """

    dataset: str = "scifact"
    split: str = "test"
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    retrievers: List[str] = field(
        default_factory=lambda: ["granite_dense", "bm25", "st_dense"]
    )
    results_path: Path = Path("results/benchmark_results.csv")
    index_cache_dir: Path | None = None
    chunk_size: int = 512
    chunk_overlap: int = 50
    pooling: str = "max"
    chunk_unit: str = "word"
    dense_fanout: int = 10
    embedding_model_id: str | None = None
    append: bool = False
    per_query_out: Path | None = None
    per_query_metric: str = "ndcg@10"
    k_rrf: int = 60
    alpha: float = 0.5
    reranker_model_id: str = DEFAULT_RERANKER_MODEL_ID
    rerank_pool: int | None = None
    index_type: str = "flat"
    ef_search: int = 64
    nprobe: int = 8


def build_run(
    retriever: Retriever,
    queries: Dict[str, str],
    pooling: str = "max",
    top_n_docs: int | None = None,
) -> Run:
    """Aggregate a retriever's chunk-level results into a doc-level ``Run``.

    For each query, call ``retriever.retrieve`` and collapse the returned chunks
    to one score per ``doc_id``. ``pooling`` selects the aggregation:

    - ``"max"`` (default, contract 3): a document's score is the highest score
      among its retrieved chunks.
    - ``"mean"``: the average score over the document's *retrieved* chunks (an
      ablation knob; only chunks the retriever returned are averaged).

    ``top_n_docs`` (optional) keeps only the top-N docs per query by pooled
    score. The dense retrievers over-fetch chunks (``dense_fanout``) so that,
    after pooling, there are at least ``max(k)`` distinct docs even when several
    chunks come from the same document; trimming back to ``max(k)`` here makes
    every retriever's run the same depth — symmetric with BM25, which returns
    ``max(k)`` docs directly — so the @k metrics and MRR are comparable and not
    inflated by the larger candidate pool. ``None`` (default) keeps every pooled
    doc, preserving the original behaviour for callers that don't pass it.

    On unchunked corpora (e.g. SciFact) every doc has one chunk, so both the
    pooling choice and the trim are pass-throughs; they only bite once documents
    are split into chunks (e.g. NQ). The result is keyed by query id, ready to
    hand to ``eval.ir_metrics.evaluate_run`` (contract 2).
    """
    if pooling not in ("max", "mean"):
        raise ValueError(
            f"Unknown pooling {pooling!r}; expected 'max' or 'mean'."
        )
    run: Run = {}
    for query_id, query_text in queries.items():
        chunks_by_doc: Dict[str, List[float]] = {}
        for chunk in retriever.retrieve(query_text):
            chunks_by_doc.setdefault(chunk.doc_id, []).append(chunk.score)
        if pooling == "max":
            doc_scores = {did: max(scores) for did, scores in chunks_by_doc.items()}
        else:
            doc_scores = {
                did: sum(scores) / len(scores) for did, scores in chunks_by_doc.items()
            }
        if top_n_docs is not None and len(doc_scores) > top_n_docs:
            doc_scores = dict(
                sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n_docs]
            )
        run[query_id] = doc_scores
    return run


def evaluate_one(
    retriever: Retriever,
    data: BenchmarkData,
    k_values: List[int] | None = None,
    pooling: str = "max",
) -> Dict[str, float]:
    """Score one retriever on a benchmark.

    Builds the retriever's doc-level ``Run`` (:func:`build_run`, contract 3,
    aggregated by ``pooling`` and trimmed to the top ``max(k)`` docs so every
    retriever is scored at the same depth), then scores it against the
    benchmark's qrels with ``eval.ir_metrics.evaluate_run`` (contract 2).
    Returns the metric suite (precision/recall/nDCG at each k, plus MRR).
    """
    ks = k_values if k_values is not None else [1, 3, 5, 10]
    run = build_run(retriever, data.queries, pooling=pooling, top_n_docs=max(ks))
    return evaluate_run(run, data.qrels, k_values)


def write_results_csv(
    results: Dict[str, Dict[str, float]],
    path: Path,
    config_columns: Dict[str, object] | None = None,
    append: bool = False,
) -> None:
    """Write the system-vs-baseline comparison table to ``path`` as CSV.

    ``results`` maps each retriever name to its metric suite (the output of
    :func:`evaluate_one`). Emits one row per retriever: the metric columns,
    preceded by a ``retriever`` column — the table the report and the P7 plots
    consume. All retrievers are scored on the same ``k_values``, so they share
    metric keys and line up into a rectangular table.

    ``config_columns`` (optional) tags every row with the run's configuration
    (e.g. ``{"dataset": ..., "chunk_size": ..., "pooling": ...}``) for ablation
    sweeps; those columns lead, then ``retriever``, then the metrics, so many
    runs accumulate into one analyzable table (contract 6).

    ``append`` (optional): when ``True`` and the file already exists, rows are
    appended under the existing header instead of overwriting — so a sweep loop
    that calls this once per config builds a single master CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    config_columns = config_columns or {}
    metric_names = list(next(iter(results.values())).keys())
    fieldnames = list(config_columns) + ["retriever"] + metric_names
    appending = append and path.exists()
    with open(path, "a" if appending else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not appending:
            writer.writeheader()
        for name, metrics in results.items():
            writer.writerow({**config_columns, "retriever": name, **metrics})


def write_per_query_csv(
    per_query: Dict[str, Dict[str, float]],
    path: Path,
) -> None:
    """Write per-query scores as a wide CSV: ``qid`` then one column per retriever.

    ``per_query`` maps ``{retriever: {query_id: score}}``. Rows are the union of
    query ids (sorted); a blank cell means that retriever has no score for that
    query. This is the input to :mod:`eval.significance` — paired significance
    testing and failure analysis read it back with ``load_per_query_csv``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    retrievers = list(per_query)
    qids = sorted({qid for scores in per_query.values() for qid in scores})
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid"] + retrievers)
        for qid in qids:
            writer.writerow([qid] + [per_query[name].get(qid, "") for name in retrievers])


def _cache_key(config: BenchmarkConfig, name: str) -> str:
    """Cache-file stem for retriever ``name``'s dense index under ``config``.

    Folds the ablation params (chunk size/overlap, embedding model) into the key
    alongside dataset + retriever name, so indexes built for different configs do
    not collide — a chunk-size sweep with ``--cache-dir`` would otherwise reload a
    stale index built at a different size.
    """
    model_slug = (config.embedding_model_id or "default").replace("/", "-")
    # flat keeps the original key (existing caches / reproducibility); ANN types add
    # a suffix so a flat and an HNSW index for the same corpus never collide.
    index_suffix = "" if config.index_type == "flat" else f"__{config.index_type}"
    return (
        f"{config.dataset}__{name}"
        f"__cs{config.chunk_size}_ov{config.chunk_overlap}__{model_slug}"
        f"__{config.chunk_unit}{index_suffix}"
    )


def _load_or_build_index(
    indexer: VectorIndexer,
    chunks: List[Chunk],
    cache_path: Path,
) -> FaissIndex:
    """Load a persisted FAISS index from ``cache_path`` if present, otherwise
    build it from ``chunks`` and save it there — so repeat runs skip the
    expensive re-embedding step. The ``.faiss`` suffix is what
    ``VectorIndexer.save`` writes.
    """
    if Path(f"{cache_path}.faiss").exists():
        return indexer.load(cache_path)
    index = indexer.build(chunks)
    indexer.save(index, cache_path)
    return index


def _load_or_build_sparse_index(
    encoder: SpladeEncoder,
    corpus: List[str],
    doc_ids: List[str],
    cache_path: Path,
) -> SparseIndex:
    """Load a persisted SPLADE CSR index from ``cache_path`` if present, otherwise encode
    the corpus, build it, and save it — so repeat runs skip re-encoding (the expensive
    step). The ``.npz`` suffix is what ``SparseIndex.save`` writes.
    """
    if Path(f"{cache_path}.npz").exists():
        return SparseIndex.load(cache_path)
    index = SparseIndex.build(encoder.encode(corpus), doc_ids, encoder.vocab_size)
    index.save(cache_path)
    return index


@dataclass(frozen=True)
class DenseSpec:
    """How to construct a named dense retriever's embedder.

    Attributes
    ----------
    backend:
        ``Embedder`` backend: ``"granite"`` or ``"sentence-transformers"``.
    model_id:
        Embedding model to load. ``None`` defers to the backend's own default
        (Granite r2 / MiniLM, honouring the ``*_EMBEDDING_MODEL_ID`` env vars) so
        the original ``granite_dense`` / ``st_dense`` runs reproduce unchanged.
    query_prefix, doc_prefix:
        Strings prepended to queries / documents before encoding. Some
        instruction-tuned baselines need them (e5 wants ``"query: "`` /
        ``"passage: "``, bge wants a query instruction); omitting them quietly
        cripples the model and turns a "strong baseline" into an accidental
        strawman. Granite r2 and gte need none.
    """

    backend: str
    model_id: str | None = None
    query_prefix: str = ""
    doc_prefix: str = ""


# Named dense retrievers the benchmark can build. granite_dense (the delivered
# system) and st_dense (the original MiniLM baseline) keep model_id=None so their
# existing results reproduce; gte/e5/bge are modern, same-class open baselines —
# fair peers to Granite — each pinned to a model and carrying the exact prefixes
# its model card requires.
DENSE_SPECS: Dict[str, DenseSpec] = {
    "granite_dense": DenseSpec(backend="granite"),
    "granite_small_dense": DenseSpec(
        backend="granite", model_id="ibm-granite/granite-embedding-small-english-r2"
    ),
    "st_dense": DenseSpec(backend="sentence-transformers"),
    "gte_dense": DenseSpec(
        backend="sentence-transformers", model_id="thenlper/gte-base"
    ),
    "e5_dense": DenseSpec(
        backend="sentence-transformers",
        model_id="intfloat/e5-base-v2",
        query_prefix="query: ",
        doc_prefix="passage: ",
    ),
    "bge_dense": DenseSpec(
        backend="sentence-transformers",
        model_id="BAAI/bge-base-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
}


def _dense_spec(name: str, model_override: str | None = None) -> DenseSpec:
    """The :class:`DenseSpec` for retriever ``name``, with an optional model swap.

    ``model_override`` (from ``--embedding-model-id``) replaces the spec's model
    but keeps its backend and prefixes — so a Granite-variant (or e5/bge variant)
    sweep still applies the right prefixes. It applies to whichever dense
    retrievers are in the run, so sweep one dense retriever at a time.
    """
    spec = DENSE_SPECS[name]
    if model_override is not None:
        spec = replace(spec, model_id=model_override)
    return spec


# Hybrid retrievers: name -> the component retrievers fused with RRF
# (:class:`~src.retrieval.hybrid.HybridRetriever`). Each component must itself be a
# buildable name (``bm25`` or a DENSE_SPECS key). Motivated by the failure analysis:
# granite (dense) and BM25 (lexical) win different queries on SciFact, so fusing
# them should recover documents either misses alone.
HYBRID_SPECS: Dict[str, List[str]] = {
    "hybrid_granite_bm25": ["granite_dense", "bm25"],
    "hybrid_granite_small_bm25": ["granite_small_dense", "bm25"],
}


# Convex-combination hybrids: name -> [dense_name, lexical_name]. Unlike HYBRID_SPECS
# (RRF over rankings), these fuse per-query min-max-normalised SCORES with weight
# config.alpha (the dense weight), via ConvexHybridRetriever. The fix for finding #4's
# RRF-loses result: convex combination beats RRF and is tunable (Bruch et al., 2023).
CONVEX_HYBRID_SPECS: Dict[str, List[str]] = {
    "convex_hybrid_granite_bm25": ["granite_dense", "bm25"],
    "convex_hybrid_granite_strong_bm25": ["granite_dense", "strong_bm25"],
    "convex_hybrid_granite_splade": ["granite_dense", "splade"],
}


# LLM query-side augmentation: name -> (base retriever name, transform kind).
# HyDE / Query2Doc wrap a base retriever with an LLM query-transform
# (:mod:`src.retrieval.query_transform`); because the wrapper is itself a
# ``Retriever`` it is measured on nDCG (here) AND cover-EM (``run_rag``). The
# generating LLM is injected (reused from the RAG generator) rather than loaded a
# second time.
HYDE_SPECS: Dict[str, Tuple[str, str]] = {
    "hyde_granite": ("granite_dense", "hyde"),
    "q2d_granite": ("granite_dense", "query2doc"),
    "hyde_strong_bm25": ("strong_bm25", "hyde"),  # HyDE also expands the lexical arm
}


def _build_component(
    name: str,
    config: BenchmarkConfig,
    data: BenchmarkData,
    corpus: List[str],
    doc_ids: List[str],
    top_k: int,
) -> Retriever:
    """Build a single, non-hybrid retriever: ``bm25`` or a ``DENSE_SPECS`` dense one.

    - ``bm25`` builds its own term index from the corpus and returns docs directly
      (``max(k)`` of them).
    - a dense name chunks + embeds the corpus, builds/loads a FAISS index, and wraps
      it in a :class:`~src.retrieval.retriever.DenseRetriever` fetching
      ``max(k) * dense_fanout`` chunks (so chunk->doc pooling still yields ``max(k)``
      distinct docs). ``config.chunk_unit == "token"`` caps chunks at the model's
      ``max_seq_length``; building a dense retriever loads its embedding model.
    """
    if name == "bm25":
        return BM25Retriever(corpus, doc_ids, top_k=top_k)
    if name == "strong_bm25":
        return StrongBM25Retriever(corpus, doc_ids, top_k=top_k)
    if name in DENSE_SPECS:
        spec = _dense_spec(name, config.embedding_model_id)
        embedder = Embedder(
            backend=spec.backend,
            model_id=spec.model_id,
            query_prefix=spec.query_prefix,
            doc_prefix=spec.doc_prefix,
        )
        chunk_kwargs = _chunk_kwargs_for(config, embedder)
        chunks = [
            chunk
            for did, text in data.corpus.items()
            for chunk in chunk_document(did, text, **chunk_kwargs)
        ]
        indexer = VectorIndexer(
            embedder,
            index_type=config.index_type,
            ef_search=config.ef_search,
            nprobe=config.nprobe,
        )
        if config.index_cache_dir is not None:
            cache_path = config.index_cache_dir / _cache_key(config, name)
            index = _load_or_build_index(indexer, chunks, cache_path)
        else:
            index = indexer.build(chunks)
        return DenseRetriever(embedder, index, top_k=top_k * config.dense_fanout)
    if name == "splade":
        encoder = SpladeEncoder()
        if config.index_cache_dir is not None:
            cache_path = config.index_cache_dir / f"{config.dataset}__splade"
            index = _load_or_build_sparse_index(encoder, corpus, doc_ids, cache_path)
        else:
            index = SparseIndex.build(encoder.encode(corpus), doc_ids, encoder.vocab_size)
        return SparseRetriever(encoder, index, corpus, top_k=top_k)
    raise ValueError(
        f"Unknown retriever {name!r}; expected 'bm25', 'strong_bm25', 'splade', "
        f"one of {sorted(DENSE_SPECS)}, or a hybrid {sorted(HYBRID_SPECS)}."
    )


# Two-stage rerank retrievers: name -> the first-stage retriever name whose
# candidate pool a cross-encoder (:class:`~src.retrieval.reranker.Reranker`)
# re-ranks. The first stage may be a dense retriever OR a hybrid — reranking a
# hybrid pool lets the cross-encoder pick the complementary BM25 finds that rank
# fusion alone could not. Stays all-Granite with the default reranker.
RERANK_SPECS: Dict[str, str] = {
    "granite_rerank": "granite_dense",
    "granite_small_rerank": "granite_small_dense",
    "hybrid_granite_bm25_rerank": "hybrid_granite_bm25",
}


# LLM listwise rerankers: name -> first-stage retriever name. A RankGPT-style
# :class:`~src.retrieval.reranker.LLMListwiseReranker` (the project's own generative
# LLM) re-ranks the first stage's pool — retesting the "reranking fails" finding
# with a *listwise* reranker instead of the pointwise cross-encoder. The candidate
# pool is capped by ``config.rerank_pool`` (LLM calls scale with it); the LLM is
# reused from the injected generator when there is one.
LLM_RERANK_SPECS: Dict[str, str] = {
    "granite_listrank": "granite_dense",
    "strong_bm25_listrank": "strong_bm25",
}


def _build_named(
    name: str,
    config: BenchmarkConfig,
    data: BenchmarkData,
    corpus: List[str],
    doc_ids: List[str],
    top_k: int,
    llm=None,
) -> Retriever:
    """Build a single retriever or a hybrid by name (the part a reranker can wrap).

    ``bm25`` / a ``DENSE_SPECS`` name -> :func:`_build_component`; a ``HYBRID_SPECS``
    name -> its components fused with RRF (:class:`~src.retrieval.hybrid.HybridRetriever`);
    a ``HYDE_SPECS`` name -> its base wrapped in an LLM query-transform (reusing the
    injected ``llm`` if given, else constructing one).
    """
    if name in HYDE_SPECS:
        base_name, kind = HYDE_SPECS[name]
        base = _build_component(base_name, config, data, corpus, doc_ids, top_k)
        from src.llm_client import LLMClient
        from src.retrieval.query_transform import (
            HyDETransform,
            Query2DocTransform,
            TransformingRetriever,
        )

        client = llm if llm is not None else LLMClient()
        transform = (
            Query2DocTransform(client)
            if kind == "query2doc"
            else HyDETransform(client)
        )
        return TransformingRetriever(base, transform)
    if name in HYBRID_SPECS:
        components = [
            _build_component(part, config, data, corpus, doc_ids, top_k)
            for part in HYBRID_SPECS[name]
        ]
        return HybridRetriever(
            components, top_k=top_k * config.dense_fanout, k_rrf=config.k_rrf
        )
    if name in CONVEX_HYBRID_SPECS:
        dense_name, lexical_name = CONVEX_HYBRID_SPECS[name]
        pool = top_k * config.dense_fanout  # ~100 candidates/arm for fusion to reorder
        # dense over-fetches internally (top_k * dense_fanout chunks); BM25 returns
        # `pool` docs directly — so both arms surface ~`pool` docs.
        dense = _build_component(dense_name, config, data, corpus, doc_ids, top_k)
        lexical = _build_component(lexical_name, config, data, corpus, doc_ids, pool)
        return ConvexHybridRetriever(dense, lexical, alpha=config.alpha, top_k=pool)
    return _build_component(name, config, data, corpus, doc_ids, top_k)


def _build_retrievers(
    config: BenchmarkConfig,
    data: BenchmarkData,
    llm=None,
) -> Dict[str, Retriever]:
    """Construct the retrievers named in ``config`` over ``data.corpus``.

    Each name is a single retriever (``bm25`` or a ``DENSE_SPECS`` dense one; see
    :func:`_build_component`), a hybrid (a ``HYBRID_SPECS`` key, fused with RRF), or
    a two-stage rerank retriever (a ``RERANK_SPECS`` key: build its first stage —
    dense or hybrid — and wrap it in a
    :class:`~src.retrieval.reranker.TwoStageRetriever` with the Granite cross-encoder).
    Dense index caches are shared by cache key, so reusing a dense retriever across
    standalone / hybrid / rerank runs does not re-embed the corpus.
    """
    doc_ids = list(data.corpus.keys())
    corpus = list(data.corpus.values())
    top_k = max(config.k_values)
    pool = top_k * config.dense_fanout  # first-stage / rerank candidate-pool depth

    retrievers: Dict[str, Retriever] = {}
    for name in config.retrievers:
        if name in RERANK_SPECS:
            first = _build_named(
                RERANK_SPECS[name], config, data, corpus, doc_ids, top_k, llm=llm
            )
            retrievers[name] = TwoStageRetriever(
                first, Reranker(config.reranker_model_id), top_k=pool, candidates=pool
            )
        elif name in LLM_RERANK_SPECS:
            first = _build_named(
                LLM_RERANK_SPECS[name], config, data, corpus, doc_ids, top_k, llm=llm
            )
            from src.llm_client import LLMClient

            client = llm if llm is not None else LLMClient()
            retrievers[name] = TwoStageRetriever(
                first,
                LLMListwiseReranker(client),
                top_k=pool,
                candidates=config.rerank_pool or pool,
            )
        else:
            retrievers[name] = _build_named(
                name, config, data, corpus, doc_ids, top_k, llm=llm
            )
    return retrievers


def _chunk_kwargs_for(config: BenchmarkConfig, embedder: Embedder) -> dict:
    """Chunking kwargs for ``chunk_document``, honouring ``config.chunk_unit``.

    Word mode (default) just forwards the configured size/overlap. Token mode
    passes the embedder's tokenizer and caps the chunk size at the model's
    ``max_seq_length`` (and the overlap below that), so no chunk exceeds what the
    model encodes — preventing silent truncation and keeping a chunk-size
    ablation meaningful up to the model limit.
    """
    if config.chunk_unit == "word":
        return {
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
        }
    if config.chunk_unit != "token":
        raise ValueError(
            f"Unknown chunk_unit {config.chunk_unit!r}; expected 'word' or 'token'."
        )

    max_len = embedder.max_seq_length
    size = min(config.chunk_size, max_len) if max_len else config.chunk_size
    overlap = min(config.chunk_overlap, size - 1)
    return {
        "chunk_size": size,
        "chunk_overlap": overlap,
        "tokenizer": embedder.tokenizer,
    }


def run(
    config: BenchmarkConfig,
    retrievers: Dict[str, Retriever] | None = None,
    data: BenchmarkData | None = None,
) -> None:
    """Run the system-vs-baseline benchmark and write the comparison CSV.

    Loads the benchmark, builds each retriever's Run once (reused for the aggregate
    table and, when ``config.per_query_out`` is set, the per-query CSV), and writes
    the table with :func:`write_results_csv`.

    ``data`` and ``retrievers`` are injectable so the orchestration is testable
    without the real loader or real retrievers; in normal use both are left
    ``None`` and built from ``config`` via :func:`_build_retrievers` (BM25 and
    the dense retrievers are all wired; dense construction downloads its model).
    """
    if data is None:
        data = load_benchmark(config.dataset, split=config.split)
    if retrievers is None:
        retrievers = _build_retrievers(config, data)

    # Build each retriever's run once, timing the query phase (end-to-end retrieval
    # latency) — the basis for the flat-vs-ANN speedup comparison at scale.
    n_queries = max(1, len(data.queries))
    runs: Dict[str, Run] = {}
    ms_per_query: Dict[str, float] = {}
    for name, retriever in retrievers.items():
        start = time.perf_counter()
        runs[name] = build_run(
            retriever, data.queries, pooling=config.pooling, top_n_docs=max(config.k_values)
        )
        ms_per_query[name] = (time.perf_counter() - start) / n_queries * 1000.0
    results = {}
    for name, run_ in runs.items():
        metrics = evaluate_run(run_, data.qrels, config.k_values)
        metrics["ms_per_query"] = ms_per_query[name]
        results[name] = metrics
    config_columns = None
    if config.append:  # sweep mode: self-describe each row and accumulate
        config_columns = {
            "dataset": config.dataset,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "pooling": config.pooling,
            "chunk_unit": config.chunk_unit,
            "embedding_model_id": config.embedding_model_id or "default",
        }
    write_results_csv(
        results,
        config.results_path,
        config_columns=config_columns,
        append=config.append,
    )
    if config.per_query_out is not None:  # Tier-1 input: per-query scores for significance
        per_query = {
            name: per_query_scores(run_, data.qrels, config.per_query_metric)
            for name, run_ in runs.items()
        }
        write_per_query_csv(per_query, config.per_query_out)


def _parse_args(argv: List[str] | None = None) -> BenchmarkConfig:
    """Parse command-line arguments into a :class:`BenchmarkConfig`.

    Example::

        python -m eval.run_benchmark --dataset scifact --retrievers bm25 st_dense
    """
    defaults = BenchmarkConfig()
    parser = argparse.ArgumentParser(
        prog="python -m eval.run_benchmark",
        description="Run the system-vs-baselines retrieval benchmark and write a CSV.",
    )
    parser.add_argument(
        "--dataset",
        default=defaults.dataset,
        help="Benchmark name to evaluate on (default: %(default)s).",
    )
    parser.add_argument(
        "--split",
        default=defaults.split,
        help="Dataset split (default: %(default)s). MS MARCO / NQ use 'dev'.",
    )
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=defaults.retrievers,
        help=(
            "Retrievers to compare (default: %(default)s). Available: 'bm25', "
            "dense baselines " + ", ".join(sorted(DENSE_SPECS)) + ", RRF hybrids "
            + ", ".join(sorted(HYBRID_SPECS)) + ", and two-stage rerankers "
            + ", ".join(sorted(RERANK_SPECS)) + " (hybrids fuse a dense retriever "
            "with BM25; *_rerank add the Granite cross-encoder)."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=defaults.k_values,
        dest="k_values",
        help="Cut-offs for precision/recall/nDCG (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=defaults.results_path,
        dest="results_path",
        help="Where to write the comparison CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=defaults.index_cache_dir,
        dest="index_cache_dir",
        help="Cache dense indexes here to skip re-embedding on reruns (default: off).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=defaults.chunk_size,
        dest="chunk_size",
        help="Token window for chunking documents (default: %(default)s). "
             "Note: SciFact abstracts are short — chunk ablations are more "
             "informative on long-document corpora such as NQ.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=defaults.chunk_overlap,
        dest="chunk_overlap",
        help="Overlap tokens between consecutive chunks (default: %(default)s).",
    )
    parser.add_argument(
        "--pooling",
        default=defaults.pooling,
        dest="pooling",
        choices=["max", "mean"],
        help="Chunk-to-doc score aggregation: max (default) or mean.",
    )
    parser.add_argument(
        "--chunk-unit",
        default=defaults.chunk_unit,
        dest="chunk_unit",
        choices=["word", "token"],
        help="Unit for chunk size/overlap: 'word' (default, whitespace split) or "
        "'token' (the embedding model's tokens, capped at its max_seq_length so "
        "long docs are split rather than silently truncated). Use 'token' for "
        "long-document corpora such as NQ.",
    )
    parser.add_argument(
        "--dense-fanout",
        type=int,
        default=defaults.dense_fanout,
        dest="dense_fanout",
        help="Chunks the dense retrievers fetch per query as a multiple of "
        "max(k), so the chunk->doc pooling still yields max(k) distinct docs on "
        "multi-chunk corpora (default: %(default)s; no effect on SciFact).",
    )
    parser.add_argument(
        "--embedding-model-id",
        default=defaults.embedding_model_id,
        dest="embedding_model_id",
        help="Override the dense embedding model, e.g. a Granite variant "
        "(default: backend default).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=defaults.append,
        help="Sweep mode: tag rows with the config and append to the CSV "
        "instead of overwriting (default: off).",
    )
    parser.add_argument(
        "--per-query-out",
        type=Path,
        default=defaults.per_query_out,
        dest="per_query_out",
        help="Also write per-query scores to this CSV (qid x retriever) for "
        "eval.significance — paired significance testing / failure analysis "
        "(default: off).",
    )
    parser.add_argument(
        "--per-query-metric",
        default=defaults.per_query_metric,
        dest="per_query_metric",
        help="Metric for --per-query-out (default: %(default)s).",
    )
    parser.add_argument(
        "--k-rrf",
        type=int,
        default=defaults.k_rrf,
        dest="k_rrf",
        help="RRF damping constant for hybrid retrievers (default: %(default)s).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=defaults.alpha,
        dest="alpha",
        help="Dense weight for convex-combination hybrids "
        "(0 = BM25 only, 1 = dense only; default: %(default)s).",
    )
    parser.add_argument(
        "--reranker-model-id",
        default=defaults.reranker_model_id,
        dest="reranker_model_id",
        help="Cross-encoder model for the *_rerank retrievers (default: %(default)s).",
    )
    parser.add_argument(
        "--rerank-pool",
        type=int,
        default=defaults.rerank_pool,
        dest="rerank_pool",
        help="Candidate-pool size for the LLM listwise rerankers (*_listrank); "
        "default is the dense fan-out pool. Lower it to bound LLM calls per query.",
    )
    parser.add_argument(
        "--index-type",
        default=defaults.index_type,
        dest="index_type",
        choices=["flat", "hnsw", "ivf"],
        help="FAISS index for dense retrievers: 'flat' (exact, default) or the ANN "
        "indexes 'hnsw'/'ivf' — far faster on large corpora at ~no recall loss.",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=defaults.ef_search,
        dest="ef_search",
        help="HNSW search breadth (recall/speed knob; default: %(default)s).",
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=defaults.nprobe,
        dest="nprobe",
        help="IVF cells probed per query (recall/speed knob; default: %(default)s).",
    )
    args = parser.parse_args(argv)
    return BenchmarkConfig(
        dataset=args.dataset,
        split=args.split,
        retrievers=args.retrievers,
        k_values=args.k_values,
        results_path=args.results_path,
        index_cache_dir=args.index_cache_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        pooling=args.pooling,
        chunk_unit=args.chunk_unit,
        dense_fanout=args.dense_fanout,
        embedding_model_id=args.embedding_model_id,
        append=args.append,
        per_query_out=args.per_query_out,
        per_query_metric=args.per_query_metric,
        k_rrf=args.k_rrf,
        alpha=args.alpha,
        reranker_model_id=args.reranker_model_id,
        rerank_pool=args.rerank_pool,
        index_type=args.index_type,
        ef_search=args.ef_search,
        nprobe=args.nprobe,
    )


def main(argv: List[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
