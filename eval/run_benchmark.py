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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List

from eval.benchmarks.loader import BenchmarkData, load_benchmark
from eval.ir_metrics import Run, evaluate_run
from src.ingestion.chunker import Chunk, chunk_document
from src.ingestion.indexer import FaissIndex, VectorIndexer
from src.retrieval.base import Retriever
from src.retrieval.bm25_baseline import BM25Retriever
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import DenseRetriever


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
    """

    dataset: str = "scifact"
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


def _cache_key(config: BenchmarkConfig, name: str) -> str:
    """Cache-file stem for retriever ``name``'s dense index under ``config``.

    Folds the ablation params (chunk size/overlap, embedding model) into the key
    alongside dataset + retriever name, so indexes built for different configs do
    not collide — a chunk-size sweep with ``--cache-dir`` would otherwise reload a
    stale index built at a different size.
    """
    model_slug = (config.embedding_model_id or "default").replace("/", "-")
    return (
        f"{config.dataset}__{name}"
        f"__cs{config.chunk_size}_ov{config.chunk_overlap}__{model_slug}"
        f"__{config.chunk_unit}"
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


def _build_retrievers(
    config: BenchmarkConfig,
    data: BenchmarkData,
) -> Dict[str, Retriever]:
    """Construct the retrievers named in ``config`` over ``data.corpus``.

    - ``bm25`` builds its own term index from the corpus and returns docs
      directly, so it fetches ``max(k)`` docs.
    - ``granite_dense`` / ``st_dense`` chunk the corpus, embed it with the
      matching :class:`~src.retrieval.embedder.Embedder` backend, build a FAISS
      index (:class:`~src.ingestion.indexer.VectorIndexer`, contract 5), and wrap
      it in a :class:`~src.retrieval.retriever.DenseRetriever`. They rank
      *chunks*, so they fetch ``max(k) * dense_fanout`` chunks to guarantee at
      least ``max(k)`` distinct docs survive the chunk->doc pooling (the run is
      trimmed back to ``max(k)`` docs in :func:`build_run`).

    When ``config.chunk_unit == "token"`` the corpus is split on each embedder's
    own tokenizer with the chunk size capped at the model's ``max_seq_length``,
    so chunks are never silently truncated at encode time; otherwise it is split
    on whitespace words (the default). Building a dense retriever loads its
    embedding model (downloaded from Hugging Face on first use), so that is the
    one path needing network/models.
    """
    doc_ids = list(data.corpus.keys())
    corpus = list(data.corpus.values())
    top_k = max(config.k_values)

    retrievers: Dict[str, Retriever] = {}
    for name in config.retrievers:
        if name == "bm25":
            retrievers[name] = BM25Retriever(corpus, doc_ids, top_k=top_k)
        elif name in DENSE_SPECS:
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
            indexer = VectorIndexer(embedder)
            if config.index_cache_dir is not None:
                cache_path = config.index_cache_dir / _cache_key(config, name)
                index = _load_or_build_index(indexer, chunks, cache_path)
            else:
                index = indexer.build(chunks)
            retrievers[name] = DenseRetriever(
                embedder, index, top_k=top_k * config.dense_fanout
            )
        else:
            raise ValueError(
                f"Unknown retriever {name!r}; expected 'bm25' or one of "
                f"{sorted(DENSE_SPECS)}."
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

    Loads the benchmark, scores every retriever with :func:`evaluate_one`, and
    writes the table with :func:`write_results_csv`.

    ``data`` and ``retrievers`` are injectable so the orchestration is testable
    without the real loader or real retrievers; in normal use both are left
    ``None`` and built from ``config`` via :func:`_build_retrievers` (BM25 and
    the dense retrievers are all wired; dense construction downloads its model).
    """
    if data is None:
        data = load_benchmark(config.dataset)
    if retrievers is None:
        retrievers = _build_retrievers(config, data)

    results = {
        name: evaluate_one(retriever, data, config.k_values, pooling=config.pooling)
        for name, retriever in retrievers.items()
    }
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
        "--retrievers",
        nargs="+",
        default=defaults.retrievers,
        help=(
            "Retrievers to compare (default: %(default)s). Available: 'bm25' plus "
            "dense baselines " + ", ".join(sorted(DENSE_SPECS)) + " (gte/e5/bge are "
            "modern same-class peers to Granite; their prefixes are wired in)."
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
    args = parser.parse_args(argv)
    return BenchmarkConfig(
        dataset=args.dataset,
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
    )


def main(argv: List[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
