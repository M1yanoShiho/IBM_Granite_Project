"""Tests for the P6 benchmark orchestration (``eval/run_benchmark.py``).

Covers the parts of P6 that are unblocked by the shared contracts and need no
real retriever (P3/P4) or index (P5): the chunk->doc aggregation (``build_run``,
contract 3), the per-retriever scoring glue, the comparison-table assembly, and
the CSV output. A ``FakeRetriever`` (canned per-query responses) stands in for
the real retrievers — it satisfies the ``Retriever`` Protocol structurally.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import pytest

from src.retrieval.base import RetrievedChunk, Retriever
from src.ingestion.chunker import Chunk
from eval.benchmarks.loader import BenchmarkData
from eval.run_benchmark import (
    BenchmarkConfig,
    _build_retrievers,
    _cache_key,
    _chunk_kwargs_for,
    _load_or_build_index,
    _parse_args,
    build_run,
    evaluate_one,
    run,
    write_results_csv,
)


class FakeRetriever:
    """A retriever with canned responses, keyed by query text.

    Satisfies the ``Retriever`` Protocol just by defining ``retrieve`` — no
    subclassing needed (the Protocol is structural / ``runtime_checkable``).
    """

    def __init__(self, responses: Dict[str, List[RetrievedChunk]]) -> None:
        self._responses = responses

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        return self._responses.get(query, [])


def test_build_run_max_pools_chunks_of_same_doc() -> None:
    # d1 surfaces via two chunks (0.4 and 0.9); contract 3 says the doc's score
    # is the MAX of its chunk scores, not the first/last/sum. d2 has one chunk.
    retriever = FakeRetriever(
        {
            "some query": [
                RetrievedChunk(doc_id="d1", text="chunk a", score=0.4),
                RetrievedChunk(doc_id="d1", text="chunk b", score=0.9),
                RetrievedChunk(doc_id="d2", text="other", score=0.5),
            ],
        }
    )

    run = build_run(retriever, {"q1": "some query"})

    assert run == {"q1": {"d1": 0.9, "d2": 0.5}}


def test_build_run_mean_pools_chunks_of_same_doc() -> None:
    # With pooling="mean", a doc's score is the MEAN of its retrieved chunk
    # scores (over the chunks actually returned), not the max. d1: mean(0.4, 0.9)
    # = 0.65; d2 has a single chunk, so its score is unchanged.
    retriever = FakeRetriever(
        {
            "some query": [
                RetrievedChunk(doc_id="d1", text="chunk a", score=0.4),
                RetrievedChunk(doc_id="d1", text="chunk b", score=0.9),
                RetrievedChunk(doc_id="d2", text="other", score=0.5),
            ],
        }
    )

    run = build_run(retriever, {"q1": "some query"}, pooling="mean")

    assert run == {"q1": {"d1": pytest.approx(0.65), "d2": pytest.approx(0.5)}}


def test_build_run_trims_to_top_n_docs() -> None:
    # With many distinct docs but top_n_docs=2, only the two highest-scored docs
    # survive — the trim that keeps dense (which over-fetches chunks) the same
    # depth as BM25 so MRR / @k are comparable.
    retriever = FakeRetriever(
        {
            "some query": [
                RetrievedChunk(doc_id="d1", text="", score=0.9),
                RetrievedChunk(doc_id="d2", text="", score=0.8),
                RetrievedChunk(doc_id="d3", text="", score=0.7),
            ],
        }
    )

    run = build_run(retriever, {"q1": "some query"}, top_n_docs=2)

    assert run == {"q1": {"d1": 0.9, "d2": 0.8}}


def test_build_run_without_top_n_keeps_all_docs() -> None:
    # Default (top_n_docs=None) preserves the original behaviour: no trimming.
    retriever = FakeRetriever(
        {
            "some query": [
                RetrievedChunk(doc_id="d1", text="", score=0.9),
                RetrievedChunk(doc_id="d2", text="", score=0.8),
                RetrievedChunk(doc_id="d3", text="", score=0.7),
            ],
        }
    )

    run = build_run(retriever, {"q1": "some query"})

    assert set(run["q1"]) == {"d1", "d2", "d3"}


def test_evaluate_one_scores_a_perfect_retriever_as_one() -> None:
    # Each query has a single relevant doc; the retriever ranks it first. So the
    # rank-based metrics are exactly 1.0. (precision@k for k>1 is not — only one
    # relevant doc exists — so we don't assert on it: that would be flaky.)
    data = BenchmarkData(
        corpus={"d1": "doc one", "d2": "doc two", "d3": "doc three"},
        queries={"q1": "qtext1", "q2": "qtext2"},
        qrels={"q1": {"d1": 1}, "q2": {"d2": 1}},
    )
    retriever = FakeRetriever(
        {
            "qtext1": [
                RetrievedChunk(doc_id="d1", text="doc one", score=0.9),
                RetrievedChunk(doc_id="d3", text="doc three", score=0.5),
            ],
            "qtext2": [
                RetrievedChunk(doc_id="d2", text="doc two", score=0.9),
                RetrievedChunk(doc_id="d3", text="doc three", score=0.5),
            ],
        }
    )

    metrics = evaluate_one(retriever, data, k_values=[1, 10])

    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["recall@10"] == pytest.approx(1.0)
    assert metrics["ndcg@10"] == pytest.approx(1.0)
    assert metrics["precision@1"] == pytest.approx(1.0)


def test_write_results_csv_round_trips(tmp_path) -> None:
    # One row per retriever; the retriever name is a column; values survive the
    # write+read round-trip. Read back with DictReader so column order is free.
    results = {
        "bm25": {"precision@1": 0.5, "recall@10": 0.8, "ndcg@10": 0.6, "mrr": 0.7},
        "granite_dense": {"precision@1": 0.9, "recall@10": 0.95, "ndcg@10": 0.92, "mrr": 0.93},
    }
    path = tmp_path / "benchmark_results.csv"

    write_results_csv(results, path)

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    by_name = {row["retriever"]: row for row in rows}
    assert set(by_name) == {"bm25", "granite_dense"}
    assert float(by_name["granite_dense"]["mrr"]) == 0.93
    assert float(by_name["bm25"]["precision@1"]) == 0.5


def test_write_results_csv_prepends_config_columns(tmp_path) -> None:
    # When sweeping, each row is tagged with the run's config (dataset, chunk
    # size, pooling, ...) so many runs accumulate in one analyzable table. The
    # config columns lead, then "retriever", then the metric columns.
    results = {"bm25": {"ndcg@10": 0.6, "mrr": 0.7}}
    path = tmp_path / "ablation.csv"

    write_results_csv(
        results,
        path,
        config_columns={"dataset": "scifact", "chunk_size": 256, "pooling": "mean"},
    )

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)

    assert header[:4] == ["dataset", "chunk_size", "pooling", "retriever"]
    row = rows[0]
    assert row["dataset"] == "scifact"
    assert row["chunk_size"] == "256"
    assert row["pooling"] == "mean"
    assert row["retriever"] == "bm25"
    assert float(row["ndcg@10"]) == 0.6


def test_write_results_csv_appends_without_duplicating_header(tmp_path) -> None:
    # A sweep calls write_results_csv repeatedly with append=True so rows from
    # many configs accumulate in one file under a single header.
    path = tmp_path / "ablation.csv"

    write_results_csv({"bm25": {"ndcg@10": 0.6}}, path, config_columns={"chunk_size": 256})
    write_results_csv(
        {"bm25": {"ndcg@10": 0.5}}, path, config_columns={"chunk_size": 128}, append=True
    )

    with open(path, newline="") as f:
        non_empty = [ln for ln in f.read().splitlines() if ln]
    assert len(non_empty) == 3  # one header + two data rows

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["chunk_size"] for r in rows] == ["256", "128"]
    assert [r["ndcg@10"] for r in rows] == ["0.6", "0.5"]


def test_run_writes_one_row_per_injected_retriever(tmp_path) -> None:
    # End-to-end orchestration with injected fakes (no real loader / retriever):
    # run() scores each retriever and writes the comparison CSV. "perfect" ranks
    # the relevant doc first (mrr 1.0); "wrong" only finds an irrelevant one (0.0).
    data = BenchmarkData(
        corpus={"d1": "doc one", "d2": "doc two"},
        queries={"q1": "qtext1"},
        qrels={"q1": {"d1": 1}},
    )
    retrievers = {
        "perfect": FakeRetriever({"qtext1": [RetrievedChunk("d1", "doc one", 0.9)]}),
        "wrong": FakeRetriever({"qtext1": [RetrievedChunk("d2", "doc two", 0.9)]}),
    }
    config = BenchmarkConfig(k_values=[1], results_path=tmp_path / "out.csv")

    run(config, retrievers=retrievers, data=data)

    with open(config.results_path, newline="") as f:
        rows = {row["retriever"]: row for row in csv.DictReader(f)}
    assert set(rows) == {"perfect", "wrong"}
    assert float(rows["perfect"]["mrr"]) == 1.0
    assert float(rows["wrong"]["mrr"]) == 0.0


def test_run_respects_pooling_config(tmp_path) -> None:
    # d1 (relevant) surfaces via two chunks (0.9, 0.1); d2 (irrelevant) via one
    # (0.6). Max-pool: d1=0.9 > d2=0.6 -> d1 first -> mrr 1.0. Mean-pool:
    # d1=0.5 < d2=0.6 -> d1 second -> mrr 0.5. The flipped result proves pooling
    # is threaded run -> evaluate_one -> build_run.
    data = BenchmarkData(
        corpus={"d1": "doc one", "d2": "doc two"},
        queries={"q1": "qtext1"},
        qrels={"q1": {"d1": 1}},
    )
    retrievers = {
        "r": FakeRetriever(
            {
                "qtext1": [
                    RetrievedChunk("d1", "a", 0.9),
                    RetrievedChunk("d1", "b", 0.1),
                    RetrievedChunk("d2", "c", 0.6),
                ]
            }
        )
    }

    def mrr_for(pooling: str) -> float:
        config = BenchmarkConfig(
            k_values=[10], pooling=pooling, results_path=tmp_path / f"{pooling}.csv"
        )
        run(config, retrievers=retrievers, data=data)
        with open(config.results_path, newline="") as f:
            return float(next(csv.DictReader(f))["mrr"])

    assert mrr_for("max") == pytest.approx(1.0)
    assert mrr_for("mean") == pytest.approx(0.5)


def test_run_tags_and_appends_when_append_set(tmp_path) -> None:
    # Sweep mode: run() tags each row with its config and appends, so two runs
    # with different configs accumulate in one master CSV.
    data = BenchmarkData(
        corpus={"d1": "doc one"}, queries={"q1": "qtext1"}, qrels={"q1": {"d1": 1}}
    )
    retrievers = {"r": FakeRetriever({"qtext1": [RetrievedChunk("d1", "x", 0.9)]})}
    out = tmp_path / "sweep.csv"

    run(
        BenchmarkConfig(k_values=[1], chunk_size=512, pooling="max", results_path=out, append=True),
        retrievers=retrievers,
        data=data,
    )
    run(
        BenchmarkConfig(k_values=[1], chunk_size=256, pooling="mean", results_path=out, append=True),
        retrievers=retrievers,
        data=data,
    )

    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["chunk_size"] for r in rows] == ["512", "256"]
    assert [r["pooling"] for r in rows] == ["max", "mean"]
    assert rows[0]["retriever"] == "r"


def test_run_builds_real_bm25_and_writes_scored_row(tmp_path) -> None:
    # No injected retrievers -> run() builds the real BM25Retriever itself
    # (via _build_retrievers). The query terms appear only in d1, the relevant
    # doc, so BM25 ranks it first and the scored row is a perfect result.
    data = BenchmarkData(
        corpus={
            "d1": "granite dense retrieval embeddings",
            "d2": "banana cake recipe with sugar",
            "d3": "weather forecast tomorrow rain",
        },
        queries={"q1": "granite retrieval"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(
        retrievers=["bm25"],
        k_values=[1],
        results_path=tmp_path / "out.csv",
    )

    run(config, data=data)

    with open(config.results_path, newline="") as f:
        rows = {row["retriever"]: row for row in csv.DictReader(f)}
    assert set(rows) == {"bm25"}
    assert float(rows["bm25"]["mrr"]) == 1.0


class FakeSentenceTransformer:
    """Deterministic stand-in for SentenceTransformer (no model download).

    Embeds text as a bag-of-words count over a tiny fixed vocabulary, so a query
    lands closest to the doc that shares its words.
    """

    _VOCAB = ("granite", "retrieval", "banana", "cake")

    def __init__(self, model_id, cache_folder=None) -> None:
        pass

    def encode(
        self,
        texts,
        convert_to_numpy: bool = False,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
    ):
        if isinstance(texts, str):
            texts = [texts]
        return [[float(t.lower().split().count(w)) for w in self._VOCAB] for t in texts]


def test_build_retrievers_wires_dense_over_corpus(monkeypatch) -> None:
    # Dense branch: _build_retrievers builds an Embedder, chunks the corpus,
    # builds a FAISS index, and wraps it in a DenseRetriever. The embedding model
    # is monkeypatched so nothing is downloaded.
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSentenceTransformer
    )
    data = BenchmarkData(
        corpus={"d1": "granite retrieval", "d2": "banana cake"},
        queries={"q1": "granite retrieval"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(retrievers=["st_dense"], k_values=[1])

    retrievers = _build_retrievers(config, data)

    assert set(retrievers) == {"st_dense"}
    dense = retrievers["st_dense"]
    assert isinstance(dense, Retriever)
    results = dense.retrieve("granite retrieval")
    assert all(isinstance(r, RetrievedChunk) for r in results)
    # dense over-fetches (max(k) * dense_fanout chunks) so the chunk->doc pooling
    # still yields enough distinct docs; the relevant doc must rank first.
    assert results[0].doc_id == "d1"


def test_build_retrievers_dense_over_fetches_by_fanout(monkeypatch) -> None:
    # The dense retriever fetches max(k) * dense_fanout chunks so the chunk->doc
    # pooling still yields enough distinct docs on multi-chunk corpora.
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSentenceTransformer
    )
    data = BenchmarkData(
        corpus={"d1": "granite retrieval", "d2": "banana cake"},
        queries={"q1": "granite retrieval"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(retrievers=["st_dense"], k_values=[5], dense_fanout=4)

    retrievers = _build_retrievers(config, data)

    assert retrievers["st_dense"].top_k == 20  # max(k)=5 * fanout 4


def test_chunk_kwargs_for_word_mode_forwards_size_and_overlap() -> None:
    config = BenchmarkConfig(chunk_unit="word", chunk_size=512, chunk_overlap=50)
    kwargs = _chunk_kwargs_for(config, object())  # embedder unused in word mode
    assert kwargs == {"chunk_size": 512, "chunk_overlap": 50}


def test_chunk_kwargs_for_token_mode_caps_at_max_seq_length() -> None:
    # Token mode passes the embedder's tokenizer and caps the chunk size at the
    # model's max_seq_length so chunks are never silently truncated at encode time.
    config = BenchmarkConfig(chunk_unit="token", chunk_size=512, chunk_overlap=50)
    embedder = SimpleNamespace(max_seq_length=256, tokenizer="TOK")

    kwargs = _chunk_kwargs_for(config, embedder)

    assert kwargs == {"chunk_size": 256, "chunk_overlap": 50, "tokenizer": "TOK"}


def test_chunk_kwargs_for_token_mode_clamps_overlap_below_size() -> None:
    config = BenchmarkConfig(chunk_unit="token", chunk_size=512, chunk_overlap=400)
    embedder = SimpleNamespace(max_seq_length=128, tokenizer="TOK")

    kwargs = _chunk_kwargs_for(config, embedder)

    assert kwargs["chunk_size"] == 128
    assert kwargs["chunk_overlap"] == 127  # clamped below the capped size


def test_chunk_kwargs_for_rejects_unknown_unit() -> None:
    config = BenchmarkConfig(chunk_unit="bogus")
    with pytest.raises(ValueError, match="Unknown chunk_unit"):
        _chunk_kwargs_for(config, object())


def test_build_retrievers_threads_chunk_params(monkeypatch) -> None:
    # Ablation: chunk size/overlap from the config must reach chunk_document,
    # else a chunk-size sweep silently re-chunks at the default 512/50.
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSentenceTransformer
    )
    calls = []

    def spy_chunk_document(doc_id, text, chunk_size=512, chunk_overlap=50):
        calls.append((chunk_size, chunk_overlap))
        return [Chunk(chunk_id=f"{doc_id}::0", doc_id=doc_id, text=text)]

    monkeypatch.setattr("eval.run_benchmark.chunk_document", spy_chunk_document)

    data = BenchmarkData(
        corpus={"d1": "granite retrieval"},
        queries={"q1": "granite retrieval"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(
        retrievers=["st_dense"], k_values=[1], chunk_size=256, chunk_overlap=32
    )

    _build_retrievers(config, data)

    assert calls == [(256, 32)]


def test_build_retrievers_threads_embedding_model_id(monkeypatch) -> None:
    # The model override must reach the embedding model, so a Granite-variant
    # sweep actually swaps models (not only the cache key).
    captured = {}

    class CapturingST(FakeSentenceTransformer):
        def __init__(self, model_id, cache_folder=None) -> None:
            captured["model_id"] = model_id
            super().__init__(model_id, cache_folder)

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", CapturingST)

    data = BenchmarkData(
        corpus={"d1": "granite retrieval"},
        queries={"q1": "granite retrieval"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(
        retrievers=["granite_dense"],
        k_values=[1],
        embedding_model_id="ibm-granite/granite-embedding-small-english-r2",
    )

    _build_retrievers(config, data)

    assert captured["model_id"] == "ibm-granite/granite-embedding-small-english-r2"


def test_parse_args_uses_config_defaults() -> None:
    config = _parse_args([])
    assert config.dataset == "scifact"
    assert config.retrievers == ["granite_dense", "bm25", "st_dense"]
    assert config.k_values == [1, 3, 5, 10]
    assert config.results_path == Path("results/benchmark_results.csv")


def test_parse_args_overrides_from_argv() -> None:
    config = _parse_args(
        ["--dataset", "nq", "--retrievers", "bm25", "--k", "1", "10", "--out", "x/y.csv"]
    )
    assert config.dataset == "nq"
    assert config.retrievers == ["bm25"]
    assert config.k_values == [1, 10]
    assert config.results_path == Path("x/y.csv")


def test_parse_args_parses_ablation_flags() -> None:
    # Defaults reproduce today's behavior.
    d = _parse_args([])
    assert d.chunk_size == 512
    assert d.chunk_overlap == 50
    assert d.pooling == "max"
    assert d.chunk_unit == "word"
    assert d.dense_fanout == 10
    assert d.embedding_model_id is None
    assert d.append is False

    config = _parse_args(
        [
            "--chunk-size", "256",
            "--overlap", "0",
            "--pooling", "mean",
            "--chunk-unit", "token",
            "--dense-fanout", "5",
            "--embedding-model-id", "ibm-granite/x",
            "--append",
        ]
    )
    assert config.chunk_size == 256
    assert config.chunk_overlap == 0
    assert config.pooling == "mean"
    assert config.chunk_unit == "token"
    assert config.dense_fanout == 5
    assert config.embedding_model_id == "ibm-granite/x"
    assert config.append is True


class RecordingIndexer:
    """Fake VectorIndexer that records build/load calls (no real FAISS/model)."""

    def __init__(self) -> None:
        self.built = 0
        self.loaded = 0
        self._index = object()

    def build(self, chunks):
        self.built += 1
        return self._index

    def save(self, index, path) -> None:
        Path(f"{path}.faiss").write_text("stub")

    def load(self, path):
        self.loaded += 1
        return self._index


def test_load_or_build_index_builds_and_saves_when_cold(tmp_path) -> None:
    indexer = RecordingIndexer()
    cache_path = tmp_path / "scifact__st_dense"

    idx = _load_or_build_index(indexer, ["chunk"], cache_path)

    assert indexer.built == 1
    assert indexer.loaded == 0
    assert Path(f"{cache_path}.faiss").exists()
    assert idx is indexer._index


def test_load_or_build_index_loads_when_warm(tmp_path) -> None:
    indexer = RecordingIndexer()
    cache_path = tmp_path / "scifact__st_dense"

    _load_or_build_index(indexer, ["chunk"], cache_path)        # cold: build + save
    idx = _load_or_build_index(indexer, ["chunk"], cache_path)  # warm: load only

    assert indexer.built == 1          # did NOT rebuild
    assert indexer.loaded == 1
    assert idx is indexer._index


def test_cache_key_distinguishes_chunk_and_model_configs() -> None:
    # Different ablation configs must map to different cache keys, else a cached
    # index built for one config is silently reused for another (corrupt sweep).
    base = BenchmarkConfig(dataset="scifact")
    key = _cache_key(base, "granite_dense")

    assert _cache_key(base, "granite_dense") == key  # stable -> warm reruns hit cache
    assert _cache_key(replace(base, chunk_size=256), "granite_dense") != key
    assert _cache_key(replace(base, chunk_overlap=0), "granite_dense") != key
    assert (
        _cache_key(replace(base, embedding_model_id="ibm-granite/other"), "granite_dense")
        != key
    )
    assert _cache_key(replace(base, chunk_unit="token"), "granite_dense") != key
    assert _cache_key(base, "st_dense") != key  # different retriever


def test_build_retrievers_dense_uses_cache_path(monkeypatch, tmp_path) -> None:
    # With a cache dir set, the dense branch routes index construction through
    # _load_or_build_index, keyed by "<dataset>__<name>". Stub that helper so the
    # test checks the wiring/keying without depending on faiss writing to disk
    # (the save logic itself is covered by the _load_or_build_index tests above).
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSentenceTransformer
    )
    captured = {}

    def fake_load_or_build(indexer, chunks, cache_path):
        captured["cache_path"] = cache_path
        return indexer.build(chunks)

    monkeypatch.setattr("eval.run_benchmark._load_or_build_index", fake_load_or_build)

    data = BenchmarkData(
        corpus={"d1": "granite retrieval"},
        queries={"q1": "granite retrieval"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(
        retrievers=["st_dense"], k_values=[1], index_cache_dir=tmp_path
    )

    _build_retrievers(config, data)

    assert captured["cache_path"] == tmp_path / "scifact__st_dense__cs512_ov50__default__word"


def test_parse_args_cache_dir() -> None:
    assert _parse_args([]).index_cache_dir is None
    assert _parse_args(["--cache-dir", "tmp/idx"]).index_cache_dir == Path("tmp/idx")


def test_build_run_unknown_pooling_raises() -> None:
    retriever = FakeRetriever({"q": []})
    with pytest.raises(ValueError, match="Unknown pooling"):
        build_run(retriever, {"q1": "q"}, pooling="sum")


def test_evaluate_one_with_mean_pooling() -> None:
    data = BenchmarkData(
        corpus={"d1": "doc one", "d2": "doc two"},
        queries={"q1": "qtext"},
        qrels={"q1": {"d1": 1}},
    )
    # d1 has two chunks; mean of (0.6, 0.8) = 0.7 beats d2's single chunk 0.5
    retriever = FakeRetriever(
        {
            "qtext": [
                RetrievedChunk(doc_id="d1", text="chunk a", score=0.6),
                RetrievedChunk(doc_id="d1", text="chunk b", score=0.8),
                RetrievedChunk(doc_id="d2", text="other", score=0.5),
            ]
        }
    )
    metrics = evaluate_one(retriever, data, k_values=[1], pooling="mean")
    assert metrics["mrr"] == pytest.approx(1.0)


def test_build_retrievers_builds_gte_baseline_with_pinned_model(monkeypatch) -> None:
    # gte_dense is a modern, same-class open baseline (a fair peer to Granite,
    # unlike the older/smaller MiniLM). It must be a first-class retriever name,
    # pinned to gte-base, so one benchmark run compares Granite against a current
    # peer. gte needs no query/passage prefix -> apples-to-apples with Granite.
    captured = {}

    class CapturingST(FakeSentenceTransformer):
        def __init__(self, model_id, cache_folder=None) -> None:
            captured["model_id"] = model_id
            super().__init__(model_id, cache_folder)

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", CapturingST)

    data = BenchmarkData(
        corpus={"d1": "granite retrieval", "d2": "banana cake"},
        queries={"q1": "granite retrieval"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(retrievers=["gte_dense"], k_values=[1])

    retrievers = _build_retrievers(config, data)

    assert captured["model_id"] == "thenlper/gte-base"
    dense = retrievers["gte_dense"]
    assert isinstance(dense, Retriever)
    assert dense.retrieve("granite retrieval")[0].doc_id == "d1"


def test_build_retrievers_e5_baseline_prefixes_query_and_passages(monkeypatch) -> None:
    # e5 models are crippled without their "query: " / "passage: " prefixes, which
    # would make the "strong baseline" an accidental strawman. e5_dense must thread
    # those prefixes through to the encoder for documents (at indexing) and for the
    # query (at retrieval).
    recorded: List[str] = []

    class RecordingST(FakeSentenceTransformer):
        def encode(self, texts, **kwargs):
            recorded.extend([texts] if isinstance(texts, str) else list(texts))
            return super().encode(texts, **kwargs)

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", RecordingST)

    data = BenchmarkData(
        corpus={"d1": "granite retrieval"},
        queries={"q1": "granite retrieval"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(retrievers=["e5_dense"], k_values=[1])

    _build_retrievers(config, data)["e5_dense"].retrieve("granite retrieval")

    assert any(t.startswith("passage: ") for t in recorded)  # documents, at indexing
    assert any(t.startswith("query: ") for t in recorded)     # query, at retrieval


def test_build_retrievers_supports_multiple_dense_models_in_one_run(monkeypatch) -> None:
    # The Tier-0 goal: a single run compares Granite against several baselines at
    # once. st_dense (MiniLM) and gte_dense (gte-base) share the sentence-
    # transformers backend but must each get their own model -- impossible under
    # the old single embedding_model_id override.
    seen: List[str] = []

    class CapturingST(FakeSentenceTransformer):
        def __init__(self, model_id, cache_folder=None) -> None:
            seen.append(model_id)
            super().__init__(model_id, cache_folder)

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", CapturingST)

    data = BenchmarkData(
        corpus={"d1": "granite retrieval", "d2": "banana cake"},
        queries={"q1": "granite retrieval"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(retrievers=["st_dense", "gte_dense"], k_values=[1])

    retrievers = _build_retrievers(config, data)

    assert set(retrievers) == {"st_dense", "gte_dense"}
    assert "thenlper/gte-base" in seen
    assert len(seen) == 2 and len(set(seen)) == 2  # two distinct models, one run


def test_build_retrievers_builds_granite_small_baseline_with_pinned_model(monkeypatch) -> None:
    # granite_small_dense pins the ~47M small Granite embedding model so the
    # efficiency question -- "does a much smaller Granite stay competitive with the
    # full r2 and with gte-base?" -- is measured in the same run. Granite embeddings
    # need no query/passage prefix.
    captured = {}

    class CapturingST(FakeSentenceTransformer):
        def __init__(self, model_id, cache_folder=None) -> None:
            captured["model_id"] = model_id
            super().__init__(model_id, cache_folder)

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", CapturingST)

    data = BenchmarkData(
        corpus={"d1": "granite retrieval", "d2": "banana cake"},
        queries={"q1": "granite retrieval"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(retrievers=["granite_small_dense"], k_values=[1])

    retrievers = _build_retrievers(config, data)

    assert captured["model_id"] == "ibm-granite/granite-embedding-small-english-r2"
    dense = retrievers["granite_small_dense"]
    assert isinstance(dense, Retriever)
    assert dense.retrieve("granite retrieval")[0].doc_id == "d1"


def test_write_per_query_csv_round_trips(tmp_path) -> None:
    # Wide format: first column qid, one column per retriever; the union of query
    # ids becomes the rows. This is what eval.significance.load_per_query_csv reads.
    from eval.run_benchmark import write_per_query_csv

    per_query = {
        "granite_dense": {"q1": 1.0, "q2": 0.0},
        "bm25": {"q1": 0.5, "q2": 0.5},
    }
    path = tmp_path / "per_query.csv"

    write_per_query_csv(per_query, path)

    with open(path, newline="") as f:
        by_qid = {row["qid"]: row for row in csv.DictReader(f)}
    assert set(by_qid) == {"q1", "q2"}
    assert float(by_qid["q1"]["granite_dense"]) == 1.0
    assert float(by_qid["q2"]["bm25"]) == 0.5


def test_run_writes_per_query_csv_when_requested(tmp_path) -> None:
    # With per_query_out set, run() also dumps per-query ndcg@10 (one column per
    # retriever) -- the input to eval.significance. "perfect" ranks the relevant doc
    # first on both queries (ndcg@10 = 1.0); "wrong" never finds it (0.0).
    data = BenchmarkData(
        corpus={"d1": "doc one", "d2": "doc two"},
        queries={"q1": "qa", "q2": "qb"},
        qrels={"q1": {"d1": 1}, "q2": {"d2": 1}},
    )
    retrievers = {
        "perfect": FakeRetriever(
            {"qa": [RetrievedChunk("d1", "", 0.9)], "qb": [RetrievedChunk("d2", "", 0.9)]}
        ),
        "wrong": FakeRetriever(
            {"qa": [RetrievedChunk("d2", "", 0.9)], "qb": [RetrievedChunk("d1", "", 0.9)]}
        ),
    }
    pq_path = tmp_path / "per_query.csv"
    config = BenchmarkConfig(
        k_values=[10],
        results_path=tmp_path / "agg.csv",
        per_query_out=pq_path,
    )

    run(config, retrievers=retrievers, data=data)

    with open(pq_path, newline="") as f:
        rows = {row["qid"]: row for row in csv.DictReader(f)}
    assert set(rows) == {"q1", "q2"}
    assert float(rows["q1"]["perfect"]) == pytest.approx(1.0)
    assert float(rows["q2"]["perfect"]) == pytest.approx(1.0)
    assert float(rows["q1"]["wrong"]) == pytest.approx(0.0)
    assert float(rows["q2"]["wrong"]) == pytest.approx(0.0)


def test_parse_args_per_query_flags() -> None:
    assert _parse_args([]).per_query_out is None
    config = _parse_args(
        ["--per-query-out", "results/pq.csv", "--per-query-metric", "recall@10"]
    )
    assert config.per_query_out == Path("results/pq.csv")
    assert config.per_query_metric == "recall@10"


def test_build_retrievers_builds_hybrid_of_dense_and_bm25(monkeypatch) -> None:
    # hybrid_granite_bm25 fuses the granite dense retriever and BM25 with RRF — the
    # complementarity the failure analysis found, turned into one retriever. The
    # embedding model is monkeypatched so nothing downloads.
    from src.retrieval.hybrid import HybridRetriever

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSentenceTransformer
    )
    data = BenchmarkData(
        corpus={"d1": "granite retrieval", "d2": "banana cake"},
        queries={"q1": "granite retrieval"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(retrievers=["hybrid_granite_bm25"], k_values=[1])

    retrievers = _build_retrievers(config, data)

    hybrid = retrievers["hybrid_granite_bm25"]
    assert isinstance(hybrid, HybridRetriever)
    assert len(hybrid.retrievers) == 2  # a dense retriever + BM25
    assert hybrid.retrieve("granite retrieval")[0].doc_id == "d1"


def test_parse_args_k_rrf() -> None:
    assert _parse_args([]).k_rrf == 60
    assert _parse_args(["--k-rrf", "30"]).k_rrf == 30


def test_parse_args_max_docs_and_queries() -> None:
    assert _parse_args([]).max_docs is None
    assert _parse_args([]).max_queries is None
    config = _parse_args(["--max-docs", "1000000", "--max-queries", "100"])
    assert config.max_docs == 1000000
    assert config.max_queries == 100


class FakeCrossEncoder:
    """Deterministic stand-in for sentence-transformers CrossEncoder (no download).

    Scores a (query, doc) pair by shared word count, so the relevant doc — which
    shares the query's words — reranks to the top.
    """

    def __init__(self, model_id, cache_folder=None) -> None:
        pass

    def predict(self, pairs):
        return [
            float(len(set(q.lower().split()) & set(t.lower().split())))
            for q, t in pairs
        ]


def test_build_retrievers_builds_granite_rerank_two_stage(monkeypatch) -> None:
    # granite_rerank = Granite dense first stage + Granite cross-encoder reranker,
    # as one TwoStageRetriever. Both models are monkeypatched so nothing downloads.
    from src.retrieval.reranker import TwoStageRetriever

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSentenceTransformer
    )
    monkeypatch.setattr("sentence_transformers.CrossEncoder", FakeCrossEncoder)
    data = BenchmarkData(
        corpus={"d1": "granite retrieval", "d2": "banana cake"},
        queries={"q1": "granite retrieval"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(retrievers=["granite_rerank"], k_values=[1])

    retrievers = _build_retrievers(config, data)

    rr = retrievers["granite_rerank"]
    assert isinstance(rr, TwoStageRetriever)
    assert isinstance(rr.retriever, Retriever)  # the first stage
    assert rr.retrieve("granite retrieval")[0].doc_id == "d1"


def test_parse_args_reranker_model_id() -> None:
    assert (
        _parse_args([]).reranker_model_id
        == "ibm-granite/granite-embedding-reranker-english-r2"
    )
    assert _parse_args(["--reranker-model-id", "x/y"]).reranker_model_id == "x/y"


def test_build_retrievers_threads_index_type(monkeypatch) -> None:
    # --index-type reaches the VectorIndexer: a dense retriever built with "hnsw"
    # gets an ANN (HNSW) faiss index, not the exact flat one.
    import faiss

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSentenceTransformer
    )
    data = BenchmarkData(
        corpus={"d1": "granite retrieval", "d2": "banana cake"},
        queries={"q1": "granite retrieval"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(retrievers=["granite_dense"], k_values=[1], index_type="hnsw")

    dense = _build_retrievers(config, data)["granite_dense"]

    assert isinstance(dense.index._index, faiss.IndexHNSWFlat)


def test_cache_key_appends_non_flat_index_type() -> None:
    # Flat keeps the original cache key (back-compat / existing caches); ANN index
    # types get a suffix so a flat and an HNSW index never collide in the cache.
    base = BenchmarkConfig(dataset="scifact")
    flat_key = _cache_key(base, "granite_dense")
    assert "__hnsw" not in flat_key and "__flat" not in flat_key
    assert _cache_key(replace(base, index_type="hnsw"), "granite_dense") == flat_key + "__hnsw"


def test_parse_args_index_type() -> None:
    d = _parse_args([])
    assert d.index_type == "flat" and d.ef_search == 64 and d.nprobe == 8
    config = _parse_args(["--index-type", "hnsw", "--ef-search", "128", "--nprobe", "16"])
    assert config.index_type == "hnsw"
    assert config.ef_search == 128
    assert config.nprobe == 16


def test_run_records_ms_per_query(tmp_path) -> None:
    # Every results row carries an end-to-end query-latency column (ms_per_query) —
    # the basis for the flat-vs-ANN speedup figure. It's wall-clock, so assert only
    # that it is present and non-negative, not a specific time.
    data = BenchmarkData(
        corpus={"d1": "doc one"}, queries={"q1": "qtext1"}, qrels={"q1": {"d1": 1}}
    )
    retrievers = {"r": FakeRetriever({"qtext1": [RetrievedChunk("d1", "x", 0.9)]})}
    config = BenchmarkConfig(k_values=[1], results_path=tmp_path / "out.csv")

    run(config, retrievers=retrievers, data=data)

    with open(config.results_path, newline="") as f:
        row = next(csv.DictReader(f))
    assert "ms_per_query" in row
    assert float(row["ms_per_query"]) >= 0.0


def test_parse_args_split() -> None:
    assert _parse_args([]).split == "test"
    assert _parse_args(["--split", "dev"]).split == "dev"


def test_run_passes_split_to_loader(monkeypatch, tmp_path) -> None:
    # MS MARCO has no public 'test' qrels — it uses 'dev'. run() must thread
    # config.split — and the corpus/query caps used by the scale sweep — through
    # to the loader so those datasets and sizes are reachable.
    captured = {}

    def fake_load(name, split="test", max_docs=None, max_queries=None):
        captured["name"] = name
        captured["split"] = split
        captured["max_docs"] = max_docs
        captured["max_queries"] = max_queries
        return BenchmarkData(
            corpus={"d1": "x"}, queries={"q1": "q"}, qrels={"q1": {"d1": 1}}
        )

    monkeypatch.setattr("eval.run_benchmark.load_benchmark", fake_load)
    config = BenchmarkConfig(
        dataset="msmarco",
        split="dev",
        retrievers=["bm25"],
        k_values=[1],
        results_path=tmp_path / "o.csv",
        max_docs=500,
        max_queries=100,
    )

    run(config)

    assert captured == {
        "name": "msmarco",
        "split": "dev",
        "max_docs": 500,
        "max_queries": 100,
    }


def test_build_retrievers_builds_convex_hybrid_of_dense_and_bm25(monkeypatch) -> None:
    # convex_hybrid_granite_bm25 fuses the granite dense retriever and BM25 by convex
    # combination of normalised scores (alpha = dense weight). At alpha=1 it ranks by
    # the dense arm alone. Embedding model monkeypatched so nothing downloads.
    from src.retrieval.hybrid import ConvexHybridRetriever

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSentenceTransformer
    )
    data = BenchmarkData(
        corpus={"d1": "granite retrieval", "d2": "banana cake"},
        queries={"q1": "granite retrieval"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(
        retrievers=["convex_hybrid_granite_bm25"], k_values=[1], alpha=1.0
    )

    retrievers = _build_retrievers(config, data)

    hybrid = retrievers["convex_hybrid_granite_bm25"]
    assert isinstance(hybrid, ConvexHybridRetriever)
    assert hybrid.alpha == 1.0
    assert hybrid.retrieve("granite retrieval")[0].doc_id == "d1"


def test_parse_args_alpha() -> None:
    assert _parse_args([]).alpha == 0.5
    assert _parse_args(["--alpha", "0.7"]).alpha == 0.7


class FakeSpladeEncoder:
    """Deterministic stand-in for SpladeEncoder (no model download).

    Encodes text as term counts over a tiny fixed vocabulary, so a query lands on the
    doc that shares its words — the sparse analogue of FakeSentenceTransformer.
    """

    _VOCAB = {"granite": 0, "retrieval": 1, "banana": 2, "cake": 3}
    vocab_size = 4

    def __init__(self, *args, **kwargs) -> None:
        pass

    def encode(self, texts):
        out = []
        for t in texts:
            tw: Dict[int, float] = {}
            for w in t.lower().split():
                if w in self._VOCAB:
                    tw[self._VOCAB[w]] = tw.get(self._VOCAB[w], 0.0) + 1.0
            out.append(tw)
        return out


def test_build_retrievers_builds_splade_sparse_retriever(monkeypatch) -> None:
    # splade builds a SparseRetriever (SPLADE encoder + scipy CSR index). The encoder is
    # monkeypatched so nothing downloads; the query shares words only with d1.
    from src.retrieval.sparse_retriever import SparseRetriever

    monkeypatch.setattr("eval.run_benchmark.SpladeEncoder", FakeSpladeEncoder)
    data = BenchmarkData(
        corpus={"d1": "granite retrieval", "d2": "banana cake"},
        queries={"q1": "granite retrieval"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(retrievers=["splade"], k_values=[1])

    retrievers = _build_retrievers(config, data)

    splade = retrievers["splade"]
    assert isinstance(splade, SparseRetriever)
    assert splade.retrieve("granite retrieval")[0].doc_id == "d1"


def test_build_retrievers_builds_convex_hybrid_granite_splade(monkeypatch) -> None:
    # convex_hybrid_granite_splade fuses the granite dense retriever with the SPLADE arm
    # by convex combination. Both models are monkeypatched so nothing downloads.
    from src.retrieval.hybrid import ConvexHybridRetriever

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSentenceTransformer
    )
    monkeypatch.setattr("eval.run_benchmark.SpladeEncoder", FakeSpladeEncoder)
    data = BenchmarkData(
        corpus={"d1": "granite retrieval", "d2": "banana cake"},
        queries={"q1": "granite retrieval"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(
        retrievers=["convex_hybrid_granite_splade"], k_values=[1], alpha=0.5
    )

    retrievers = _build_retrievers(config, data)

    hybrid = retrievers["convex_hybrid_granite_splade"]
    assert isinstance(hybrid, ConvexHybridRetriever)
    assert hybrid.retrieve("granite retrieval")[0].doc_id == "d1"


def test_build_retrievers_splade_uses_sparse_index_cache(monkeypatch, tmp_path) -> None:
    # With a cache dir, the splade branch encodes once and persists the CSR index
    # (keyed by dataset) so repeat runs skip re-encoding the corpus.
    monkeypatch.setattr("eval.run_benchmark.SpladeEncoder", FakeSpladeEncoder)
    data = BenchmarkData(
        corpus={"d1": "granite retrieval", "d2": "banana cake"},
        queries={"q1": "granite retrieval"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(
        retrievers=["splade"], k_values=[1], index_cache_dir=tmp_path
    )

    _build_retrievers(config, data)

    assert (tmp_path / "scifact__splade.npz").exists()


def test_build_retrievers_builds_strong_bm25() -> None:
    # strong_bm25 = the fair lexical baseline (tuned k1/b + a real analyzer). Like
    # bm25 it needs no model, so _build_retrievers builds it directly over the
    # corpus. Query terms appear only in d1 (a minority term -> positive BM25 idf).
    from src.retrieval.strong_bm25 import StrongBM25Retriever

    data = BenchmarkData(
        corpus={
            "d1": "granite dense retrieval embeddings",
            "d2": "banana cake recipe with sugar",
            "d3": "weather forecast tomorrow rain",
        },
        queries={"q1": "granite retrieval"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(retrievers=["strong_bm25"], k_values=[1])

    retrievers = _build_retrievers(config, data)

    assert isinstance(retrievers["strong_bm25"], StrongBM25Retriever)
    assert retrievers["strong_bm25"].retrieve("granite retrieval")[0].doc_id == "d1"


def test_build_retrievers_builds_convex_hybrid_granite_strong_bm25(monkeypatch) -> None:
    # convex_hybrid_granite_strong_bm25 fuses the granite dense retriever with the
    # FAIR bm25 arm (not the naive one) by convex combination — the deployment
    # hybrid that no longer rests on a strawman lexical baseline. Embedding model
    # monkeypatched so nothing downloads; >=4 docs so bm25 idf is positive.
    from src.retrieval.hybrid import ConvexHybridRetriever

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSentenceTransformer
    )
    data = BenchmarkData(
        corpus={
            "d1": "granite retrieval",
            "d2": "banana cake",
            "d3": "weather forecast",
            "d4": "quantum physics",
        },
        queries={"q1": "granite retrieval"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(
        retrievers=["convex_hybrid_granite_strong_bm25"], k_values=[1], alpha=0.5
    )

    retrievers = _build_retrievers(config, data)

    hybrid = retrievers["convex_hybrid_granite_strong_bm25"]
    assert isinstance(hybrid, ConvexHybridRetriever)
    assert hybrid.retrieve("granite retrieval")[0].doc_id == "d1"


def test_build_retrievers_hyde_reuses_injected_llm(monkeypatch) -> None:
    # hyde_granite wraps the granite dense retriever with an LLM query-transform.
    # When run_rag injects its generator LLM, the transform must REUSE that one
    # client -- not load a second model (a second load would OOM the 6GB GPU).
    from src.retrieval.query_transform import TransformingRetriever

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSentenceTransformer
    )

    class InjectedLLM:
        def generate(self, prompt):
            return "granite retrieval passage"

    injected = InjectedLLM()
    data = BenchmarkData(
        corpus={"d1": "granite retrieval", "d2": "banana cake"},
        queries={"q1": "granite"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(retrievers=["hyde_granite"], k_values=[1])

    retrievers = _build_retrievers(config, data, llm=injected)

    hyde = retrievers["hyde_granite"]
    assert isinstance(hyde, TransformingRetriever)
    assert hyde.transform.llm is injected  # the fix: reuse, not a second load
    assert hyde.retrieve("granite")[0].doc_id == "d1"


def test_build_retrievers_hyde_builds_own_llm_when_none_injected(monkeypatch) -> None:
    # Standalone nDCG eval (run_benchmark) has no generator LLM, so HyDE builds one
    # itself -- exactly once. Faked so nothing downloads.
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSentenceTransformer
    )
    built = []

    class FakeLLMClient:
        def __init__(self, *args, **kwargs):
            built.append(1)

        def generate(self, prompt):
            return "granite retrieval"

    monkeypatch.setattr("src.llm_client.LLMClient", FakeLLMClient)

    data = BenchmarkData(
        corpus={"d1": "granite retrieval", "d2": "banana cake"},
        queries={"q1": "granite"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(retrievers=["hyde_granite"], k_values=[1])

    retrievers = _build_retrievers(config, data)  # no llm injected

    assert built == [1]  # constructed exactly one client itself
    assert isinstance(retrievers["hyde_granite"].transform.llm, FakeLLMClient)


def test_build_retrievers_builds_granite_listrank_reusing_injected_llm(monkeypatch) -> None:
    # granite_listrank = granite dense first stage + an LLM listwise reranker, as one
    # TwoStageRetriever. Retests the "reranking fails" claim with a listwise reranker
    # instead of the pointwise cross-encoder. Reuses the injected LLM (no 2nd load).
    from src.retrieval.reranker import LLMListwiseReranker, TwoStageRetriever

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSentenceTransformer
    )

    class InjectedLLM:
        def generate(self, prompt):
            return "1 > 2"  # keep the first-stage order

    injected = InjectedLLM()
    data = BenchmarkData(
        corpus={"d1": "granite retrieval", "d2": "banana cake"},
        queries={"q1": "granite"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(retrievers=["granite_listrank"], k_values=[1])

    retrievers = _build_retrievers(config, data, llm=injected)

    rr = retrievers["granite_listrank"]
    assert isinstance(rr, TwoStageRetriever)
    assert isinstance(rr.reranker, LLMListwiseReranker)
    assert rr.reranker.llm is injected  # reuse, not a second load
    assert rr.retrieve("granite")[0].doc_id == "d1"


def test_build_retrievers_listrank_respects_rerank_pool(monkeypatch) -> None:
    # --rerank-pool bounds the candidate pool the (expensive) LLM reranker sees.
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSentenceTransformer
    )

    class InjectedLLM:
        def generate(self, prompt):
            return "1"

    data = BenchmarkData(
        corpus={"d1": "granite retrieval", "d2": "banana cake"},
        queries={"q1": "granite"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(
        retrievers=["granite_listrank"], k_values=[1], rerank_pool=5
    )

    rr = _build_retrievers(config, data, llm=InjectedLLM())["granite_listrank"]

    assert rr.candidates == 5


def test_parse_args_rerank_pool() -> None:
    assert _parse_args([]).rerank_pool is None
    assert _parse_args(["--rerank-pool", "20"]).rerank_pool == 20
