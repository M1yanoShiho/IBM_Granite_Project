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
from typing import Dict, List

import pytest

from src.retrieval.base import RetrievedChunk, Retriever
from src.ingestion.chunker import Chunk
from eval.benchmarks.loader import BenchmarkData
from eval.run_benchmark import (
    BenchmarkConfig,
    _build_retrievers,
    _cache_key,
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
    assert [r.doc_id for r in results] == ["d1"]


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
    assert d.embedding_model_id is None
    assert d.append is False

    config = _parse_args(
        [
            "--chunk-size", "256",
            "--overlap", "0",
            "--pooling", "mean",
            "--embedding-model-id", "ibm-granite/x",
            "--append",
        ]
    )
    assert config.chunk_size == 256
    assert config.chunk_overlap == 0
    assert config.pooling == "mean"
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

    assert captured["cache_path"] == tmp_path / "scifact__st_dense__cs512_ov50__default"


def test_parse_args_cache_dir() -> None:
    assert _parse_args([]).index_cache_dir is None
    assert _parse_args(["--cache-dir", "tmp/idx"]).index_cache_dir == Path("tmp/idx")


# --- chunk / pool ablation tests ---


def test_build_run_mean_pools_chunks_of_same_doc() -> None:
    retriever = FakeRetriever(
        {
            "q": [
                RetrievedChunk(doc_id="d1", text="chunk a", score=0.4),
                RetrievedChunk(doc_id="d1", text="chunk b", score=0.8),
                RetrievedChunk(doc_id="d2", text="other", score=0.5),
            ]
        }
    )
    result = build_run(retriever, {"q1": "q"}, pool_strategy="mean")
    assert result == {"q1": {"d1": pytest.approx(0.6), "d2": 0.5}}


def test_build_run_unknown_pool_strategy_raises() -> None:
    retriever = FakeRetriever({"q": []})
    with pytest.raises(ValueError, match="Unknown pool_strategy"):
        build_run(retriever, {"q1": "q"}, pool_strategy="sum")


def test_parse_args_chunk_and_pool_defaults() -> None:
    config = _parse_args([])
    assert config.chunk_size == 512
    assert config.chunk_overlap == 50
    assert config.pool_strategy == "max"


def test_parse_args_chunk_and_pool_overrides() -> None:
    config = _parse_args(["--chunk-size", "128", "--chunk-overlap", "0", "--pool", "mean"])
    assert config.chunk_size == 128
    assert config.chunk_overlap == 0
    assert config.pool_strategy == "mean"


def test_build_retrievers_passes_chunk_params_to_chunker(monkeypatch) -> None:
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeSentenceTransformer)
    captured_calls: list[dict] = []

    original_chunk_document = __import__(
        "src.ingestion.chunker", fromlist=["chunk_document"]
    ).chunk_document

    def recording_chunk_document(doc_id, text, chunk_size=512, chunk_overlap=50):
        captured_calls.append({"chunk_size": chunk_size, "chunk_overlap": chunk_overlap})
        return original_chunk_document(doc_id, text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    monkeypatch.setattr("eval.run_benchmark.chunk_document", recording_chunk_document)

    data = BenchmarkData(
        corpus={"d1": "word " * 300},
        queries={"q1": "word"},
        qrels={"q1": {"d1": 1}},
    )
    config = BenchmarkConfig(retrievers=["st_dense"], k_values=[1], chunk_size=128, chunk_overlap=0)
    _build_retrievers(config, data)

    assert all(c["chunk_size"] == 128 and c["chunk_overlap"] == 0 for c in captured_calls)


def test_evaluate_one_with_mean_pool() -> None:
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
    metrics = evaluate_one(retriever, data, k_values=[1], pool_strategy="mean")
    assert metrics["mrr"] == pytest.approx(1.0)
