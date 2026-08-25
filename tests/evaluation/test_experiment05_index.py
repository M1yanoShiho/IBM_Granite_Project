from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from evidence_rag.evaluation.experiment05_index import (
    DenseIndexParameters,
    build_dense_index,
    corpus_rows,
    freeze_bm25_manifest,
    iter_dpr_passages,
    iter_kilt_passages,
    normalize_corpus,
    pyserini_index_command,
    training_ordinals,
)


def test_kilt_passages_preserve_paragraph_spans_and_split_long_units(tmp_path: Path) -> None:
    source = tmp_path / "kilt.jsonl"
    page = {
        "wikipedia_id": "42",
        "wikipedia_title": "A Title",
        "text": [
            "A Title",
            "one two three",
            "four five",
            "six seven eight nine ten eleven",
        ],
    }
    source.write_text(json.dumps(page) + "\n", encoding="utf-8")

    rows = list(iter_kilt_passages(source, max_words=5))

    assert rows == [
        {
            "id": "kilt-000000000",
            "contents": "A Title\none two three four five",
            "title": "A Title",
            "source_kind": "kilt-paragraph",
            "source_id": "42",
            "start_unit": 1,
            "end_unit": 2,
        },
        {
            "id": "kilt-000000001",
            "contents": "A Title\nsix seven eight nine ten",
            "title": "A Title",
            "source_kind": "kilt-paragraph",
            "source_id": "42",
            "start_unit": 3,
            "end_unit": 3,
        },
        {
            "id": "kilt-000000002",
            "contents": "A Title\neleven",
            "title": "A Title",
            "source_kind": "kilt-paragraph",
            "source_id": "42",
            "start_unit": 3,
            "end_unit": 3,
        },
    ]


def test_dpr_passages_preserve_official_ids_titles_and_text(tmp_path: Path) -> None:
    source = tmp_path / "psgs.tsv.gz"
    with gzip.open(source, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("id", "text", "title"))
        writer.writerow(("7", "A passage.", "Page"))

    assert list(iter_dpr_passages(source)) == [
        {
            "id": "dpr-000000000",
            "contents": "Page\nA passage.",
            "title": "Page",
            "source_kind": "dpr-passage",
            "source_id": "7",
            "start_unit": None,
            "end_unit": None,
        }
    ]


def test_normalization_is_sharded_deterministic_and_hash_audited(tmp_path: Path) -> None:
    source = tmp_path / "kilt.jsonl"
    source.write_text(
        "".join(
            json.dumps(
                {
                    "wikipedia_id": str(index),
                    "wikipedia_title": f"Title {index}",
                    "text": [f"Title {index}", "one two three"],
                }
            )
            + "\n"
            for index in range(3)
        ),
        encoding="utf-8",
    )

    first = normalize_corpus(
        dataset="kilt-wikipedia-20190801",
        source_kind="kilt",
        source_path=source,
        output_dir=tmp_path / "first",
        max_words=100,
        shard_size=2,
    )
    second = normalize_corpus(
        dataset="kilt-wikipedia-20190801",
        source_kind="kilt",
        source_path=source,
        output_dir=tmp_path / "second",
        max_words=100,
        shard_size=2,
    )

    assert first["schema_version"] == "experiment05.corpus_manifest.v1"
    assert first["passage_count"] == 3
    assert first["shard_count"] == 2
    assert [row["sha256"] for row in first["shards"]] == [
        row["sha256"] for row in second["shards"]
    ]
    assert first["ordered_passage_ids_sha256"] == second["ordered_passage_ids_sha256"]
    first_row = json.loads((tmp_path / "first/shards/part-00000.jsonl").read_text().splitlines()[0])
    assert first_row["id"] == "kilt-000000000"
    assert len(first_row["contents_sha256"]) == 64


def test_corpus_reader_rejects_tampered_shards(tmp_path: Path) -> None:
    source = tmp_path / "kilt.jsonl"
    source.write_text(
        json.dumps(
            {"wikipedia_id": "1", "wikipedia_title": "Title", "text": ["Title", "body"]}
        )
        + "\n",
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus"
    normalize_corpus(
        dataset="kilt-wikipedia-20190801",
        source_kind="kilt",
        source_path=source,
        output_dir=corpus,
    )
    with (corpus / "shards/part-00000.jsonl").open("ab") as handle:
        handle.write(b" ")

    with pytest.raises(ValueError, match="shard hash differs"):
        list(corpus_rows(corpus))


def test_training_ordinals_are_deterministic_unique_and_cover_the_full_range() -> None:
    first = training_ordinals(100, 12, seed=13)
    second = training_ordinals(100, 12, seed=13)

    assert first == second
    assert len(first) == len(set(first)) == 12
    assert min(first) >= 0
    assert max(first) < 100
    assert training_ordinals(5, 20, seed=13) == (0, 1, 2, 3, 4)


class _FakeEncoder:
    def encode(self, texts, **kwargs):  # noqa: ANN001, ANN201
        return np.asarray(
            [[float(len(text)), 1.0, float(index), 2.0] for index, text in enumerate(texts)],
            dtype="float32",
        )


class _FakeIndex:
    def __init__(self) -> None:
        self.is_trained = False
        self.ntotal = 0
        self.cp = type("CP", (), {"seed": None, "niter": None})()
        self.pq = type("PQ", (), {"cp": type("CP", (), {"seed": None, "niter": None})()})()

    def train(self, vectors) -> None:  # noqa: ANN001
        assert vectors.dtype == np.float32
        self.is_trained = True

    def add(self, vectors) -> None:  # noqa: ANN001
        assert self.is_trained
        self.ntotal += len(vectors)


class _FakeFaiss:
    METRIC_INNER_PRODUCT = 0

    @staticmethod
    def IndexFlatIP(dimension: int) -> object:  # noqa: N802
        return {"dimension": dimension}

    @staticmethod
    def IndexIVFPQ(*args) -> _FakeIndex:  # noqa: ANN002, N802
        return _FakeIndex()

    @staticmethod
    def write_index(index: _FakeIndex, path: str) -> None:
        Path(path).write_text(json.dumps({"ntotal": index.ntotal}), encoding="utf-8")


def test_dense_builder_trains_adds_every_passage_and_freezes_manifest(tmp_path: Path) -> None:
    source = tmp_path / "kilt.jsonl"
    source.write_text(
        "".join(
            json.dumps(
                {
                    "wikipedia_id": str(index),
                    "wikipedia_title": f"Title {index}",
                    "text": [f"Title {index}", "one two three"],
                }
            )
            + "\n"
            for index in range(6)
        ),
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus"
    normalize_corpus(
        dataset="kilt-wikipedia-20190801",
        source_kind="kilt",
        source_path=source,
        output_dir=corpus,
    )
    params = DenseIndexParameters(
        dimension=4,
        nlist=2,
        m=2,
        nbits=8,
        nprobe=1,
        train_size=4,
        batch_size=2,
        checkpoint_interval=3,
        seed=13,
    )

    manifest = build_dense_index(
        corpus_dir=corpus,
        index_dir=tmp_path / "dense",
        model_snapshot=tmp_path / "model",
        parameters=params,
        encoder=_FakeEncoder(),
        faiss_module=_FakeFaiss(),
    )

    assert manifest["status"] == "FROZEN_FULL_CORPUS_INDEX"
    assert manifest["passage_count"] == 6
    assert manifest["faiss_ntotal"] == 6
    assert manifest["parameters"]["metric"] == "inner_product_on_l2_normalized_vectors"
    assert len(manifest["index_sha256"]) == 64


def test_pyserini_command_indexes_raw_json_for_runtime_provenance(tmp_path: Path) -> None:
    command = pyserini_index_command(
        corpus_dir=tmp_path / "corpus", index_dir=tmp_path / "bm25", threads=8
    )

    assert command[:3] == [sys.executable, "-m", "pyserini.index.lucene"]
    assert "--storeRaw" in command
    assert command[command.index("--threads") + 1] == "8"


def test_bm25_manifest_binds_every_index_file_and_full_corpus(tmp_path: Path) -> None:
    source = tmp_path / "kilt.jsonl"
    source.write_text(
        json.dumps(
            {"wikipedia_id": "1", "wikipedia_title": "Title", "text": ["Title", "body"]}
        )
        + "\n",
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus"
    normalize_corpus(
        dataset="kilt-wikipedia-20190801",
        source_kind="kilt",
        source_path=source,
        output_dir=corpus,
    )
    index = tmp_path / "bm25"
    index.mkdir()
    (index / "segments_1").write_bytes(b"lucene")
    (index / "_0.si").write_bytes(b"segment")

    manifest = freeze_bm25_manifest(
        corpus_dir=corpus,
        index_dir=index,
        document_count=1,
        pyserini_version="1.2.0",
        k1=0.9,
        b=0.4,
    )

    assert manifest["status"] == "FROZEN_FULL_CORPUS_INDEX"
    assert manifest["document_count"] == manifest["passage_count"] == 1
    assert manifest["search_parameters"] == {"b": 0.4, "k1": 0.9}
    assert len(manifest["index_tree_sha256"]) == 64
    assert [row["file"] for row in manifest["index_files"]] == ["_0.si", "segments_1"]
