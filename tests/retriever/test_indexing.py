import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evidence_rag.contracts.models import Document, Query
from evidence_rag.infrastructure.corpus import CorpusBuilder, CorpusSnapshot, WordChunker
from evidence_rag.retriever.bm25 import BM25Retriever
from evidence_rag.retriever.indexing import (
    IndexManifest,
    load_index,
    read_index_manifest,
    write_index,
)

BM25_PARAMS = {"k1": 1.5, "b": 0.75}


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
        documents,
        dataset_signature="dataset-signature",
    )


def build_bm25_index(directory: Path, snapshot: CorpusSnapshot, **params: float) -> None:
    write_index(
        snapshot,
        directory,
        implementation="bm25",
        implementation_version="bm25-v1",
        parameters=params or BM25_PARAMS,
    )


def load_bm25(directory: Path, snapshot: CorpusSnapshot, **overrides: object) -> CorpusSnapshot:
    kwargs = {
        "expected_corpus_signature": snapshot.manifest.corpus_signature,
        "expected_implementation": "bm25",
        "expected_implementation_version": "bm25-v1",
        "expected_parameters": BM25_PARAMS,
    }
    kwargs.update(overrides)
    return load_index(directory, **kwargs)  # type: ignore[arg-type]


def replace_json_field(path: Path, field: str, value: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")


def remove_json_field(path: Path, field: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload[field]
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_write_persists_v1_manifest_and_corpus_snapshot(tmp_path: Path) -> None:
    snapshot = corpus()

    build_bm25_index(tmp_path, snapshot)

    manifest_path = tmp_path / "index_manifest.json"
    snapshot_path = tmp_path / "corpus_snapshot.json"
    manifest = IndexManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    assert manifest.schema_version == "1.0"
    assert manifest.implementation == "bm25"
    assert manifest.implementation_version == "bm25-v1"
    assert manifest.corpus_signature == snapshot.manifest.corpus_signature
    assert manifest.parameters == BM25_PARAMS
    assert manifest.snapshot_filename == "corpus_snapshot.json"
    assert len(manifest.index_signature) == 64
    assert json.loads(snapshot_path.read_text(encoding="utf-8")) == snapshot.model_dump(mode="json")
    assert manifest_path.read_text(encoding="utf-8").endswith("\n")
    assert snapshot_path.read_text(encoding="utf-8").endswith("\n")


def test_load_returns_snapshot_that_rebuilds_identical_rankings(tmp_path: Path) -> None:
    snapshot = corpus()
    build_bm25_index(tmp_path, snapshot, k1=1.2, b=0.6)

    loaded = load_index(
        tmp_path,
        expected_corpus_signature=snapshot.manifest.corpus_signature,
        expected_implementation="bm25",
        expected_implementation_version="bm25-v1",
        expected_parameters={"k1": 1.2, "b": 0.6},
    )

    query = Query(query_id="q-1", text="revenue policy")
    from_snapshot = BM25Retriever.from_corpus(loaded, k1=1.2, b=0.6)
    from_original = BM25Retriever.from_corpus(snapshot, k1=1.2, b=0.6)
    assert from_snapshot.retrieve(query, top_k=10) == from_original.retrieve(query, top_k=10)


def test_load_rejects_expected_corpus_signature_mismatch(tmp_path: Path) -> None:
    snapshot = corpus()
    build_bm25_index(tmp_path, snapshot)

    with pytest.raises(ValueError, match=r"corpus signature mismatch.*index_manifest\.json"):
        load_bm25(tmp_path, snapshot, expected_corpus_signature="different-corpus")


def test_load_rejects_implementation_mismatch(tmp_path: Path) -> None:
    snapshot = corpus()
    build_bm25_index(tmp_path, snapshot)

    with pytest.raises(ValueError, match=r"implementation mismatch.*index_manifest\.json"):
        load_bm25(tmp_path, snapshot, expected_implementation="granite-dense")


def test_load_rejects_parameter_mismatch(tmp_path: Path) -> None:
    snapshot = corpus()
    build_bm25_index(tmp_path, snapshot)

    with pytest.raises(ValueError, match=r"parameters mismatch.*index_manifest\.json"):
        load_bm25(tmp_path, snapshot, expected_parameters={"k1": 1.6, "b": 0.75})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", "2.0"),
        ("implementation", "faiss"),
        ("implementation_version", "bm25-v2"),
    ),
)
def test_load_rejects_manifest_identity_tampering(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    snapshot = corpus()
    build_bm25_index(tmp_path, snapshot)
    manifest_path = tmp_path / "index_manifest.json"
    replace_json_field(manifest_path, field, value)

    with pytest.raises(ValueError) as raised:
        load_bm25(tmp_path, snapshot)

    assert field.split("_")[0] in str(raised.value)
    assert str(manifest_path) in str(raised.value)


@pytest.mark.parametrize(
    "field",
    ("schema_version", "implementation", "implementation_version", "snapshot_filename"),
)
def test_load_rejects_missing_manifest_identity_field(tmp_path: Path, field: str) -> None:
    snapshot = corpus()
    build_bm25_index(tmp_path, snapshot)
    manifest_path = tmp_path / "index_manifest.json"
    remove_json_field(manifest_path, field)

    with pytest.raises(ValueError) as raised:
        load_bm25(tmp_path, snapshot)

    assert field in str(raised.value)
    assert str(manifest_path) in str(raised.value)


def test_load_rejects_snapshot_corpus_signature_mismatch(tmp_path: Path) -> None:
    snapshot = corpus()
    build_bm25_index(tmp_path, snapshot)
    snapshot_path = tmp_path / "corpus_snapshot.json"
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    payload["manifest"]["corpus_signature"] = "tampered-corpus"
    snapshot_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"corpus signature mismatch.*corpus_snapshot\.json"):
        load_bm25(tmp_path, snapshot)


@pytest.mark.parametrize("tamper_target", ("manifest", "snapshot"))
def test_load_rejects_index_signature_tampering(tmp_path: Path, tamper_target: str) -> None:
    snapshot = corpus()
    build_bm25_index(tmp_path, snapshot)
    if tamper_target == "manifest":
        replace_json_field(tmp_path / "index_manifest.json", "index_signature", "0" * 64)
    else:
        snapshot_path = tmp_path / "corpus_snapshot.json"
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        payload["chunks"][0]["text"] = "tampered chunk text"
        snapshot_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"index signature mismatch.*index_manifest\.json"):
        load_bm25(tmp_path, snapshot)


def test_same_corpus_and_config_produce_deterministic_index_signature(tmp_path: Path) -> None:
    snapshot = corpus()
    build_bm25_index(tmp_path / "first", snapshot)
    build_bm25_index(tmp_path / "second", snapshot)

    first = read_index_manifest(tmp_path / "first")
    second = read_index_manifest(tmp_path / "second")
    assert first.index_signature == second.index_signature


def test_distinct_implementations_produce_distinct_signatures(tmp_path: Path) -> None:
    snapshot = corpus()
    write_index(
        snapshot,
        tmp_path / "bm25",
        implementation="bm25",
        implementation_version="bm25-v1",
        parameters={"k1": 0.9, "b": 0.4},
    )
    write_index(
        snapshot,
        tmp_path / "strong",
        implementation="strong-bm25",
        implementation_version="strong-bm25-v1",
        parameters={"k1": 0.9, "b": 0.4},
    )
    bm25 = read_index_manifest(tmp_path / "bm25")
    strong = read_index_manifest(tmp_path / "strong")
    assert bm25.index_signature != strong.index_signature


def test_index_manifest_is_strict_frozen_and_forbids_extra(tmp_path: Path) -> None:
    snapshot = corpus()
    build_bm25_index(tmp_path, snapshot)
    payload = json.loads((tmp_path / "index_manifest.json").read_text(encoding="utf-8"))
    manifest = IndexManifest.model_validate(payload)

    with pytest.raises(ValidationError):
        IndexManifest.model_validate({**payload, "unexpected": True})
    with pytest.raises(ValidationError):
        manifest.implementation = "granite-dense"


@pytest.mark.parametrize("filename", ("index_manifest.json", "corpus_snapshot.json"))
@pytest.mark.parametrize("failure", ("missing", "corrupt"))
def test_load_reports_missing_or_corrupt_file_path(
    tmp_path: Path,
    filename: str,
    failure: str,
) -> None:
    snapshot = corpus()
    build_bm25_index(tmp_path, snapshot)
    path = tmp_path / filename
    if failure == "missing":
        path.unlink()
    else:
        path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError) as raised:
        load_bm25(tmp_path, snapshot)

    assert str(path) in str(raised.value)
