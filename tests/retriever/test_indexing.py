import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from evidence_rag.contracts.models import Document, Query
from evidence_rag.infrastructure.corpus import CorpusBuilder, CorpusSnapshot, WordChunker
from evidence_rag.retriever.indexing import BM25IndexPlugin, IndexManifest


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


def replace_json_field(path: Path, field: str, value: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")


def remove_json_field(path: Path, field: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload[field]
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_persists_v1_manifest_and_corpus_snapshot(tmp_path: Path) -> None:
    snapshot = corpus()

    BM25IndexPlugin().build(snapshot, tmp_path, k1=1.5, b=0.75)

    manifest_path = tmp_path / "index_manifest.json"
    snapshot_path = tmp_path / "corpus_snapshot.json"
    manifest = IndexManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    assert manifest.schema_version == "1.0"
    assert manifest.implementation == "bm25"
    assert manifest.implementation_version == "bm25-v1"
    assert manifest.corpus_signature == snapshot.manifest.corpus_signature
    assert manifest.k1 == 1.5
    assert manifest.b == 0.75
    assert manifest.snapshot_filename == "corpus_snapshot.json"
    assert len(manifest.index_signature) == 64
    assert json.loads(snapshot_path.read_text(encoding="utf-8")) == snapshot.model_dump(mode="json")
    assert manifest_path.read_text(encoding="utf-8").endswith("\n")
    assert snapshot_path.read_text(encoding="utf-8").endswith("\n")


def test_load_reconstructs_identical_bm25_rankings(tmp_path: Path) -> None:
    snapshot = corpus()
    plugin = BM25IndexPlugin()
    built = plugin.build(snapshot, tmp_path, k1=1.2, b=0.6)

    loaded = plugin.load(
        tmp_path,
        expected_corpus_signature=snapshot.manifest.corpus_signature,
        expected_k1=1.2,
        expected_b=0.6,
    )

    query = Query(query_id="q-1", text="revenue policy")
    assert loaded.retrieve(query, top_k=10) == built.retrieve(query, top_k=10)


def test_load_rejects_expected_corpus_signature_mismatch(tmp_path: Path) -> None:
    snapshot = corpus()
    plugin = BM25IndexPlugin()
    plugin.build(snapshot, tmp_path, k1=1.5, b=0.75)

    try:
        plugin.load(
            tmp_path,
            expected_corpus_signature="different-corpus",
            expected_k1=1.5,
            expected_b=0.75,
        )
    except ValueError as error:
        assert "corpus signature mismatch" in str(error)
        assert str(tmp_path / "index_manifest.json") in str(error)
    else:
        raise AssertionError("mismatched corpus signature was accepted")


@pytest.mark.parametrize(
    ("expected_k1", "expected_b", "parameter"),
    ((1.6, 0.75, "k1"), (1.5, 0.8, "b")),
)
def test_load_rejects_bm25_parameter_mismatch(
    tmp_path: Path,
    expected_k1: float,
    expected_b: float,
    parameter: str,
) -> None:
    snapshot = corpus()
    plugin = BM25IndexPlugin()
    plugin.build(snapshot, tmp_path, k1=1.5, b=0.75)

    with pytest.raises(ValueError, match=rf"{parameter} mismatch.*index_manifest\.json"):
        plugin.load(
            tmp_path,
            expected_corpus_signature=snapshot.manifest.corpus_signature,
            expected_k1=expected_k1,
            expected_b=expected_b,
        )


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
    plugin = BM25IndexPlugin()
    plugin.build(snapshot, tmp_path, k1=1.5, b=0.75)
    manifest_path = tmp_path / "index_manifest.json"
    replace_json_field(manifest_path, field, value)

    with pytest.raises(ValueError) as raised:
        plugin.load(
            tmp_path,
            expected_corpus_signature=snapshot.manifest.corpus_signature,
            expected_k1=1.5,
            expected_b=0.75,
        )

    assert field in str(raised.value)
    assert str(manifest_path) in str(raised.value)


@pytest.mark.parametrize(
    "field",
    (
        "schema_version",
        "implementation",
        "implementation_version",
        "snapshot_filename",
    ),
)
def test_load_rejects_missing_manifest_identity_field(
    tmp_path: Path,
    field: str,
) -> None:
    snapshot = corpus()
    plugin = BM25IndexPlugin()
    plugin.build(snapshot, tmp_path, k1=1.5, b=0.75)
    manifest_path = tmp_path / "index_manifest.json"
    remove_json_field(manifest_path, field)

    with pytest.raises(ValueError) as raised:
        plugin.load(
            tmp_path,
            expected_corpus_signature=snapshot.manifest.corpus_signature,
            expected_k1=1.5,
            expected_b=0.75,
        )

    assert field in str(raised.value)
    assert str(manifest_path) in str(raised.value)


def test_load_rejects_snapshot_corpus_signature_mismatch(tmp_path: Path) -> None:
    snapshot = corpus()
    plugin = BM25IndexPlugin()
    plugin.build(snapshot, tmp_path, k1=1.5, b=0.75)
    snapshot_path = tmp_path / "corpus_snapshot.json"
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    payload["manifest"]["corpus_signature"] = "tampered-corpus"
    snapshot_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"corpus signature mismatch.*corpus_snapshot\.json"):
        plugin.load(
            tmp_path,
            expected_corpus_signature=snapshot.manifest.corpus_signature,
            expected_k1=1.5,
            expected_b=0.75,
        )


@pytest.mark.parametrize("tamper_target", ("manifest", "snapshot"))
def test_load_rejects_index_signature_tampering(
    tmp_path: Path,
    tamper_target: str,
) -> None:
    snapshot = corpus()
    plugin = BM25IndexPlugin()
    plugin.build(snapshot, tmp_path, k1=1.5, b=0.75)
    if tamper_target == "manifest":
        replace_json_field(tmp_path / "index_manifest.json", "index_signature", "0" * 64)
    else:
        snapshot_path = tmp_path / "corpus_snapshot.json"
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        payload["chunks"][0]["text"] = "tampered chunk text"
        snapshot_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"index signature mismatch.*index_manifest\.json"):
        plugin.load(
            tmp_path,
            expected_corpus_signature=snapshot.manifest.corpus_signature,
            expected_k1=1.5,
            expected_b=0.75,
        )


def test_same_corpus_and_config_produce_deterministic_index_signature(tmp_path: Path) -> None:
    snapshot = corpus()
    plugin = BM25IndexPlugin()
    plugin.build(snapshot, tmp_path / "first", k1=1.5, b=0.75)
    plugin.build(snapshot, tmp_path / "second", k1=1.5, b=0.75)

    first = IndexManifest.model_validate_json(
        (tmp_path / "first/index_manifest.json").read_text(encoding="utf-8")
    )
    second = IndexManifest.model_validate_json(
        (tmp_path / "second/index_manifest.json").read_text(encoding="utf-8")
    )
    assert first.index_signature == second.index_signature


def test_index_manifest_is_strict_frozen_extra_forbid_and_finite(tmp_path: Path) -> None:
    snapshot = corpus()
    BM25IndexPlugin().build(snapshot, tmp_path, k1=1.5, b=0.75)
    payload = json.loads((tmp_path / "index_manifest.json").read_text(encoding="utf-8"))
    manifest = IndexManifest.model_validate(payload)

    with pytest.raises(ValidationError):
        IndexManifest.model_validate({**payload, "k1": "1.5"})
    with pytest.raises(ValidationError):
        IndexManifest.model_validate({**payload, "unexpected": True})
    with pytest.raises(ValidationError):
        IndexManifest.model_validate({**payload, "b": float("inf")})
    with pytest.raises(ValidationError):
        manifest.b = 0.5


@pytest.mark.parametrize(
    ("k1", "b", "parameter"),
    (
        (-1.0, 0.75, "k1"),
        (math.inf, 0.75, "k1"),
        (1.5, -0.1, "b"),
        (1.5, 1.1, "b"),
        (1.5, math.nan, "b"),
    ),
)
@pytest.mark.parametrize("operation", ("build", "load"))
def test_index_plugin_validates_bm25_parameters_everywhere(
    tmp_path: Path,
    k1: float,
    b: float,
    parameter: str,
    operation: str,
) -> None:
    snapshot = corpus()
    plugin = BM25IndexPlugin()
    if operation == "load":
        plugin.build(snapshot, tmp_path, k1=1.5, b=0.75)

    with pytest.raises(ValueError, match=parameter):
        if operation == "build":
            plugin.build(snapshot, tmp_path, k1=k1, b=b)
        else:
            plugin.load(
                tmp_path,
                expected_corpus_signature=snapshot.manifest.corpus_signature,
                expected_k1=k1,
                expected_b=b,
            )


@pytest.mark.parametrize("filename", ("index_manifest.json", "corpus_snapshot.json"))
@pytest.mark.parametrize("failure", ("missing", "corrupt"))
def test_load_reports_missing_or_corrupt_file_path(
    tmp_path: Path,
    filename: str,
    failure: str,
) -> None:
    snapshot = corpus()
    plugin = BM25IndexPlugin()
    plugin.build(snapshot, tmp_path, k1=1.5, b=0.75)
    path = tmp_path / filename
    if failure == "missing":
        path.unlink()
    else:
        path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError) as raised:
        plugin.load(
            tmp_path,
            expected_corpus_signature=snapshot.manifest.corpus_signature,
            expected_k1=1.5,
            expected_b=0.75,
        )

    assert str(path) in str(raised.value)
