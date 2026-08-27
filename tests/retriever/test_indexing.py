import json
from hashlib import sha256
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


# Golden constants over `corpus()`, recorded 2026-08-09. Two of them, because a reshape of the
# signature payload breaks two different things and only one is visible from inside the process.
GOLDEN_INDEX_SIGNATURE = "7868c59e2b3cc305a8b79be655abb0c30c2b57d7718ff4f019321dbd96304fb9"
GOLDEN_MANIFEST_SHA256 = "51204c1e9acafb18433fd1d72bf72b788682c2f2f8084cf222f76d467d2877ea"


def test_index_identity_is_pinned_so_a_reshape_cannot_pass_silently(tmp_path: Path) -> None:
    """A payload reshape must fail here, in seconds, not on the cluster hours later.

    `d5f7908` moved BM25's `k1`/`b` from flat `IndexManifest` fields into a nested
    `parameters` dict. Same corpus, same retriever, same k1/b -- but `_index_signature`
    hashes that payload, so the identity changed and `index_manifest.json` changed
    byte-for-byte. `implementation_version` stayed `bm25-v1`, so nothing announced it.

    Every run directory prepared before that commit became permanently unloadable, because
    `ArtifactStore` records the sha256 of the manifest FILE in `run_manifest.json.metadata.json`
    and that file had been deleted and rebuilt. The failure does not surface at the commit; it
    surfaces as an opaque `upstream artifact hashes mismatch` on a compute node -- job 18235971
    reached it after burning 5h19m on the arm that ran first.

    Hence two constants. `index_signature` is the identity two runs are compared on;
    `GOLDEN_MANIFEST_SHA256` is the exact quantity `_validate_stored_manifest` compares, and it
    moves under changes the signature alone would not catch (key order, separators, the trailing
    newline `write_index` appends).

    WHEN THIS GOES RED the change is not necessarily wrong -- but it is no longer invisible.
    Either revert it, or bump `implementation_version` in the same commit and re-record both
    constants below. Note what the bump costs: every index manifest already on disk says
    `bm25-v1`, so bumping makes `load_index` reject all of them. That belongs at a deliberate
    re-index boundary, not inside a refactor.

    The first thing this test caught was not a reshape. `GOLDEN_MANIFEST_SHA256` was first
    recorded on Windows, where `write_text` had been translating the trailing "\\n" to "\\r\\n" --
    so the persisted bytes, and therefore `upstream_artifact_hashes`, depended on which machine
    prepared the index. Green locally, red in CI, and on a mixed-platform team it would have
    surfaced as yet another uninterpretable hash mismatch. `write_index` now writes bytes; the
    no-CR assertion below is what keeps that fixed rather than merely fixed once.
    """

    snapshot = corpus()
    build_bm25_index(tmp_path, snapshot)
    raw = (tmp_path / "index_manifest.json").read_bytes()

    assert b"\r" not in raw, "persisted artifacts must not carry platform line endings"
    assert read_index_manifest(tmp_path).index_signature == GOLDEN_INDEX_SIGNATURE
    assert sha256(raw).hexdigest() == GOLDEN_MANIFEST_SHA256


def test_pinned_identity_actually_discriminates(tmp_path: Path) -> None:
    """The pin is worthless if it holds under a real parameter change; prove it does not."""

    snapshot = corpus()
    build_bm25_index(tmp_path, snapshot, k1=1.2, b=0.75)
    raw = (tmp_path / "index_manifest.json").read_bytes()

    assert read_index_manifest(tmp_path).index_signature != GOLDEN_INDEX_SIGNATURE
    assert sha256(raw).hexdigest() != GOLDEN_MANIFEST_SHA256
