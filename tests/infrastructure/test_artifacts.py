import json
import shutil
from hashlib import sha256
from pathlib import Path

import pytest

from evidence_rag.contracts.models import CandidateSet, Query
from evidence_rag.infrastructure.artifacts import (
    ArtifactMetadata,
    ArtifactStore,
    RunManifest,
    run_manifest_signature,
)
from evidence_rag.infrastructure.config import ModuleConfig


def manifest() -> RunManifest:
    module = ModuleConfig(name="bm25", parameters={"k1": 1.5})
    return RunManifest(
        dataset_id="annual-reports",
        dataset_version="2026-01",
        split="test",
        dataset_signature="dataset-signature",
        corpus_signature="corpus-signature",
        chunker_name="WordChunker",
        chunker_version="word-v1",
        chunk_size=120,
        overlap=20,
        index_implementation="bm25",
        index_implementation_version="bm25-v1",
        index_signature="1" * 64,
        retriever=module,
        selector=ModuleConfig(name="top-k"),
        generator=ModuleConfig(name="extractive"),
        top_k=3,
        max_selected=2,
        seed=7,
        git_commit="abc123",
        git_dirty=False,
        source_tree_signature="2" * 64,
    )


def candidate_set() -> CandidateSet:
    return CandidateSet(query_id="q-1", candidates=())


def metadata_path(root: Path, filename: str) -> Path:
    return root / f"{filename}.metadata.json"


def refresh_content_hash(root: Path, filename: str) -> None:
    sidecar_path = metadata_path(root, filename)
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["content_sha256"] = sha256((root / filename).read_bytes()).hexdigest()
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")


def test_manifest_round_trip_and_compatibility_validation(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run")
    expected = manifest()

    store.write_manifest(expected)

    assert (
        store.read_manifest(
            expected_dataset_signature="dataset-signature",
            expected_corpus_signature="corpus-signature",
        )
        == expected
    )

    with pytest.raises(ValueError, match="dataset signature mismatch"):
        store.read_manifest(expected_dataset_signature="other-dataset")
    with pytest.raises(ValueError, match="corpus signature mismatch"):
        store.read_manifest(expected_corpus_signature="other-corpus")


def test_artifact_sidecar_binds_canonical_content_and_run_provenance(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "run")
    expected_manifest = manifest()
    store.write_manifest(expected_manifest)

    store.write_jsonl(
        "candidates.jsonl",
        (candidate_set(),),
        producer="ExperimentWorkflow",
        stage="retriever",
        upstream_artifact_hashes={"queries.jsonl": "2" * 64},
    )

    artifact_path = store.root / "candidates.jsonl"
    metadata = ArtifactMetadata.model_validate_json(
        metadata_path(store.root, "candidates.jsonl").read_text(encoding="utf-8")
    )
    assert metadata.artifact_filename == "candidates.jsonl"
    assert metadata.content_sha256 == sha256(artifact_path.read_bytes()).hexdigest()
    assert metadata.dataset_signature == expected_manifest.dataset_signature
    assert metadata.corpus_signature == expected_manifest.corpus_signature
    assert metadata.run_manifest_signature == run_manifest_signature(expected_manifest)
    assert metadata.producer == "ExperimentWorkflow"
    assert metadata.stage == "retriever"
    assert metadata.upstream_artifact_hashes == {"queries.jsonl": "2" * 64}


def test_read_rejects_tampered_content_before_parsing(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run")
    store.write_manifest(manifest())
    store.write_jsonl("candidates.jsonl", (candidate_set(),))
    (store.root / "candidates.jsonl").write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"content SHA-256 mismatch.*candidates\.jsonl"):
        store.read_jsonl(
            "candidates.jsonl",
            CandidateSet,
            expected_dataset_signature="dataset-signature",
            expected_corpus_signature="corpus-signature",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("content_sha256", "0" * 64, "content SHA-256 mismatch"),
        ("run_manifest_signature", "0" * 64, "run manifest signature mismatch"),
        ("artifact_filename", "../candidates.jsonl", "artifact filename"),
    ),
)
def test_read_rejects_tampered_or_path_unsafe_sidecar(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    store = ArtifactStore(tmp_path / "run")
    store.write_manifest(manifest())
    store.write_jsonl("candidates.jsonl", (candidate_set(),))
    sidecar_path = metadata_path(store.root, "candidates.jsonl")
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload[field] = value
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        store.read_jsonl(
            "candidates.jsonl",
            CandidateSet,
            expected_dataset_signature="dataset-signature",
            expected_corpus_signature="corpus-signature",
        )


def test_read_rejects_cross_run_artifact_and_sidecar_substitution(tmp_path: Path) -> None:
    first = ArtifactStore(tmp_path / "first")
    second = ArtifactStore(tmp_path / "second")
    first.write_manifest(manifest())
    second_manifest = manifest().model_copy(update={"seed": 99})
    second.write_manifest(second_manifest)
    first.write_jsonl("candidate_sets.jsonl", (candidate_set(),))
    second.write_jsonl("candidate_sets.jsonl", (candidate_set(),))

    for filename in ("candidate_sets.jsonl", "candidate_sets.jsonl.metadata.json"):
        shutil.copyfile(second.root / filename, first.root / filename)

    with pytest.raises(ValueError, match="run manifest signature mismatch"):
        first.read_jsonl(
            "candidate_sets.jsonl",
            CandidateSet,
            expected_dataset_signature="dataset-signature",
            expected_corpus_signature="corpus-signature",
        )


def test_read_rejects_tampered_ancestor_provenance(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run")
    store.write_manifest(manifest())
    store.write_jsonl("queries.jsonl", (Query(query_id="q-1", text="question"),))
    store.write_jsonl(
        "candidate_sets.jsonl",
        (candidate_set(),),
        upstream_artifact_hashes={
            "queries.jsonl": store.artifact_hash("queries.jsonl")
        },
    )
    store.write_jsonl(
        "selected.jsonl",
        (candidate_set(),),
        upstream_artifact_hashes={
            "candidate_sets.jsonl": store.artifact_hash("candidate_sets.jsonl")
        },
    )
    candidate_sidecar = metadata_path(store.root, "candidate_sets.jsonl")
    payload = json.loads(candidate_sidecar.read_text(encoding="utf-8"))
    payload["upstream_artifact_hashes"]["queries.jsonl"] = "0" * 64
    candidate_sidecar.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="upstream artifact hash mismatch"):
        store.read_jsonl(
            "selected.jsonl",
            CandidateSet,
            expected_dataset_signature="dataset-signature",
            expected_corpus_signature="corpus-signature",
        )


def test_read_requires_sidecar_before_parsing(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run")
    store.write_manifest(manifest())
    (store.root / "candidates.jsonl").write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"metadata.*candidates\.jsonl\.metadata\.json"):
        store.read_jsonl(
            "candidates.jsonl",
            CandidateSet,
            expected_dataset_signature="dataset-signature",
            expected_corpus_signature="corpus-signature",
        )


def test_jsonl_round_trip_for_query_and_candidate_set(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run")
    queries = (Query(query_id="q-1", text="What changed?"),)
    candidates = (candidate_set(),)

    store.write_manifest(manifest())
    store.write_jsonl("queries.jsonl", queries)
    store.write_jsonl("candidates.jsonl", candidates)

    assert (
        store.read_jsonl(
            "queries.jsonl",
            Query,
            expected_dataset_signature="dataset-signature",
            expected_corpus_signature="corpus-signature",
        )
        == queries
    )
    assert (
        store.read_jsonl(
            "candidates.jsonl",
            CandidateSet,
            expected_dataset_signature="dataset-signature",
            expected_corpus_signature="corpus-signature",
        )
        == candidates
    )


def test_invalid_jsonl_reports_filename_and_one_based_line(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run")
    store.write_manifest(manifest())
    store.write_jsonl("queries.jsonl", (Query(query_id="q-1", text="valid"),))
    path = tmp_path / "run" / "queries.jsonl"
    path.write_text("\n{bad json}\n", encoding="utf-8")
    refresh_content_hash(store.root, "queries.jsonl")

    with pytest.raises(ValueError, match=r"queries\.jsonl:2"):
        store.read_jsonl(
            "queries.jsonl",
            Query,
            expected_dataset_signature="dataset-signature",
            expected_corpus_signature="corpus-signature",
        )


def test_invalid_utf8_jsonl_reports_filename_and_one_based_line(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run")
    store.write_manifest(manifest())
    store.write_jsonl("queries.jsonl", (Query(query_id="q-1", text="valid"),))
    (tmp_path / "run" / "queries.jsonl").write_bytes(b'{"query_id":"q-1","text":"valid"}\n\xff\n')
    refresh_content_hash(store.root, "queries.jsonl")

    with pytest.raises(ValueError, match=r"queries\.jsonl:2"):
        store.read_jsonl(
            "queries.jsonl",
            Query,
            expected_dataset_signature="dataset-signature",
            expected_corpus_signature="corpus-signature",
        )


@pytest.mark.parametrize(
    ("reader", "filename", "model_type"),
    (
        ("read_jsonl", "queries.jsonl", Query),
        ("read_json", "manifest-copy.json", RunManifest),
    ),
)
def test_stage_readers_require_expected_signatures(
    tmp_path: Path,
    reader: str,
    filename: str,
    model_type: type[Query] | type[RunManifest],
) -> None:
    store = ArtifactStore(tmp_path / "run")

    with pytest.raises(TypeError):
        getattr(store, reader)(filename, model_type)


@pytest.mark.parametrize(
    (
        "reader",
        "filename",
        "model_type",
        "expected_dataset_signature",
        "expected_corpus_signature",
        "mismatch_message",
    ),
    (
        (
            "read_jsonl",
            "queries.jsonl",
            Query,
            "wrong-dataset-signature",
            "corpus-signature",
            "dataset signature mismatch",
        ),
        (
            "read_jsonl",
            "queries.jsonl",
            Query,
            "dataset-signature",
            "wrong-corpus-signature",
            "corpus signature mismatch",
        ),
        (
            "read_json",
            "manifest-copy.json",
            RunManifest,
            "wrong-dataset-signature",
            "corpus-signature",
            "dataset signature mismatch",
        ),
        (
            "read_json",
            "manifest-copy.json",
            RunManifest,
            "dataset-signature",
            "wrong-corpus-signature",
            "corpus signature mismatch",
        ),
    ),
)
def test_stage_reader_rejects_manifest_mismatch_before_parsing_artifact(
    tmp_path: Path,
    reader: str,
    filename: str,
    model_type: type[Query] | type[RunManifest],
    expected_dataset_signature: str,
    expected_corpus_signature: str,
    mismatch_message: str,
) -> None:
    store = ArtifactStore(tmp_path / "run")
    store.write_manifest(manifest())
    if reader == "read_jsonl":
        store.write_jsonl(filename, ())
    else:
        store.write_json(filename, manifest())
    (tmp_path / "run" / filename).write_bytes(b"not valid JSON")
    refresh_content_hash(store.root, filename)

    with pytest.raises(ValueError, match=mismatch_message):
        getattr(store, reader)(
            filename,
            model_type,
            expected_dataset_signature=expected_dataset_signature,
            expected_corpus_signature=expected_corpus_signature,
        )


@pytest.mark.parametrize("filename", ("../outside.jsonl", "/tmp/outside.jsonl", "..\\x.jsonl"))
def test_artifact_paths_reject_traversal_and_absolute_paths(
    tmp_path: Path,
    filename: str,
) -> None:
    store = ArtifactStore(tmp_path / "run")

    with pytest.raises(ValueError, match="artifact filename"):
        store.write_jsonl(filename, ())


def test_json_output_is_canonical_and_typed(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run")
    expected = manifest()

    store.write_manifest(expected)
    store.write_json("manifest-copy.json", expected)

    assert json.loads((tmp_path / "run" / "manifest-copy.json").read_text()) == expected.model_dump(
        mode="json"
    )
    assert (
        store.read_json(
            "manifest-copy.json",
            RunManifest,
            expected_dataset_signature="dataset-signature",
            expected_corpus_signature="corpus-signature",
        )
        == expected
    )
