import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from evidence_rag.contracts.models import (
    CandidateSet,
    Document,
    EvidenceCandidate,
    Query,
    RetrieverProvenance,
)
from evidence_rag.infrastructure.artifacts import ArtifactStore, ProducerProvenance, RunManifest
from evidence_rag.infrastructure.config import ModuleConfig
from evidence_rag.infrastructure.corpus import CorpusBuilder, WordChunker
from evidence_rag.infrastructure.datasets import (
    DatasetManifest,
    GoldCase,
    JsonlDatasetAdapter,
)
from evidence_rag.materializer.selector_pool import (
    EXACT_RECOVERY_ROLE_PINS,
    EXPECTED_RETRIEVER_PARAMETERS_SHA256,
    SELECTOR_POOL_MANIFEST_FILE,
    SelectorCandidatePoolManifestV2,
    build_selector_candidate_pool_manifest_v2,
    candidate_set_sha256,
    freeze_selector_candidate_pool_manifest_v2,
    query_signature,
    read_selector_candidate_pool_manifest_v2,
    verify_selector_candidate_pool_v2,
)
from evidence_rag.retriever.indexing import write_index

HYBRID_PARAMETERS: dict[str, object] = {
    "fusion": "rrf",
    "retrievers": [
        {
            "name": "strong-bm25",
            "implementation_version": "strong-bm25-v1",
            "parameters": {"k1": 0.9, "b": 0.4},
        },
        {
            "name": "granite-dense",
            "implementation_version": "granite-dense-v1",
            "parameters": {
                "embedder_model_id": "ibm-granite/granite-embedding-english-r2",
                "query_prefix": "",
                "document_prefix": "",
            },
        },
    ],
    "pool_size": None,
    "k": 60,
}


@dataclass(frozen=True)
class PoolFixture:
    pool: Path
    dataset_manifest: Path


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, allow_nan=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _write_jsonl(path: Path, records: tuple[object, ...]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                record.model_dump(mode="json"),  # type: ignore[attr-defined]
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def make_pool(
    tmp_path: Path,
    *,
    top_k: int = 20,
    candidate_count: int = 20,
    rank_start: int = 1,
    reverse_queries: bool = False,
    candidate_retriever: bool = True,
    wrong_candidate_text: bool = False,
    index_parameters: dict[str, object] | None = None,
) -> PoolFixture:
    dataset_directory = tmp_path / "dataset"
    dataset_directory.mkdir()
    documents = tuple(
        Document(
            document_id=f"d-{index:02d}",
            text=f"unique document text {index}",
            source_uri=f"fixture://d-{index:02d}",
        )
        for index in range(20)
    )
    queries = (Query(query_id="q-1", text="first"), Query(query_id="q-2", text="second"))
    gold_cases = (
        GoldCase(query_id="q-1", relevant_document_ids=("d-00",)),
        GoldCase(query_id="q-2", relevant_document_ids=("d-01",)),
    )
    dataset_manifest = DatasetManifest(
        dataset_id="fixture/selector-pool",
        dataset_version="v1",
        split="dev",
        documents_file="documents.jsonl",
        queries_file="queries.jsonl",
        gold_cases_file="gold_cases.jsonl",
    )
    manifest_path = dataset_directory / "manifest.json"
    manifest_path.write_text(dataset_manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _write_jsonl(dataset_directory / "documents.jsonl", documents)
    _write_jsonl(dataset_directory / "queries.jsonl", queries)
    _write_jsonl(dataset_directory / "gold_cases.jsonl", gold_cases)
    dataset = JsonlDatasetAdapter.load(manifest_path)
    corpus = CorpusBuilder(WordChunker(chunk_size=100, overlap=0)).build(
        dataset.documents, dataset.dataset_signature
    )

    pool = tmp_path / "pool"
    parameters = index_parameters or HYBRID_PARAMETERS
    index_manifest = write_index(
        corpus,
        pool / "index",
        implementation="hybrid",
        implementation_version="hybrid-v1",
        parameters=parameters,
    )
    retriever_config = ModuleConfig(
        name="hybrid",
        parameters={
            "fusion": "rrf",
            "k": 60,
            "retrievers": [{"name": "strong-bm25"}, {"name": "granite-dense"}],
        },
    )
    run_manifest = RunManifest(
        dataset_id=dataset.manifest.dataset_id,
        dataset_version=dataset.manifest.dataset_version,
        split=dataset.manifest.split,
        dataset_signature=dataset.dataset_signature,
        corpus_signature=corpus.manifest.corpus_signature,
        chunker_name=corpus.manifest.chunker_name,
        chunker_version=corpus.manifest.chunker_version,
        chunk_size=corpus.manifest.chunk_size,
        overlap=corpus.manifest.overlap,
        index_implementation="hybrid",
        index_implementation_version="hybrid-v1",
        index_signature=index_manifest.index_signature,
        retriever=retriever_config,
        selector=ModuleConfig(name="top-k"),
        generator=ModuleConfig(name="extractive"),
        top_k=top_k,
        max_selected=10,
        seed=13,
        git_commit="a" * 40,
        git_dirty=False,
        source_tree_signature="b" * 64,
    )
    store = ArtifactStore(pool)
    store.write_manifest(run_manifest)
    store.write_json("corpus_snapshot.json", corpus, stage="prepare")
    store.write_jsonl("queries.jsonl", dataset.queries, stage="prepare")

    provenance = RetrieverProvenance(
        name="hybrid",
        implementation_version="hybrid-v1",
        parameters_sha256=_canonical_digest(parameters),
    )
    chunks = corpus.chunks[:candidate_count]
    windows = tuple(
        CandidateSet(
            query_id=query.query_id,
            retriever=provenance if candidate_retriever else None,
            candidates=tuple(
                EvidenceCandidate(
                    evidence_id=chunk.evidence_id,
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    text=(
                        "tampered candidate text"
                        if wrong_candidate_text and rank == rank_start
                        else chunk.text
                    ),
                    source_uri=chunk.source_uri,
                    metadata=chunk.metadata,
                    retrieval_score=1.0 / rank,
                    retrieval_rank=rank,
                )
                for rank, chunk in enumerate(chunks, start=rank_start)
            ),
        )
        for query in dataset.queries
    )
    if reverse_queries:
        windows = tuple(reversed(windows))
    producer = ProducerProvenance(
        git_commit=run_manifest.git_commit,
        git_dirty=run_manifest.git_dirty,
        source_tree_signature=run_manifest.source_tree_signature,
        module=run_manifest.retriever,
        execution=ModuleConfig(name="retriever", parameters={"top_k": top_k}),
    )
    store.write_jsonl(
        "candidate_sets.jsonl",
        windows,
        producer="ExperimentWorkflow",
        stage="retriever",
        producer_provenance=producer,
        upstream_artifact_hashes={
            "corpus_snapshot.json": store.artifact_hash("corpus_snapshot.json"),
            "queries.jsonl": store.artifact_hash("queries.jsonl"),
        },
    )
    return PoolFixture(pool=pool, dataset_manifest=manifest_path)


def test_hybrid_rrf_top20_pool_builds_freezes_and_verifies(tmp_path: Path) -> None:
    fixture = make_pool(tmp_path)

    manifest = build_selector_candidate_pool_manifest_v2(
        fixture.pool,
        fixture.dataset_manifest,
        data_role="niah-dev",
        recovery_status="new-v2-run",
    )
    path = freeze_selector_candidate_pool_manifest_v2(fixture.pool, manifest)

    assert path.name == SELECTOR_POOL_MANIFEST_FILE
    assert manifest.data_role == "niah-dev"
    assert manifest.top_n == 20
    assert manifest.n_queries == 2
    assert manifest.retriever.parameters_sha256 == EXPECTED_RETRIEVER_PARAMETERS_SHA256
    assert tuple(window.query_id for window in manifest.windows) == ("q-1", "q-2")
    assert read_selector_candidate_pool_manifest_v2(fixture.pool) == manifest
    assert verify_selector_candidate_pool_v2(fixture.pool, fixture.dataset_manifest) == manifest
    assert not (fixture.pool / "candidate_freeze.json").exists()


def test_freeze_is_write_once(tmp_path: Path) -> None:
    fixture = make_pool(tmp_path)
    manifest = build_selector_candidate_pool_manifest_v2(
        fixture.pool,
        fixture.dataset_manifest,
        data_role="niah-train",
        recovery_status="new-v2-run",
    )
    freeze_selector_candidate_pool_manifest_v2(fixture.pool, manifest)

    with pytest.raises(ValueError, match="already frozen"):
        freeze_selector_candidate_pool_manifest_v2(fixture.pool, manifest)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"top_k": 50}, "native top_k=20"),
        ({"candidate_count": 19}, "19 candidates, expected 20"),
        ({"rank_start": 2}, "ordered consecutive ranks 1..20"),
        ({"reverse_queries": True}, "query order/completeness"),
        ({"candidate_retriever": False}, "name no retriever"),
        ({"wrong_candidate_text": True}, "signed corpus chunk on text"),
    ],
)
def test_build_fails_closed_on_pool_integrity_errors(
    tmp_path: Path, options: dict[str, object], message: str
) -> None:
    fixture = make_pool(tmp_path, **options)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=message):
        build_selector_candidate_pool_manifest_v2(
            fixture.pool,
            fixture.dataset_manifest,
            data_role="niah-dev",
            recovery_status="new-v2-run",
        )


def test_build_rejects_any_hybrid_parameter_digest_other_than_the_frozen_one(
    tmp_path: Path,
) -> None:
    parameters = dict(HYBRID_PARAMETERS)
    parameters["pool_size"] = 50
    fixture = make_pool(tmp_path, index_parameters=parameters)

    with pytest.raises(ValueError, match="expected the frozen Selector-v2 digest"):
        build_selector_candidate_pool_manifest_v2(
            fixture.pool,
            fixture.dataset_manifest,
            data_role="2wiki-train",
            recovery_status="new-v2-run",
        )


def test_candidate_metadata_must_match_run_and_candidate_bytes(tmp_path: Path) -> None:
    fixture = make_pool(tmp_path)
    path = fixture.pool / "candidate_sets.jsonl.metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["dataset_signature"] = "0" * 64
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="candidate metadata dataset_signature"):
        build_selector_candidate_pool_manifest_v2(
            fixture.pool,
            fixture.dataset_manifest,
            data_role="2wiki-dev",
            recovery_status="new-v2-run",
        )


def test_verify_detects_candidate_file_change_after_freeze(tmp_path: Path) -> None:
    fixture = make_pool(tmp_path)
    manifest = build_selector_candidate_pool_manifest_v2(
        fixture.pool,
        fixture.dataset_manifest,
        data_role="niah-sealed600",
        recovery_status="new-v2-run",
    )
    freeze_selector_candidate_pool_manifest_v2(fixture.pool, manifest)
    path = fixture.pool / "candidate_sets.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])
    payload["candidates"][0]["retrieval_score"] = 0.123
    lines[0] = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="raw sha256 mismatch"):
        verify_selector_candidate_pool_v2(fixture.pool, fixture.dataset_manifest)


def test_verify_detects_dataset_file_change_after_freeze(tmp_path: Path) -> None:
    fixture = make_pool(tmp_path)
    manifest = build_selector_candidate_pool_manifest_v2(
        fixture.pool,
        fixture.dataset_manifest,
        data_role="2wiki-heldout",
        recovery_status="new-v2-run",
    )
    freeze_selector_candidate_pool_manifest_v2(fixture.pool, manifest)
    queries = fixture.dataset_manifest.parent / "queries.jsonl"
    payload = json.loads(queries.read_text(encoding="utf-8").splitlines()[0])
    payload["text"] = "changed after freeze"
    lines = queries.read_text(encoding="utf-8").splitlines()
    lines[0] = json.dumps(payload, sort_keys=True)
    queries.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="queries.jsonl raw sha256 mismatch"):
        verify_selector_candidate_pool_v2(fixture.pool, fixture.dataset_manifest)


def test_query_and_window_hashes_are_order_sensitive_and_content_complete(tmp_path: Path) -> None:
    fixture = make_pool(tmp_path)
    candidate_lines = (fixture.pool / "candidate_sets.jsonl").read_text(encoding="utf-8").splitlines()
    first = CandidateSet.model_validate_json(candidate_lines[0])
    changed = first.model_copy(
        update={
            "candidates": (
                first.candidates[0].model_copy(update={"retrieval_score": 0.123}),
                *first.candidates[1:],
            )
        }
    )
    queries = (Query(query_id="a", text="x"), Query(query_id="b", text="y"))

    assert candidate_set_sha256(first) != candidate_set_sha256(changed)
    assert query_signature(queries) != query_signature(tuple(reversed(queries)))


def test_manifest_schema_rejects_duplicate_window_pins(tmp_path: Path) -> None:
    fixture = make_pool(tmp_path)
    manifest = build_selector_candidate_pool_manifest_v2(
        fixture.pool,
        fixture.dataset_manifest,
        data_role="niah-dev",
        recovery_status="new-v2-run",
    )
    payload = manifest.model_dump(mode="json")
    payload["windows"] = (payload["windows"][0], payload["windows"][0])

    with pytest.raises(ValidationError, match="duplicate query IDs"):
        SelectorCandidatePoolManifestV2.model_validate(payload)


def test_exact_recovery_rejects_a_new_pool_with_an_unknown_historical_hash(
    tmp_path: Path,
) -> None:
    fixture = make_pool(tmp_path)

    with pytest.raises(ValidationError, match="is not the audited exact-recovery pool"):
        build_selector_candidate_pool_manifest_v2(
            fixture.pool,
            fixture.dataset_manifest,
            data_role="niah-dev",
            recovery_status="exact-recovery",
        )


def test_exact_recovery_rejects_a_valid_historical_identity_under_the_wrong_role(
    tmp_path: Path,
) -> None:
    fixture = make_pool(tmp_path)
    manifest = build_selector_candidate_pool_manifest_v2(
        fixture.pool,
        fixture.dataset_manifest,
        data_role="niah-dev",
        recovery_status="new-v2-run",
    )
    candidate_sha256, dataset_signature, n_queries = EXACT_RECOVERY_ROLE_PINS["niah-train"]
    payload = manifest.model_dump(mode="json")
    payload.update(
        {
            "recovery_status": "exact-recovery",
            "data_role": "niah-dev",
            "candidate_sha256": candidate_sha256,
            "dataset_signature": dataset_signature,
            "n_queries": n_queries,
            "windows": tuple(
                {
                    "query_id": f"historical-{index}",
                    "candidate_set_sha256": "0" * 64,
                }
                for index in range(n_queries)
            ),
        }
    )

    with pytest.raises(ValidationError, match="data_role 'niah-dev'.*expected"):
        SelectorCandidatePoolManifestV2.model_validate(payload)


def test_pool_root_queries_must_equal_dataset_queries_in_content_and_order(tmp_path: Path) -> None:
    fixture = make_pool(tmp_path)
    path = fixture.pool / "queries.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])
    payload["text"] = "different pool-root query"
    lines[0] = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="pool-root queries.jsonl content/order"):
        build_selector_candidate_pool_manifest_v2(
            fixture.pool,
            fixture.dataset_manifest,
            data_role="2wiki-dev",
            recovery_status="new-v2-run",
        )


def test_candidate_metadata_must_name_exact_root_corpus_and_query_upstreams(
    tmp_path: Path,
) -> None:
    fixture = make_pool(tmp_path)
    path = fixture.pool / "candidate_sets.jsonl.metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["upstream_artifact_hashes"] = {"queries.jsonl": "0" * 64}
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="upstream hashes must point exactly"):
        build_selector_candidate_pool_manifest_v2(
            fixture.pool,
            fixture.dataset_manifest,
            data_role="2wiki-dev",
            recovery_status="new-v2-run",
        )
