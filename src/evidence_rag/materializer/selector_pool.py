"""Write-once integrity manifest for the recovered Selector-v2 candidate pools.

This freeze is deliberately separate from :mod:`evidence_rag.materializer.sealed600`.
The historical ``CandidateFreeze`` certifies the BM25 pool used by Graph 2.0; changing
that contract to admit Hybrid would make two different experiments share one green
check.  Selector-v2 instead admits exactly the recovered Hybrid-RRF configuration used
by its six data roles and stores its manifest beside ``candidate_sets.jsonl``.

The verifier is intentionally fail closed.  A plausible-looking Top-20 JSONL is not
enough: the run, candidate metadata and index must agree on their dataset/corpus/index
signatures; every window must carry the index-derived retriever provenance; and every
candidate must resolve back to the exact chunk in the signed index snapshot.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from evidence_rag.contracts.models import CandidateSet, Query, RetrieverProvenance
from evidence_rag.infrastructure.artifacts import (
    ArtifactMetadata,
    RunManifest,
    run_manifest_signature,
)
from evidence_rag.infrastructure.corpus import Chunk, CorpusSnapshot
from evidence_rag.infrastructure.datasets import DatasetManifest, JsonlDatasetAdapter
from evidence_rag.retriever.indexing import IndexManifest, load_index

SELECTOR_POOL_MANIFEST_FILE: Literal["selector_candidate_pool_manifest_v2.json"] = (
    "selector_candidate_pool_manifest_v2.json"
)
CANDIDATE_FILE: Literal["candidate_sets.jsonl"] = "candidate_sets.jsonl"
CANDIDATE_METADATA_FILE: Literal["candidate_sets.jsonl.metadata.json"] = (
    "candidate_sets.jsonl.metadata.json"
)
RUN_MANIFEST_FILE: Literal["run_manifest.json"] = "run_manifest.json"
CORPUS_SNAPSHOT_FILE: Literal["corpus_snapshot.json"] = "corpus_snapshot.json"
CORPUS_SNAPSHOT_METADATA_FILE: Literal["corpus_snapshot.json.metadata.json"] = (
    "corpus_snapshot.json.metadata.json"
)
QUERIES_FILE: Literal["queries.jsonl"] = "queries.jsonl"
QUERIES_METADATA_FILE: Literal["queries.jsonl.metadata.json"] = (
    "queries.jsonl.metadata.json"
)
INDEX_MANIFEST_FILE: Literal["index/index_manifest.json"] = "index/index_manifest.json"
INDEX_CORPUS_SNAPSHOT_FILE: Literal["index/corpus_snapshot.json"] = (
    "index/corpus_snapshot.json"
)

EXPECTED_RETRIEVER_NAME: Literal["hybrid"] = "hybrid"
EXPECTED_RETRIEVER_VERSION: Literal["hybrid-v1"] = "hybrid-v1"
EXPECTED_RETRIEVER_PARAMETERS_SHA256 = (
    "67333c6f3fcf6567756b382048961bc150da0ee26af4f9cecc59f58f30b5d78d"
)
EXPECTED_FUSION: Literal["rrf"] = "rrf"
EXPECTED_RRF_K: Literal[60] = 60
EXPECTED_TOP_N: Literal[20] = 20
PROTOCOL_VERSION: Literal["selector-candidate-pool-v2"] = "selector-candidate-pool-v2"

DataRole: TypeAlias = Literal[
    "niah-train",
    "niah-dev",
    "niah-sealed600",
    "2wiki-train",
    "2wiki-dev",
    "2wiki-heldout",
]
RecoveryStatus: TypeAlias = Literal["exact-recovery", "new-v2-run"]

# These are the six byte-identical pools audited in R001A/B.  ``exact-recovery`` is
# therefore a verifiable statement, not a free-form label: a role accepts only the
# historical candidate bytes, dataset signature and query count recorded here.
EXACT_RECOVERY_ROLE_PINS: Mapping[DataRole, tuple[str, str, int]] = {
    "niah-train": (
        "09e8c9b4972f48a67661c8b06dcf220f178528156699da671a16dd8d08fb908f",
        "9c389b69d5a74daba69a8577145ee65e27a092de8916ccd0f8e20772050a04e0",
        2000,
    ),
    "niah-dev": (
        "89ede8249e0682568ea0a85d00eac4881323c992d8d16362016810bd8cb96ee3",
        "eb7760674bf3aace707acd932c67c86694d560510f949e3c73dea1ca7353db87",
        2000,
    ),
    "niah-sealed600": (
        "777391fac4854448a47a7cdc9543cd77710b17f4e960084a26f4640f340ae590",
        "57b89ef69e409623f38ac111311dddb47248c9c2ed84f4cdba0d4854ae5ec7bb",
        600,
    ),
    "2wiki-train": (
        "0ab0fc92f95add1c5d514f531e7b567c4e67d1e5430c401f37ec6f719dbc8887",
        "036385693999c5bdf6932cd96a32019be6053fc2ac85f8ad3d458d1de1485f86",
        3000,
    ),
    "2wiki-dev": (
        "26442003e230c93e53fe71f4b9a16d269fc5b02674ba6137f6cfceabb721607f",
        "69dce7a2d909808ddf6c509005181a4741f22c471551ecf117b399eddef38afc",
        2000,
    ),
    "2wiki-heldout": (
        "fcca691ed0867bfdc8c6491452d82b4a5701218c8bac7ead0bb486a8469ee219",
        "ba8b6c950962191a3a3799fb4b7fe44f65ef416e3fcd5f32644957e745c689cc",
        2000,
    ),
}

NonEmpty = Annotated[str, Field(min_length=1)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PositiveInteger = Annotated[int, Field(gt=0)]
ModelT = TypeVar("ModelT", bound=BaseModel)


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class SelectorCandidateWindowPinV2(FrozenModel):
    query_id: NonEmpty
    candidate_set_sha256: Digest


class SelectorCandidatePoolManifestV2(FrozenModel):
    """The complete, portable identity of one recovered Hybrid-RRF Top-20 pool."""

    schema_version: Literal["2.0"] = "2.0"
    protocol_version: Literal["selector-candidate-pool-v2"] = PROTOCOL_VERSION
    recovery_status: Literal["exact-recovery", "new-v2-run"]
    data_role: DataRole

    candidate_file: Literal["candidate_sets.jsonl"] = CANDIDATE_FILE
    candidate_metadata_file: Literal["candidate_sets.jsonl.metadata.json"] = (
        CANDIDATE_METADATA_FILE
    )
    run_manifest_file: Literal["run_manifest.json"] = RUN_MANIFEST_FILE
    corpus_snapshot_file: Literal["corpus_snapshot.json"] = CORPUS_SNAPSHOT_FILE
    index_manifest_file: Literal["index/index_manifest.json"] = INDEX_MANIFEST_FILE
    index_corpus_snapshot_file: Literal["index/corpus_snapshot.json"] = (
        INDEX_CORPUS_SNAPSHOT_FILE
    )

    candidate_sha256: Digest
    candidate_metadata_sha256: Digest
    run_manifest_sha256: Digest
    corpus_snapshot_sha256: Digest
    pool_queries_sha256: Digest
    index_manifest_sha256: Digest
    index_corpus_snapshot_sha256: Digest

    dataset_manifest_sha256: Digest
    dataset_documents_sha256: Digest
    dataset_queries_sha256: Digest
    dataset_gold_cases_sha256: Digest
    dataset_id: NonEmpty
    dataset_version: NonEmpty
    dataset_split: NonEmpty
    dataset_signature: Digest
    query_signature: Digest
    corpus_signature: Digest
    index_signature: Digest

    retriever: RetrieverProvenance
    fusion: Literal["rrf"] = EXPECTED_FUSION
    rrf_k: Literal[60] = EXPECTED_RRF_K
    top_n: Literal[20] = EXPECTED_TOP_N
    n_queries: PositiveInteger
    windows: tuple[SelectorCandidateWindowPinV2, ...]

    @model_validator(mode="after")
    def windows_are_complete_and_unique(self) -> SelectorCandidatePoolManifestV2:
        query_ids = tuple(window.query_id for window in self.windows)
        if not query_ids:
            raise ValueError("selector candidate pool manifest must contain at least one window")
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("selector candidate pool manifest has duplicate query IDs")
        if len(query_ids) != self.n_queries:
            raise ValueError(
                f"manifest has {len(query_ids)} window pins but n_queries={self.n_queries}"
            )
        if self.recovery_status == "exact-recovery":
            candidate_sha256, dataset_signature, n_queries = EXACT_RECOVERY_ROLE_PINS[
                self.data_role
            ]
            mismatches: list[str] = []
            if self.candidate_sha256 != candidate_sha256:
                mismatches.append(
                    f"candidate_sha256={self.candidate_sha256}, expected {candidate_sha256}"
                )
            if self.dataset_signature != dataset_signature:
                mismatches.append(
                    f"dataset_signature={self.dataset_signature}, expected {dataset_signature}"
                )
            if self.n_queries != n_queries:
                mismatches.append(f"n_queries={self.n_queries}, expected {n_queries}")
            if mismatches:
                raise ValueError(
                    f"data_role {self.data_role!r} is not the audited exact-recovery pool: "
                    + "; ".join(mismatches)
                )
        return self


@dataclass(frozen=True)
class _DatasetEvidence:
    manifest: DatasetManifest
    signature: str
    queries: tuple[Query, ...]
    query_signature: str
    manifest_sha256: str
    documents_sha256: str
    queries_sha256: str
    gold_cases_sha256: str


@dataclass(frozen=True)
class _PoolEvidence:
    directory: Path
    run_manifest: RunManifest
    candidate_metadata: ArtifactMetadata
    corpus_snapshot_metadata: ArtifactMetadata
    pool_queries_metadata: ArtifactMetadata
    index_manifest: IndexManifest
    corpus: CorpusSnapshot
    candidate_sets: tuple[CandidateSet, ...]
    pool_queries: tuple[Query, ...]
    dataset: _DatasetEvidence
    candidate_sha256: str
    candidate_metadata_sha256: str
    run_manifest_sha256: str
    corpus_snapshot_sha256: str
    pool_queries_sha256: str
    index_manifest_sha256: str
    index_corpus_snapshot_sha256: str
    corpus_snapshot_bundle_sha256: str
    pool_queries_bundle_sha256: str


def _canonical_json(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_digest(value: object) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def query_signature(queries: Sequence[Query]) -> str:
    """Hash the complete, ordered query list after checking its IDs are unique."""

    if not queries:
        raise ValueError("query signature is undefined for an empty query list")
    query_ids = tuple(query.query_id for query in queries)
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query signature input contains duplicate query IDs")
    return _canonical_digest([query.model_dump(mode="json") for query in queries])


def candidate_set_sha256(candidate_set: CandidateSet) -> str:
    """Canonical per-query digest, independent of JSONL whitespace and line endings."""

    return _canonical_digest(candidate_set)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"unable to hash required selector-pool artifact {path}: {error}") from error
    return digest.hexdigest()


def _artifact_hashes(path: Path, metadata: ArtifactMetadata) -> tuple[str, str]:
    """Return the raw-content and ArtifactStore bundle hashes in one streaming pass."""

    content_digest = sha256()
    bundle_digest = sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                content_digest.update(block)
                bundle_digest.update(block)
    except OSError as error:
        raise ValueError(f"unable to hash required selector-pool artifact {path}: {error}") from error
    bundle_digest.update(b"\0")
    bundle_digest.update((_canonical_json(metadata) + "\n").encode("utf-8"))
    return content_digest.hexdigest(), bundle_digest.hexdigest()


def _read_model(path: Path, model_type: type[ModelT]) -> ModelT:
    try:
        content = Path(path).read_text(encoding="utf-8")
        return model_type.model_validate_json(content)
    except OSError as error:
        raise ValueError(f"unable to read required selector-pool artifact {path}: {error}") from error
    except (ValidationError, ValueError) as error:
        raise ValueError(f"invalid selector-pool artifact {path}: {error}") from error


def _read_candidate_sets(path: Path) -> tuple[tuple[CandidateSet, ...], str]:
    records: list[CandidateSet] = []
    digest = sha256()
    try:
        with Path(path).open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                if not raw_line.strip():
                    raise ValueError(
                        f"blank JSONL record at {path}:{line_number}; native retrieval output "
                        "contains exactly one CandidateSet per line"
                    )
                try:
                    records.append(CandidateSet.model_validate_json(raw_line))
                except (ValidationError, ValueError) as error:
                    raise ValueError(
                        f"invalid CandidateSet at {path}:{line_number}: {error}"
                    ) from error
    except OSError as error:
        raise ValueError(f"unable to read required selector-pool artifact {path}: {error}") from error
    if not records:
        raise ValueError(f"candidate pool {path} is empty")
    return tuple(records), digest.hexdigest()


def _read_queries(path: Path) -> tuple[Query, ...]:
    records: list[Query] = []
    try:
        with Path(path).open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    raise ValueError(f"blank query JSONL record at {path}:{line_number}")
                try:
                    records.append(Query.model_validate_json(raw_line))
                except (ValidationError, ValueError) as error:
                    raise ValueError(f"invalid Query at {path}:{line_number}: {error}") from error
    except OSError as error:
        raise ValueError(f"unable to read required selector-pool artifact {path}: {error}") from error
    if not records:
        raise ValueError(f"pool query artifact {path} is empty")
    return tuple(records)


def _dataset_path(root: Path, filename: str) -> Path:
    root = root.resolve()
    path = (root / filename).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"dataset artifact {filename!r} escapes its manifest directory {root}"
        ) from error
    return path


def _read_dataset_evidence(manifest_path: Path) -> _DatasetEvidence:
    manifest_path = Path(manifest_path).resolve()
    manifest = _read_model(manifest_path, DatasetManifest)
    root = manifest_path.parent
    documents_path = _dataset_path(root, manifest.documents_file)
    queries_path = _dataset_path(root, manifest.queries_file)
    gold_cases_path = _dataset_path(root, manifest.gold_cases_file)

    # JsonlDatasetAdapter performs the canonical ID/alignment validation and computes the
    # exact dataset signature used by RunManifest.  Only the small query tuple is retained;
    # the documents are released before the (potentially large) corpus snapshot is loaded.
    bundle = JsonlDatasetAdapter.load(manifest_path)
    queries = bundle.queries
    evidence = _DatasetEvidence(
        manifest=bundle.manifest,
        signature=bundle.dataset_signature,
        queries=queries,
        query_signature=query_signature(queries),
        manifest_sha256=_sha256_file(manifest_path),
        documents_sha256=_sha256_file(documents_path),
        queries_sha256=_sha256_file(queries_path),
        gold_cases_sha256=_sha256_file(gold_cases_path),
    )
    return evidence


def _load_pool_evidence(pool_directory: Path, dataset_manifest_path: Path) -> _PoolEvidence:
    directory = Path(pool_directory).resolve()
    if not directory.is_dir():
        raise ValueError(f"selector candidate pool directory does not exist: {directory}")

    run_path = directory / RUN_MANIFEST_FILE
    candidate_path = directory / CANDIDATE_FILE
    candidate_metadata_path = directory / CANDIDATE_METADATA_FILE
    corpus_path = directory / CORPUS_SNAPSHOT_FILE
    corpus_metadata_path = directory / CORPUS_SNAPSHOT_METADATA_FILE
    queries_path = directory / QUERIES_FILE
    queries_metadata_path = directory / QUERIES_METADATA_FILE
    index_manifest_path = directory / INDEX_MANIFEST_FILE
    index_corpus_path = directory / INDEX_CORPUS_SNAPSHOT_FILE

    run_manifest = _read_model(run_path, RunManifest)
    candidate_metadata = _read_model(candidate_metadata_path, ArtifactMetadata)
    corpus_metadata = _read_model(corpus_metadata_path, ArtifactMetadata)
    queries_metadata = _read_model(queries_metadata_path, ArtifactMetadata)
    index_manifest = _read_model(index_manifest_path, IndexManifest)
    candidate_sets, candidate_digest = _read_candidate_sets(candidate_path)
    pool_queries = _read_queries(queries_path)
    dataset = _read_dataset_evidence(dataset_manifest_path)
    corpus_snapshot_digest, corpus_snapshot_bundle_digest = _artifact_hashes(
        corpus_path, corpus_metadata
    )
    pool_queries_digest, pool_queries_bundle_digest = _artifact_hashes(
        queries_path, queries_metadata
    )

    # load_index recomputes the signed index identity and reads its corpus snapshot once.
    # Later candidate-to-chunk validation scans that returned snapshot once, storing only
    # chunks that occur in the candidate pool rather than constructing a second full index.
    corpus = load_index(
        directory / "index",
        expected_corpus_signature=index_manifest.corpus_signature,
        expected_implementation=index_manifest.implementation,
        expected_implementation_version=index_manifest.implementation_version,
        expected_parameters=index_manifest.parameters,
    )

    return _PoolEvidence(
        directory=directory,
        run_manifest=run_manifest,
        candidate_metadata=candidate_metadata,
        corpus_snapshot_metadata=corpus_metadata,
        pool_queries_metadata=queries_metadata,
        index_manifest=index_manifest,
        corpus=corpus,
        candidate_sets=candidate_sets,
        pool_queries=pool_queries,
        dataset=dataset,
        candidate_sha256=candidate_digest,
        candidate_metadata_sha256=_sha256_file(candidate_metadata_path),
        run_manifest_sha256=_sha256_file(run_path),
        corpus_snapshot_sha256=corpus_snapshot_digest,
        pool_queries_sha256=pool_queries_digest,
        index_manifest_sha256=_sha256_file(index_manifest_path),
        index_corpus_snapshot_sha256=_sha256_file(index_corpus_path),
        corpus_snapshot_bundle_sha256=corpus_snapshot_bundle_digest,
        pool_queries_bundle_sha256=pool_queries_bundle_digest,
    )


def _expected_retriever(index_manifest: IndexManifest) -> RetrieverProvenance:
    return RetrieverProvenance(
        name=index_manifest.implementation,
        implementation_version=index_manifest.implementation_version,
        parameters_sha256=_canonical_digest(index_manifest.parameters),
    )


def _candidate_chunk_problems(
    candidate_sets: Sequence[CandidateSet], corpus: CorpusSnapshot
) -> list[str]:
    wanted = {
        candidate.evidence_id
        for candidate_set in candidate_sets
        for candidate in candidate_set.candidates
    }
    chunks: dict[str, Chunk] = {}
    duplicate_chunks: set[str] = set()
    for chunk in corpus.chunks:
        if chunk.evidence_id not in wanted:
            continue
        if chunk.evidence_id in chunks:
            duplicate_chunks.add(chunk.evidence_id)
        else:
            chunks[chunk.evidence_id] = chunk

    problems = [
        f"corpus snapshot contains duplicate evidence_id {evidence_id!r}"
        for evidence_id in sorted(duplicate_chunks)
    ]
    for candidate_set in candidate_sets:
        for candidate in candidate_set.candidates:
            matched_chunk = chunks.get(candidate.evidence_id)
            if matched_chunk is None:
                problems.append(
                    f"{candidate_set.query_id}: candidate {candidate.evidence_id!r} is absent "
                    "from the signed corpus snapshot"
                )
                continue
            mismatches: list[str] = []
            for field, found, expected in (
                ("document_id", candidate.document_id, matched_chunk.document_id),
                ("chunk_id", candidate.chunk_id, matched_chunk.chunk_id),
                ("text", candidate.text, matched_chunk.text),
                ("source_uri", candidate.source_uri, matched_chunk.source_uri),
                ("metadata", candidate.metadata, matched_chunk.metadata),
            ):
                if found != expected:
                    mismatches.append(field)
            if mismatches:
                problems.append(
                    f"{candidate_set.query_id}: candidate {candidate.evidence_id!r} disagrees "
                    f"with its signed corpus chunk on {', '.join(mismatches)}"
                )
    return problems


def _verification_problems(
    manifest: SelectorCandidatePoolManifestV2, evidence: _PoolEvidence
) -> tuple[str, ...]:
    problems: list[str] = []
    run = evidence.run_manifest
    metadata = evidence.candidate_metadata
    index = evidence.index_manifest
    dataset = evidence.dataset
    corpus = evidence.corpus

    raw_checks = (
        ("candidate_sets.jsonl", manifest.candidate_sha256, evidence.candidate_sha256),
        (
            "candidate_sets.jsonl.metadata.json",
            manifest.candidate_metadata_sha256,
            evidence.candidate_metadata_sha256,
        ),
        ("run_manifest.json", manifest.run_manifest_sha256, evidence.run_manifest_sha256),
        (
            "corpus_snapshot.json",
            manifest.corpus_snapshot_sha256,
            evidence.corpus_snapshot_sha256,
        ),
        ("queries.jsonl", manifest.pool_queries_sha256, evidence.pool_queries_sha256),
        (
            "index/index_manifest.json",
            manifest.index_manifest_sha256,
            evidence.index_manifest_sha256,
        ),
        (
            "index/corpus_snapshot.json",
            manifest.index_corpus_snapshot_sha256,
            evidence.index_corpus_snapshot_sha256,
        ),
        (
            "dataset manifest",
            manifest.dataset_manifest_sha256,
            dataset.manifest_sha256,
        ),
        (
            dataset.manifest.documents_file,
            manifest.dataset_documents_sha256,
            dataset.documents_sha256,
        ),
        (
            dataset.manifest.queries_file,
            manifest.dataset_queries_sha256,
            dataset.queries_sha256,
        ),
        (
            dataset.manifest.gold_cases_file,
            manifest.dataset_gold_cases_sha256,
            dataset.gold_cases_sha256,
        ),
    )
    for label, expected, found in raw_checks:
        if expected != found:
            problems.append(f"{label} raw sha256 mismatch: expected {expected}, found {found}")

    if evidence.corpus_snapshot_sha256 != evidence.index_corpus_snapshot_sha256:
        problems.append(
            "root and index corpus snapshots differ byte-for-byte; the candidate metadata and "
            "the signed index do not point at one recovered corpus artifact"
        )

    for label, artifact_metadata, expected_filename, content_sha256 in (
        (
            "pool-root corpus metadata",
            evidence.corpus_snapshot_metadata,
            CORPUS_SNAPSHOT_FILE,
            evidence.corpus_snapshot_sha256,
        ),
        (
            "pool-root query metadata",
            evidence.pool_queries_metadata,
            QUERIES_FILE,
            evidence.pool_queries_sha256,
        ),
    ):
        if artifact_metadata.artifact_filename != expected_filename:
            problems.append(
                f"{label} names {artifact_metadata.artifact_filename!r}, "
                f"expected {expected_filename!r}"
            )
        if artifact_metadata.content_sha256 != content_sha256:
            problems.append(f"{label} content_sha256 does not match its artifact bytes")
        if artifact_metadata.run_manifest_signature != run_manifest_signature(run):
            problems.append(f"{label} run_manifest_signature does not match RunManifest")
        if artifact_metadata.dataset_signature != run.dataset_signature:
            problems.append(f"{label} dataset_signature does not match RunManifest")
        if artifact_metadata.corpus_signature != run.corpus_signature:
            problems.append(f"{label} corpus_signature does not match RunManifest")

    identity_checks = (
        ("dataset_id", manifest.dataset_id, dataset.manifest.dataset_id),
        ("dataset_version", manifest.dataset_version, dataset.manifest.dataset_version),
        ("dataset_split", manifest.dataset_split, dataset.manifest.split),
        ("dataset_signature", manifest.dataset_signature, dataset.signature),
        ("query_signature", manifest.query_signature, dataset.query_signature),
        ("corpus_signature", manifest.corpus_signature, corpus.manifest.corpus_signature),
        ("index_signature", manifest.index_signature, index.index_signature),
    )
    for label, expected, found in identity_checks:
        if expected != found:
            problems.append(f"{label} mismatch: expected {expected!r}, found {found!r}")

    if corpus.manifest.dataset_signature != dataset.signature:
        problems.append(
            "signed corpus dataset_signature does not match the supplied dataset manifest"
        )
    for label, found, expected in (
        ("RunManifest dataset_id", run.dataset_id, dataset.manifest.dataset_id),
        ("RunManifest dataset_version", run.dataset_version, dataset.manifest.dataset_version),
        ("RunManifest split", run.split, dataset.manifest.split),
        ("RunManifest dataset_signature", run.dataset_signature, dataset.signature),
        (
            "RunManifest corpus_signature",
            run.corpus_signature,
            corpus.manifest.corpus_signature,
        ),
        ("IndexManifest corpus_signature", index.corpus_signature, corpus.manifest.corpus_signature),
        ("RunManifest index_signature", run.index_signature, index.index_signature),
    ):
        if found != expected:
            problems.append(f"{label} mismatch: expected {expected!r}, found {found!r}")

    if run.top_k != EXPECTED_TOP_N:
        problems.append(
            f"RunManifest top_k={run.top_k}; Selector-v2 recovery requires native top_k=20 "
            "and does not admit a post-hoc Top-50 truncation"
        )
    if run.index_implementation != EXPECTED_RETRIEVER_NAME:
        problems.append(
            f"RunManifest index implementation is {run.index_implementation!r}, expected 'hybrid'"
        )
    if run.index_implementation_version != EXPECTED_RETRIEVER_VERSION:
        problems.append(
            "RunManifest index implementation version is "
            f"{run.index_implementation_version!r}, expected 'hybrid-v1'"
        )
    if run.retriever.name != EXPECTED_RETRIEVER_NAME:
        problems.append(f"RunManifest retriever is {run.retriever.name!r}, expected 'hybrid'")

    if index.implementation != EXPECTED_RETRIEVER_NAME:
        problems.append(f"IndexManifest implementation is {index.implementation!r}, expected 'hybrid'")
    if index.implementation_version != EXPECTED_RETRIEVER_VERSION:
        problems.append(
            f"IndexManifest implementation version is {index.implementation_version!r}, "
            "expected 'hybrid-v1'"
        )
    if index.parameters.get("fusion") != EXPECTED_FUSION:
        problems.append(
            f"IndexManifest fusion is {index.parameters.get('fusion')!r}, expected 'rrf'"
        )
    if index.parameters.get("k") != EXPECTED_RRF_K:
        problems.append(f"IndexManifest RRF k is {index.parameters.get('k')!r}, expected 60")
    arms = index.parameters.get("retrievers")
    if not isinstance(arms, list) or not arms:
        problems.append("IndexManifest Hybrid-RRF configuration has no retriever arms")

    expected_retriever = _expected_retriever(index)
    if expected_retriever.parameters_sha256 != EXPECTED_RETRIEVER_PARAMETERS_SHA256:
        problems.append(
            "IndexManifest parameters digest is "
            f"{expected_retriever.parameters_sha256}, expected the frozen Selector-v2 digest "
            f"{EXPECTED_RETRIEVER_PARAMETERS_SHA256}"
        )
    frozen_retriever = RetrieverProvenance(
        name=EXPECTED_RETRIEVER_NAME,
        implementation_version=EXPECTED_RETRIEVER_VERSION,
        parameters_sha256=EXPECTED_RETRIEVER_PARAMETERS_SHA256,
    )
    if manifest.retriever != frozen_retriever:
        problems.append("manifest retriever does not equal the frozen Hybrid-v1 parameter identity")
    if expected_retriever != frozen_retriever:
        problems.append("index-derived retriever identity does not equal the frozen Selector-v2 identity")

    if metadata.artifact_filename != CANDIDATE_FILE:
        problems.append(
            f"candidate metadata names {metadata.artifact_filename!r}, expected {CANDIDATE_FILE!r}"
        )
    if metadata.content_sha256 != evidence.candidate_sha256:
        problems.append("candidate metadata content_sha256 does not match candidate_sets.jsonl")
    if metadata.dataset_signature != dataset.signature:
        problems.append("candidate metadata dataset_signature does not match the dataset")
    if metadata.corpus_signature != corpus.manifest.corpus_signature:
        problems.append("candidate metadata corpus_signature does not match the signed corpus")
    if metadata.run_manifest_signature != run_manifest_signature(run):
        problems.append("candidate metadata run_manifest_signature does not match RunManifest")
    if metadata.producer != "ExperimentWorkflow" or metadata.stage != "retriever":
        problems.append(
            f"candidate metadata producer/stage is {metadata.producer!r}/{metadata.stage!r}, "
            "expected 'ExperimentWorkflow'/'retriever'"
        )
    producer = metadata.producer_provenance
    if producer.git_commit != run.git_commit:
        problems.append("candidate producer git_commit does not match RunManifest")
    if producer.git_dirty != run.git_dirty:
        problems.append("candidate producer git_dirty does not match RunManifest")
    if producer.source_tree_signature != run.source_tree_signature:
        problems.append("candidate producer source_tree_signature does not match RunManifest")
    if producer.module != run.retriever:
        problems.append("candidate producer retriever config does not match RunManifest")
    if producer.execution.name != "retriever" or producer.execution.parameters != {
        "top_k": EXPECTED_TOP_N
    }:
        problems.append(
            "candidate producer execution must be exactly retriever(top_k=20); "
            f"found {producer.execution.model_dump(mode='json')!r}"
        )
    expected_upstream = {
        CORPUS_SNAPSHOT_FILE: evidence.corpus_snapshot_bundle_sha256,
        QUERIES_FILE: evidence.pool_queries_bundle_sha256,
    }
    if metadata.upstream_artifact_hashes != expected_upstream:
        problems.append(
            "candidate metadata upstream hashes must point exactly at the pool-root "
            f"corpus/query artifacts: expected {expected_upstream}, "
            f"found {metadata.upstream_artifact_hashes}"
        )

    expected_query_ids = tuple(query.query_id for query in dataset.queries)
    if evidence.pool_queries != dataset.queries:
        problems.append(
            "pool-root queries.jsonl content/order does not exactly match the supplied dataset queries"
        )
    found_query_ids = tuple(candidate_set.query_id for candidate_set in evidence.candidate_sets)
    if len(found_query_ids) != len(set(found_query_ids)):
        problems.append("candidate pool contains duplicate query IDs")
    if found_query_ids != expected_query_ids:
        missing = sorted(set(expected_query_ids) - set(found_query_ids))[:5]
        extra = sorted(set(found_query_ids) - set(expected_query_ids))[:5]
        problems.append(
            "candidate query order/completeness does not exactly match dataset queries: "
            f"missing={missing}, extra={extra}, ordered_match=False"
        )
    if manifest.n_queries != len(evidence.candidate_sets):
        problems.append(
            f"manifest n_queries={manifest.n_queries}, pool has {len(evidence.candidate_sets)}"
        )
    pinned_query_ids = tuple(window.query_id for window in manifest.windows)
    if pinned_query_ids != expected_query_ids:
        problems.append("manifest window order/completeness does not exactly match dataset queries")

    actual_window_digests = tuple(
        SelectorCandidateWindowPinV2(
            query_id=candidate_set.query_id,
            candidate_set_sha256=candidate_set_sha256(candidate_set),
        )
        for candidate_set in evidence.candidate_sets
    )
    if manifest.windows != actual_window_digests:
        problems.append("one or more canonical per-query CandidateSet hashes do not match")

    producers = {candidate_set.retriever for candidate_set in evidence.candidate_sets}
    if None in producers:
        problems.append("one or more CandidateSet windows name no retriever")
    named_producers = {item for item in producers if item is not None}
    if len(named_producers) != 1:
        problems.append(
            f"candidate pool names {len(named_producers)} distinct non-null retriever identities"
        )
    elif next(iter(named_producers)) != frozen_retriever:
        problems.append("CandidateSet retriever identity does not match the frozen index-derived one")

    for candidate_set in evidence.candidate_sets:
        if len(candidate_set.candidates) != EXPECTED_TOP_N:
            problems.append(
                f"{candidate_set.query_id}: {len(candidate_set.candidates)} candidates, expected 20"
            )
        ranks = tuple(candidate.retrieval_rank for candidate in candidate_set.candidates)
        if ranks != tuple(range(1, EXPECTED_TOP_N + 1)):
            problems.append(
                f"{candidate_set.query_id}: candidate tuple ranks are {ranks[:5]}..., expected "
                "ordered consecutive ranks 1..20"
            )

    problems.extend(_candidate_chunk_problems(evidence.candidate_sets, corpus))
    return tuple(problems)


def _raise_on_problems(problems: Sequence[str]) -> None:
    if not problems:
        return
    detail = "\n".join(f"- {problem}" for problem in problems)
    raise ValueError(f"SelectorCandidatePoolManifestV2 verification failed:\n{detail}")


def build_selector_candidate_pool_manifest_v2(
    pool_directory: Path,
    dataset_manifest_path: Path,
    *,
    data_role: DataRole,
    recovery_status: RecoveryStatus,
) -> SelectorCandidatePoolManifestV2:
    """Validate recovered artifacts and construct, but do not yet write, their freeze."""

    evidence = _load_pool_evidence(pool_directory, dataset_manifest_path)
    retriever = RetrieverProvenance(
        name=EXPECTED_RETRIEVER_NAME,
        implementation_version=EXPECTED_RETRIEVER_VERSION,
        parameters_sha256=EXPECTED_RETRIEVER_PARAMETERS_SHA256,
    )
    manifest = SelectorCandidatePoolManifestV2(
        recovery_status=recovery_status,
        data_role=data_role,
        candidate_sha256=evidence.candidate_sha256,
        candidate_metadata_sha256=evidence.candidate_metadata_sha256,
        run_manifest_sha256=evidence.run_manifest_sha256,
        corpus_snapshot_sha256=evidence.corpus_snapshot_sha256,
        pool_queries_sha256=evidence.pool_queries_sha256,
        index_manifest_sha256=evidence.index_manifest_sha256,
        index_corpus_snapshot_sha256=evidence.index_corpus_snapshot_sha256,
        dataset_manifest_sha256=evidence.dataset.manifest_sha256,
        dataset_documents_sha256=evidence.dataset.documents_sha256,
        dataset_queries_sha256=evidence.dataset.queries_sha256,
        dataset_gold_cases_sha256=evidence.dataset.gold_cases_sha256,
        dataset_id=evidence.dataset.manifest.dataset_id,
        dataset_version=evidence.dataset.manifest.dataset_version,
        dataset_split=evidence.dataset.manifest.split,
        dataset_signature=evidence.dataset.signature,
        query_signature=evidence.dataset.query_signature,
        corpus_signature=evidence.corpus.manifest.corpus_signature,
        index_signature=evidence.index_manifest.index_signature,
        retriever=retriever,
        n_queries=len(evidence.candidate_sets),
        windows=tuple(
            SelectorCandidateWindowPinV2(
                query_id=candidate_set.query_id,
                candidate_set_sha256=candidate_set_sha256(candidate_set),
            )
            for candidate_set in evidence.candidate_sets
        ),
    )
    _raise_on_problems(_verification_problems(manifest, evidence))
    return manifest


def freeze_selector_candidate_pool_manifest_v2(
    pool_directory: Path, manifest: SelectorCandidatePoolManifestV2
) -> Path:
    """Write the v2 freeze once; replacing it requires an explicit external action."""

    directory = Path(pool_directory)
    if not directory.is_dir():
        raise ValueError(f"selector candidate pool directory does not exist: {directory}")
    path = directory / SELECTOR_POOL_MANIFEST_FILE
    if path.exists():
        raise ValueError(
            f"{path} is already frozen; refusing to replace a Selector-v2 candidate pool manifest"
        )
    payload = json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    path.write_bytes((payload + "\n").encode("utf-8"))
    return path


def read_selector_candidate_pool_manifest_v2(
    pool_directory: Path,
) -> SelectorCandidatePoolManifestV2:
    path = Path(pool_directory) / SELECTOR_POOL_MANIFEST_FILE
    if not path.is_file():
        raise ValueError(f"missing Selector-v2 candidate pool manifest: {path}")
    return _read_model(path, SelectorCandidatePoolManifestV2)


def verify_selector_candidate_pool_v2(
    pool_directory: Path, dataset_manifest_path: Path
) -> SelectorCandidatePoolManifestV2:
    """Re-read and verify every frozen dependency before any Selector scorer runs."""

    manifest = read_selector_candidate_pool_manifest_v2(pool_directory)
    evidence = _load_pool_evidence(pool_directory, dataset_manifest_path)
    _raise_on_problems(_verification_problems(manifest, evidence))
    return manifest
