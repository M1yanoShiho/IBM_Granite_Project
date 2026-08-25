"""Deterministic, signature-verified persistence for a retriever index.

An "index" here is the corpus snapshot plus the fully-normalised retriever
configuration. v1 persists no numeric structures (BM25 tables, dense vectors are
rebuilt on load), so the manifest captures *identity* — which retriever
implementation, at which version, with which parameters, over which corpus — and a
SHA-256 ``index_signature`` binding all of it together. Concrete retriever
construction lives in :mod:`evidence_rag.composition`; this module is
implementation-agnostic.
"""

import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from evidence_rag.infrastructure.corpus import CorpusSnapshot

NonEmpty = Annotated[str, Field(min_length=1)]
ModelT = TypeVar("ModelT", bound=BaseModel)
SNAPSHOT_FILENAME: Literal["corpus_snapshot.json"] = "corpus_snapshot.json"
MANIFEST_FILENAME: Literal["index_manifest.json"] = "index_manifest.json"
SCHEMA_VERSION: Literal["1.0"] = "1.0"


class IndexManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["1.0"]
    implementation: NonEmpty
    implementation_version: NonEmpty
    corpus_signature: NonEmpty
    parameters: dict[str, Any]
    snapshot_filename: Literal["corpus_snapshot.json"]
    index_signature: NonEmpty


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


def _index_signature(
    corpus: CorpusSnapshot,
    *,
    implementation: str,
    implementation_version: str,
    parameters: Mapping[str, Any],
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "implementation": implementation,
        "implementation_version": implementation_version,
        "corpus_signature": corpus.manifest.corpus_signature,
        "parameters": dict(parameters),
        "snapshot_filename": SNAPSHOT_FILENAME,
        "corpus_snapshot": corpus.model_dump(mode="json"),
    }
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _read_model(path: Path, model_type: type[ModelT]) -> ModelT:
    try:
        return model_type.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"unable to read index file {path}: {error}") from error
    except (ValidationError, ValueError) as error:
        raise ValueError(f"invalid index file {path}: {error}") from error


def read_index_manifest(directory: Path) -> IndexManifest:
    return _read_model(Path(directory) / MANIFEST_FILENAME, IndexManifest)


def write_index(
    corpus: CorpusSnapshot,
    directory: Path,
    *,
    implementation: str,
    implementation_version: str,
    parameters: Mapping[str, Any],
) -> IndexManifest:
    """Persist the corpus snapshot and a signed manifest; return the manifest."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = IndexManifest(
        schema_version=SCHEMA_VERSION,
        implementation=implementation,
        implementation_version=implementation_version,
        corpus_signature=corpus.manifest.corpus_signature,
        parameters=dict(parameters),
        snapshot_filename=SNAPSHOT_FILENAME,
        index_signature=_index_signature(
            corpus,
            implementation=implementation,
            implementation_version=implementation_version,
            parameters=parameters,
        ),
    )
    # write_bytes, not write_text: text mode translates "\n" to os.linesep, so the same index
    # persisted on Windows and on Linux differs by one byte per file. That is not cosmetic here
    # -- `ExperimentWorkflow` records the sha256 of these files as `upstream_artifact_hashes`
    # and rejects a run whose index no longer hashes to what was stored, so a platform-dependent
    # byte makes a platform-dependent guard. The module docstring's "deterministic" has to mean
    # deterministic across machines, not just across runs on one. Linux output is unchanged, so
    # every index already built on the cluster keeps its recorded hash.
    (directory / SNAPSHOT_FILENAME).write_bytes((_canonical_json(corpus) + "\n").encode("utf-8"))
    (directory / MANIFEST_FILENAME).write_bytes(
        (_canonical_json(manifest) + "\n").encode("utf-8")
    )
    return manifest


def load_index(
    directory: Path,
    *,
    expected_corpus_signature: str,
    expected_implementation: str,
    expected_implementation_version: str,
    expected_parameters: Mapping[str, Any],
) -> CorpusSnapshot:
    """Validate a persisted index against expectations and return its snapshot.

    Raises ``ValueError`` on any mismatch (corpus signature, implementation,
    implementation version, parameters, or a tampered index signature), so the
    caller can safely reconstruct the retriever from the returned snapshot.
    """

    directory = Path(directory)
    manifest_path = directory / MANIFEST_FILENAME
    manifest = _read_model(manifest_path, IndexManifest)
    expected = dict(expected_parameters)
    for name, want, found in (
        ("corpus signature", expected_corpus_signature, manifest.corpus_signature),
        ("implementation", expected_implementation, manifest.implementation),
        (
            "implementation_version",
            expected_implementation_version,
            manifest.implementation_version,
        ),
    ):
        if found != want:
            raise ValueError(
                f"{name} mismatch in {manifest_path}: expected {want}, found {found}"
            )
    if manifest.parameters != expected:
        raise ValueError(
            f"parameters mismatch in {manifest_path}: "
            f"expected {expected}, found {manifest.parameters}"
        )
    snapshot_path = directory / SNAPSHOT_FILENAME
    corpus = _read_model(snapshot_path, CorpusSnapshot)
    if corpus.manifest.corpus_signature != manifest.corpus_signature:
        raise ValueError(
            f"corpus signature mismatch in {snapshot_path}: "
            f"expected {manifest.corpus_signature}, "
            f"found {corpus.manifest.corpus_signature}"
        )
    expected_signature = _index_signature(
        corpus,
        implementation=manifest.implementation,
        implementation_version=manifest.implementation_version,
        parameters=manifest.parameters,
    )
    if manifest.index_signature != expected_signature:
        raise ValueError(
            f"index signature mismatch in {manifest_path}: "
            f"expected {expected_signature}, found {manifest.index_signature}"
        )
    return corpus
