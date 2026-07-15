import json
from collections.abc import Iterable, Mapping
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from typing import Annotated, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from evidence_rag.infrastructure.config import ModuleConfig

NonEmpty = Annotated[str, Field(min_length=1)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PositiveInteger = Annotated[int, Field(gt=0)]
NonNegativeInteger = Annotated[int, Field(ge=0)]
ModelT = TypeVar("ModelT", bound=BaseModel)


def _validate_artifact_filename(filename: str) -> str:
    path = Path(filename)
    windows_path = PureWindowsPath(filename)
    if (
        not filename
        or "\\" in filename
        or path.is_absolute()
        or windows_path.is_absolute()
        or path.as_posix() != filename
        or path == Path(".")
        or ".." in path.parts
        or ".." in windows_path.parts
    ):
        raise ValueError(f"artifact filename must be relative and traversal-free: {filename}")
    return filename


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class RunManifest(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    contract_version: Literal["1.0"] = "1.0"
    dataset_id: NonEmpty
    dataset_version: NonEmpty
    split: NonEmpty
    dataset_signature: NonEmpty
    corpus_signature: NonEmpty
    chunker_name: NonEmpty
    chunker_version: NonEmpty
    chunk_size: PositiveInteger
    overlap: NonNegativeInteger
    index_implementation: Literal["bm25"]
    index_implementation_version: Literal["bm25-v1"]
    index_signature: Digest
    retriever: ModuleConfig
    selector: ModuleConfig
    generator: ModuleConfig
    top_k: PositiveInteger
    max_selected: PositiveInteger
    seed: int
    git_commit: NonEmpty
    git_dirty: bool
    source_tree_signature: Digest

    def validate_compatibility(
        self,
        expected_dataset_signature: str | None = None,
        expected_corpus_signature: str | None = None,
    ) -> None:
        if (
            expected_dataset_signature is not None
            and self.dataset_signature != expected_dataset_signature
        ):
            raise ValueError(
                "dataset signature mismatch: "
                f"expected {expected_dataset_signature}, found {self.dataset_signature}"
            )
        if (
            expected_corpus_signature is not None
            and self.corpus_signature != expected_corpus_signature
        ):
            raise ValueError(
                "corpus signature mismatch: "
                f"expected {expected_corpus_signature}, found {self.corpus_signature}"
            )


class ProducerProvenance(FrozenModel):
    git_commit: NonEmpty
    git_dirty: bool
    source_tree_signature: Digest
    module: ModuleConfig
    execution: ModuleConfig


class ArtifactMetadata(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_filename: NonEmpty
    content_sha256: Digest
    dataset_signature: NonEmpty
    corpus_signature: NonEmpty
    run_manifest_signature: Digest
    producer: NonEmpty
    stage: NonEmpty
    producer_provenance: ProducerProvenance
    upstream_artifact_hashes: dict[str, Digest]

    @field_validator("artifact_filename")
    @classmethod
    def artifact_filename_must_be_safe(cls, value: str) -> str:
        return _validate_artifact_filename(value)

    @field_validator("upstream_artifact_hashes")
    @classmethod
    def upstream_filenames_must_be_safe(cls, value: dict[str, str]) -> dict[str, str]:
        for filename in value:
            _validate_artifact_filename(filename)
        return value


def _canonical_json(model: BaseModel) -> str:
    return json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_bytes(model: BaseModel) -> bytes:
    return f"{_canonical_json(model)}\n".encode()


def run_manifest_signature(manifest: RunManifest) -> str:
    return sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()


def _artifact_bundle_hash(content: bytes, metadata: ArtifactMetadata) -> str:
    return sha256(content + b"\0" + _canonical_bytes(metadata)).hexdigest()


def metadata_filename(filename: str) -> str:
    _validate_artifact_filename(filename)
    return f"{filename}.metadata.json"


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_manifest(
        self,
        manifest: RunManifest,
        *,
        producer: str = "ExperimentWorkflow",
        stage: str = "prepare",
        producer_provenance: ProducerProvenance | None = None,
        upstream_artifact_hashes: Mapping[str, str] | None = None,
    ) -> None:
        self._write_bound_artifact(
            "run_manifest.json",
            _canonical_bytes(manifest),
            manifest,
            producer=producer,
            stage=stage,
            producer_provenance=producer_provenance,
            upstream_artifact_hashes=upstream_artifact_hashes or {},
        )

    def read_manifest(
        self,
        *,
        expected_dataset_signature: str | None = None,
        expected_corpus_signature: str | None = None,
        expected_upstream_artifact_hashes: Mapping[str, str] | None = None,
    ) -> RunManifest:
        filename = "run_manifest.json"
        path = self._path_for(filename)
        metadata = self._read_metadata(filename)
        self._validate_metadata_filename(metadata, filename)
        content = self._read_content(path)
        self._validate_content_hash(metadata, content, path)
        manifest = self._parse_json(path, content, RunManifest)
        self._validate_metadata_manifest(metadata, manifest)
        if (
            expected_upstream_artifact_hashes is not None
            and metadata.upstream_artifact_hashes
            != dict(expected_upstream_artifact_hashes)
        ):
            raise ValueError(
                f"upstream artifact hashes mismatch for {path}: "
                f"expected {dict(expected_upstream_artifact_hashes)}, "
                f"found {metadata.upstream_artifact_hashes}"
            )
        manifest.validate_compatibility(
            expected_dataset_signature=expected_dataset_signature,
            expected_corpus_signature=expected_corpus_signature,
        )
        return manifest

    def write_jsonl(
        self,
        filename: str,
        records: Iterable[BaseModel],
        *,
        producer: str = "ArtifactStore",
        stage: str = "unspecified",
        producer_provenance: ProducerProvenance | None = None,
        upstream_artifact_hashes: Mapping[str, str] | None = None,
    ) -> None:
        _validate_artifact_filename(filename)
        manifest = self.read_manifest()
        content = "".join(f"{_canonical_json(record)}\n" for record in records).encode("utf-8")
        self._write_bound_artifact(
            filename,
            content,
            manifest,
            producer=producer,
            stage=stage,
            producer_provenance=producer_provenance,
            upstream_artifact_hashes=upstream_artifact_hashes or {},
        )

    def read_jsonl(
        self,
        filename: str,
        model_type: type[ModelT],
        *,
        expected_dataset_signature: str,
        expected_corpus_signature: str,
        expected_producer: str | None = None,
        expected_stage: str | None = None,
        expected_upstream_artifact_hashes: Mapping[str, str] | None = None,
    ) -> tuple[ModelT, ...]:
        _validate_artifact_filename(filename)
        manifest = self.read_manifest(
            expected_dataset_signature=expected_dataset_signature,
            expected_corpus_signature=expected_corpus_signature,
        )
        path, content = self._read_verified_artifact(
            filename,
            manifest,
            expected_producer=expected_producer,
            expected_stage=expected_stage,
            expected_upstream_artifact_hashes=expected_upstream_artifact_hashes,
        )

        records: list[ModelT] = []
        for line_number, raw_line in enumerate(content.splitlines(), start=1):
            if not raw_line.strip():
                continue
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"invalid UTF-8 JSONL record at {path}:{line_number}: {error}"
                ) from error
            try:
                records.append(model_type.model_validate_json(line))
            except (ValidationError, ValueError) as error:
                raise ValueError(
                    f"invalid JSONL record at {path}:{line_number}: {error}"
                ) from error
        return tuple(records)

    def write_json(
        self,
        filename: str,
        model: BaseModel,
        *,
        producer: str = "ArtifactStore",
        stage: str = "unspecified",
        producer_provenance: ProducerProvenance | None = None,
        upstream_artifact_hashes: Mapping[str, str] | None = None,
    ) -> None:
        _validate_artifact_filename(filename)
        manifest = self.read_manifest()
        self._write_bound_artifact(
            filename,
            _canonical_bytes(model),
            manifest,
            producer=producer,
            stage=stage,
            producer_provenance=producer_provenance,
            upstream_artifact_hashes=upstream_artifact_hashes or {},
        )

    def read_json(
        self,
        filename: str,
        model_type: type[ModelT],
        *,
        expected_dataset_signature: str,
        expected_corpus_signature: str,
        expected_producer: str | None = None,
        expected_stage: str | None = None,
        expected_upstream_artifact_hashes: Mapping[str, str] | None = None,
    ) -> ModelT:
        _validate_artifact_filename(filename)
        manifest = self.read_manifest(
            expected_dataset_signature=expected_dataset_signature,
            expected_corpus_signature=expected_corpus_signature,
        )
        path, content = self._read_verified_artifact(
            filename,
            manifest,
            expected_producer=expected_producer,
            expected_stage=expected_stage,
            expected_upstream_artifact_hashes=expected_upstream_artifact_hashes,
        )
        return self._parse_json(path, content, model_type)

    def artifact_hash(self, filename: str) -> str:
        manifest = self.read_manifest()
        _, _, _, bundle_hash = self._read_bound_artifact(
            filename,
            manifest,
            ancestors=frozenset(),
        )
        return bundle_hash

    def _write_bound_artifact(
        self,
        filename: str,
        content: bytes,
        manifest: RunManifest,
        *,
        producer: str,
        stage: str,
        producer_provenance: ProducerProvenance | None,
        upstream_artifact_hashes: Mapping[str, str],
    ) -> None:
        path = self._path_for(filename)
        metadata_path = self._path_for(metadata_filename(filename))
        metadata = ArtifactMetadata(
            artifact_filename=filename,
            content_sha256=sha256(content).hexdigest(),
            dataset_signature=manifest.dataset_signature,
            corpus_signature=manifest.corpus_signature,
            run_manifest_signature=run_manifest_signature(manifest),
            producer=producer,
            stage=stage,
            producer_provenance=(
                producer_provenance
                if producer_provenance is not None
                else ProducerProvenance(
                    git_commit=manifest.git_commit,
                    git_dirty=manifest.git_dirty,
                    source_tree_signature=manifest.source_tree_signature,
                    module=ModuleConfig(name=producer),
                    execution=ModuleConfig(name=stage),
                )
            ),
            upstream_artifact_hashes=dict(upstream_artifact_hashes),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        metadata_path.write_bytes(_canonical_bytes(metadata))

    def _read_verified_artifact(
        self,
        filename: str,
        manifest: RunManifest,
        *,
        expected_producer: str | None = None,
        expected_stage: str | None = None,
        expected_upstream_artifact_hashes: Mapping[str, str] | None = None,
    ) -> tuple[Path, bytes]:
        path, content, metadata, _ = self._read_bound_artifact(
            filename,
            manifest,
            ancestors=frozenset(),
        )
        if expected_producer is not None and metadata.producer != expected_producer:
            raise ValueError(
                f"artifact producer mismatch for {path}: expected {expected_producer}, "
                f"found {metadata.producer}"
            )
        if expected_stage is not None and metadata.stage != expected_stage:
            raise ValueError(
                f"artifact stage mismatch for {path}: expected {expected_stage}, "
                f"found {metadata.stage}"
            )
        if (
            expected_upstream_artifact_hashes is not None
            and metadata.upstream_artifact_hashes != dict(expected_upstream_artifact_hashes)
        ):
            raise ValueError(
                f"upstream artifact hashes mismatch for {path}: "
                f"expected {dict(expected_upstream_artifact_hashes)}, "
                f"found {metadata.upstream_artifact_hashes}"
            )
        return path, content

    def _read_bound_artifact(
        self,
        filename: str,
        manifest: RunManifest,
        *,
        ancestors: frozenset[str],
    ) -> tuple[Path, bytes, ArtifactMetadata, str]:
        if filename in ancestors:
            raise ValueError(f"artifact provenance cycle detected at {filename}")
        path = self._path_for(filename)
        metadata = self._read_metadata(filename)
        self._validate_metadata_filename(metadata, filename)
        self._validate_metadata_manifest(metadata, manifest)
        content = self._read_content(path)
        self._validate_content_hash(metadata, content, path)

        next_ancestors = ancestors | {filename}
        for upstream_filename, expected_hash in metadata.upstream_artifact_hashes.items():
            _, _, _, found_hash = self._read_bound_artifact(
                upstream_filename,
                manifest,
                ancestors=next_ancestors,
            )
            if expected_hash != found_hash:
                raise ValueError(
                    f"upstream artifact hash mismatch for {path}: "
                    f"{upstream_filename} expected {expected_hash}, found {found_hash}"
                )
        return path, content, metadata, _artifact_bundle_hash(content, metadata)

    def _read_metadata(self, filename: str) -> ArtifactMetadata:
        path = self._path_for(metadata_filename(filename))
        try:
            content = path.read_bytes()
        except OSError as error:
            raise ValueError(f"unable to read artifact metadata file {path}: {error}") from error
        return self._parse_metadata(path, content)

    @staticmethod
    def _parse_metadata(path: Path, content: bytes) -> ArtifactMetadata:
        try:
            return ArtifactMetadata.model_validate_json(content)
        except (ValidationError, ValueError) as error:
            raise ValueError(f"invalid artifact metadata file {path}: {error}") from error

    @staticmethod
    def _parse_json(
        path: Path,
        content: bytes,
        model_type: type[ModelT],
    ) -> ModelT:
        try:
            return model_type.model_validate_json(content)
        except (ValidationError, ValueError) as error:
            raise ValueError(f"invalid JSON file {path}: {error}") from error

    @staticmethod
    def _read_content(path: Path) -> bytes:
        try:
            return path.read_bytes()
        except OSError as error:
            raise ValueError(f"unable to read artifact file {path}: {error}") from error

    @staticmethod
    def _validate_metadata_filename(
        metadata: ArtifactMetadata,
        expected_filename: str,
    ) -> None:
        if metadata.artifact_filename != expected_filename:
            raise ValueError(
                "artifact filename mismatch in metadata: "
                f"expected {expected_filename}, found {metadata.artifact_filename}"
            )

    @staticmethod
    def _validate_content_hash(
        metadata: ArtifactMetadata,
        content: bytes,
        path: Path,
    ) -> None:
        found = sha256(content).hexdigest()
        if metadata.content_sha256 != found:
            raise ValueError(
                f"content SHA-256 mismatch for {path}: "
                f"expected {metadata.content_sha256}, found {found}"
            )

    @staticmethod
    def _validate_metadata_manifest(
        metadata: ArtifactMetadata,
        manifest: RunManifest,
    ) -> None:
        if metadata.dataset_signature != manifest.dataset_signature:
            raise ValueError(
                "dataset signature mismatch in artifact metadata: "
                f"expected {manifest.dataset_signature}, "
                f"found {metadata.dataset_signature}"
            )
        if metadata.corpus_signature != manifest.corpus_signature:
            raise ValueError(
                "corpus signature mismatch in artifact metadata: "
                f"expected {manifest.corpus_signature}, "
                f"found {metadata.corpus_signature}"
            )
        expected_signature = run_manifest_signature(manifest)
        if metadata.run_manifest_signature != expected_signature:
            raise ValueError(
                "run manifest signature mismatch in artifact metadata: "
                f"expected {expected_signature}, "
                f"found {metadata.run_manifest_signature}"
            )

    def _path_for(self, filename: str) -> Path:
        return self.root / _validate_artifact_filename(filename)
