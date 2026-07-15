import json
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, ValidationError

from evidence_rag.contracts.protocols import Retriever
from evidence_rag.infrastructure.corpus import CorpusSnapshot
from evidence_rag.retriever.bm25 import BM25Retriever, validate_bm25_parameters

NonEmpty = Annotated[str, Field(min_length=1)]
ModelT = TypeVar("ModelT", bound=BaseModel)
SNAPSHOT_FILENAME: Literal["corpus_snapshot.json"] = "corpus_snapshot.json"
MANIFEST_FILENAME: Literal["index_manifest.json"] = "index_manifest.json"


class IndexManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["1.0"]
    implementation: Literal["bm25"]
    implementation_version: Literal["bm25-v1"]
    corpus_signature: NonEmpty
    k1: Annotated[FiniteFloat, Field(ge=0)]
    b: Annotated[FiniteFloat, Field(ge=0, le=1)]
    snapshot_filename: Literal["corpus_snapshot.json"]
    index_signature: NonEmpty


class IndexPlugin(Protocol):
    def build(
        self,
        corpus: CorpusSnapshot,
        directory: Path,
        *,
        k1: float,
        b: float,
    ) -> Retriever: ...

    def load(
        self,
        directory: Path,
        *,
        expected_corpus_signature: str,
        expected_k1: float,
        expected_b: float,
    ) -> Retriever: ...


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


def _index_signature(corpus: CorpusSnapshot, *, k1: float, b: float) -> str:
    payload = {
        "schema_version": "1.0",
        "implementation": "bm25",
        "implementation_version": "bm25-v1",
        "corpus_signature": corpus.manifest.corpus_signature,
        "k1": k1,
        "b": b,
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


class BM25IndexPlugin:
    def build(
        self,
        corpus: CorpusSnapshot,
        directory: Path,
        *,
        k1: float,
        b: float,
    ) -> Retriever:
        k1, b = validate_bm25_parameters(k1, b)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = IndexManifest(
            schema_version="1.0",
            implementation="bm25",
            implementation_version="bm25-v1",
            corpus_signature=corpus.manifest.corpus_signature,
            k1=k1,
            b=b,
            snapshot_filename=SNAPSHOT_FILENAME,
            index_signature=_index_signature(corpus, k1=k1, b=b),
        )
        (directory / SNAPSHOT_FILENAME).write_text(
            _canonical_json(corpus) + "\n",
            encoding="utf-8",
        )
        (directory / MANIFEST_FILENAME).write_text(
            _canonical_json(manifest) + "\n",
            encoding="utf-8",
        )
        return BM25Retriever.from_corpus(corpus, k1=k1, b=b)

    def load(
        self,
        directory: Path,
        *,
        expected_corpus_signature: str,
        expected_k1: float,
        expected_b: float,
    ) -> Retriever:
        expected_k1, expected_b = validate_bm25_parameters(expected_k1, expected_b)
        directory = Path(directory)
        manifest_path = directory / MANIFEST_FILENAME
        manifest = _read_model(manifest_path, IndexManifest)
        if manifest.corpus_signature != expected_corpus_signature:
            raise ValueError(
                f"corpus signature mismatch in {manifest_path}: "
                f"expected {expected_corpus_signature}, found {manifest.corpus_signature}"
            )
        for name, expected, found in (
            ("k1", expected_k1, manifest.k1),
            ("b", expected_b, manifest.b),
        ):
            if found != expected:
                raise ValueError(
                    f"{name} mismatch in {manifest_path}: expected {expected}, found {found}"
                )
        snapshot_path = directory / SNAPSHOT_FILENAME
        corpus = _read_model(snapshot_path, CorpusSnapshot)
        if corpus.manifest.corpus_signature != manifest.corpus_signature:
            raise ValueError(
                f"corpus signature mismatch in {snapshot_path}: "
                f"expected {manifest.corpus_signature}, "
                f"found {corpus.manifest.corpus_signature}"
            )
        expected_index_signature = _index_signature(
            corpus,
            k1=manifest.k1,
            b=manifest.b,
        )
        if manifest.index_signature != expected_index_signature:
            raise ValueError(
                f"index signature mismatch in {manifest_path}: "
                f"expected {expected_index_signature}, found {manifest.index_signature}"
            )
        return BM25Retriever.from_corpus(corpus, k1=manifest.k1, b=manifest.b)
