import json
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from typing import Annotated, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from evidence_rag.contracts.models import Document, Query

NonEmpty = Annotated[str, Field(min_length=1)]
ModelT = TypeVar("ModelT", bound=BaseModel)


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class GoldCase(FrozenModel):
    query_id: NonEmpty
    relevant_document_ids: tuple[NonEmpty, ...] | None = None
    reference_answers: tuple[NonEmpty, ...] | None = None

    @field_validator("relevant_document_ids", "reference_answers", mode="before")
    @classmethod
    def normalize_legacy_lists(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("query_id")
    @classmethod
    def normalize_query_id(cls, value: str) -> str:
        return _required_boundary_value(value, "query ID")

    @field_validator("relevant_document_ids")
    @classmethod
    def normalize_relevant_document_ids(
        cls, value: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        if value is None:
            return None
        return tuple(
            _required_boundary_value(item, "relevant document ID") for item in value
        )

    @field_validator("reference_answers")
    @classmethod
    def normalize_reference_answers(
        cls, value: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        if value is None:
            return None
        return tuple(_required_boundary_value(item, "reference answer") for item in value)

    @model_validator(mode="after")
    def labels_are_unique(self) -> "GoldCase":
        for values in (self.relevant_document_ids, self.reference_answers):
            if values is not None and len(values) != len(set(values)):
                raise ValueError("gold labels must be unique")
        return self


class DatasetManifest(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset_id: NonEmpty
    dataset_version: NonEmpty
    split: NonEmpty
    documents_file: NonEmpty
    queries_file: NonEmpty
    gold_cases_file: NonEmpty

    @field_validator("dataset_id", "dataset_version", "split")
    @classmethod
    def normalize_dataset_identity(cls, value: str, info: ValidationInfo) -> str:
        labels = {
            "dataset_id": "dataset ID",
            "dataset_version": "dataset version",
            "split": "dataset split",
        }
        field_name = info.field_name
        if field_name is None:
            raise ValueError("dataset identity field is unavailable")
        return _required_boundary_value(value, labels[field_name])

    @field_validator("documents_file", "queries_file", "gold_cases_file")
    @classmethod
    def data_files_must_be_relative(cls, value: str) -> str:
        if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
            raise ValueError("dataset data files must be relative")
        return value


class DatasetBundle(FrozenModel):
    manifest: DatasetManifest
    dataset_signature: NonEmpty
    documents: tuple[Document, ...]
    queries: tuple[Query, ...]
    gold_cases: tuple[GoldCase, ...]


def _canonical_signature(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _required_boundary_value(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must not be blank")
    return normalized


def normalize_document(document: Document) -> Document:
    return Document(
        schema_version=document.schema_version,
        document_id=_required_boundary_value(document.document_id, "document ID"),
        text=_required_boundary_value(document.text, "document text"),
        source_uri=_required_boundary_value(document.source_uri, "document source URI"),
    )


def normalize_query(query: Query) -> Query:
    return Query(
        schema_version=query.schema_version,
        query_id=_required_boundary_value(query.query_id, "query ID"),
        text=_required_boundary_value(query.text, "query text"),
    )


def _load_jsonl(
    path: Path,
    model_type: type[ModelT],
    normalize: Callable[[ModelT], ModelT] | None = None,
) -> tuple[ModelT, ...]:
    records: list[ModelT] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"unable to read JSONL file {path}: {error}") from error

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = model_type.model_validate_json(line)
            if normalize is not None:
                record = normalize(record)
        except (ValidationError, ValueError) as error:
            raise ValueError(f"invalid JSONL record at {path}:{line_number}: {error}") from error
        records.append(record)
    return tuple(records)


def _validate_unique_ids(
    records: tuple[ModelT, ...],
    identifier: Callable[[ModelT], str],
    label: str,
) -> None:
    seen: set[str] = set()
    for record in records:
        value = identifier(record)
        if value in seen:
            raise ValueError(f"duplicate {label} ID: {value}")
        seen.add(value)


class JsonlDatasetAdapter:
    @staticmethod
    def load(manifest_path: Path) -> DatasetBundle:
        manifest_path = Path(manifest_path)
        try:
            manifest = DatasetManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError, ValueError) as error:
            raise ValueError(f"invalid dataset manifest {manifest_path}: {error}") from error

        root = manifest_path.parent
        documents = _load_jsonl(root / manifest.documents_file, Document, normalize_document)
        queries = _load_jsonl(root / manifest.queries_file, Query, normalize_query)
        gold_cases = _load_jsonl(root / manifest.gold_cases_file, GoldCase)

        _validate_unique_ids(documents, lambda document: document.document_id, "document")
        _validate_unique_ids(queries, lambda query: query.query_id, "query")
        _validate_unique_ids(gold_cases, lambda gold_case: gold_case.query_id, "gold query")

        document_ids = {document.document_id for document in documents}
        query_ids = {query.query_id for query in queries}
        for gold_case in gold_cases:
            if gold_case.query_id not in query_ids:
                raise ValueError(f"gold case references unknown query ID: {gold_case.query_id}")
            for document_id in gold_case.relevant_document_ids or ():
                if document_id not in document_ids:
                    raise ValueError(f"unknown relevant document ID: {document_id}")

        signature = _canonical_signature(
            {
                "dataset": {
                    "schema_version": manifest.schema_version,
                    "dataset_id": manifest.dataset_id,
                    "dataset_version": manifest.dataset_version,
                    "split": manifest.split,
                },
                "documents": [document.model_dump(mode="json") for document in documents],
                "queries": [query.model_dump(mode="json") for query in queries],
                "gold_cases": [gold_case.model_dump(mode="json") for gold_case in gold_cases],
            }
        )
        return DatasetBundle(
            manifest=manifest,
            dataset_signature=signature,
            documents=documents,
            queries=queries,
            gold_cases=gold_cases,
        )
