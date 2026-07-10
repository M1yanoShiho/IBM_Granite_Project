"""Strict, metadata-only loaders and leakage-safe splits for selector datasets."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


Identifier = str | int
_CONTRACT_SPLITS = ("train", "dev", "test")
_CONTRACT_CHOICES = ("Contradiction", "Entailment", "NotMentioned")
_RAMDOCS_TYPES = ("correct", "misinfo", "noise")


@dataclass(frozen=True)
class FinanceBenchRecord:
    """FinanceBench query metadata needed for grouped splitting."""

    financebench_id: Identifier
    company: str
    doc_name: str
    evidence_count: int


@dataclass(frozen=True)
class FinanceBenchAudit:
    """Aggregate FinanceBench source counts."""

    question_count: int
    company_count: int
    document_count: int
    evidence_count: int
    pdf_count: int


@dataclass(frozen=True)
class ContractNLIAnnotation:
    """One official ContractNLI label without hypothesis or contract text."""

    hypothesis_id: str
    choice: str
    evidence_span_indices: tuple[int, ...]


@dataclass(frozen=True)
class ContractNLIDocumentRecord:
    """Metadata and official labels for one contract."""

    document_id: Identifier
    split: str
    span_count: int
    annotations: tuple[ContractNLIAnnotation, ...]


@dataclass(frozen=True)
class ContractNLISplitAudit:
    """ContractNLI counts for one official split."""

    split: str
    document_count: int
    hypothesis_count: int
    annotation_choice_count_items: tuple[tuple[str, int], ...]
    evidence_span_count: int

    @property
    def annotation_choice_counts(self) -> dict[str, int]:
        return dict(self.annotation_choice_count_items)


@dataclass(frozen=True)
class ContractNLIAudit:
    """ContractNLI audit summaries and the common hypothesis-key set."""

    split_summaries: tuple[ContractNLISplitAudit, ...]
    hypothesis_ids: tuple[str, ...]

    def split(self, split_name: str) -> ContractNLISplitAudit:
        for summary in self.split_summaries:
            if summary.split == split_name:
                return summary
        raise KeyError(split_name)


@dataclass(frozen=True)
class RAMDocsDocumentRecord:
    """Stable metadata for one RAMDocs candidate."""

    document_id: str
    document_type: str


@dataclass(frozen=True)
class RAMDocsRecord:
    """Stable query and candidate metadata for one RAMDocs example."""

    query_id: str
    documents: tuple[RAMDocsDocumentRecord, ...]


@dataclass(frozen=True)
class RAMDocsAudit:
    """Aggregate official RAMDocs counts."""

    example_count: int
    document_count: int
    min_documents_per_query: int
    max_documents_per_query: int
    document_type_count_items: tuple[tuple[str, int], ...]

    @property
    def document_type_counts(self) -> dict[str, int]:
        return dict(self.document_type_count_items)


@dataclass(frozen=True)
class NIAHMetadataRecord:
    """The three identifiers needed to form leakage-safe NIAH components."""

    query_id: str
    parent_page_id: str
    synthetic_family_id: str


def _id_sort_key(value: Identifier) -> tuple[str, str]:
    return type(value).__name__, str(value)


def _stable_hash(seed: int, *parts: object) -> str:
    material = "\x1f".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a file's SHA-256 using bounded-memory streaming reads."""

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            while chunk := stream.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"Unable to hash file {path}: {exc}") from exc
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except UnicodeError as exc:
        raise ValueError(f"{path} is not valid UTF-8: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} contains invalid JSON: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to read required JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise ValueError(f"{path} line {line_number} must not be blank")
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path} line {line_number} contains invalid JSON: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{path} line {line_number} must contain a JSON object"
                    )
                rows.append(value)
    except UnicodeError as exc:
        raise ValueError(f"{path} is not valid UTF-8: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to read required JSONL file {path}: {exc}") from exc
    return rows


def _required(mapping: Mapping[str, object], field: str, context: str) -> object:
    if field not in mapping:
        raise ValueError(f"{context} is missing required field {field!r}")
    return mapping[field]


def _object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _list(value: object, context: str, *, non_empty: bool = False) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    if non_empty and not value:
        raise ValueError(f"{context} must be a non-empty list")
    return value


def _non_empty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _identifier(value: object, context: str) -> Identifier:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{context} must be a string or integer ID")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{context} must be a non-empty ID")
    return value


def validate_unique_ids(ids: Iterable[Identifier], *, id_kind: str) -> None:
    """Reject duplicate query, document, or other identifiers."""

    seen: set[Identifier] = set()
    for value in ids:
        if value in seen:
            raise ValueError(f"duplicate {id_kind} ID {value!r}")
        seen.add(value)


def validate_no_cross_split_leakage(
    split_ids: Mapping[str, Iterable[Identifier]], *, id_kind: str
) -> None:
    """Reject duplicate IDs within a split and overlap between splits."""

    owner: dict[Identifier, str] = {}
    for split, values_iterable in split_ids.items():
        values = list(values_iterable)
        validate_unique_ids(values, id_kind=id_kind)
        for value in values:
            previous = owner.get(value)
            if previous is not None:
                raise ValueError(
                    f"{id_kind} ID {value!r} crosses splits {previous!r} and {split!r}"
                )
            owner[value] = split


def validate_unique_query_document_ids(
    query_ids: Iterable[Identifier], document_ids: Iterable[Identifier]
) -> None:
    """Validate query IDs and document IDs as two independent namespaces."""

    validate_unique_ids(query_ids, id_kind="query")
    validate_unique_ids(document_ids, id_kind="document")


def load_financebench(
    questions_path: str | Path,
    document_information_path: str | Path,
    pdf_dir: str | Path,
) -> tuple[tuple[FinanceBenchRecord, ...], FinanceBenchAudit]:
    """Load and audit the official FinanceBench JSONL metadata and PDF directory."""

    question_rows = _load_jsonl(Path(questions_path))
    records: list[FinanceBenchRecord] = []
    for index, row in enumerate(question_rows, start=1):
        context = f"{questions_path} line {index}"
        financebench_id = _identifier(
            _required(row, "financebench_id", context), f"{context}.financebench_id"
        )
        company = _non_empty_string(
            _required(row, "company", context), f"{context}.company"
        )
        doc_name = _non_empty_string(
            _required(row, "doc_name", context), f"{context}.doc_name"
        )
        _non_empty_string(_required(row, "question", context), f"{context}.question")
        _non_empty_string(_required(row, "answer", context), f"{context}.answer")
        evidence = _list(_required(row, "evidence", context), f"{context}.evidence")
        for evidence_index, item in enumerate(evidence):
            _object(item, f"{context}.evidence[{evidence_index}]")
        records.append(
            FinanceBenchRecord(
                financebench_id=financebench_id,
                company=company,
                doc_name=doc_name,
                evidence_count=len(evidence),
            )
        )
    try:
        validate_unique_ids(
            (record.financebench_id for record in records),
            id_kind="financebench_id",
        )
    except ValueError as exc:
        message = str(exc).replace("financebench_id ID", "financebench_id")
        raise ValueError(message) from exc

    document_rows = _load_jsonl(Path(document_information_path))
    document_names: list[str] = []
    for index, row in enumerate(document_rows, start=1):
        context = f"{document_information_path} line {index}"
        document_names.append(
            _non_empty_string(
                _required(row, "doc_name", context), f"{context}.doc_name"
            )
        )
        _non_empty_string(_required(row, "company", context), f"{context}.company")
    validate_unique_ids(document_names, id_kind="document")

    pdf_path = Path(pdf_dir)
    if not pdf_path.is_dir():
        raise ValueError(f"Required FinanceBench PDF directory does not exist: {pdf_path}")
    pdf_count = sum(
        1
        for path in pdf_path.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    audit = FinanceBenchAudit(
        question_count=len(records),
        company_count=len({record.company for record in records}),
        document_count=len(document_rows),
        evidence_count=sum(record.evidence_count for record in records),
        pdf_count=pdf_count,
    )
    return tuple(records), audit


def _balanced_company_folds(
    records: Sequence[FinanceBenchRecord],
    *,
    n_folds: int,
    seed: int,
    namespace: str,
) -> list[list[str]]:
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2:
        raise ValueError("n_folds must be an integer of at least 2")
    query_counts: dict[str, int] = {}
    for record in records:
        query_counts[record.company] = query_counts.get(record.company, 0) + 1
    ordered_companies = sorted(
        query_counts,
        key=lambda company: (
            -query_counts[company],
            _stable_hash(seed, namespace, "company", company),
            company,
        ),
    )
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    fold_counts = [0] * n_folds
    for company in ordered_companies:
        minimum = min(fold_counts)
        candidates = [index for index, count in enumerate(fold_counts) if count == minimum]
        target = min(
            candidates,
            key=lambda index: (
                _stable_hash(seed, namespace, company, "fold", index),
                index,
            ),
        )
        folds[target].append(company)
        fold_counts[target] += query_counts[company]
    return [sorted(companies) for companies in folds]


def _question_ids(
    records: Sequence[FinanceBenchRecord], companies: set[str]
) -> list[Identifier]:
    return sorted(
        (
            record.financebench_id
            for record in records
            if record.company in companies
        ),
        key=_id_sort_key,
    )


def build_financebench_nested_folds(
    records: Sequence[FinanceBenchRecord], *, seed: int = 42, n_folds: int = 5
) -> list[dict[str, object]]:
    """Build deterministic nested query-balanced folds grouped by company."""

    validate_unique_ids(
        (record.financebench_id for record in records), id_kind="financebench_id"
    )
    all_companies = {record.company for record in records}
    if len(all_companies) < 2:
        raise ValueError("FinanceBench nested folds require at least two companies")
    outer_company_folds = _balanced_company_folds(
        records, n_folds=n_folds, seed=seed, namespace="outer"
    )
    output: list[dict[str, object]] = []
    for outer_index, outer_test_list in enumerate(outer_company_folds):
        outer_test = set(outer_test_list)
        outer_train = all_companies - outer_test
        outer_train_records = [
            record for record in records if record.company in outer_train
        ]
        inner_company_folds = _balanced_company_folds(
            outer_train_records,
            n_folds=n_folds,
            seed=seed,
            namespace=f"outer-{outer_index}-inner",
        )
        inner_folds: list[dict[str, object]] = []
        for inner_index, validation_list in enumerate(inner_company_folds):
            validation = set(validation_list)
            inner_train = outer_train - validation
            inner_folds.append(
                {
                    "fold": inner_index,
                    "train_companies": sorted(inner_train),
                    "train_question_ids": _question_ids(records, inner_train),
                    "validation_companies": sorted(validation),
                    "validation_question_ids": _question_ids(records, validation),
                }
            )
        output.append(
            {
                "fold": outer_index,
                "train_companies": sorted(outer_train),
                "train_question_ids": _question_ids(records, outer_train),
                "test_companies": sorted(outer_test),
                "test_question_ids": _question_ids(records, outer_test),
                "inner_folds": inner_folds,
            }
        )
    return output


def load_contractnli(
    contract_dir: str | Path,
) -> tuple[dict[str, tuple[ContractNLIDocumentRecord, ...]], ContractNLIAudit]:
    """Load and cross-audit the official ContractNLI train/dev/test files."""

    root = Path(contract_dir)
    records_by_split: dict[str, tuple[ContractNLIDocumentRecord, ...]] = {}
    label_keys_by_split: dict[str, tuple[str, ...]] = {}
    summaries: list[ContractNLISplitAudit] = []
    for split in _CONTRACT_SPLITS:
        path = root / f"{split}.json"
        payload = _load_json(path)
        documents = _list(
            _required(payload, "documents", str(path)), f"{path}.documents"
        )
        labels = _object(_required(payload, "labels", str(path)), f"{path}.labels")
        label_keys: list[str] = []
        for raw_key, raw_label in labels.items():
            key = _non_empty_string(raw_key, f"{path}.labels key")
            label = _object(raw_label, f"{path}.labels[{key!r}]")
            _non_empty_string(
                _required(label, "hypothesis", f"{path}.labels[{key!r}]"),
                f"{path}.labels[{key!r}].hypothesis",
            )
            label_keys.append(key)
        sorted_label_keys = tuple(sorted(label_keys))
        label_keys_by_split[split] = sorted_label_keys

        split_records: list[ContractNLIDocumentRecord] = []
        choice_counts = {choice: 0 for choice in _CONTRACT_CHOICES}
        evidence_count = 0
        for document_index, raw_document in enumerate(documents):
            context = f"{path}.documents[{document_index}]"
            document = _object(raw_document, context)
            document_id = _identifier(
                _required(document, "id", context), f"{context}.id"
            )
            _non_empty_string(
                _required(document, "text", context), f"{context}.text"
            )
            spans = _list(_required(document, "spans", context), f"{context}.spans")
            for span_index, raw_span in enumerate(spans):
                span = _list(raw_span, f"{context}.spans[{span_index}]")
                if (
                    len(span) != 2
                    or any(isinstance(value, bool) or not isinstance(value, int) for value in span)
                    or span[0] < 0
                    or span[1] < span[0]
                ):
                    raise ValueError(
                        f"{context}.spans[{span_index}] must be [start, end] non-negative integers"
                    )
            annotation_sets = _list(
                _required(document, "annotation_sets", context),
                f"{context}.annotation_sets",
                non_empty=True,
            )
            first_set = _object(annotation_sets[0], f"{context}.annotation_sets[0]")
            annotations = _object(
                _required(first_set, "annotations", f"{context}.annotation_sets[0]"),
                f"{context}.annotation_sets[0].annotations",
            )
            if set(annotations) != set(sorted_label_keys):
                raise ValueError(
                    f"{context} annotation keys must match label keys; "
                    f"got {sorted(annotations)}, expected {list(sorted_label_keys)}"
                )
            normalized_annotations: list[ContractNLIAnnotation] = []
            for hypothesis_id in sorted_label_keys:
                annotation_context = f"{context}.annotations[{hypothesis_id!r}]"
                annotation = _object(annotations[hypothesis_id], annotation_context)
                choice = _non_empty_string(
                    _required(annotation, "choice", annotation_context),
                    f"{annotation_context}.choice",
                )
                if choice not in _CONTRACT_CHOICES:
                    raise ValueError(
                        f"{annotation_context}.choice {choice!r} must be one of "
                        f"{list(_CONTRACT_CHOICES)}"
                    )
                evidence_spans = _list(
                    _required(annotation, "spans", annotation_context),
                    f"{annotation_context}.spans",
                )
                normalized_evidence: list[int] = []
                for raw_index in evidence_spans:
                    if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                        raise ValueError(
                            f"{annotation_context}.spans entries must be integer span indices"
                        )
                    if raw_index < 0 or raw_index >= len(spans):
                        raise ValueError(
                            f"{annotation_context} span index {raw_index} is out of range "
                            f"for {len(spans)} spans"
                        )
                    normalized_evidence.append(raw_index)
                choice_counts[choice] += 1
                evidence_count += len(normalized_evidence)
                normalized_annotations.append(
                    ContractNLIAnnotation(
                        hypothesis_id=hypothesis_id,
                        choice=choice,
                        evidence_span_indices=tuple(normalized_evidence),
                    )
                )
            split_records.append(
                ContractNLIDocumentRecord(
                    document_id=document_id,
                    split=split,
                    span_count=len(spans),
                    annotations=tuple(normalized_annotations),
                )
            )
        validate_unique_ids(
            (record.document_id for record in split_records), id_kind="document"
        )
        records_by_split[split] = tuple(split_records)
        summaries.append(
            ContractNLISplitAudit(
                split=split,
                document_count=len(split_records),
                hypothesis_count=len(sorted_label_keys),
                annotation_choice_count_items=tuple(
                    (choice, choice_counts[choice]) for choice in _CONTRACT_CHOICES
                ),
                evidence_span_count=evidence_count,
            )
        )

    reference_keys = label_keys_by_split["train"]
    for split in ("dev", "test"):
        if label_keys_by_split[split] != reference_keys:
            raise ValueError(
                f"label keys for {split} do not match train: "
                f"{list(label_keys_by_split[split])} != {list(reference_keys)}"
            )
    validate_no_cross_split_leakage(
        {
            split: [record.document_id for record in records_by_split[split]]
            for split in _CONTRACT_SPLITS
        },
        id_kind="document",
    )
    return records_by_split, ContractNLIAudit(tuple(summaries), reference_keys)


def load_ramdocs(
    path: str | Path,
) -> tuple[tuple[RAMDocsRecord, ...], RAMDocsAudit]:
    """Load RAMDocs and derive stable metadata-only query/document IDs."""

    rows = _load_jsonl(Path(path))
    records: list[RAMDocsRecord] = []
    type_counts = {document_type: 0 for document_type in _RAMDOCS_TYPES}
    document_counts: list[int] = []
    for line_index, row in enumerate(rows):
        context = f"{path} line {line_index + 1}"
        _non_empty_string(
            _required(row, "question", context), f"{context}.question"
        )
        documents = _list(
            _required(row, "documents", context),
            f"{context}.documents",
            non_empty=True,
        )
        for answer_field in ("gold_answers", "wrong_answers"):
            answers = _list(
                _required(row, answer_field, context),
                f"{context}.{answer_field}",
                non_empty=True,
            )
            for answer_index, answer in enumerate(answers):
                _non_empty_string(
                    answer, f"{context}.{answer_field}[{answer_index}]"
                )
        query_id = f"ramdocs-{line_index:06d}"
        normalized_documents: list[RAMDocsDocumentRecord] = []
        for document_index, raw_document in enumerate(documents):
            document_context = f"{context}.documents[{document_index}]"
            document = _object(raw_document, document_context)
            _non_empty_string(
                _required(document, "text", document_context),
                f"{document_context}.text",
            )
            _non_empty_string(
                _required(document, "answer", document_context),
                f"{document_context}.answer",
            )
            document_type = _non_empty_string(
                _required(document, "type", document_context),
                f"{document_context}.type",
            )
            if document_type not in _RAMDOCS_TYPES:
                raise ValueError(
                    f"{document_context}.type {document_type!r} must be one of "
                    f"{list(_RAMDOCS_TYPES)}"
                )
            type_counts[document_type] += 1
            normalized_documents.append(
                RAMDocsDocumentRecord(
                    document_id=f"{query_id}-doc-{document_index:03d}",
                    document_type=document_type,
                )
            )
        document_counts.append(len(normalized_documents))
        records.append(RAMDocsRecord(query_id, tuple(normalized_documents)))
    validate_unique_query_document_ids(
        (record.query_id for record in records),
        (
            document.document_id
            for record in records
            for document in record.documents
        ),
    )
    audit = RAMDocsAudit(
        example_count=len(records),
        document_count=sum(document_counts),
        min_documents_per_query=min(document_counts, default=0),
        max_documents_per_query=max(document_counts, default=0),
        document_type_count_items=tuple(
            (document_type, type_counts[document_type])
            for document_type in _RAMDOCS_TYPES
        ),
    )
    return tuple(records), audit


def _normalize_niah_record(
    value: NIAHMetadataRecord | Mapping[str, object], index: int
) -> NIAHMetadataRecord:
    if isinstance(value, NIAHMetadataRecord):
        record = value
    elif isinstance(value, Mapping):
        context = f"NIAH metadata record {index}"
        record = NIAHMetadataRecord(
            query_id=_non_empty_string(
                _required(value, "query_id", context), f"{context}.query_id"
            ),
            parent_page_id=_non_empty_string(
                _required(value, "parent_page_id", context),
                f"{context}.parent_page_id",
            ),
            synthetic_family_id=_non_empty_string(
                _required(value, "synthetic_family_id", context),
                f"{context}.synthetic_family_id",
            ),
        )
    else:
        raise ValueError(f"NIAH metadata record {index} must be an object")
    for field_name in ("query_id", "parent_page_id", "synthetic_family_id"):
        _non_empty_string(getattr(record, field_name), f"NIAH record {index}.{field_name}")
    return record


def _niah_components(records: Sequence[NIAHMetadataRecord]) -> list[list[int]]:
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    seen_parent: dict[str, int] = {}
    seen_family: dict[str, int] = {}
    for index, record in enumerate(records):
        if record.parent_page_id in seen_parent:
            union(index, seen_parent[record.parent_page_id])
        else:
            seen_parent[record.parent_page_id] = index
        if record.synthetic_family_id in seen_family:
            union(index, seen_family[record.synthetic_family_id])
        else:
            seen_family[record.synthetic_family_id] = index
    components: dict[int, list[int]] = {}
    for index in range(len(records)):
        components.setdefault(find(index), []).append(index)
    return list(components.values())


def _greedy_component_assignment(
    components: Sequence[Sequence[int]],
    split_names: Sequence[str],
    targets: Mapping[str, int],
    seed: int,
) -> list[int] | None:
    remaining = [targets[name] for name in split_names]
    assignments: list[int] = []
    for component_index, component in enumerate(components):
        size = len(component)
        candidates = [
            index for index, capacity in enumerate(remaining) if capacity >= size
        ]
        if not candidates:
            return None
        target = min(
            candidates,
            key=lambda index: (
                -(remaining[index] / targets[split_names[index]]),
                _stable_hash(seed, "niah", component_index, split_names[index]),
                index,
            ),
        )
        assignments.append(target)
        remaining[target] -= size
    return assignments if not any(remaining) else None


def _exact_component_assignment(
    components: Sequence[Sequence[int]],
    split_names: Sequence[str],
    targets: Mapping[str, int],
    seed: int,
) -> list[int] | None:
    if len(components) > 100:
        return None
    initial = tuple(targets[name] for name in split_names)
    failed: set[tuple[int, tuple[int, ...]]] = set()

    def search(index: int, remaining: tuple[int, ...]) -> list[int] | None:
        state = (index, remaining)
        if state in failed:
            return None
        if index == len(components):
            return [] if not any(remaining) else None
        size = len(components[index])
        candidates = [
            split_index
            for split_index, capacity in enumerate(remaining)
            if capacity >= size
        ]
        candidates.sort(
            key=lambda split_index: (
                -(remaining[split_index] / targets[split_names[split_index]]),
                _stable_hash(seed, "niah-exact", index, split_names[split_index]),
                split_index,
            )
        )
        tried_capacities: set[int] = set()
        for split_index in candidates:
            if remaining[split_index] in tried_capacities:
                continue
            tried_capacities.add(remaining[split_index])
            updated = list(remaining)
            updated[split_index] -= size
            suffix = search(index + 1, tuple(updated))
            if suffix is not None:
                return [split_index, *suffix]
        failed.add(state)
        return None

    return search(0, initial)


def assign_niah_grouped_splits(
    records: Sequence[NIAHMetadataRecord | Mapping[str, object]],
    *,
    targets: Mapping[str, int],
    seed: int = 42,
) -> dict[str, tuple[str, ...]]:
    """Assign connected parent/family components to exact train/dev/test targets."""

    split_names = ("train", "dev", "test")
    if set(targets) != set(split_names):
        raise ValueError("targets must contain exactly train, dev, and test")
    for split in split_names:
        target = targets[split]
        if isinstance(target, bool) or not isinstance(target, int) or target < 0:
            raise ValueError(f"target for {split} must be a non-negative integer")
    normalized = tuple(
        _normalize_niah_record(record, index) for index, record in enumerate(records)
    )
    validate_unique_ids(
        (record.query_id for record in normalized), id_kind="query"
    )
    if sum(targets.values()) != len(normalized):
        raise ValueError(
            f"target counts total {sum(targets.values())} but there are "
            f"{len(normalized)} NIAH records"
        )
    components = _niah_components(normalized)
    required_non_empty = sum(targets[split] > 0 for split in split_names)
    if len(components) < required_non_empty:
        raise ValueError(
            f"insufficient independent groups: {len(components)} components cannot "
            f"populate {required_non_empty} non-empty splits"
        )
    components.sort(
        key=lambda component: (
            -len(component),
            _stable_hash(
                seed,
                "niah-component",
                *(sorted(normalized[index].query_id for index in component)),
            ),
        )
    )
    assignments = _greedy_component_assignment(
        components, split_names, targets, seed
    )
    if assignments is None:
        assignments = _exact_component_assignment(
            components, split_names, targets, seed
        )
    if assignments is None:
        sizes = sorted((len(component) for component in components), reverse=True)
        raise ValueError(
            f"connected component sizes {sizes} cannot satisfy exact targets "
            f"{dict(targets)}; insufficient independently assignable groups"
        )
    output: dict[str, list[str]] = {split: [] for split in split_names}
    for component, split_index in zip(components, assignments):
        output[split_names[split_index]].extend(
            normalized[index].query_id for index in component
        )
    result = {
        split: tuple(sorted(query_ids)) for split, query_ids in output.items()
    }
    validate_no_cross_split_leakage(result, id_kind="query")
    return result
