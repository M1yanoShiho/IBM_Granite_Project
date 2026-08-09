"""Leakage-safe NIAH assignments for the three-class Beam Selector.

The historical NIAH directories all claim ``split=dev`` and therefore cannot be trusted as
train/dev declarations.  This module derives the supervised grouping axes from the actual
query, mutation and document contents, reserves sealed600 first, then dev, and only then admits
training rows.  The split label is never read.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from evidence_rag.materializer.provenance import read_provenance
from evidence_rag.materializer.source_parent import parse_parent
from evidence_rag.relations.task_probe import synthetic_family


class NiahSelectorAssignment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    required_document_ids: tuple[str, ...]
    harmful_document_id: str
    source_parent_ids: tuple[str, ...]
    synthetic_family: str


def sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> Iterable[Mapping[str, object]]:
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"expected an object at {path}:{number}")
        yield row


def _gold_by_query(path: Path) -> dict[str, tuple[str, ...]]:
    values: dict[str, tuple[str, ...]] = {}
    for row in _jsonl(path):
        query_id = str(row["query_id"])
        if query_id in values:
            raise ValueError(f"duplicate gold query ID: {query_id}")
        raw_documents = row.get("relevant_document_ids")
        if not isinstance(raw_documents, list):
            raise ValueError(f"query {query_id} has invalid required document IDs")
        documents = tuple(str(item) for item in raw_documents)
        if not documents:
            raise ValueError(f"query {query_id} has no required document IDs")
        values[query_id] = documents
    return values


def _selected_document_texts(path: Path, wanted: set[str]) -> dict[str, str]:
    texts: dict[str, str] = {}
    for row in _jsonl(path):
        document_id = str(row["document_id"])
        if document_id in wanted:
            texts[document_id] = str(row["text"])
    missing = sorted(wanted - set(texts))
    if missing:
        raise ValueError(f"documents file is missing labeled IDs, e.g. {missing[:3]}")
    return texts


def load_assignments(directory: Path) -> tuple[NiahSelectorAssignment, ...]:
    directory = Path(directory)
    records = read_provenance(directory / "provenance.jsonl")
    gold = _gold_by_query(directory / "gold_cases.jsonl")
    if len({record.query_id for record in records}) != len(records):
        raise ValueError(f"duplicate provenance query ID in {directory}")
    record_queries = {record.query_id for record in records}
    missing_gold = sorted(record_queries - set(gold))
    if missing_gold:
        raise ValueError(f"provenance queries have no gold case, e.g. {missing_gold[:3]}")

    wanted = {
        document_id
        for record in records
        for document_id in (*gold[record.query_id], record.counterfactual_document_id)
    }
    texts = _selected_document_texts(directory / "documents.jsonl", wanted)
    assignments: list[NiahSelectorAssignment] = []
    for record in records:
        required = tuple(dict.fromkeys(gold[record.query_id]))
        labeled_ids = (*required, record.counterfactual_document_id)
        parents: set[str] = set()
        for document_id in labeled_ids:
            parent = parse_parent(texts[document_id])
            if parent is None:
                raise ValueError(
                    f"labeled document {document_id} for query {record.query_id} has no parent"
                )
            parents.add(parent)
        assignments.append(
            NiahSelectorAssignment(
                query_id=record.query_id,
                required_document_ids=required,
                harmful_document_id=record.counterfactual_document_id,
                source_parent_ids=tuple(sorted(parents)),
                synthetic_family=synthetic_family(record),
            )
        )
    return tuple(sorted(assignments, key=lambda item: item.query_id))


def _axes(
    assignments: Sequence[NiahSelectorAssignment],
) -> dict[str, set[str]]:
    return {
        "query_id": {item.query_id for item in assignments},
        "source_parent_id": {
            parent for item in assignments for parent in item.source_parent_ids
        },
        "synthetic_family": {item.synthetic_family for item in assignments},
    }


def _overlap(
    left: Sequence[NiahSelectorAssignment],
    right: Sequence[NiahSelectorAssignment],
) -> dict[str, int]:
    left_axes = _axes(left)
    right_axes = _axes(right)
    return {name: len(left_axes[name] & right_axes[name]) for name in left_axes}


def _filter_reserved(
    assignments: Sequence[NiahSelectorAssignment],
    reserved: Sequence[NiahSelectorAssignment],
) -> tuple[tuple[NiahSelectorAssignment, ...], dict[str, tuple[str, ...]]]:
    reserved_axes = _axes(reserved)
    kept: list[NiahSelectorAssignment] = []
    dropped: dict[str, list[str]] = {
        "query_id": [],
        "source_parent_id": [],
        "synthetic_family": [],
    }
    for item in assignments:
        reasons: list[str] = []
        if item.query_id in reserved_axes["query_id"]:
            reasons.append("query_id")
        if set(item.source_parent_ids) & reserved_axes["source_parent_id"]:
            reasons.append("source_parent_id")
        if item.synthetic_family in reserved_axes["synthetic_family"]:
            reasons.append("synthetic_family")
        if reasons:
            for reason in reasons:
                dropped[reason].append(item.query_id)
        else:
            kept.append(item)
    return tuple(kept), {
        name: tuple(sorted(query_ids)) for name, query_ids in dropped.items()
    }


def build_splits(
    *,
    train: Sequence[NiahSelectorAssignment],
    dev: Sequence[NiahSelectorAssignment],
    sealed: Sequence[NiahSelectorAssignment],
) -> tuple[
    tuple[NiahSelectorAssignment, ...],
    tuple[NiahSelectorAssignment, ...],
    dict[str, object],
]:
    dev_kept, dev_dropped = _filter_reserved(dev, sealed)
    train_kept, train_dropped = _filter_reserved(train, (*sealed, *dev_kept))
    final_overlaps = {
        "train_dev": _overlap(train_kept, dev_kept),
        "train_sealed": _overlap(train_kept, sealed),
        "dev_sealed": _overlap(dev_kept, sealed),
    }
    if any(value for pair in final_overlaps.values() for value in pair.values()):
        raise AssertionError("internal leakage filter failed")
    if not train_kept or not dev_kept or not sealed:
        raise ValueError("train, dev and sealed assignments must all be non-empty")
    report: dict[str, object] = {
        "schema_version": "1.0",
        "counts": {
            "source_train": len(train),
            "source_dev": len(dev),
            "sealed": len(sealed),
            "train": len(train_kept),
            "dev": len(dev_kept),
        },
        "overlap_before_filtering": {
            "train_dev": _overlap(train, dev),
            "train_sealed": _overlap(train, sealed),
            "dev_sealed": _overlap(dev, sealed),
        },
        "dropped": {
            "train": train_dropped,
            "dev": dev_dropped,
        },
        "overlap_after_filtering": final_overlaps,
    }
    return train_kept, dev_kept, report


def write_assignments(path: Path, values: Sequence[NiahSelectorAssignment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(item.model_dump_json() for item in values)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")
