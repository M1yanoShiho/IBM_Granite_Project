"""Leakage-safe inputs for the three-class Beam Selector.

Only the question and candidate passage text leave this module as model inputs.  Gold document
IDs and NIAH provenance are used here to create integer labels, then discarded.  This makes the
train/inference boundary explicit instead of relying on callers to remember which fields are safe.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate
from evidence_rag.infrastructure.datasets import JsonlDatasetAdapter
from evidence_rag.materializer.selector_beam_split import NiahSelectorAssignment
from evidence_rag.selector.beam_three_class import BeamCandidate


class EvidenceLabel(IntEnum):
    IRRELEVANT = 0
    REQUIRED = 1
    HARMFUL = 2


@dataclass(frozen=True)
class BeamSelectorCase:
    query_id: str
    question: str
    candidates: tuple[BeamCandidate, ...]
    labels: tuple[EvidenceLabel, ...]
    dataset: str
    required_document_ids: tuple[str, ...]
    harmful_document_id: str | None = None

    def __post_init__(self) -> None:
        if len(self.candidates) != len(self.labels):
            raise ValueError("candidate and label lengths differ")
        if len({item.evidence_id for item in self.candidates}) != len(self.candidates):
            raise ValueError(f"duplicate candidate ID for {self.query_id}")
        if self.harmful_document_id in set(self.required_document_ids):
            raise ValueError(f"required and harmful documents overlap for {self.query_id}")


@dataclass(frozen=True)
class BeamTrainingExample:
    query_id: str
    question: str
    selected_passages: tuple[str, ...]
    candidate_passage: str
    label: EvidenceLabel
    hop: int
    dataset: str


def _read_candidates(path: Path) -> dict[str, CandidateSet]:
    values: dict[str, CandidateSet] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            candidate_set = CandidateSet.model_validate_json(line)
        except ValueError as error:
            raise ValueError(f"invalid candidate set at {path}:{number}: {error}") from error
        if candidate_set.query_id in values:
            raise ValueError(f"duplicate candidate query ID: {candidate_set.query_id}")
        if len(candidate_set.candidates) != 20:
            raise ValueError(f"{candidate_set.query_id} does not have exactly Top-20")
        if {item.retrieval_rank for item in candidate_set.candidates} != set(range(1, 21)):
            raise ValueError(f"{candidate_set.query_id} does not have exact ranks 1-20")
        if candidate_set.retriever is None or candidate_set.retriever.name != "hybrid":
            raise ValueError(f"{candidate_set.query_id} is not from the frozen Hybrid Retriever")
        values[candidate_set.query_id] = candidate_set
    return values


def _safe_candidates(values: Sequence[EvidenceCandidate]) -> tuple[BeamCandidate, ...]:
    # Retrieval rank is the frozen order.  No metadata, source URI, cf:: marker or provenance is
    # copied into the model-side record.
    return tuple(
        BeamCandidate(
            evidence_id=item.evidence_id,
            document_id=item.document_id,
            text=item.text,
            retrieval_rank=item.retrieval_rank,
            retrieval_score=float(item.retrieval_score),
        )
        for item in sorted(values, key=lambda candidate: candidate.retrieval_rank)
    )


def _assignment_records(path: Path) -> tuple[NiahSelectorAssignment, ...]:
    records: list[NiahSelectorAssignment] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(NiahSelectorAssignment.model_validate_json(line))
        except ValueError as error:
            raise ValueError(f"invalid NIAH assignment at {path}:{number}: {error}") from error
    if len({item.query_id for item in records}) != len(records):
        raise ValueError("duplicate NIAH assignment query ID")
    return tuple(records)


def load_niah_cases(
    *,
    manifest_path: Path,
    candidate_path: Path,
    assignment_path: Path,
    expected_dataset_signature: str | None = None,
) -> tuple[BeamSelectorCase, ...]:
    """Load only the leakage-filtered NIAH assignments from a possibly larger source pool."""

    bundle = JsonlDatasetAdapter.load(manifest_path)
    if (
        expected_dataset_signature is not None
        and bundle.dataset_signature != expected_dataset_signature
    ):
        raise ValueError("NIAH dataset content differs from the version frozen by M0")
    source_document_ids = {document.document_id for document in bundle.documents}
    queries = {query.query_id: query for query in bundle.queries}
    candidate_sets = _read_candidates(candidate_path)
    cases: list[BeamSelectorCase] = []
    for assignment in _assignment_records(assignment_path):
        query = queries.get(assignment.query_id)
        candidate_set = candidate_sets.get(assignment.query_id)
        if query is None or candidate_set is None:
            raise ValueError(f"assignment {assignment.query_id} is missing its query or Top-20")
        labeled = {*assignment.required_document_ids, assignment.harmful_document_id}
        if not labeled <= source_document_ids:
            raise ValueError(
                f"NIAH assignment {assignment.query_id} references unknown source documents"
            )
        overlap = set(assignment.required_document_ids) & {assignment.harmful_document_id}
        if overlap:
            raise ValueError(f"NIAH labels overlap for {assignment.query_id}: {sorted(overlap)}")
        candidates = _safe_candidates(candidate_set.candidates)
        labels = tuple(
            EvidenceLabel.REQUIRED
            if item.document_id in assignment.required_document_ids
            else (
                EvidenceLabel.HARMFUL
                if item.document_id == assignment.harmful_document_id
                else EvidenceLabel.IRRELEVANT
            )
            for item in candidates
        )
        cases.append(
            BeamSelectorCase(
                query_id=query.query_id,
                question=query.text,
                candidates=candidates,
                labels=labels,
                dataset="niah",
                required_document_ids=assignment.required_document_ids,
                harmful_document_id=assignment.harmful_document_id,
            )
        )
    return tuple(cases)


def load_twowiki_cases(
    *,
    manifest_path: Path,
    candidate_path: Path,
    expected_dataset_signature: str | None = None,
) -> tuple[BeamSelectorCase, ...]:
    bundle = JsonlDatasetAdapter.load(manifest_path)
    if (
        expected_dataset_signature is not None
        and bundle.dataset_signature != expected_dataset_signature
    ):
        raise ValueError("2Wiki dataset content differs from the version frozen by M0")
    candidate_sets = _read_candidates(candidate_path)
    gold = {item.query_id: set(item.relevant_document_ids or ()) for item in bundle.gold_cases}
    cases: list[BeamSelectorCase] = []
    for query in bundle.queries:
        candidate_set = candidate_sets.get(query.query_id)
        if candidate_set is None:
            raise ValueError(f"2Wiki query {query.query_id} is missing its Top-20")
        candidates = _safe_candidates(candidate_set.candidates)
        labels = tuple(
            EvidenceLabel.REQUIRED
            if item.document_id in gold[query.query_id]
            else EvidenceLabel.IRRELEVANT
            for item in candidates
        )
        cases.append(
            BeamSelectorCase(
                query_id=query.query_id,
                question=query.text,
                candidates=candidates,
                labels=labels,
                dataset="2wiki",
                required_document_ids=tuple(sorted(gold[query.query_id])),
            )
        )
    return tuple(cases)


def _chosen_irrelevant(
    case: BeamSelectorCase,
    *,
    excluded: set[int],
    count: int,
    seed: int,
    hop: int,
) -> tuple[int, ...]:
    # Draw only from the high-ranked half of Top-20: these are genuine hard negatives, not random
    # corpus passages.  The fixed seed changes ties without making the selection irreproducible.
    pool = [
        index
        for index, label in enumerate(case.labels)
        if label == EvidenceLabel.IRRELEVANT and index not in excluded
    ][: max(count * 2, count)]
    generator = random.Random(f"{seed}:{case.query_id}:{hop}")
    generator.shuffle(pool)
    return tuple(sorted(pool[:count], key=lambda index: case.candidates[index].retrieval_rank))


def training_examples(
    cases: Iterable[BeamSelectorCase],
    *,
    hard_negatives_per_query: int,
    seed: int,
) -> tuple[BeamTrainingExample, ...]:
    if hard_negatives_per_query < 1:
        raise ValueError("hard_negatives_per_query must be positive")
    examples: list[BeamTrainingExample] = []
    for case in cases:
        required = [
            index for index, label in enumerate(case.labels) if label == EvidenceLabel.REQUIRED
        ]
        harmful = [
            index for index, label in enumerate(case.labels) if label == EvidenceLabel.HARMFUL
        ]
        if not required and not harmful:
            # A Top-20 miss is a Retriever-quality outcome, not a fabricated all-negative
            # supervision case.  It remains in evaluation but contributes no training example.
            continue
        selected: list[int] = []
        # Include the post-gold hop as a learned stopping state: once every required passage has
        # been chosen, remaining items must not be promoted merely to fill ten slots.
        for hop in range(len(required) + 1):
            excluded = set(selected)
            remaining_required = [index for index in required if index not in excluded]
            included = [*remaining_required, *[index for index in harmful if index not in excluded]]
            included.extend(
                _chosen_irrelevant(
                    case,
                    excluded=excluded,
                    count=hard_negatives_per_query,
                    seed=seed,
                    hop=hop,
                )
            )
            for index in dict.fromkeys(included):
                candidate = case.candidates[index]
                examples.append(
                    BeamTrainingExample(
                        query_id=case.query_id,
                        question=case.question,
                        selected_passages=tuple(
                            case.candidates[item].text for item in selected
                        ),
                        candidate_passage=candidate.text,
                        label=case.labels[index],
                        hop=hop,
                        dataset=case.dataset,
                    )
                )
            if hop < len(required):
                selected.append(required[hop])
    return tuple(examples)


def write_example_manifest(path: Path, examples: Sequence[BeamTrainingExample]) -> None:
    counts = {label.name: 0 for label in EvidenceLabel}
    datasets: dict[str, int] = {}
    for example in examples:
        counts[example.label.name] += 1
        datasets[example.dataset] = datasets.get(example.dataset, 0) + 1
    payload = {
        "schema_version": "1.0",
        "examples": len(examples),
        "queries": len({item.query_id for item in examples}),
        "labels": counts,
        "datasets": datasets,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
