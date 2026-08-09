import json
from pathlib import Path

import pytest

from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate, RetrieverProvenance
from evidence_rag.materializer.selector_beam_data import (
    EvidenceLabel,
    load_niah_cases,
    training_examples,
)
from evidence_rag.materializer.selector_beam_split import NiahSelectorAssignment


def _candidate(query_id: str, document_id: str, rank: int) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=f"e-{query_id}-{rank}",
        document_id=document_id,
        chunk_id=f"c-{rank}",
        text=f"safe passage text {rank}",
        source_uri="source",
        retrieval_score=1.0 / rank,
        retrieval_rank=rank,
    )


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    root.mkdir()
    documents = [
        {"document_id": f"d{i}", "text": f"doc {i}", "source_uri": "source"}
        for i in range(20)
    ]
    (root / "documents.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in documents), encoding="utf-8"
    )
    (root / "queries.jsonl").write_text(
        '{"query_id":"allowed","text":"question only"}\n'
        '{"query_id":"forbidden","text":"must not enter training"}\n',
        encoding="utf-8",
    )
    (root / "gold_cases.jsonl").write_text(
        '{"query_id":"allowed","relevant_document_ids":["d0"]}\n'
        '{"query_id":"forbidden","relevant_document_ids":["d0"]}\n',
        encoding="utf-8",
    )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "niah/test",
                "dataset_version": "v1",
                "split": "wrong-historical-label",
                "documents_file": "documents.jsonl",
                "queries_file": "queries.jsonl",
                "gold_cases_file": "gold_cases.jsonl",
            }
        ),
        encoding="utf-8",
    )
    provenance = RetrieverProvenance(
        name="hybrid", implementation_version="hybrid-v1", parameters_sha256="0" * 64
    )
    pools = []
    for query_id in ("allowed", "forbidden"):
        pools.append(
            CandidateSet(
                query_id=query_id,
                candidates=tuple(
                    _candidate(query_id, f"d{index}", index + 1) for index in range(20)
                ),
                retriever=provenance,
            )
        )
    candidates = root / "candidate_sets.jsonl"
    candidates.write_text(
        "".join(item.model_dump_json() + "\n" for item in pools), encoding="utf-8"
    )
    assignment = root / "assignments.jsonl"
    assignment.write_text(
        NiahSelectorAssignment(
            query_id="allowed",
            required_document_ids=("d0",),
            harmful_document_id="d1",
            source_parent_ids=("p",),
            synthetic_family="f",
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    return manifest, candidates, assignment


def test_niah_loader_uses_assignment_as_the_only_allow_list(tmp_path: Path) -> None:
    manifest, candidates, assignment = _fixture(tmp_path / "data")
    cases = load_niah_cases(
        manifest_path=manifest,
        candidate_path=candidates,
        assignment_path=assignment,
    )
    assert [case.query_id for case in cases] == ["allowed"]
    assert cases[0].labels[:3] == (
        EvidenceLabel.REQUIRED,
        EvidenceLabel.HARMFUL,
        EvidenceLabel.IRRELEVANT,
    )
    assert "forbidden" not in {case.query_id for case in cases}


def test_training_examples_never_expose_ids_or_provenance_in_text(tmp_path: Path) -> None:
    manifest, candidates, assignment = _fixture(tmp_path / "data")
    case = load_niah_cases(
        manifest_path=manifest,
        candidate_path=candidates,
        assignment_path=assignment,
    )[0]
    examples = training_examples((case,), hard_negatives_per_query=2, seed=13)
    assert {example.label for example in examples} == set(EvidenceLabel)
    assert any(example.hop == 1 for example in examples)
    model_text = " ".join(
        [
            *(example.question for example in examples),
            *(example.candidate_passage for example in examples),
            *(text for example in examples for text in example.selected_passages),
        ]
    )
    assert "cf::" not in model_text
    assert "source_parent" not in model_text
    assert "forbidden" not in model_text


def test_niah_loader_rejects_unknown_assignment_document(tmp_path: Path) -> None:
    manifest, candidates, assignment = _fixture(tmp_path / "data")
    row = json.loads(assignment.read_text(encoding="utf-8"))
    row["harmful_document_id"] = "not-in-source"
    assignment.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown source documents"):
        load_niah_cases(
            manifest_path=manifest,
            candidate_path=candidates,
            assignment_path=assignment,
        )


def test_loader_rejects_dataset_content_not_frozen_by_m0(tmp_path: Path) -> None:
    manifest, candidates, assignment = _fixture(tmp_path / "data")
    with pytest.raises(ValueError, match="differs from the version frozen by M0"):
        load_niah_cases(
            manifest_path=manifest,
            candidate_path=candidates,
            assignment_path=assignment,
            expected_dataset_signature="wrong-signature",
        )
