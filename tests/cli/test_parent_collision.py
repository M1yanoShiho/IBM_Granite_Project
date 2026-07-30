import json
from pathlib import Path

import pytest

from evidence_rag.cli.parent_collision import main
from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate


def _candidate(evidence_id: str, document_id: str, rank: int) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=evidence_id,
        document_id=document_id,
        chunk_id=f"{document_id}::c0",
        text="t",
        source_uri=f"s://{document_id}",
        retrieval_score=1.0 / rank,
        retrieval_rank=rank,
    )


def _candidates(tmp_path: Path) -> Path:
    path = tmp_path / "candidate_sets.jsonl"
    rows = [
        CandidateSet(
            query_id="q1",
            candidates=(_candidate("e1", "d1", 1), _candidate("e2", "d2", 2)),
        ),
        CandidateSet(
            query_id="q2",
            candidates=(_candidate("e3", "d3", 1), _candidate("e4", "d4", 2)),
        ),
    ]
    path.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
    )
    return path


def _index(tmp_path: Path, mapping: dict[str, str]) -> Path:
    path = tmp_path / "source_parent.jsonl"
    path.write_text(
        "".join(
            json.dumps({"document_id": key, "source_parent_id": value}, sort_keys=True) + "\n"
            for key, value in mapping.items()
        ),
        encoding="utf-8",
    )
    return path


def test_reports_collision_rate_over_queries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    index = _index(
        tmp_path, {"d1": "page a", "d2": "page a", "d3": "page b", "d4": "page c"}
    )
    assert (
        main(
            [
                "--candidates", str(_candidates(tmp_path)),
                "--parent-index", str(index),
                "--top-n", "20",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["n_queries"] == 2
    assert report["n_queries_with_collision"] == 1
    assert report["collision_rate"] == 0.5
    assert report["mean_documents_per_window"] == 2.0
    assert report["mean_parents_per_window"] == 1.5
    assert report["n_unresolved_documents"] == 0


def test_unresolved_documents_do_not_count_as_collisions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unmapped document is its own parent, so it must never look like a duplicate."""
    index = _index(tmp_path, {"d1": "page a"})
    main(
        [
            "--candidates", str(_candidates(tmp_path)),
            "--parent-index", str(index),
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert report["n_queries_with_collision"] == 0
    assert report["n_unresolved_documents"] == 3


def test_top_n_truncates_the_window(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    index = _index(
        tmp_path, {"d1": "page a", "d2": "page a", "d3": "page b", "d4": "page c"}
    )
    main(
        [
            "--candidates", str(_candidates(tmp_path)),
            "--parent-index", str(index),
            "--top-n", "1",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert report["mean_documents_per_window"] == 1.0
    assert report["n_queries_with_collision"] == 0
