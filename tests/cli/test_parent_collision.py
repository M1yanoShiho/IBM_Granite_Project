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


def _provenance(tmp_path: Path, query_ids: tuple[str, ...] = ("q1",)) -> Path:
    from evidence_rag.materializer.provenance import MutationRecord, write_provenance

    path = tmp_path / "provenance.jsonl"
    write_provenance(
        path,
        [
            MutationRecord(
                query_id=qid, needle_document_id="d1",
                counterfactual_document_id="cf::d1", gold_value="Kennedy",
                gold_alias_used="Kennedy", replacement_value="Nixon", string_class="name-1",
                seed=42, char_span=(0, 7), text_hash_before="a", text_hash_after="b",
                answer_bank_hash="h",
            )
            for qid in query_ids
        ],
    )
    return path


def _injected_candidates(tmp_path: Path, extra_document_id: str) -> Path:
    """Window = needle d1, its counterfactual twin, and one more passage."""
    path = tmp_path / "injected_sets.jsonl"
    row = CandidateSet(
        query_id="q1",
        candidates=(
            _candidate("e1", "d1", 1),
            _candidate("e2", "cf::d1", 2),
            _candidate("e3", extra_document_id, 3),
        ),
    )
    path.write_text(row.model_dump_json() + "\n", encoding="utf-8")
    return path


def test_needle_parent_inflation_detected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The gate's behaviour turns on whether the GOLD cluster was inflated, which the
    window-wide rate cannot show. Here d9 is another passage of the needle's own article."""
    index = _index(tmp_path, {"d1": "page a", "cf::d1": "page a", "d9": "page a"})
    main([
        "--candidates", str(_injected_candidates(tmp_path, "d9")),
        "--parent-index", str(index),
        "--provenance", str(_provenance(tmp_path)),
    ])
    report = json.loads(capsys.readouterr().out)
    assert report["n_injected_scored"] == 1
    assert report["needle_parent_inflated"] == 1
    assert report["needle_parent_inflation_rate"] == 1.0
    assert report["cf_shares_needle_parent"] == 1


def test_counterfactual_twin_alone_is_not_inflation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The twin shares the needle's parent BY CONSTRUCTION (it is a copy of that passage), so
    counting it would report injector bookkeeping as corpus structure."""
    index = _index(tmp_path, {"d1": "page a", "cf::d1": "page a", "d9": "page b"})
    main([
        "--candidates", str(_injected_candidates(tmp_path, "d9")),
        "--parent-index", str(index),
        "--provenance", str(_provenance(tmp_path)),
    ])
    report = json.loads(capsys.readouterr().out)
    assert report["needle_parent_inflated"] == 0
    assert report["needle_parent_inflation_rate"] == 0.0
    assert report["cf_shares_needle_parent"] == 1


def test_injected_diagnostics_absent_without_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main([
        "--candidates", str(_candidates(tmp_path)),
        "--parent-index", str(_index(tmp_path, {"d1": "page a"})),
    ])
    report = json.loads(capsys.readouterr().out)
    assert "needle_parent_inflation_rate" not in report
