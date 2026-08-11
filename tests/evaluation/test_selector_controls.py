import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from evidence_rag.contracts.models import EvidenceCandidate
from evidence_rag.evaluation.selector_components import (
    build_selector_component_artifacts,
    freeze_selector_component_artifacts,
)
from evidence_rag.evaluation.selector_controls import (
    COUNT_MATCHED_MANIFEST_FILE,
    COUNT_MATCHED_PROTOCOL_FILE,
    TOPK_MANIFEST_FILE,
    TOPK_REPORT_FILE,
    TOPK_ROWS_FILE,
    build_count_matched_protocol_artifacts,
    build_topk_control_artifacts,
    count_matched_bottom_rank_drop,
    count_matched_random_drop,
    freeze_count_matched_protocol_artifacts,
    freeze_topk_control_artifacts,
    generate_count_matched_drops,
    repeat_seed_token,
    verify_count_matched_protocol_artifacts,
    verify_topk_control_artifacts,
)


@dataclass(frozen=True)
class _Fixture:
    manifest: Path
    parents: Path
    candidates: Path
    assignments: Path | None


def _jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _candidate_docs(query_id: str, ranked: dict[int, str]) -> list[str]:
    values: list[str] = []
    for rank in range(1, 21):
        values.append(ranked.get(rank, f"{query_id}-noise-{rank}"))
    assert len(values) == len(set(values))
    return values


def _fixture(
    tmp_path: Path,
    *,
    cases: dict[str, tuple[tuple[str, ...], list[str]]],
    assignments: list[dict[str, object]] | None,
    dataset_id: str,
) -> _Fixture:
    root = tmp_path / dataset_id
    root.mkdir()
    document_ids = {
        document_id for gold, candidates in cases.values() for document_id in (*gold, *candidates)
    }
    if assignments is not None:
        document_ids.update(str(row["harmful_document_id"]) for row in assignments)
    _jsonl(
        root / "documents.jsonl",
        [
            {
                "schema_version": "1.0",
                "document_id": document_id,
                "text": f"text for {document_id}",
                "source_uri": f"toy://{document_id}",
            }
            for document_id in sorted(document_ids)
        ],
    )
    _jsonl(
        root / "queries.jsonl",
        [
            {"schema_version": "1.0", "query_id": query_id, "text": f"Question {query_id}?"}
            for query_id in sorted(cases)
        ],
    )
    _jsonl(
        root / "gold_cases.jsonl",
        [
            {"query_id": query_id, "relevant_document_ids": list(cases[query_id][0])}
            for query_id in sorted(cases)
        ],
    )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_id": dataset_id,
                "dataset_version": "toy-v1",
                "split": "source-split",
                "documents_file": "documents.jsonl",
                "queries_file": "queries.jsonl",
                "gold_cases_file": "gold_cases.jsonl",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    parents = _jsonl(
        root / "source_parent.jsonl",
        [
            {"document_id": document_id, "source_parent_id": f"parent-{document_id}"}
            for document_id in sorted(document_ids)
        ],
    )
    retriever = {
        "schema_version": "1.0",
        "name": "hybrid",
        "implementation_version": "hybrid-v1",
        "parameters_sha256": "a" * 64,
    }
    candidates = _jsonl(
        root / "candidate_sets.jsonl",
        [
            {
                "schema_version": "1.0",
                "query_id": query_id,
                "retriever": retriever,
                "candidates": [
                    {
                        "schema_version": "1.0",
                        "evidence_id": f"{query_id}-evidence-{rank}",
                        "document_id": document_id,
                        "chunk_id": f"{query_id}-chunk-{rank}",
                        "text": f"candidate {document_id}",
                        "source_uri": f"toy://{document_id}",
                        "retrieval_score": float(21 - rank),
                        "retrieval_rank": rank,
                    }
                    for rank, document_id in enumerate(cases[query_id][1], start=1)
                ],
            }
            for query_id in sorted(cases)
        ],
    )
    assignment_path = (
        _jsonl(root / "assignments.jsonl", assignments) if assignments is not None else None
    )
    return _Fixture(manifest, parents, candidates, assignment_path)


def _freeze_components(
    tmp_path: Path,
    fixture: _Fixture,
    *,
    dataset_kind: str,
) -> Path:
    output = tmp_path / f"components-{dataset_kind}"
    artifacts = build_selector_component_artifacts(
        dataset_kind=dataset_kind,  # type: ignore[arg-type]
        source_split="dev",
        dataset_manifest_path=fixture.manifest,
        source_parent_path=fixture.parents,
        candidate_pool_path=fixture.candidates,
        assignment_path=fixture.assignments,
    )
    freeze_selector_component_artifacts(output, artifacts)
    return output


def _rows(payload: bytes) -> list[dict[str, object]]:
    return [json.loads(line) for line in payload.decode("utf-8").splitlines()]


def _niah_fixture(tmp_path: Path) -> _Fixture:
    cases = {
        "q1": (
            ("q1-gold",),
            _candidate_docs("q1", {9: "q1-harm", 10: "q1-gold"}),
        ),
        "q2": (
            ("q2-gold",),
            _candidate_docs("q2", {1: "q2-gold", 15: "q2-harm"}),
        ),
        "q3": (("q3-gold",), _candidate_docs("q3", {1: "q3-gold"})),
        "q-unused": (
            ("unused-gold",),
            _candidate_docs("q-unused", {1: "unused-gold"}),
        ),
    }
    assignments: list[dict[str, object]] = []
    for query_id in ("q1", "q2", "q3"):
        harmful = f"{query_id}-harm"
        gold = f"{query_id}-gold"
        assignments.append(
            {
                "query_id": query_id,
                "required_document_ids": [gold],
                "harmful_document_id": harmful,
                "source_parent_ids": [f"parent-{gold}", f"parent-{harmful}"],
                "synthetic_family": f"family-{query_id}",
            }
        )
    return _fixture(
        tmp_path,
        cases=cases,
        assignments=assignments,
        dataset_id="niah-toy",
    )


def test_topk_controls_use_s0_prefix_document_metrics_and_three_harm_denominators(
    tmp_path: Path,
) -> None:
    fixture = _niah_fixture(tmp_path)
    components = _freeze_components(tmp_path, fixture, dataset_kind="niah")
    artifacts = build_topk_control_artifacts(
        dataset_kind="niah",
        source_split="dev",
        dataset_manifest_path=fixture.manifest,
        source_parent_path=fixture.parents,
        candidate_pool_path=fixture.candidates,
        component_directory=components,
        assignment_path=fixture.assignments,
    )
    rows = _rows(artifacts.files[TOPK_ROWS_FILE])
    assert len(rows) == 12  # assignment subset only; q-unused is deliberately absent
    q1_k9 = next(row for row in rows if row["query_id"] == "q1" and row["k"] == 9)
    assert q1_k9["selected_evidence_ids"] == [f"q1-evidence-{rank}" for rank in range(1, 10)]
    assert "q1-evidence-11" not in q1_k9["selected_evidence_ids"]
    assert q1_k9["document_recall"] == 0.0
    assert q1_k9["topk10_relative_recall_loss"] == 1.0
    assert q1_k9["conditional_chain_loss"] == 1.0
    assert q1_k9["pool_conditional_harmful_exposure"] == 1.0

    topk8 = artifacts.report["overall"]["topk8"]  # type: ignore[index]
    assert topk8["harmful"]["denominators"] == {  # type: ignore[index]
        "pool_conditional": 2,
        "baseline_exposed": 1,
        "unconditional": 3,
    }
    assert topk8["harmful"]["pool_conditional_reduction_from_topk10"] == 0.5  # type: ignore[index]
    assert topk8["harmful"]["baseline_exposed_deletion"] == 1.0  # type: ignore[index]
    assert topk8["harmful"]["unconditional_exposure"] == 0.0  # type: ignore[index]
    assert topk8["conditional_chain"]["n_topk10_chain_eligible"] == 3  # type: ignore[index]

    paired = artifacts.report["paired_vs_topk10"]["topk10_vs_topk8"]  # type: ignore[index]
    assert paired["recall_loss"]["delta"] == pytest.approx(1 / 3)  # type: ignore[index]
    assert paired["conditional_chain_loss"]["delta"] == pytest.approx(1 / 3)  # type: ignore[index]
    pool_paired = paired["pool_conditional_harmful_reduction"]  # type: ignore[index]
    assert pool_paired["delta"] == 0.5
    assert pool_paired["n_queries"] == 2
    assert pool_paired["n_total"] == 3
    assert pool_paired["n_unscored"] == 1
    baseline_paired = paired["baseline_exposed_harmful_deletion"]  # type: ignore[index]
    assert baseline_paired["delta"] == 1.0
    assert baseline_paired["n_queries"] == 1
    assert paired["unconditional_harmful_reduction"]["delta"] == pytest.approx(1 / 3)  # type: ignore[index]
    for role, role_comparisons in artifacts.report["paired_vs_topk10_by_role"].items():  # type: ignore[union-attr]
        role_query_count = sum(row["role"] == role and row["k"] == 10 for row in rows)
        role_recall = role_comparisons["topk10_vs_topk8"]["recall_loss"]
        assert role_recall["n_total"] == role_query_count
        assert role_recall["n_clusters"] >= 1

    manifest = json.loads(artifacts.files[TOPK_MANIFEST_FILE])
    assert (
        manifest["inputs"]["selector_components"]["selector_components_manifest.json"]["sha256"]
        == hashlib.sha256(
            (components / "selector_components_manifest.json").read_bytes()
        ).hexdigest()
    )
    assert (
        manifest["outputs"][TOPK_ROWS_FILE]["sha256"]
        == hashlib.sha256(artifacts.files[TOPK_ROWS_FILE]).hexdigest()
    )


def test_twowiki_topk_rows_do_not_invent_harmful_metrics(tmp_path: Path) -> None:
    fixture = _fixture(
        tmp_path,
        cases={
            "q": (
                ("gold-1", "gold-2"),
                _candidate_docs("q", {1: "gold-1", 10: "gold-2"}),
            )
        },
        assignments=None,
        dataset_id="2wiki-toy",
    )
    components = _freeze_components(tmp_path, fixture, dataset_kind="2wiki")
    artifacts = build_topk_control_artifacts(
        dataset_kind="2wiki",
        source_split="dev",
        dataset_manifest_path=fixture.manifest,
        source_parent_path=fixture.parents,
        candidate_pool_path=fixture.candidates,
        component_directory=components,
    )
    rows = _rows(artifacts.files[TOPK_ROWS_FILE])
    assert len(rows) == 4
    assert all("harmful_document_id" not in row for row in rows)
    assert all("pool_conditional_harmful_exposure" not in row for row in rows)
    assert all("harmful" not in system for system in artifacts.report["overall"].values())  # type: ignore[union-attr]


def test_topk_builder_rejects_sealed_or_heldout_even_when_called_directly(
    tmp_path: Path,
) -> None:
    fixture = _niah_fixture(tmp_path)
    with pytest.raises(ValueError, match="sealed/heldout is forbidden"):
        build_topk_control_artifacts(
            dataset_kind="niah",
            source_split="heldout",  # type: ignore[arg-type]
            dataset_manifest_path=fixture.manifest,
            source_parent_path=fixture.parents,
            candidate_pool_path=fixture.candidates,
            component_directory=tmp_path / "unused-components",
            assignment_path=fixture.assignments,
        )


def test_topk_artifacts_are_write_once_and_verify_by_full_recomputation(tmp_path: Path) -> None:
    fixture = _niah_fixture(tmp_path)
    components = _freeze_components(tmp_path, fixture, dataset_kind="niah")
    arguments = {
        "dataset_kind": "niah",
        "source_split": "dev",
        "dataset_manifest_path": fixture.manifest,
        "source_parent_path": fixture.parents,
        "candidate_pool_path": fixture.candidates,
        "component_directory": components,
        "assignment_path": fixture.assignments,
    }
    first = build_topk_control_artifacts(**arguments)  # type: ignore[arg-type]
    second = build_topk_control_artifacts(**arguments)  # type: ignore[arg-type]
    assert first.files == second.files
    output = tmp_path / "topk-output"
    freeze_topk_control_artifacts(output, first)
    verify_topk_control_artifacts(output_directory=output, **arguments)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="refusing to overwrite"):
        freeze_topk_control_artifacts(output, first)
    (output / TOPK_REPORT_FILE).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from recomputation"):
        verify_topk_control_artifacts(output_directory=output, **arguments)  # type: ignore[arg-type]


def _candidate(rank: int) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=f"e{rank}",
        document_id=f"d{rank}",
        chunk_id=f"c{rank}",
        text=f"candidate {rank}",
        source_uri=f"toy://d{rank}",
        retrieval_score=float(11 - rank),
        retrieval_rank=rank,
    )


def test_count_matched_seed_golden_vectors_and_nested_sha_priority() -> None:
    assert repeat_seed_token(0) == (
        "2a1bf681ff4a25881dafb3008a682541b7124a4ed58772ae3da19e8fa4a14ec6"
    )
    assert repeat_seed_token(99) == (
        "e43e3c63caf51d9d55c9fbe3c8cceb53e1965e9b55550584c9243ed71b07c611"
    )
    s0 = tuple(_candidate(rank) for rank in range(1, 11))
    common = {
        "repeat_index": 7,
        "dataset_id": "dataset",
        "dataset_signature": "a" * 64,
        "pool_sha256": "b" * 64,
        "query_id": "q",
    }
    one = count_matched_random_drop(s0, deletion_count=1, **common)  # type: ignore[arg-type]
    two = count_matched_random_drop(s0, deletion_count=2, **common)  # type: ignore[arg-type]
    three = count_matched_random_drop(s0, deletion_count=3, **common)  # type: ignore[arg-type]
    assert one == two[:1] == three[:1]
    assert two == three[:2]
    assert count_matched_bottom_rank_drop(s0, deletion_count=3) == ("e10", "e9", "e8")
    generated = generate_count_matched_drops(
        s0,
        deletion_count=3,
        **common,  # type: ignore[arg-type]
    )
    assert generated.random_dropped_evidence_ids == three
    assert generated.bottom_rank_dropped_evidence_ids == ("e10", "e9", "e8")
    with pytest.raises(ValueError, match="deletion_count"):
        count_matched_bottom_rank_drop(s0, deletion_count=4)


def test_count_matched_protocol_has_100_seeds_no_fake_results_and_is_write_once(
    tmp_path: Path,
) -> None:
    artifacts = build_count_matched_protocol_artifacts()
    protocol = json.loads(artifacts.files[COUNT_MATCHED_PROTOCOL_FILE])
    assert protocol["repeat_count"] == 100
    assert len(protocol["repeats"]) == 100
    assert protocol["status"] == "GENERATOR_PROTOCOL_ONLY_NO_SELECTOR_TRACE"
    assert "results" not in protocol
    manifest = json.loads(artifacts.files[COUNT_MATCHED_MANIFEST_FILE])
    assert manifest["inputs"] == {}
    assert (
        manifest["outputs"][COUNT_MATCHED_PROTOCOL_FILE]["sha256"]
        == hashlib.sha256(artifacts.files[COUNT_MATCHED_PROTOCOL_FILE]).hexdigest()
    )
    assert manifest["golden_vectors"]["random_priority"]["priority_sha256"] == (
        "e942586160b9b9f619719eaa23163f63447641e6afd9597ee634a2fc166e669b"
    )

    output = tmp_path / "protocol"
    freeze_count_matched_protocol_artifacts(output, artifacts)
    verify_count_matched_protocol_artifacts(output)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        freeze_count_matched_protocol_artifacts(output, artifacts)
    (output / COUNT_MATCHED_PROTOCOL_FILE).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from recomputation"):
        verify_count_matched_protocol_artifacts(output)
