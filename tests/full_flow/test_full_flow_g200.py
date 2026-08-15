from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from evidence_rag.contracts.models import EvidenceCandidate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_flow_g200 as g200  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _candidate(rank: int) -> dict[str, object]:
    if rank == 1:
        document_id = "cf::harm"
        source_uri = "synthetic://cf/harm"
        text = "The answer is deliberately wrong."
    elif rank == 5:
        document_id = "support"
        source_uri = "memory://support"
        text = "The event happened in 2009."
    else:
        document_id = f"benign-{rank}"
        source_uri = f"memory://benign-{rank}"
        text = f"Unrelated benign context {rank}."
    return EvidenceCandidate(
        evidence_id=f"e{rank}",
        document_id=document_id,
        chunk_id=f"c{rank}",
        text=text,
        source_uri=source_uri,
        retrieval_score=float(11 - rank),
        retrieval_rank=rank,
    ).model_dump(mode="json")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    query_ids = ("train", "val")
    paths = {
        name: tmp_path / name
        for name in (
            "queries.jsonl",
            "candidate_sets.jsonl",
            "gold_cases.jsonl",
            "roles.jsonl",
            "components.jsonl",
            "selection.jsonl",
            "qa2d.jsonl",
        )
    }
    _write_jsonl(
        paths["queries.jsonl"],
        [
            {"schema_version": "1.0", "query_id": query_id, "text": "When did it happen?"}
            for query_id in query_ids
        ],
    )
    _write_jsonl(
        paths["candidate_sets.jsonl"],
        [
            {
                "schema_version": "1.0",
                "query_id": query_id,
                "candidates": [_candidate(rank) for rank in range(1, 11)],
            }
            for query_id in query_ids
        ],
    )
    _write_jsonl(
        paths["gold_cases.jsonl"],
        [
            {
                "query_id": query_id,
                "reference_answers": ["2009"],
                "relevant_document_ids": ["support"],
            }
            for query_id in query_ids
        ],
    )
    roles = {"train": g200.TRAIN_ROLE, "val": g200.VALIDATION_ROLE}
    _write_jsonl(
        paths["roles.jsonl"],
        [
            {
                "schema_version": "1.0",
                "query_id": query_id,
                "role": roles[query_id],
                "component_id": f"component-{query_id}",
            }
            for query_id in query_ids
        ],
    )
    _write_jsonl(
        paths["components.jsonl"],
        [
            {
                "schema_version": "1.0",
                "query_id": query_id,
                "component_id": f"component-{query_id}",
                "component_root": f"root-{query_id}",
                "component_size": 1,
                "allowed_keys": [],
            }
            for query_id in query_ids
        ],
    )
    _write_jsonl(
        paths["selection.jsonl"],
        [
            {
                "schema_version": "full-flow-g200-selection-row-v1",
                "query_id": query_id,
                "role": roles[query_id],
                "component_id": f"component-{query_id}",
                "selected_evidence_ids": [f"e{rank}" for rank in range(2, 11)],
                "dropped_evidence_ids": ["e1"],
            }
            for query_id in query_ids
        ],
    )
    _write_jsonl(
        paths["qa2d.jsonl"],
        [
            {
                "schema_version": "full-flow-g200-qa2d-target-v1",
                "query_id": query_id,
                "role": roles[query_id],
                "component_id": f"component-{query_id}",
                "question": "When did it happen?",
                "answer": "2009",
                "declarative": "The event happened in 2009.",
                "answer_preserved": True,
                "model": g200.QA2D_MODEL_ID,
                "revision": g200.QA2D_REVISION,
            }
            for query_id in query_ids
        ],
    )
    return paths


def test_materialize_builds_equal_clean_and_mixed_draft_examples(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "output"

    manifest = g200.materialize(
        queries_path=paths["queries.jsonl"],
        candidate_pool_path=paths["candidate_sets.jsonl"],
        gold_path=paths["gold_cases.jsonl"],
        role_assignments_path=paths["roles.jsonl"],
        component_map_path=paths["components.jsonl"],
        selection_trace_path=paths["selection.jsonl"],
        qa2d_targets_path=paths["qa2d.jsonl"],
        target_support_audit_path=None,
        output_dir=output,
    )

    assert manifest["train_queries"] == 1
    assert manifest["validation_queries"] == 1
    assert manifest["gc_examples"] == manifest["gm_examples"] == len(g200.VARIANT_NAMES)
    train = json.loads((output / "train_cases.jsonl").read_text(encoding="utf-8"))
    variants = train["variants"]
    assert set(variants) == set(g200.VARIANT_NAMES)
    assert variants["support_only"]["target"] == "The event happened in 2009 [1]."
    assert variants["topk"]["target"] == "The event happened in 2009 [5]."
    assert variants["legacy_selected"]["target"] == "The event happened in 2009 [4]."
    assert variants["support_first"]["target"] == "The event happened in 2009 [1]."
    assert variants["support_last"]["target"] == "The event happened in 2009 [10]."
    assert "Answer the question using only the evidence below." in variants["topk"]["prompt"]
    assert "(e5) The event happened in 2009." in variants["topk"]["prompt"]
    assert {train["semantic_target"]} == {
        variants[name]["target"].rsplit(" [", 1)[0] + "." for name in g200.VARIANT_NAMES
    }


def test_variant_is_rejected_when_selector_removed_answer_support() -> None:
    context = tuple(
        EvidenceCandidate.model_validate(_candidate(rank))
        for rank in range(1, 11)
        if rank != 5
    )

    variant = g200._variant_row(
        question="When did it happen?",
        answer="2009",
        declarative="The event happened in 2009.",
        candidates=context,
        relevant_document_ids=frozenset({"support"}),
    )

    assert variant is None


def test_materialize_rejects_component_crossing_splits(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    rows = [json.loads(line) for line in paths["roles.jsonl"].read_text().splitlines()]
    rows[1]["component_id"] = rows[0]["component_id"]
    _write_jsonl(paths["roles.jsonl"], rows)

    with pytest.raises(ValueError, match="components cross train/model-val"):
        g200.materialize(
            queries_path=paths["queries.jsonl"],
            candidate_pool_path=paths["candidate_sets.jsonl"],
            gold_path=paths["gold_cases.jsonl"],
            role_assignments_path=paths["roles.jsonl"],
            component_map_path=paths["components.jsonl"],
            selection_trace_path=paths["selection.jsonl"],
            qa2d_targets_path=paths["qa2d.jsonl"],
            target_support_audit_path=None,
            output_dir=tmp_path / "output",
        )


def test_qa2d_export_records_answer_preservation(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    def fake_generate(pairs: list[tuple[str, str]]) -> list[str]:
        assert pairs == [("When did it happen?", "2009"), ("When did it happen?", "2009")]
        return ["The event happened in 2009 .", "The date was omitted."]

    manifest = g200.export_qa2d_targets(
        queries_path=paths["queries.jsonl"],
        gold_path=paths["gold_cases.jsonl"],
        role_assignments_path=paths["roles.jsonl"],
        output_dir=tmp_path / "qa2d-output",
        model_id=g200.QA2D_MODEL_ID,
        revision=g200.QA2D_REVISION,
        batch_size=2,
        generator=fake_generate,
    )

    assert manifest["answer_preserved"] == 1
    assert manifest["answer_not_preserved"] == 1
    target_rows = [
        json.loads(line)
        for line in (tmp_path / "qa2d-output/qa2d_targets.jsonl").read_text().splitlines()
    ]
    assert target_rows[0]["declarative"] == "The event happened in 2009."
    assert target_rows[0]["answer_preserved"] is True
    assert target_rows[1]["answer_preserved"] is False


def test_answer_matching_requires_token_boundaries_and_rejects_unknown() -> None:
    assert g200._contains_normalised("Heather West", "Heather West won.") is True
    assert g200._contains_normalised("Heather West", "Sheather West won.") is False
    assert g200._is_unknown_reference("unknown") is True
    assert g200._is_unknown_reference("a known answer") is False


def test_target_support_audit_records_true_scores(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    preaudit = tmp_path / "preaudit"
    g200.materialize(
        queries_path=paths["queries.jsonl"],
        candidate_pool_path=paths["candidate_sets.jsonl"],
        gold_path=paths["gold_cases.jsonl"],
        role_assignments_path=paths["roles.jsonl"],
        component_map_path=paths["components.jsonl"],
        selection_trace_path=paths["selection.jsonl"],
        qa2d_targets_path=paths["qa2d.jsonl"],
        target_support_audit_path=None,
        output_dir=preaudit,
    )

    scores = iter((0.9, 0.4))
    manifest = g200.audit_target_support(
        train_cases_path=preaudit / "train_cases.jsonl",
        validation_cases_path=preaudit / "validation_cases.jsonl",
        candidate_pool_path=paths["candidate_sets.jsonl"],
        true_snapshot=tmp_path / "true",
        output_dir=tmp_path / "audit",
        scorer=lambda _premise, _hypothesis: next(scores),
    )

    assert manifest["queries"] == 2
    assert manifest["entailed"] == 1
    assert manifest["not_entailed"] == 1
