from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import pytest

import evidence_rag.evaluation.selector_lean as lean
from evidence_rag.contracts.models import EvidenceCandidate
from evidence_rag.evaluation.selector_lean import (
    DevelopmentCandidateResult,
    FrozenPolicy,
    PolicyCandidate,
    apply_seed_policy,
    build_evaluation_projection,
    build_policy_candidates,
    combine_development_metrics,
    combine_projection_sha256,
    evaluate_seed_policy,
    evidence_inference,
    freeze_policy,
    macro_answer_inference,
    nearest_rank_float32,
    require_frozen_final_policy,
    select_development_policy,
    thresholds_from_train_scores,
)
from evidence_rag.selector.models import CandidateRiskScore


def _result(
    *,
    quantile: float = 0.99,
    cap: int = 1,
    mean_harm: tuple[float, float] = (0.04, 0.03),
    mean_deletions: float = 0.3,
    seed13_loss: float = 0.5,
    seed42_loss: float = 2.0,
    controls: bool = True,
) -> DevelopmentCandidateResult:
    return DevelopmentCandidateResult(
        quantile=quantile,
        cap=cap,
        thresholds_by_seed=((13, 0.75), (42, 0.5)),
        mean_deletions_per_query=mean_deletions,
        harmful_reduction_seed13=mean_harm[0],
        harmful_reduction_seed42=mean_harm[1],
        niah_recall_loss_seed13_pp=seed13_loss,
        niah_recall_loss_seed42_pp=seed42_loss,
        twowiki_recall_loss_seed13_pp=seed13_loss,
        twowiki_recall_loss_seed42_pp=seed42_loss,
        niah_chain_loss_seed13_pp=seed13_loss,
        niah_chain_loss_seed42_pp=seed42_loss,
        twowiki_chain_loss_seed13_pp=seed13_loss,
        twowiki_chain_loss_seed42_pp=seed42_loss,
        seed13_harm_beats_random=controls,
        seed13_harm_beats_bottom=controls,
        seed13_precision_beats_random=controls,
        seed13_precision_beats_bottom=controls,
    )


def test_nearest_rank_uses_canonical_float32_and_ceil_index() -> None:
    values = [0.1, 0.2, 0.3, 0.4]
    expected = struct.unpack("!f", struct.pack("!f", 0.3))[0]

    assert nearest_rank_float32(values, 0.75) == expected
    with pytest.raises(ValueError, match="empty"):
        nearest_rank_float32([], 0.95)
    with pytest.raises(ValueError, match="greater than zero"):
        nearest_rank_float32(values, 0.0)


def test_thresholds_and_policy_grid_are_global_per_seed_and_keep_duplicate_values() -> None:
    thresholds = thresholds_from_train_scores({13: [0.25] * 100, 42: [0.5] * 100})
    policies = build_policy_candidates(thresholds)

    assert len(thresholds) == 8
    assert [row.quantile for row in thresholds[:4]] == list(lean.LEAN_QUANTILES)
    assert [row.nearest_rank for row in thresholds[:4]] == [100, 99, 98, 95]
    assert policies[0].policy_enabled is False
    assert len(policies) == 9
    assert [row.policy_id for row in policies[1:3]] == ["Q0.995_CAP1", "Q0.995_CAP2"]
    assert policies[1].thresholds_by_seed == ((13, 0.25), (42, 0.5))

    with pytest.raises(ValueError, match="exactly seeds"):
        thresholds_from_train_scores({13: [0.1]})


def test_combined_projection_hash_binds_both_datasets_and_one_role() -> None:
    niah = lean.EvaluationProjection(
        dataset_kind="niah",
        role="crc-calibration",
        rows=(),
        sha256="a" * 64,
    )
    twowiki = lean.EvaluationProjection(
        dataset_kind="2wiki",
        role="crc-calibration",
        rows=(),
        sha256="b" * 64,
    )
    digest = combine_projection_sha256({"niah": niah, "2wiki": twowiki})
    assert len(digest) == 64
    with pytest.raises(ValueError, match="common role"):
        combine_projection_sha256(
            {
                "niah": niah,
                "2wiki": lean.EvaluationProjection(
                    dataset_kind="2wiki",
                    role="decision-dev",
                    rows=(),
                    sha256="c" * 64,
                ),
            }
        )


def _policy_projection(dataset_kind: str) -> lean.EvaluationProjection:
    candidates = tuple(
        EvidenceCandidate(
            evidence_id=f"{dataset_kind}-ev-{rank}",
            document_id=(
                f"{dataset_kind}-harm"
                if dataset_kind == "niah" and rank == 2
                else f"{dataset_kind}-doc-{rank}"
            ),
            chunk_id=f"{dataset_kind}-chunk-{rank}",
            text=f"candidate {rank}",
            source_uri=f"fixture://{dataset_kind}/{rank}",
            retrieval_score=1.0 / rank,
            retrieval_rank=rank,
        )
        for rank in range(1, 11)
    )
    row = lean.EvaluationQuery(
        dataset_kind=dataset_kind,  # type: ignore[arg-type]
        dataset_id=f"{dataset_kind}/toy",
        dataset_signature="a" * 64,
        pool_sha256="b" * 64,
        role="crc-calibration",
        query_id=f"{dataset_kind}-query",
        question="question",
        component_id=f"{dataset_kind}-component",
        chain_eligible_topk10=True,
        topk10=candidates,
        required_document_ids=(f"{dataset_kind}-doc-1",),
        reference_answers=("answer",),
        harmful_document_id=f"{dataset_kind}-harm" if dataset_kind == "niah" else None,
        harmful_in_top20_pool=dataset_kind == "niah",
    )
    return lean.EvaluationProjection(
        dataset_kind=dataset_kind,  # type: ignore[arg-type]
        role="crc-calibration",
        rows=(row,),
        sha256=("c" if dataset_kind == "niah" else "d") * 64,
    )


def test_real_policy_metrics_reward_harmful_drop_without_losing_required_evidence() -> None:
    projections = {"niah": _policy_projection("niah"), "2wiki": _policy_projection("2wiki")}
    scores: dict[str, dict[str, dict[str, CandidateRiskScore]]] = {}
    for dataset_kind, projection in projections.items():
        row = projection.rows[0]
        scores[dataset_kind] = {
            row.query_id: {
                candidate.evidence_id: CandidateRiskScore(
                    protect_score=0.05,
                    harm_score=0.95,
                )
                if candidate.document_id.endswith("-harm")
                else CandidateRiskScore(protect_score=0.95, harm_score=0.05)
                for candidate in row.topk10
            }
        }

    seed13 = evaluate_seed_policy(
        seed=13,
        projections=projections,  # type: ignore[arg-type]
        scores_by_dataset=scores,  # type: ignore[arg-type]
        threshold=0.8,
        cap=1,
        include_controls=True,
    )
    seed42 = evaluate_seed_policy(
        seed=42,
        projections=projections,  # type: ignore[arg-type]
        scores_by_dataset=scores,  # type: ignore[arg-type]
        threshold=0.8,
        cap=1,
        include_controls=False,
    )
    result = combine_development_metrics(
        quantile=0.99,
        cap=1,
        thresholds_by_seed=((13, 0.8), (42, 0.8)),
        seed13=seed13,
        seed42=seed42,
    )

    assert seed13.niah_harmful_reduction == 1.0
    assert seed13.niah_deletion_precision == 1.0
    assert seed13.niah_recall_loss_pp == 0.0
    assert seed13.twowiki_recall_loss_pp == 0.0
    assert result.passes_gate is True

    decisions = apply_seed_policy(
        projections=projections,  # type: ignore[arg-type]
        scores_by_dataset=scores,  # type: ignore[arg-type]
        threshold=0.8,
        cap=1,
    )
    inference = evidence_inference(
        decisions=decisions,
        projections=projections,  # type: ignore[arg-type]
        include_controls=True,
        random_repeats=10,
    )
    harm = inference["niah_harmful_reduction"]
    assert isinstance(harm, dict)
    selector_vs_topk = harm["selector_vs_topk10"]
    assert isinstance(selector_vs_topk, dict)
    assert selector_vs_topk["delta"] == 1.0
    assert selector_vs_topk["ci_low"] == 1.0


def test_macro_answer_inference_weights_datasets_equally() -> None:
    result = macro_answer_inference(
        selector_by_dataset={"niah": {"n1": 1.0, "n2": 0.0}, "2wiki": {"w1": 1.0}},
        topk10_by_dataset={"niah": {"n1": 0.0, "n2": 0.0}, "2wiki": {"w1": 1.0}},
        components_by_dataset={
            "niah": {"n1": "nc1", "n2": "nc2"},
            "2wiki": {"w1": "wc1"},
        },
        iterations=100,
        seed=13,
    )
    assert result["dataset_delta"] == {"niah": 0.5, "2wiki": 0.0}
    assert result["macro_delta"] == 0.25


def _grid() -> tuple[
    tuple[PolicyCandidate, ...], dict[tuple[float, int], DevelopmentCandidateResult]
]:
    thresholds = thresholds_from_train_scores({13: [0.75] * 100, 42: [0.5] * 100})
    policies = build_policy_candidates(thresholds)
    results = {
        (row.quantile, row.cap): _result(quantile=row.quantile, cap=row.cap)
        for row in policies
        if row.policy_enabled and row.quantile is not None and row.cap is not None
    }
    return policies, results


def test_development_selection_applies_gates_then_conservative_equivalence_tie_break() -> None:
    global_best = _result(
        quantile=0.95,
        cap=2,
        mean_harm=(0.06, 0.05),
        mean_deletions=0.8,
    )
    within_half_pp = _result(
        quantile=0.99,
        cap=1,
        mean_harm=(0.055, 0.05),
        mean_deletions=0.2,
    )
    outside_band = _result(
        quantile=0.995,
        cap=1,
        mean_harm=(0.04, 0.04),
        mean_deletions=0.1,
    )
    unsafe = _result(
        quantile=0.975,
        cap=1,
        mean_harm=(0.2, 0.2),
        seed13_loss=1.1,
    )

    policies, results = _grid()
    for result in (global_best, within_half_pp, outside_band, unsafe):
        results[(result.quantile, result.cap)] = result
    selection = select_development_policy(list(results.values()), policy_candidates=policies)

    assert selection is not None
    assert selection.global_max_mean_harmful_reduction == pytest.approx(0.055)
    assert selection.selected is within_half_pp
    all_unsafe = [_result(quantile=quantile, cap=cap, seed13_loss=1.1) for quantile, cap in results]
    assert select_development_policy(all_unsafe, policy_candidates=policies) is None


def test_development_gate_requires_both_seed_directions_controls_and_nonzero_action() -> None:
    failing = (
        _result(quantile=0.995, cap=1, mean_harm=(0.0, 0.1)),
        _result(quantile=0.995, cap=2, mean_harm=(0.1, -0.01)),
        _result(quantile=0.99, cap=1, mean_deletions=0.0),
        _result(quantile=0.99, cap=2, controls=False),
        _result(quantile=0.975, cap=1, seed42_loss=3.01),
    )

    policies, results = _grid()
    results = {
        identity: _result(
            quantile=identity[0],
            cap=identity[1],
            seed13_loss=1.1,
        )
        for identity in results
    }
    for result in failing:
        results[(result.quantile, result.cap)] = result
    assert select_development_policy(list(results.values()), policy_candidates=policies) is None


@pytest.mark.parametrize(
    "field",
    (
        "niah_recall_loss_seed13_pp",
        "twowiki_recall_loss_seed13_pp",
        "niah_chain_loss_seed13_pp",
        "twowiki_chain_loss_seed13_pp",
        "niah_recall_loss_seed42_pp",
        "twowiki_recall_loss_seed42_pp",
        "niah_chain_loss_seed42_pp",
        "twowiki_chain_loss_seed42_pp",
    ),
)
def test_each_protection_metric_has_its_own_seed_specific_gate(field: str) -> None:
    values: dict[str, Any] = {
        "quantile": 0.99,
        "cap": 1,
        "thresholds_by_seed": ((13, 0.75), (42, 0.5)),
        "mean_deletions_per_query": 0.2,
        "harmful_reduction_seed13": 0.04,
        "harmful_reduction_seed42": 0.03,
        "niah_recall_loss_seed13_pp": 0.5,
        "niah_recall_loss_seed42_pp": 2.0,
        "twowiki_recall_loss_seed13_pp": 0.5,
        "twowiki_recall_loss_seed42_pp": 2.0,
        "niah_chain_loss_seed13_pp": 0.5,
        "niah_chain_loss_seed42_pp": 2.0,
        "twowiki_chain_loss_seed13_pp": 0.5,
        "twowiki_chain_loss_seed42_pp": 2.0,
        "seed13_harm_beats_random": True,
        "seed13_harm_beats_bottom": True,
        "seed13_precision_beats_random": True,
        "seed13_precision_beats_bottom": True,
    }
    values[field] = 1.01 if "seed13" in field else 3.01
    assert DevelopmentCandidateResult(**values).passes_gate is False


def test_frozen_policy_roundtrip_and_final_guard_cannot_change_grid() -> None:
    policies, results = _grid()
    selection = select_development_policy(list(results.values()), policy_candidates=policies)
    assert selection is not None
    digest = "a" * 64
    policy = freeze_policy(
        selection,
        variant="NLI-base",
        checkpoint_sha256_by_seed={13: "b" * 64, 42: "c" * 64},
        code_commit="d" * 40,
        final_input_sha256=digest,
        development_projection_sha256="e" * 64,
        generator_sha256="f" * 64,
    )

    restored = FrozenPolicy.from_dict(policy.to_dict())
    assert restored == policy
    observed: dict[str, Any] = {
        "checkpoint_sha256_by_seed": {13: "b" * 64, 42: "c" * 64},
        "code_commit": "d" * 40,
        "final_input_sha256": digest,
        "development_projection_sha256": "e" * 64,
        "generator_sha256": "f" * 64,
    }
    assert require_frozen_final_policy(restored, **observed) is restored
    with pytest.raises(ValueError, match="quantile"):
        require_frozen_final_policy(restored, requested_quantile=0.95, **observed)
    with pytest.raises(ValueError, match="cap"):
        require_frozen_final_policy(restored, requested_cap=2, **observed)
    with pytest.raises(ValueError, match="development_projection_sha256"):
        require_frozen_final_policy(
            restored,
            **{**observed, "development_projection_sha256": "0" * 64},
        )


def _jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _projection_fixture(tmp_path: Path) -> dict[str, Path]:
    dataset = tmp_path / "dataset"
    components = tmp_path / "components"
    pool = tmp_path / "pool"
    dataset.mkdir()
    components.mkdir()
    pool.mkdir()
    query_ids = ("q-dev", "q-final")
    candidate_docs: dict[str, list[str]] = {}
    for query_id in query_ids:
        clean_id = f"{query_id}-doc-1"
        candidate_docs[query_id] = [
            clean_id,
            f"cf::{query_id}::{clean_id}",
            *(f"{query_id}-doc-{rank}" for rank in range(3, 21)),
        ]
    all_documents = sorted(
        document_id for values in candidate_docs.values() for document_id in values
    )
    document_texts = {
        document_id: (
            "wrong"
            if document_id.startswith("cf::")
            else "gold"
            if document_id.endswith("-doc-1")
            else f"candidate {document_id}"
        )
        for document_id in all_documents
    }
    _jsonl(
        dataset / "documents.jsonl",
        [
            {
                "schema_version": "1.0",
                "document_id": document_id,
                "text": document_texts[document_id],
                "source_uri": f"fixture://{document_id}",
            }
            for document_id in all_documents
        ],
    )
    _jsonl(
        dataset / "queries.jsonl",
        [
            {"schema_version": "1.0", "query_id": query_id, "text": f"Question {query_id}?"}
            for query_id in query_ids
        ],
    )
    _jsonl(
        dataset / "gold.jsonl",
        [
            {
                "query_id": query_id,
                "relevant_document_ids": [candidate_docs[query_id][0]],
                "reference_answers": ["gold"],
            }
            for query_id in query_ids
        ],
    )
    manifest = dataset / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_id": "niah/toy",
                "dataset_version": "v1",
                "split": "dev",
                "documents_file": "documents.jsonl",
                "queries_file": "queries.jsonl",
                "gold_cases_file": "gold.jsonl",
            }
        ),
        encoding="utf-8",
    )
    parents = _jsonl(
        dataset / "source_parent.jsonl",
        [
            {"document_id": document_id, "source_parent_id": f"parent-{document_id}"}
            for document_id in all_documents
        ],
    )
    candidates: list[dict[str, object]] = []
    for query_id in query_ids:
        candidates.append(
            {
                "schema_version": "1.0",
                "query_id": query_id,
                "candidates": [
                    {
                        "schema_version": "1.0",
                        "evidence_id": f"{query_id}-ev-{rank}",
                        "document_id": document_id,
                        "chunk_id": f"chunk-{rank}",
                        "text": document_texts[document_id],
                        "source_uri": f"fixture://{document_id}",
                        "retrieval_score": 1.0 / rank,
                        "retrieval_rank": rank,
                    }
                    for rank, document_id in enumerate(candidate_docs[query_id], start=1)
                ],
            }
        )
    _jsonl(pool / "candidate_sets.jsonl", candidates)
    _jsonl(
        components / "role_assignments.jsonl",
        [
            {
                "schema_version": "1.0",
                "dataset_id": "niah/toy",
                "query_id": "q-dev",
                "component_id": "component-dev",
                "fold": 0,
                "role": "crc-calibration",
                "chain_eligible_topk10": True,
            },
            {
                "schema_version": "1.0",
                "dataset_id": "niah/toy",
                "query_id": "q-final",
                "component_id": "component-final",
                "fold": 1,
                "role": "decision-dev",
                "chain_eligible_topk10": True,
            },
        ],
    )
    assignment = _jsonl(
        tmp_path / "assignment.jsonl",
        [
            {
                "query_id": query_id,
                "required_document_ids": [candidate_docs[query_id][0]],
                "harmful_document_id": candidate_docs[query_id][1],
                "source_parent_ids": [f"parent-{candidate_docs[query_id][0]}"],
                "synthetic_family": "toy",
            }
            for query_id in query_ids
        ],
    )
    provenance = _jsonl(
        tmp_path / "provenance.jsonl",
        [
            {
                "query_id": query_id,
                "needle_document_id": candidate_docs[query_id][0],
                "counterfactual_document_id": candidate_docs[query_id][1],
                "gold_value": "gold",
                "gold_alias_used": "gold",
                "replacement_value": "wrong",
                "string_class": "text",
                "seed": 1,
                "char_span": [0, 4],
                "text_hash_before": hashlib.sha256(b"gold").hexdigest(),
                "text_hash_after": hashlib.sha256(b"wrong").hexdigest(),
                "answer_bank_hash": "b" * 64,
            }
            for query_id in query_ids
        ],
    )
    return {
        "manifest": manifest,
        "parents": parents,
        "pool": pool,
        "components": components,
        "assignment": assignment,
        "provenance": provenance,
    }


def test_projection_uses_frozen_role_and_topk10_without_reading_other_role(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _projection_fixture(tmp_path)
    verified: dict[str, object] = {}

    def fake_verify(**kwargs: object) -> object:
        verified.update(kwargs)
        return object()

    monkeypatch.setattr(lean, "verify_selector_component_artifacts", fake_verify)
    projection = build_evaluation_projection(
        dataset_kind="niah",
        role="crc-calibration",
        dataset_manifest_path=fixture["manifest"],
        source_parent_path=fixture["parents"],
        candidate_pool_path=fixture["pool"],
        component_directory=fixture["components"],
        assignment_path=fixture["assignment"],
        provenance_path=fixture["provenance"],
    )

    assert verified["source_split"] == "dev"
    assert verified["source_parent_path"] == fixture["parents"]
    assert projection.role == "crc-calibration"
    assert len(projection.rows) == 1
    row = projection.rows[0]
    assert row.query_id == "q-dev"
    assert row.component_id == "component-dev"
    assert tuple(candidate.retrieval_rank for candidate in row.topk10) == tuple(range(1, 11))
    assert row.required_document_ids == ("q-dev-doc-1",)
    assert row.harmful_document_id == "cf::q-dev::q-dev-doc-1"
    assert len(projection.sha256) == 64


def test_projection_rejects_tampered_counterfactual_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _projection_fixture(tmp_path)
    monkeypatch.setattr(lean, "verify_selector_component_artifacts", lambda **_: object())
    rows = [
        json.loads(line) for line in fixture["provenance"].read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["text_hash_after"] = "0" * 64
    _jsonl(fixture["provenance"], rows)

    with pytest.raises(ValueError, match="counterfactual text hash mismatch"):
        build_evaluation_projection(
            dataset_kind="niah",
            role="crc-calibration",
            dataset_manifest_path=fixture["manifest"],
            source_parent_path=fixture["parents"],
            candidate_pool_path=fixture["pool"],
            component_directory=fixture["components"],
            assignment_path=fixture["assignment"],
            provenance_path=fixture["provenance"],
        )


def test_projection_keeps_dataset_specific_inputs_separate(tmp_path: Path) -> None:
    fixture = _projection_fixture(tmp_path)
    with pytest.raises(ValueError, match="requires assignment"):
        build_evaluation_projection(
            dataset_kind="niah",
            role="crc-calibration",
            dataset_manifest_path=fixture["manifest"],
            source_parent_path=fixture["parents"],
            candidate_pool_path=fixture["pool"],
            component_directory=fixture["components"],
        )
    with pytest.raises(ValueError, match="must not receive"):
        build_evaluation_projection(
            dataset_kind="2wiki",
            role="crc-calibration",
            dataset_manifest_path=fixture["manifest"],
            source_parent_path=fixture["parents"],
            candidate_pool_path=fixture["pool"],
            component_directory=fixture["components"],
            assignment_path=fixture["assignment"],
            provenance_path=fixture["provenance"],
        )
