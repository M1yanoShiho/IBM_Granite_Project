from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from evidence_rag.contracts.models import EvidenceCandidate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import full_flow_g400_niah_qualification as g400  # noqa: E402


def _sha256(path: Path) -> str:
    return g400._sha256(path)


def _run_row(answer: str, *, baseline: bool) -> dict[str, object]:
    row: dict[str, object] = {
        "generation": {
            "schema_version": "1.0",
            "query_id": "q1",
            "answer": answer,
            "cited_evidence_ids": ["e1"] if answer else [],
        },
        "trace": {
            "claims": [
                {
                    "final_sentence": answer,
                    "citation": "e1",
                    "routing_outcome": "verified",
                }
            ]
        },
        "error": None,
    }
    if not baseline:
        row["routing"] = [
            {
                "sentence": answer,
                "citation": "e1",
                "outcome": "verified",
            }
        ]
    return row


def _candidate_file(tmp_path: Path, seed: int, answer: str) -> Path:
    output_dir = tmp_path / f"seed{seed}"
    output_dir.mkdir()
    evidence = EvidenceCandidate(
        evidence_id="e1",
        document_id="d1",
        chunk_id="c1",
        text="right",
        source_uri="fixture://1",
        retrieval_score=1.0,
        retrieval_rank=1,
    )
    row = {
        "schema_version": g400.SCHEMA_GENERATION_ROW,
        "task_id": "full::K_topk::q1",
        "scope": "full",
        "context": "K_topk",
        "dataset": "niah",
        "case_id": "q1",
        "query_id": "q1",
        "component_id": "component-1",
        "selector_changed": False,
        "variant_name": "K_topk",
        "answerable": True,
        "evidence_ids": ["e1"],
        "question": "question?",
        "evidence": [evidence.model_dump(mode="json")],
        "arm_order": [g400.CANDIDATE_ARM],
        "arms": {g400.CANDIDATE_ARM: _run_row(answer, baseline=False)},
    }
    generations = output_dir / "generations.jsonl"
    generations.write_text(json.dumps(row) + "\n", encoding="utf-8")
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": g400.SCHEMA_RUN_MANIFEST,
                "status": "COMPLETE",
                "seed": seed,
                "arms": [g400.CANDIDATE_ARM],
                "generations_sha256": _sha256(generations),
            }
        ),
        encoding="utf-8",
    )
    return generations


def test_validate_g330_run_checks_adapter_hashes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    adapter = run_dir / "adapter"
    adapter.mkdir(parents=True)
    weights = adapter / "adapter_model.safetensors"
    config = adapter / "adapter_config.json"
    weights.write_bytes(b"weights")
    config.write_text("{}", encoding="utf-8")
    manifest = {
        "schema_version": g400.SCHEMA_G300_TRAINING,
        "status": "COMPLETE",
        "run_kind": "formal",
        "recipe": "gr-c",
        "seed": 42,
        "entry_mode": "controlled_continuation_limited_internal_screen",
        "clean_freeze_ready": False,
        "g223_status": "CONTROLLED_CONTINUATION_READY",
        "g223_controlled_continuation_ready": True,
        "g223_twowiki_modelval_groups": g400.EXPECTED_TWOWIKI_MODELVAL_GROUPS,
        "full_data_train_groups": g400.EXPECTED_TRAIN_GROUPS,
        "full_data_validation_groups": g400.EXPECTED_VALIDATION_GROUPS,
        "training_examples": g400.EXPECTED_TRAINING_EXAMPLES,
        "optimizer_steps": g400.EXPECTED_OPTIMIZER_STEPS,
        "model_snapshot_config_sha256": "model-hash",
        "dev_read": False,
        "decision_dev_used": False,
        "sealed_or_heldout_read": False,
        "utility_labels_started": False,
        "adapter_scope": "draft generation call only",
        "claim_splitter_scope": "frozen Granite base with adapters disabled",
        "reload_check": {"status": "PASS"},
        "adapter_weights_sha256": _sha256(weights),
        "adapter_config_sha256": _sha256(config),
    }
    (run_dir / "training_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    audit = g400._validate_g330_run(
        run_dir,
        seed=42,
        model_config_sha256="model-hash",
    )

    assert audit["adapter_weights_sha256"] == _sha256(weights)
    manifest["seed"] = 13
    (run_dir / "training_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="seed"):
        g400._validate_g330_run(run_dir, seed=42, model_config_sha256="model-hash")


def test_score_reuses_fixed_g0_and_scores_three_grc_seeds(tmp_path: Path) -> None:
    a002 = tmp_path / "a002.jsonl"
    a002.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "component_id": "component-1",
                "arms": {"K_topk_base": [_run_row("wrong", baseline=True)]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    b100 = tmp_path / "b100.jsonl"
    b100.write_text("", encoding="utf-8")
    seeds = {
        seed: _candidate_file(tmp_path, seed, "right")
        for seed in g400.ALLOWED_SEEDS
    }
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "reference_answers": ["right"],
                "relevant_document_ids": ["d1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = g400.score(
        a002_generations_path=a002,
        b100_generations_path=b100,
        seed_generations=seeds,
        gold_path=gold,
        output_json=tmp_path / "score.json",
        output_rows=tmp_path / "rows.jsonl",
        output_report=tmp_path / "REPORT.md",
        entails=lambda premise, hypothesis: hypothesis in premise,
    )

    assert report["status"] == "G400_NIAH_RESPONSIBILITY_PASS"
    assert report["aggregate"]["K_topk"]["G0"]["correct_and_cited"] == 0.0
    assert report["aggregate"]["K_topk"]["GRC13"]["correct_and_cited"] == 1.0
    assert report["family_deltas_vs_g0"]["K_topk"]["correct_and_cited"] == 1.0
    assert report["gate"]["responsibility_checks"][
        "niah_correct_and_cited_delta_gt_0"
    ] is True
    assert report["boundaries"]["gold_loaded_at_runtime"] is False
