from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = (
    ROOT
    / "docs/full-flow/experiments/04_frozen_three_module_system_evaluation_2026-08-21"
)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_goal2_model_manifest_freezes_all_runtime_and_scorer_models() -> None:
    manifest = _json(EXPERIMENT / "artifacts/goal2_model_config_manifest.json")

    assert manifest["status"] == "FROZEN_BEFORE_HELD_OUT"
    models = manifest["models"]
    assert isinstance(models, dict)
    assert set(models) == {
        "dense_embedder",
        "granite_reranker",
        "provence",
        "direct_and_grounded_base",
        "selector_backbone",
        "true_runtime_verifier",
        "minicheck_scorer_only",
    }
    for identity in models.values():
        assert isinstance(identity, dict)
        assert len(str(identity["revision"])) == 40
        assert len(str(identity["config_sha256"])) == 64
    assert set(manifest["grc_adapters"]) == {"13", "42", "73"}


def test_goal2_content_free_smoke_manifest_is_complete_and_leak_free() -> None:
    smoke = _json(EXPERIMENT / "artifacts/goal2_smoke_manifest.json")

    assert smoke["status"] == "PASS"
    assert smoke["arm_count"] == smoke["expected_arm_count"] == 10
    assert smoke["all_arms_completed"] is True
    assert smoke["formal_output_directory_empty"] is True
    arms = smoke["arms"]
    assert isinstance(arms, list) and len(arms) == 10
    for arm in arms:
        assert arm["completed_queries"] == 1
        assert arm["gold_leakage_keys"] == []
        assert arm["retrieval_trace_present"] is True
        assert arm["selected_context_present"] is True
        assert arm["answer_present"] is True
        assert arm["score_present"] is True
        assert arm["scorer_readable"] is True
        assert len(arm["canonical_output_sha256"]) == 64


def test_goal2_real_baseline_smoke_passed_with_frozen_models() -> None:
    smoke = _json(EXPERIMENT / "artifacts/goal2_real_baseline_smoke.json")

    assert smoke["status"] == "PASS"
    assert smoke["arm_count"] == 4
    assert smoke["model_hashes_verified"] is True
    assert smoke["minicheck_real_model"] is True
    assert smoke["slurm"]["state"] == "COMPLETED"
    assert smoke["slurm"]["exit_code"] == "0:0"
    assert {arm["arm_id"] for arm in smoke["arms"]} == {
        "dense_rag",
        "hybrid_rag",
        "granite_rerank_rag",
        "provence_rag",
    }
    for arm in smoke["arms"]:
        assert arm["completed_queries"] == 1
        assert arm["retrieval_trace_present"] is True
        assert arm["selected_context_present"] is True
        assert arm["answer_present"] is True
        assert arm["score_present"] is True
