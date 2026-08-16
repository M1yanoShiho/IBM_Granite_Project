from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from evidence_rag.contracts.models import EvidenceCandidate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import full_flow_g230 as g230  # noqa: E402
import full_flow_joint as joint  # noqa: E402


def _candidate(index: int) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=f"e{index}",
        document_id=f"d{index}",
        chunk_id=f"c{index}",
        text=f"Evidence {index}",
        source_uri=f"fixture://{index}",
        retrieval_score=float(10 - index),
        retrieval_rank=index,
    )


def _full_case() -> joint.JointCase:
    evidence = (_candidate(1), _candidate(2))
    return joint.JointCase("q1", "What is the answer?", "component-1", evidence, evidence)


def _stress_row() -> dict[str, object]:
    evidence = [_candidate(1).model_dump(mode="json")]
    return {
        "query_id": "q1",
        "question": "What is the answer?",
        "component_id": "component-1",
        "selector_changed": True,
        "contexts": {context: evidence for context in g230.STRESS_CONTEXTS},
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_tasks_freezes_full_topk_and_five_stress_contexts() -> None:
    tasks = g230.build_tasks([_full_case()], [_stress_row()])

    assert len(tasks) == 1 + len(g230.STRESS_CONTEXTS)
    assert tasks[0].task_id == "full::K_topk::q1"
    assert tasks[0].evidence == _full_case().topk10
    assert [task.context for task in tasks[1:]] == list(g230.STRESS_CONTEXTS)
    assert all(task.scope == "stress" for task in tasks[1:])


def test_arm_order_is_deterministic_and_resume_requires_exact_prefix() -> None:
    tasks = g230.build_tasks([_full_case()], [_stress_row()])
    first = g230.arm_order(tasks[0].task_id, ("GC", "GM"))
    assert first == g230.arm_order(tasks[0].task_id, ("GC", "GM"))
    assert set(first) == {"GC", "GM"}

    rows = [{"task_id": tasks[0].task_id, "arms": {"GC": {}, "GM": {}}}]
    g230.validate_resume_prefix(rows, tasks, ("GC", "GM"))
    with pytest.raises(ValueError, match="diverges"):
        g230.validate_resume_prefix(
            [{"task_id": "wrong", "arms": {"GC": {}, "GM": {}}}],
            tasks,
            ("GC", "GM"),
        )
    with pytest.raises(ValueError, match="wrong arms"):
        g230.validate_resume_prefix(
            [{"task_id": tasks[0].task_id, "arms": {"GC": {}}}],
            tasks,
            ("GC", "GM"),
        )


def test_g220_adapter_validation_binds_formal_manifest_and_weights(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "gc-seed13"
    adapter = run_dir / "adapter"
    adapter.mkdir(parents=True)
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    manifest = {
        "schema_version": "full-flow-g220-training-manifest-v1",
        "status": "COMPLETE",
        "run_kind": "formal",
        "arm": "gc",
        "seed": 13,
        "queries": 515,
        "training_examples": 4120,
        "validation_queries": 62,
        "optimizer_steps": 515,
        "max_length": 2304,
        "truncated_examples": 0,
        "decision_dev_used": False,
        "sealed_or_heldout_read": False,
        "adapter_scope": "draft generation call only",
        "key_fact_extraction_scope": "not used",
        "claim_splitter_scope": "frozen Granite base with adapter disabled",
        "reload_check": {"status": "PASS"},
        "adapter_weights_sha256": _sha256(weights),
    }
    (run_dir / "training_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    audit = g230._validate_g220_run(run_dir, arm="gc", seed=13)
    assert audit["adapter_weights_sha256"] == _sha256(weights)

    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="weights"):
        g230._validate_g220_run(run_dir, arm="gc", seed=13)


class _FakeClient:
    def __init__(self, adapters: tuple[str, ...]) -> None:
        self.adapter_paths = dict.fromkeys(adapters, "fixture")

    def generate(self, prompt: str) -> str:
        return prompt

    def generate_with_adapter(self, prompt: str, adapter_name: str) -> str:
        return f"{adapter_name}:{prompt}"


class _FakeNLI:
    def classify(self, premise: str, hypothesis: str) -> str:
        return "entailment"


def test_generator_construction_limits_adapter_to_the_frozen_call_scope() -> None:
    gn_client = _FakeClient(("gn",))
    gn = g230.build_generators(mode="gn", client=gn_client, nli=_FakeNLI())["GN"]
    gn_draft = gn.draft_generator
    assert gn_draft.llm is gn_client
    assert gn_draft.note_extractor.llm.adapter_name == "gn"
    assert gn_draft.claim_splitter.llm is gn_client

    pair_client = _FakeClient(("gc", "gm"))
    pair = g230.build_generators(
        mode="draft-pair", client=pair_client, nli=_FakeNLI()
    )
    for arm, adapter_name in (("GC", "gc"), ("GM", "gm")):
        draft_stage = pair[arm].draft_generator
        assert draft_stage.draft_generator.llm.adapter_name == adapter_name
        assert draft_stage.claim_splitter.llm is pair_client


def test_failure_flags_distinguish_draft_claim_and_final_empty() -> None:
    run = {
        "generation": {"query_id": "q1", "answer": "", "cited_evidence_ids": []},
        "trace": {
            "draft": {"normalized_draft_text": "A draft"},
            "claims": [],
        },
        "error": None,
    }

    flags = g230._failure_flags(run)

    assert flags == {
        "coverage": 0.0,
        "draft_empty": 0.0,
        "zero_claims": 1.0,
        "final_empty": 1.0,
        "runtime_error": 0.0,
    }


def _run_row(answer: str) -> dict[str, object]:
    return {
        "generation": {
            "schema_version": "1.0",
            "query_id": "q1",
            "answer": answer,
            "cited_evidence_ids": ["e1"] if answer else [],
        },
        "trace": {
            "draft": {"normalized_draft_text": answer},
            "claims": [{"claim_id": "c1"}] if answer else [],
        },
        "routing": [],
        "error": None,
    }


def _candidate_file(
    root: Path, name: str, arms: dict[str, str], *, seed: int | None
) -> Path:
    output_dir = root / name
    output_dir.mkdir()
    rows = []
    task_ids = ["full::K_topk::q1", *[f"stress::{item}::q1" for item in g230.STRESS_CONTEXTS]]
    for task_id in task_ids:
        rows.append(
            {
                "schema_version": "full-flow-g230-generation-row-v1",
                "task_id": task_id,
                "arms": {arm: _run_row(answer) for arm, answer in arms.items()},
            }
        )
    generations = output_dir / "generations.jsonl"
    generations.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "arms": list(arms),
                "seed": seed,
                "generations_sha256": _sha256(generations),
            }
        ),
        encoding="utf-8",
    )
    return generations


def test_score_joins_gold_only_after_all_runtime_outputs_exist(tmp_path: Path) -> None:
    a002 = tmp_path / "a002.jsonl"
    a002.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "component_id": "component-1",
                "arms": {"K_topk_base": [_run_row("")]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    b100 = tmp_path / "b100.jsonl"
    b100.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "component_id": "component-1",
                "arms": {context: _run_row("") for context in g230.STRESS_CONTEXTS},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gn = _candidate_file(tmp_path, "gn", {"GN": ""}, seed=None)
    seeds = {
        seed: _candidate_file(
            tmp_path,
            f"seed{seed}",
            {"GC": "wrong", "GM": "right"},
            seed=seed,
        )
        for seed in g230.ALLOWED_SEEDS
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

    report, cases = g230.score(
        a002_generations_path=a002,
        b100_generations_path=b100,
        gn_generations_path=gn,
        seed_generations=seeds,
        gold_path=gold,
    )

    assert report["status"] == "COMPLETE_PENDING_CITATION"
    assert report["queries"] == 1
    assert report["tasks"] == 1 + len(g230.STRESS_CONTEXTS)
    assert report["aggregate"]["G0"]["K_topk"]["answer_match"] == 0.0
    assert report["aggregate"]["GM13"]["K_topk"]["answer_match"] == 1.0
    assert report["comparisons"]["GM13_minus_G0"]["K_topk"]["answer_match"][
        "delta"
    ] == 1.0
    assert report["development_gate"]["candidate_before_citation"] == "GM"
    assert report["development_gate"]["pre_citation_pass"] is True
    assert len(cases) == 1 + len(g230.STRESS_CONTEXTS)
