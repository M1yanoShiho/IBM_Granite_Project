from __future__ import annotations

import json
import sys
from pathlib import Path

from evidence_rag.contracts.models import EvidenceCandidate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import full_flow_g410_cross_data_qualification as g410  # noqa: E402


def _run_row(answer: str) -> dict[str, object]:
    return {
        "generation": {
            "schema_version": "1.0",
            "query_id": "task-1",
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
        "routing": [
            {
                "sentence": answer,
                "citation": "e1",
                "outcome": "verified",
            }
        ],
        "error": None,
    }


def _task() -> g410.G410Task:
    return g410.G410Task(
        task_id="validation-answerable::2wiki::case1::topk",
        scope="validation-2wiki-answerable",
        dataset="2wiki",
        case_id="2wiki::case1",
        query_id="case1",
        component_id="component-1",
        context="topk",
        variant_name="topk",
        question="Who?",
        target_kind="twowiki_evidence_chain",
        evidence=(
            EvidenceCandidate(
                evidence_id="e1",
                document_id="d1",
                chunk_id="c1",
                text="right",
                source_uri="fixture://1",
                retrieval_score=1.0,
                retrieval_rank=1,
            ),
        ),
        support_evidence_ids=("e1",),
    )


def _candidate_file(tmp_path: Path, seed: int, task: g410.G410Task, answer: str) -> Path:
    output_dir = tmp_path / f"seed{seed}"
    output_dir.mkdir()
    generations = output_dir / "generations.jsonl"
    generations.write_text(
        json.dumps(
            {
                "schema_version": g410.SCHEMA_GENERATION_ROW,
                **g410._task_identity(task),
                "question": task.question,
                "evidence": [item.model_dump(mode="json") for item in task.evidence],
                "arm_order": [g410.CANDIDATE_ARM],
                "arms": {g410.CANDIDATE_ARM: _run_row(answer)},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": g410.SCHEMA_RUN_MANIFEST,
                "status": "COMPLETE",
                "seed": seed,
                "arms": [g410.CANDIDATE_ARM],
                "generations_sha256": g410._sha256(generations),
            }
        ),
        encoding="utf-8",
    )
    return generations


def test_build_2wiki_tasks_splits_references_from_generation_packet() -> None:
    row = {
        "dataset": "2wiki",
        "answerable": True,
        "case_id": "2wiki::case1",
        "query_id": "case1",
        "component_id": "component-1",
        "question": "Who?",
        "target_kind": "twowiki_evidence_chain",
        "answer": "right",
        "official_answer": "right",
        "variants": {
            "topk": {
                "prompt": "[ignored]\nEvidence:\n[1] (e1) right\n\nQuestion: Who?\nAnswer:",
                "support_evidence_ids": ["e1"],
            }
        },
    }

    tasks, references = g410.build_2wiki_tasks([row])

    assert len(tasks) == 1
    assert tasks[0].task_id == "validation-answerable::2wiki::case1::topk"
    task_row = g410._task_row(tasks[0])
    assert "reference_answers" not in task_row
    assert "answer" not in task_row
    assert references[0]["reference_answers"] == ["right"]


def test_score_reuses_g310_g0_and_scores_three_grc_seeds(tmp_path: Path) -> None:
    task = _task()
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(json.dumps(g410._task_row(task)) + "\n", encoding="utf-8")
    references_path = tmp_path / "references.jsonl"
    references_path.write_text(
        json.dumps(g410._reference_row(task, ["right"])) + "\n",
        encoding="utf-8",
    )
    baseline = tmp_path / "g310.jsonl"
    baseline.write_text(
        json.dumps(
            {
                "task_id": task.task_id,
                "arms": {"G0": _run_row("wrong")},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    seeds = {
        seed: _candidate_file(tmp_path, seed, task, "right")
        for seed in g410.ALLOWED_SEEDS
    }

    report = g410.score(
        tasks_path=tasks_path,
        references_path=references_path,
        baseline_g310_generations_path=baseline,
        seed_generations=seeds,
        output_json=tmp_path / "score.json",
        output_rows=tmp_path / "rows.jsonl",
        output_report=tmp_path / "REPORT.md",
        entails=lambda premise, hypothesis: hypothesis in premise,
    )

    assert report["status"] == "G410_CROSS_DATA_RESPONSIBILITY_PASS"
    assert report["aggregate"]["all_2wiki"]["G0"]["correct_and_cited"] == 0.0
    assert report["aggregate"]["all_2wiki"]["GRC13"]["correct_and_cited"] == 1.0
    assert report["family_deltas_vs_g0"]["all_2wiki"]["correct_and_cited"] == 1.0
    assert report["boundaries"]["references_loaded_at_runtime"] is False
