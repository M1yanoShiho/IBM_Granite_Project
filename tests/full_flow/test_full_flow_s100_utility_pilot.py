from __future__ import annotations

import json
import sys
from pathlib import Path

from evidence_rag.contracts.models import EvidenceCandidate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import full_flow_s100_utility_pilot as s100  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _case(dataset: str, case_id: str, *, answer: str) -> dict[str, object]:
    return {
        "schema_version": "full-flow-g200-case-v2",
        "dataset": dataset,
        "case_id": case_id,
        "query_id": case_id,
        "component_id": f"component-{case_id}",
        "role": "train-fit",
        "answerable": True,
        "target_kind": "fixture",
        "question": "Who?",
        "answer": answer,
        "official_answer": answer,
        "variants": {
            "topk": {
                "prompt": (
                    "Answer the question using only the evidence below.\n"
                    "Evidence:\n"
                    f"[1] ({case_id}-e1) right answer\n"
                    f"[2] ({case_id}-e2) harmless distractor\n\n"
                    "Question: Who?\n"
                    "Answer:"
                ),
                "support_evidence_ids": [f"{case_id}-e1"],
            }
        },
    }


def test_prepare_splits_generation_tasks_from_references(tmp_path: Path) -> None:
    g430 = tmp_path / "g430.json"
    _write_json(
        g430,
        {
            "schema_version": s100.SCHEMA_G430,
            "status": "GQ_FROZEN_NEW_GRC",
            "gq": {"id": "gq", "seeds": [13, 42, 73]},
            "boundaries": {"sealed_or_heldout_read": False},
        },
    )
    train_cases = tmp_path / "train_cases.jsonl"
    _write_jsonl(
        train_cases,
        [
            _case("niah", "n1", answer="right"),
            _case("2wiki", "t1", answer="right"),
        ],
    )
    ordered_ids = tmp_path / "ordered_ids.json"
    _write_json(ordered_ids, {"ids": ["n1", "t1"]})
    g223 = tmp_path / "manifest.json"
    _write_json(
        g223,
        {
            "schema_version": s100.SCHEMA_G223,
            "status": "CONTROLLED_CONTINUATION_READY",
            "dev_read": False,
            "sealed_or_heldout_read": False,
            "train_cases_sha256": s100._sha256(train_cases),
            "ordered_ids_sha256": s100._sha256(ordered_ids),
        },
    )

    manifest = s100.prepare(
        g430_manifest_path=g430,
        g223_manifest_path=g223,
        train_cases_path=train_cases,
        ordered_ids_path=ordered_ids,
        output_dir=tmp_path / "prepare",
        niah_questions=1,
        twowiki_questions=1,
    )

    assert manifest["questions"] == 2
    tasks = [
        json.loads(line)
        for line in (tmp_path / "prepare" / "generation_tasks.jsonl").read_text().splitlines()
    ]
    assert len(tasks) == 6
    assert all("reference_answers" not in row for row in tasks)
    assert all("support_evidence_ids" not in row for row in tasks)
    references = [
        json.loads(line)
        for line in (tmp_path / "prepare" / "references.jsonl").read_text().splitlines()
    ]
    assert {row["dataset"] for row in references} == {"niah", "2wiki"}
    assert references[0]["reference_answers"] == ["right"]


def _task(question_uid: str, variant_type: str, evidence: tuple[EvidenceCandidate, ...]) -> s100.S100Task:
    dropped = None
    rank = None
    if variant_type == "drop-e1":
        evidence = (evidence[1],)
        dropped = "e1"
        rank = 1
        variant_type = "leave_one_out"
    elif variant_type == "drop-e2":
        evidence = (evidence[0],)
        dropped = "e2"
        rank = 2
        variant_type = "leave_one_out"
    return s100.S100Task(
        task_id=f"{question_uid}::{variant_type if dropped is None else 'drop-rank' + str(rank)}",
        question_uid=question_uid,
        dataset="2wiki",
        case_id="case1",
        query_id="case1",
        component_id="component-1",
        context_variant="topk",
        variant_type=variant_type,
        question="Who?",
        target_kind="fixture",
        original_evidence_count=2,
        dropped_evidence_id=dropped,
        dropped_rank=rank,
        evidence=evidence,
    )


def _run_row(task_id: str, answer: str, citation: str | None) -> dict[str, object]:
    return {
        "generation": {
            "schema_version": "1.0",
            "query_id": task_id,
            "answer": answer,
            "cited_evidence_ids": [citation] if citation else [],
        },
        "trace": {
            "draft": {"raw_draft_text": f"{answer} [1]" if citation else ""},
            "claims": [
                {
                    "final_sentence": answer,
                    "citation": citation,
                    "routing_outcome": "verified" if citation else "unverified",
                }
            ],
        },
        "routing": [
            {
                "sentence": answer,
                "citation": citation,
                "outcome": "verified" if citation else "unverified",
            }
        ],
        "error": None,
    }


def _generation_file(tmp_path: Path, seed: int, tasks: list[s100.S100Task]) -> Path:
    output_dir = tmp_path / f"seed{seed}"
    output_dir.mkdir()
    generations = output_dir / "generations.jsonl"
    rows = []
    for task in tasks:
        if task.dropped_evidence_id == "e1":
            answer, citation = "", None
        else:
            answer, citation = "right", "e1"
        rows.append(
            {
                "schema_version": s100.SCHEMA_GENERATION_ROW,
                **s100._task_identity(task),
                "question": task.question,
                "evidence": [item.model_dump(mode="json") for item in task.evidence],
                "arm_order": [s100.CANDIDATE_ARM],
                "arms": {s100.CANDIDATE_ARM: _run_row(task.task_id, answer, citation)},
            }
        )
    _write_jsonl(generations, rows)
    _write_json(
        output_dir / "run_manifest.json",
        {
            "schema_version": s100.SCHEMA_RUN_MANIFEST,
            "status": "COMPLETE",
            "seed": seed,
            "arm": s100.CANDIDATE_ARM,
            "generations_sha256": s100._sha256(generations),
        },
    )
    return generations


def test_score_labels_required_support_and_safe_distractor(tmp_path: Path) -> None:
    evidence = (
        EvidenceCandidate(
            evidence_id="e1",
            document_id="d1",
            chunk_id="c1",
            text="right",
            source_uri="fixture://1",
            retrieval_score=1.0,
            retrieval_rank=1,
        ),
        EvidenceCandidate(
            evidence_id="e2",
            document_id="d2",
            chunk_id="c2",
            text="distractor",
            source_uri="fixture://2",
            retrieval_score=0.5,
            retrieval_rank=2,
        ),
    )
    question_uid = "s100::2wiki::case1::topk"
    tasks = [
        _task(question_uid, "full", evidence),
        _task(question_uid, "drop-e1", evidence),
        _task(question_uid, "drop-e2", evidence),
    ]
    tasks_path = tmp_path / "tasks.jsonl"
    _write_jsonl(tasks_path, [s100._task_row(task) for task in tasks])
    references = tmp_path / "references.jsonl"
    _write_jsonl(
        references,
        [
            {
                "schema_version": s100.SCHEMA_REFERENCE,
                "question_uid": question_uid,
                "dataset": "2wiki",
                "case_id": "case1",
                "query_id": "case1",
                "component_id": "component-1",
                "reference_answers": ["right"],
                "support_evidence_ids": ["e1"],
                "full_evidence_ids": ["e1", "e2"],
            }
        ],
    )
    generations = {
        seed: _generation_file(tmp_path, seed, tasks)
        for seed in s100.ALLOWED_SEEDS
    }

    report = s100.score(
        tasks_path=tasks_path,
        references_path=references,
        seed_generations=generations,
        output_json=tmp_path / "score.json",
        output_rows=tmp_path / "rows.jsonl",
        output_labels=tmp_path / "labels.jsonl",
        output_report=tmp_path / "REPORT.md",
        entails=lambda premise, hypothesis: hypothesis in premise,
    )

    labels = [
        json.loads(line)
        for line in (tmp_path / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_evidence = {row["evidence_id"]: row["label"] for row in labels}
    assert by_evidence == {"e1": "MUST_KEEP", "e2": "SAFE_DROP"}
    assert report["label_summary"]["label_counts"]["MUST_KEEP"] == 1
    assert report["label_summary"]["label_counts"]["SAFE_DROP"] == 1
    assert report["boundaries"]["generation_runtime_reference_answers_loaded"] is False
