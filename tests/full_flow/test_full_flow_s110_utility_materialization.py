from __future__ import annotations

import json
import sys
from pathlib import Path

from evidence_rag.contracts.models import EvidenceCandidate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import full_flow_s100_utility_pilot as s100  # noqa: E402
import full_flow_s110_utility_materialization as s110  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _case(dataset: str, case_id: str, *, role: str, answer: str) -> dict[str, object]:
    return {
        "schema_version": "full-flow-g200-case-v2",
        "dataset": dataset,
        "case_id": case_id,
        "query_id": case_id,
        "component_id": f"component-{case_id}",
        "role": role,
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


def _g430(path: Path) -> None:
    _write_json(
        path,
        {
            "schema_version": s100.SCHEMA_G430,
            "status": "GQ_FROZEN_NEW_GRC",
            "gq": {"id": "gq", "seeds": [13, 42, 73]},
            "boundaries": {"sealed_or_heldout_read": False},
        },
    )


def _g223(path: Path, train_cases: Path, ordered_ids: Path) -> None:
    _write_json(
        path,
        {
            "schema_version": s100.SCHEMA_G223,
            "status": "CONTROLLED_CONTINUATION_READY",
            "dev_read": False,
            "sealed_or_heldout_read": False,
            "train_cases_sha256": s100._sha256(train_cases),
            "ordered_ids_sha256": s100._sha256(ordered_ids),
        },
    )


def _s100_ready(report_path: Path, manifest_path: Path) -> None:
    _write_json(
        report_path,
        {
            "status": "S100_PILOT_COMPLETE",
            "s110_recommendation": "S110_READY",
        },
    )
    _write_json(
        manifest_path,
        {
            "status": "COMPLETE",
            "s100_status": "S100_PILOT_COMPLETE",
            "s110_recommendation": "S110_READY",
            "score_report_sha256": s110._sha256(report_path),
            "sealed_or_heldout_read": False,
        },
    )


def test_prepare_materializes_train_and_modelval_without_runtime_references(tmp_path: Path) -> None:
    g430 = tmp_path / "g430.json"
    _g430(g430)
    train_cases = tmp_path / "train_cases.jsonl"
    validation_cases = tmp_path / "validation_cases.jsonl"
    _write_jsonl(
        train_cases,
        [
            _case("niah", "n-train", role="train-fit", answer="right"),
            _case("2wiki", "t-train", role="train-fit", answer="right"),
        ],
    )
    _write_jsonl(
        validation_cases,
        [
            _case("niah", "n-val", role="train-modelval", answer="right"),
            _case("2wiki", "t-val", role="train-modelval", answer="right"),
        ],
    )
    ordered_ids = tmp_path / "ordered_ids.json"
    _write_json(ordered_ids, {"ids": ["n-train", "t-train", "n-val", "t-val"]})
    g223 = tmp_path / "g223.json"
    _g223(g223, train_cases, ordered_ids)
    s100_report = tmp_path / "s100_report.json"
    s100_manifest = tmp_path / "s100_manifest.json"
    _s100_ready(s100_report, s100_manifest)

    manifest = s110.prepare(
        g430_manifest_path=g430,
        g223_manifest_path=g223,
        train_cases_path=train_cases,
        validation_cases_path=validation_cases,
        ordered_ids_path=ordered_ids,
        s100_score_report_path=s100_report,
        s100_score_manifest_path=s100_manifest,
        output_dir=tmp_path / "prepare",
    )

    assert manifest["stage"] == "S110_PREPARE"
    assert manifest["questions"] == 4
    assert manifest["split_dataset_question_counts"] == {
        "modelval": {"2wiki": 1, "niah": 1},
        "train": {"2wiki": 1, "niah": 1},
    }
    tasks = [
        json.loads(line)
        for line in (tmp_path / "prepare" / "generation_tasks.jsonl").read_text().splitlines()
    ]
    assert len(tasks) == 12
    assert all(row["question_uid"].startswith("s110::") for row in tasks)
    assert all("reference_answers" not in row for row in tasks)
    assert all("support_evidence_ids" not in row for row in tasks)
    references = [
        json.loads(line)
        for line in (tmp_path / "prepare" / "references.jsonl").read_text().splitlines()
    ]
    assert {row["split"] for row in references} == {"train", "modelval"}


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


def test_score_writes_s110_report_and_split_labels(tmp_path: Path) -> None:
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
    question_uid = "s110::train::2wiki::case1::topk"
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
                "stage": "S110",
                "split": "train",
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
    generations = {seed: _generation_file(tmp_path, seed, tasks) for seed in s100.ALLOWED_SEEDS}

    report = s110.score(
        tasks_path=tasks_path,
        references_path=references,
        seed_generations=generations,
        output_json=tmp_path / "score.json",
        output_rows=tmp_path / "rows.jsonl",
        output_labels=tmp_path / "labels.jsonl",
        output_report=tmp_path / "REPORT.md",
        output_manifest=tmp_path / "manifest.json",
        entails=lambda premise, hypothesis: hypothesis in premise,
    )

    assert report["schema_version"] == s110.SCHEMA_SCORE_REPORT
    assert report["stage"] == "S110"
    assert report["status"] == "S110_MATERIALIZATION_COMPLETE"
    assert report["s200_recommendation"] == "STOP_SPLIT_WITHOUT_UTILITY_LABELS"
    labels = [
        json.loads(line)
        for line in (tmp_path / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["split"] for row in labels} == {"train"}
    assert {row["schema_version"] for row in labels} == {s110.SCHEMA_LABEL_ROW}
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == s110.SCHEMA_SCORE_MANIFEST
    assert manifest["labels_sha256"] == s110._sha256(tmp_path / "labels.jsonl")
