from __future__ import annotations

import sys
from pathlib import Path

from evidence_rag.contracts.models import EvidenceCandidate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import full_flow_g230 as g230  # noqa: E402
import full_flow_g230_citation_score as citation  # noqa: E402


def _run(*, baseline: bool) -> dict[str, object]:
    row: dict[str, object] = {
        "generation": {
            "schema_version": "1.0",
            "query_id": "q1",
            "answer": "The answer is right.",
            "cited_evidence_ids": ["e1"],
        },
        "error": None,
    }
    if baseline:
        row["trace"] = {
            "claims": [
                {
                    "final_sentence": "The answer is right.",
                    "citation": "e1",
                    "routing_outcome": "verified",
                }
            ]
        }
    else:
        row["routing"] = [
            {
                "sentence": "The answer is right.",
                "citation": "e1",
                "outcome": "verified",
            }
        ]
    return row


def test_routing_from_archived_trace_preserves_sentence_citation_pair() -> None:
    assert citation.routing_from_trace(_run(baseline=True)) == [
        {
            "sentence": "The answer is right.",
            "citation": "e1",
            "outcome": "verified",
        }
    ]


def test_score_rows_uses_minicheck_callback_for_every_frozen_config() -> None:
    task_ids = [
        "full::K_topk::q1",
        *[f"stress::{context}::q1" for context in g230.STRESS_CONTEXTS],
    ]
    configs = {
        name: {
            task_id: _run(baseline=name == "G0") for task_id in task_ids
        }
        for name in citation.CONFIG_ORDER
    }
    candidate = EvidenceCandidate(
        evidence_id="e1",
        document_id="d1",
        chunk_id="c1",
        text="The answer is right.",
        source_uri="fixture://1",
        retrieval_score=1.0,
        retrieval_rank=1,
    )
    evidence = {task_id: (candidate,) for task_id in task_ids}
    components = dict.fromkeys(task_ids, "component-1")

    report, rows = citation.score_rows(
        configs=configs,
        evidence=evidence,
        components=components,
        entails=lambda premise, hypothesis: "right" in premise and "right" in hypothesis,
    )

    assert report["status"] == "COMPLETE"
    assert report["aggregate"]["G0"]["K_topk"]["citation_precision_alce"] == 1.0
    assert report["aggregate"]["GM73"]["K_topk"]["citation_recall_alce"] == 1.0
    assert report["citation_gate"]["pass_by_family"] == {"GC": True, "GM": True}
    assert len(rows) == len(task_ids)
