from __future__ import annotations

import sys
from pathlib import Path

from evidence_rag.contracts.models import EvidenceCandidate, GenerationResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_flow_f005 as f005  # noqa: E402
import full_flow_f005_citation_score as citation  # noqa: E402


def _arm(query_id: str, sentence: str, evidence_id: str) -> dict[str, object]:
    return {
        "generation": GenerationResult(
            query_id=query_id, answer=sentence, cited_evidence_ids=(evidence_id,)
        ).model_dump(mode="json"),
        "routing": [
            {
                "outcome": "verified",
                "sentence": sentence,
                "citation": evidence_id,
            }
        ],
    }


def test_score_uses_three_f005_arms() -> None:
    candidate = EvidenceCandidate(
        evidence_id="e1",
        document_id="d1",
        chunk_id="c1",
        text="supported sentence",
        source_uri="test://d1",
        retrieval_score=1.0,
        retrieval_rank=1,
    )
    rows = [
        {
            "query_id": "q1",
            "topk10": [candidate.model_dump(mode="json")],
            "arms": {
                arm: _arm("q1", "supported sentence", "e1") for arm in f005.ARMS
            },
        }
    ]

    summary, per_case = citation.score_rows(rows, lambda premise, hypothesis: True)

    assert set(summary["aggregate"]) == set(f005.ARMS)
    assert summary["aggregate"][f005.ARMS[2]]["citation_precision_alce"] == 1.0
    assert len(per_case) == 1
