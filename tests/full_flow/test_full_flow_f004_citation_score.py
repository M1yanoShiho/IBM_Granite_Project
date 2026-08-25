from __future__ import annotations

import sys
from pathlib import Path

import pytest

from evidence_rag.contracts.models import EvidenceCandidate, GenerationResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_flow_f004_citation_score as citation  # noqa: E402


def _candidate(index: int) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=f"e{index}",
        document_id=f"d{index}",
        chunk_id=f"c{index}",
        text=f"Supported fact {index}.",
        source_uri=f"memory://{index}",
        retrieval_score=float(11 - index),
        retrieval_rank=index,
    )


def _arm(answer: str, citation_id: str | None) -> dict[str, object]:
    cited = (citation_id,) if citation_id is not None else ()
    return {
        "generation": GenerationResult(
            query_id="q1", answer=answer, cited_evidence_ids=cited
        ).model_dump(mode="json"),
        "routing": [
            {
                "claim_id": "c1",
                "outcome": "verified" if citation_id else "unverified",
                "sentence": answer,
                "citation": citation_id,
            }
        ],
        "error": None,
        "seconds": 0.1,
    }


def test_f004_citation_score_compares_all_three_verify_arms() -> None:
    row = {
        "query_id": "q1",
        "component_id": "component-1",
        "topk10": [_candidate(index).model_dump(mode="json") for index in range(1, 11)],
        "arms": {
            "G0_current": _arm("Unsupported fact.", "e1"),
            "G1_notes": _arm("Supported fact 1.", "e1"),
            "G2_guided_notes": _arm("Supported fact 1.", "e1"),
        },
    }

    summary, per_case = citation.score_rows(
        [row], lambda premise, hypothesis: hypothesis in premise
    )

    assert summary["aggregate"]["G0_current"]["citation_precision_alce"] == 0.0
    assert summary["aggregate"]["G1_notes"]["citation_precision_alce"] == 1.0
    comparison = summary["comparisons"]["G1_minus_G0"]["citation_precision"]
    assert comparison["delta"] == pytest.approx(1.0)
    assert len(per_case) == 1
