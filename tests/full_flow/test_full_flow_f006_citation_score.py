from __future__ import annotations

import sys
from pathlib import Path

from evidence_rag.contracts.models import EvidenceCandidate, GenerationResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_flow_f006_citation_score as citation  # noqa: E402


def _arm(answer: str, evidence_id: str) -> dict[str, object]:
    return {
        "generation": GenerationResult(
            query_id="q1", answer=answer, cited_evidence_ids=(evidence_id,)
        ).model_dump(mode="json"),
        "routing": [{"sentence": answer, "citation": evidence_id}],
    }


def test_score_uses_all_four_exact_routing_arms() -> None:
    evidence = EvidenceCandidate(
        evidence_id="e1",
        document_id="d1",
        chunk_id="c1",
        text="right",
        source_uri="memory://1",
        retrieval_score=1.0,
        retrieval_rank=1,
    )
    row = {
        "query_id": "q1",
        "component_id": "c1",
        "topk10": [evidence.model_dump(mode="json")],
        "arms": {
            "TopK_frozen": _arm("wrong", "e1"),
            "Base_notes_frozen": _arm("wrong", "e1"),
            "Clean_LoRA": _arm("wrong", "e1"),
            "Mixed_LoRA": _arm("right", "e1"),
        },
    }

    summary, rows = citation.score_rows([row], lambda premise, hypothesis: hypothesis in premise)

    assert summary["aggregate"]["Mixed_LoRA"]["citation_precision_alce"] == 1.0
    comparison = summary["comparisons"]["Mixed_minus_Clean"]["citation_precision"]
    assert comparison["delta"] == 1.0
    assert len(rows) == 1
