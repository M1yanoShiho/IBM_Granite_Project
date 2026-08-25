from __future__ import annotations

import sys
from pathlib import Path

from evidence_rag.contracts.models import EvidenceCandidate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_flow_f006_materialize as f006  # noqa: E402


def _candidate(index: int, *, relevant: bool = False) -> dict[str, object]:
    answer = "The answer is 2009." if relevant else f"Distractor {index}."
    return EvidenceCandidate(
        evidence_id=f"e{index}",
        document_id="relevant" if relevant else f"d{index}",
        chunk_id=f"c{index}",
        text=answer,
        source_uri=f"memory://{index}",
        retrieval_score=float(11 - index),
        retrieval_rank=index,
    ).model_dump(mode="json")


def test_build_cases_excludes_dev_and_requires_exact_relevant_answer() -> None:
    queries = [
        {"query_id": "keep", "text": "When did it happen?"},
        {"query_id": "dev", "text": "When did it happen?"},
        {"query_id": "miss", "text": "When did it happen?"},
    ]
    candidate_rows = [
        {
            "query_id": query_id,
            "candidates": [_candidate(1, relevant=True), _candidate(2)],
        }
        for query_id in ("keep", "dev", "miss")
    ]
    gold = [
        {
            "query_id": query_id,
            "reference_answers": ("not present",) if query_id == "miss" else ("2009",),
            "relevant_document_ids": ("relevant",),
        }
        for query_id in ("keep", "dev", "miss")
    ]

    cases, counts = f006.build_cases(queries, candidate_rows, gold, {"dev"})

    assert [row["query_id"] for row in cases] == ["keep"]
    assert counts["excluded_train_dev_overlap"] == 1
    assert counts["excluded_no_exact_relevant_answer"] == 1
    assert cases[0]["clean_target"] == (
        '{"facts":[{"slot":"DATE_OR_YEAR","value":"2009","evidence":[1]}]}'
    )
    assert "Distractor 2" in str(cases[0]["mixed_prompt"])
    assert "Distractor 2" not in str(cases[0]["clean_prompt"])
