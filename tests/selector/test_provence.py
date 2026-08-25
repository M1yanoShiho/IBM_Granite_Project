from __future__ import annotations

from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate, Query
from evidence_rag.selector.provence import ProvenceSelector


def _candidate(name: str, text: str, rank: int) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=f"ev-{name}",
        document_id=f"doc-{name}",
        chunk_id=f"chunk-{name}",
        text=text,
        source_uri=f"fixture://{name}",
        retrieval_score=float(10 - rank),
        retrieval_rank=rank,
    )


class _Pruner:
    def prune(self, *, question: str, title: str, text: str) -> str:
        assert question == "question"
        assert title == ""
        return "" if "drop" in text else text.replace("noise", "").strip()


def test_provence_preserves_source_mapping_and_retrieval_order() -> None:
    candidates = CandidateSet(
        query_id="q",
        candidates=(
            _candidate("one", "useful noise", 1),
            _candidate("two", "drop all", 2),
            _candidate("three", "also useful", 3),
        ),
    )
    selector = ProvenceSelector(_Pruner())

    result = selector.select_with_context(
        Query(query_id="q", text="question"),
        candidates,
        10,
    )

    assert result.dropped_empty == 1
    assert [item.evidence_id for item in result.selected.evidence] == [
        "ev-one",
        "ev-three",
    ]
    assert [item.text for item in result.selected.evidence] == ["useful", "also useful"]
    assert [item.selection_rank for item in result.selection.items] == [1, 2]
    assert result.selected.evidence[0].document_id == "doc-one"
