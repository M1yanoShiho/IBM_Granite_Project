from __future__ import annotations

from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate, Query
from evidence_rag.retriever.rerank import RerankingRetriever


def _candidate(name: str, rank: int) -> EvidenceCandidate:
    return EvidenceCandidate(
        evidence_id=f"ev-{name}",
        document_id=f"doc-{name}",
        chunk_id=f"chunk-{name}",
        text=name,
        source_uri=f"fixture://{name}",
        retrieval_score=float(10 - rank),
        retrieval_rank=rank,
    )


class _Base:
    def __init__(self) -> None:
        self.requested: list[int] = []

    def retrieve(self, query: Query, top_k: int) -> CandidateSet:
        self.requested.append(top_k)
        return CandidateSet(
            query_id=query.query_id,
            candidates=(
                _candidate("first", 1),
                _candidate("second", 2),
                _candidate("third", 3),
            ),
        )


class _Reranker:
    def score(self, query: str, passages: tuple[str, ...]) -> tuple[float, ...]:
        assert query == "question"
        assert passages == ("first", "second", "third")
        return (0.2, 0.9, 0.9)


def test_reranking_retriever_uses_frozen_deep_pool_and_stable_tie_break() -> None:
    base = _Base()
    retriever = RerankingRetriever(base, _Reranker(), pool_size=40)

    result = retriever.retrieve(Query(query_id="q", text="question"), 2)

    assert base.requested == [40]
    assert [item.evidence_id for item in result.candidates] == [
        "ev-second",
        "ev-third",
    ]
    assert [item.retrieval_rank for item in result.candidates] == [1, 2]
    assert [item.retrieval_score for item in result.candidates] == [0.9, 0.9]


def test_reranking_retriever_rejects_final_k_above_frozen_pool() -> None:
    retriever = RerankingRetriever(_Base(), _Reranker(), pool_size=2)

    try:
        retriever.retrieve(Query(query_id="q", text="question"), 3)
    except ValueError as exc:
        assert "may not exceed" in str(exc)
    else:  # pragma: no cover - assertion helper without an extra dependency
        raise AssertionError("expected final TopK guard")
