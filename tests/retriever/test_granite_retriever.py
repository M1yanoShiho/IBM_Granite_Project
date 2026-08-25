from collections.abc import Sequence

from evidence_rag.contracts.models import Document, Query
from evidence_rag.retriever.granite import GraniteDenseRetriever


class FakeEmbedder:
    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._embed(text) for text in texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> tuple[float, ...]:
        lowered = text.lower()
        return (
            1.0 if "revenue" in lowered else 0.0,
            1.0 if "profit" in lowered else 0.0,
        )


def test_granite_dense_retriever_returns_ranked_candidate_set() -> None:
    retriever = GraniteDenseRetriever(
        (
            Document(
                document_id="revenue-doc",
                text="Revenue increased by ten percent.",
                source_uri="fixture://revenue",
            ),
            Document(
                document_id="profit-doc",
                text="Profit was stable.",
                source_uri="fixture://profit",
            ),
        ),
        embedder=FakeEmbedder(),
    )

    result = retriever.retrieve(Query(query_id="q", text="revenue change"), top_k=2)

    assert result.query_id == "q"
    assert tuple(candidate.document_id for candidate in result.candidates) == (
        "revenue-doc",
        "profit-doc",
    )
    assert tuple(candidate.retrieval_rank for candidate in result.candidates) == (1, 2)
    assert result.candidates[0].retrieval_score > result.candidates[1].retrieval_score
