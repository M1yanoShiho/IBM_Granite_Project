from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate, Query
from evidence_rag.retriever.granite import Query2DocRetriever


class RecordingRetriever:
    def __init__(self) -> None:
        self.received: list[Query] = []

    def retrieve(self, query: Query, top_k: int) -> CandidateSet:
        self.received.append(query)
        return CandidateSet(
            query_id=query.query_id,
            candidates=(
                EvidenceCandidate(
                    evidence_id="ev-1",
                    document_id="doc-1",
                    chunk_id="chunk-1",
                    text=query.text,
                    source_uri="fixture://doc-1",
                    retrieval_score=1.0,
                    retrieval_rank=1,
                ),
            ),
        )


class FakeGenerator:
    def generate(self, prompt: str) -> str:
        assert "Question: revenue?" in prompt
        return "The annual report says revenue increased."


def test_query2doc_retriever_appends_generated_pseudo_document() -> None:
    base = RecordingRetriever()
    retriever = Query2DocRetriever(base, FakeGenerator())

    result = retriever.retrieve(Query(query_id="q", text="revenue?"), top_k=3)

    assert result.query_id == "q"
    assert base.received[0].text == (
        "revenue? The annual report says revenue increased."
    )
    assert result.candidates[0].text == base.received[0].text
