import pytest

from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate, Query
from evidence_rag.retriever.granite import DecomposingRetriever, HyDERetriever


class RecordingRetriever:
    def __init__(self) -> None:
        self.received: list[Query] = []

    def retrieve(self, query: Query, top_k: int) -> CandidateSet:
        self.received.append(query)
        # Score by a toy relevance signal so RRF ordering is observable: a candidate
        # per distinct query text, keyed deterministically.
        return CandidateSet(
            query_id=query.query_id,
            candidates=(
                EvidenceCandidate(
                    evidence_id=f"ev-{abs(hash(query.text)) % 1000}",
                    document_id="doc-1",
                    chunk_id="chunk-1",
                    text=query.text,
                    source_uri="fixture://doc-1",
                    retrieval_score=1.0,
                    retrieval_rank=1,
                ),
            ),
        )


class ConstantGenerator:
    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.text


def test_hyde_replaces_query_with_generated_document() -> None:
    base = RecordingRetriever()
    retriever = HyDERetriever(base, ConstantGenerator("Revenue rose in 2024."))

    retriever.retrieve(Query(query_id="q", text="did revenue rise?"), top_k=3)

    # HyDE replaces (not appends) the query text with the hypothetical document.
    assert base.received[0].text == "Revenue rose in 2024."


def test_hyde_falls_back_to_original_query_when_generation_empty() -> None:
    base = RecordingRetriever()
    retriever = HyDERetriever(base, ConstantGenerator("   "))

    retriever.retrieve(Query(query_id="q", text="original"), top_k=3)

    assert base.received[0].text == "original"


def test_decompose_retrieves_each_subquery_and_merges() -> None:
    base = RecordingRetriever()
    generator = ConstantGenerator("1. what is revenue?\n2. what is profit?")
    retriever = DecomposingRetriever(base, generator)

    result = retriever.retrieve(Query(query_id="q", text="revenue and profit?"), top_k=10)

    # Numbered-list prefixes are stripped; both sub-questions were retrieved.
    assert [q.text for q in base.received] == ["what is revenue?", "what is profit?"]
    assert result.query_id == "q"
    assert tuple(c.retrieval_rank for c in result.candidates) == tuple(
        range(1, len(result.candidates) + 1)
    )


def test_decompose_falls_back_to_original_when_no_subqueries() -> None:
    base = RecordingRetriever()
    retriever = DecomposingRetriever(base, ConstantGenerator("\n\n"))

    retriever.retrieve(Query(query_id="q", text="only question"), top_k=5)

    assert [q.text for q in base.received] == ["only question"]


def test_decompose_uses_pool_size_per_subquery() -> None:
    class TopKSpy(RecordingRetriever):
        def __init__(self) -> None:
            super().__init__()
            self.top_ks: list[int] = []

        def retrieve(self, query: Query, top_k: int) -> CandidateSet:
            self.top_ks.append(top_k)
            return super().retrieve(query, top_k)

    base = TopKSpy()
    retriever = DecomposingRetriever(
        base, ConstantGenerator("a?\nb?"), pool_size=30
    )
    retriever.retrieve(Query(query_id="q", text="x"), top_k=5)
    assert base.top_ks == [30, 30]


def test_decompose_omits_original_query_by_default() -> None:
    base = RecordingRetriever()
    retriever = DecomposingRetriever(base, ConstantGenerator("a?\nb?"))

    retriever.retrieve(Query(query_id="q", text="original?"), top_k=5)

    # The recorded 2Wiki/NQ/SciFact results were all produced without the original
    # query as an arm; the default must keep reproducing them.
    assert [q.text for q in base.received] == ["a?", "b?"]


def test_decompose_include_original_prepends_the_unmodified_query() -> None:
    base = RecordingRetriever()
    retriever = DecomposingRetriever(
        base, ConstantGenerator("a?\nb?"), include_original=True
    )

    retriever.retrieve(Query(query_id="q", text="original?"), top_k=5)

    assert [q.text for q in base.received] == ["original?", "a?", "b?"]


def test_decompose_include_original_does_not_duplicate_an_echoed_query() -> None:
    base = RecordingRetriever()
    # The prompt tells the LLM to return a simple question unchanged, so the original
    # can legitimately come back as one of the sub-questions. Fusing it twice would
    # double its RRF mass for no reason.
    retriever = DecomposingRetriever(
        base, ConstantGenerator("original?\nb?"), include_original=True
    )

    retriever.retrieve(Query(query_id="q", text="original?"), top_k=5)

    assert [q.text for q in base.received] == ["original?", "b?"]


def test_decompose_rejects_an_unknown_fusion() -> None:
    with pytest.raises(ValueError, match="decompose 'fusion' must be one of"):
        DecomposingRetriever(
            RecordingRetriever(), ConstantGenerator("a?"), fusion="mystery"
        )


def test_decompose_best_rank_fusion_prefers_the_single_arm_specialist() -> None:
    # Distinct scores per sub-query so the two fusions can disagree: "spec" is placed
    # first by one sub-query only, "broad" is mid-ranked by both. Summing (rrf) lets
    # breadth win; taking the best rank keeps the specialist on top.
    class TwoDocRetriever:
        def retrieve(self, query: Query, top_k: int) -> CandidateSet:
            if query.text == "a?":
                ranks = {"ev-spec": 1, "ev-broad": 5}
            else:
                ranks = {"ev-broad": 4}
            return CandidateSet(
                query_id=query.query_id,
                candidates=tuple(
                    EvidenceCandidate(
                        evidence_id=evidence_id,
                        document_id=evidence_id,
                        chunk_id=f"chunk-{evidence_id}",
                        text=evidence_id,
                        source_uri=f"fixture://{evidence_id}",
                        retrieval_score=1.0 / rank,
                        retrieval_rank=rank,
                    )
                    for evidence_id, rank in sorted(ranks.items(), key=lambda kv: kv[1])
                ),
            )

    query = Query(query_id="q", text="x")
    generator = ConstantGenerator("a?\nb?")

    rrf = DecomposingRetriever(TwoDocRetriever(), generator).retrieve(query, top_k=5)
    best = DecomposingRetriever(
        TwoDocRetriever(), generator, fusion="best-rank"
    ).retrieve(query, top_k=5)

    assert rrf.candidates[0].evidence_id == "ev-broad"
    assert best.candidates[0].evidence_id == "ev-spec"


def test_decompose_include_original_still_falls_back_when_no_subqueries() -> None:
    base = RecordingRetriever()
    retriever = DecomposingRetriever(
        base, ConstantGenerator("\n\n"), include_original=True
    )

    retriever.retrieve(Query(query_id="q", text="only question"), top_k=5)

    # Fallback already retrieves the original exactly once — not twice.
    assert [q.text for q in base.received] == ["only question"]
