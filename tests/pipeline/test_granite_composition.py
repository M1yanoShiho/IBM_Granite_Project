from collections.abc import Sequence

from evidence_rag.composition import build_granite_baseline, build_q2d_granite_baseline
from evidence_rag.contracts.models import Document, Query


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


class FakeLLM:
    def generate(self, prompt: str) -> str:
        if "Write a short, factual passage" in prompt:
            return "The report says revenue increased."
        return "Answer: Revenue increased by ten percent.\nEvidence: [1]"


def test_granite_baseline_composes_retriever_selector_and_generator() -> None:
    pipeline = build_granite_baseline(
        (
            Document(
                document_id="annual-report",
                text="Revenue increased by ten percent.",
                source_uri="fixture://annual-report",
            ),
        ),
        embedder=FakeEmbedder(),
        llm=FakeLLM(),
    )

    run = pipeline.run_with_trace(
        Query(query_id="q", text="revenue change"),
        top_k=3,
        max_selected=1,
    )

    assert run.candidates.candidates[0].document_id == "annual-report"
    assert run.selection.items[0].evidence_id == run.selected.evidence[0].evidence_id
    assert run.generation.answer == "Revenue increased by ten percent."
    assert run.generation.cited_evidence_ids == (run.selected.evidence[0].evidence_id,)


def test_q2d_granite_baseline_composes_with_same_contracts() -> None:
    pipeline = build_q2d_granite_baseline(
        (
            Document(
                document_id="annual-report",
                text="Revenue increased by ten percent.",
                source_uri="fixture://annual-report",
            ),
        ),
        embedder=FakeEmbedder(),
        llm=FakeLLM(),
    )

    run = pipeline.run_with_trace(
        Query(query_id="q", text="what happened"),
        top_k=3,
        max_selected=1,
    )

    assert run.candidates.candidates[0].document_id == "annual-report"
    assert run.generation.answer == "Revenue increased by ten percent."
