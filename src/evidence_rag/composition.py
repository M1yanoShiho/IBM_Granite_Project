from collections.abc import Iterable

from evidence_rag.contracts.models import Document
from evidence_rag.generator.extractive import ExtractiveGenerator
from evidence_rag.generator.granite import GraniteGenerator, GraniteLLMClient, TextGenerator
from evidence_rag.pipeline.service import EvidenceRAGPipeline
from evidence_rag.retriever.bm25 import BM25Retriever
from evidence_rag.retriever.granite import (
    GraniteDenseRetriever,
    Query2DocRetriever,
    TextEmbedder,
)
from evidence_rag.selector.corroboration import CorroborationSelector
from evidence_rag.selector.top_k import TopKSelector


def build_baseline(documents: Iterable[Document]) -> EvidenceRAGPipeline:
    return EvidenceRAGPipeline(
        retriever=BM25Retriever(documents),
        selector=TopKSelector(),
        generator=ExtractiveGenerator(),
    )


def build_granite_baseline(
    documents: Iterable[Document],
    *,
    embedder: TextEmbedder | None = None,
    llm: TextGenerator | None = None,
) -> EvidenceRAGPipeline:
    return EvidenceRAGPipeline(
        retriever=GraniteDenseRetriever(documents, embedder=embedder),
        selector=TopKSelector(),
        generator=GraniteGenerator(llm=llm),
    )


def build_q2d_granite_baseline(
    documents: Iterable[Document],
    *,
    embedder: TextEmbedder | None = None,
    llm: TextGenerator | None = None,
) -> EvidenceRAGPipeline:
    shared_llm = llm or GraniteLLMClient()
    dense_retriever = GraniteDenseRetriever(documents, embedder=embedder)
    return EvidenceRAGPipeline(
        retriever=Query2DocRetriever(dense_retriever, shared_llm),
        selector=TopKSelector(),
        generator=GraniteGenerator(llm=shared_llm),
    )


def build_q2d_corroboration_granite(
    documents: Iterable[Document],
    *,
    embedder: TextEmbedder | None = None,
    llm: TextGenerator | None = None,
    alpha: float = 0.6,
) -> EvidenceRAGPipeline:
    shared_llm = llm or GraniteLLMClient()
    dense_retriever = GraniteDenseRetriever(documents, embedder=embedder)
    return EvidenceRAGPipeline(
        retriever=Query2DocRetriever(dense_retriever, shared_llm),
        selector=CorroborationSelector(shared_llm, alpha=alpha),
        generator=GraniteGenerator(llm=shared_llm),
    )
