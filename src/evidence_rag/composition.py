from collections.abc import Iterable

from evidence_rag.contracts.models import Document
from evidence_rag.generator.extractive import ExtractiveGenerator
from evidence_rag.pipeline.service import EvidenceRAGPipeline
from evidence_rag.retriever.bm25 import BM25Retriever
from evidence_rag.selector.top_k import TopKSelector


def build_baseline(documents: Iterable[Document]) -> EvidenceRAGPipeline:
    return EvidenceRAGPipeline(
        retriever=BM25Retriever(documents),
        selector=TopKSelector(),
        generator=ExtractiveGenerator(),
    )
