"""Demo service layer: build a RAG pipeline from raw text.

The Streamlit demo (``app/main.py``) calls this thin seam so the UI stays free of
construction logic and the composition stays unit-tested. It reuses the retriever
factory (``src.retrieval.factory``) for the retrieval half and pairs it with an
injected LLM in a :class:`~src.rag_pipeline.RAGPipeline` — the same pipeline the
evaluation path uses.

The LLM is passed in rather than constructed here, so the heavy model load can be
cached by the caller (``st.cache_resource``) and so this function stays testable
without loading a real model.
"""

from __future__ import annotations

from src.llm_client import LLMClient
from src.rag_pipeline import RAGPipeline
from src.retrieval.factory import build_dense_retriever_from_text


def build_rag_pipeline_from_text(
    document_text: str,
    llm: LLMClient,
    *,
    top_k: int = 4,
    backend: str = "granite",
    embedding_model_id: str | None = None,
    doc_id: str = "document",
) -> RAGPipeline:
    """Build a retrieve-then-generate ``RAGPipeline`` over a single document.

    Embeds ``document_text`` into a dense retriever (via
    :func:`~src.retrieval.factory.build_dense_retriever_from_text`) and pairs it
    with ``llm`` for generation. ``top_k`` controls both how many chunks the
    retriever returns and how many are passed to the generator as context, so the
    answer is grounded in exactly the chunks shown as citations.
    """
    retriever = build_dense_retriever_from_text(
        document_text,
        doc_id=doc_id,
        top_k=top_k,
        backend=backend,
        embedding_model_id=embedding_model_id,
    )
    return RAGPipeline(retriever, llm, top_k=top_k)
