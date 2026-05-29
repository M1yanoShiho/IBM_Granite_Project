"""Basic Retrieval-Augmented Generation (RAG) pipeline.

This module provides scaffolding for a *retrieval* baseline to compare against
the "stuff the whole document into context" approach used in the core NIAH
test. Instead of feeding the entire haystack to the model, the RAG pipeline:

1. Splits the haystack into chunks.
2. Embeds and indexes the chunks in a vector store.
3. Retrieves the top-k chunks most relevant to the probe question.
4. Passes only those chunks to the model to answer.

Comparing the two approaches helps separate *retrieval* failures from
*long-context reasoning* failures.

The implementation is left as a documented skeleton.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.llm_client import LLMClient


@dataclass
class RAGResult:
    """Output of a single RAG query.

    Attributes
    ----------
    answer:
        The model's generated answer.
    retrieved_chunks:
        The chunks supplied to the model as context (for context-precision
        scoring).
    """

    answer: str
    retrieved_chunks: List[str]


class RAGPipeline:
    """A minimal retrieve-then-generate pipeline.

    Parameters
    ----------
    llm:
        The :class:`~src.llm_client.LLMClient` used for generation.
    top_k:
        Number of chunks to retrieve per query.
    chunk_size:
        Chunk size used when indexing documents.
    chunk_overlap:
        Overlap between consecutive chunks.
    """

    def __init__(
        self,
        llm: LLMClient,
        top_k: int = 4,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        self.llm = llm
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._vector_store = None  # populated by ``index``

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #
    def index(self, document: str) -> None:
        """Chunk, embed, and index a document for retrieval.

        Parameters
        ----------
        document:
            The haystack text to index.
        """
        raise NotImplementedError(
            "TODO: split into chunks, embed, and build a vector store."
        )

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def retrieve(self, query: str) -> List[str]:
        """Return the top-k chunks most relevant to ``query``.

        Parameters
        ----------
        query:
            The probe question.

        Returns
        -------
        list of str
            The retrieved context chunks.
        """
        raise NotImplementedError("TODO: query the vector store.")

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def query(self, question: str, prompt_template: Optional[str] = None) -> RAGResult:
        """Run the full retrieve-then-generate flow for a question.

        Parameters
        ----------
        question:
            The probe question.
        prompt_template:
            Optional template for assembling the final prompt from the
            retrieved chunks and the question.

        Returns
        -------
        RAGResult
            The answer plus the retrieved chunks used to produce it.
        """
        raise NotImplementedError(
            "TODO: retrieve chunks, build prompt, call the LLM, return RAGResult."
        )
