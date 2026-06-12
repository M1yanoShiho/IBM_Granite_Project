"""Retrieval-Augmented Generation (RAG) pipeline — a first-class deliverable.

This is the generation layer that sits **on top of the project's retrieval
core**, not a self-contained reimplementation of retrieval. It composes:

1. A :class:`~src.retrieval.base.Retriever` (the same contract the Granite dense
   retriever and the baselines implement) to fetch the most relevant chunks, and
2. An :class:`~src.llm_client.LLMClient` to generate an answer grounded in those
   chunks.

Because it reuses the canonical ``Retriever`` (CONTRACT 1), the RAG layer is
evaluated against the *same* indexed corpus and retrievers as the retrieval
benchmark — there is no second, divergent retrieval stack to keep in sync. The
"retrieve-then-generate" output (answer + supporting chunks) is then scored for
answer quality and faithfulness in ``eval/rag_metrics.py`` and attributed to
sources in ``src/explainability/citations.py``.

See ``docs/interfaces.md`` (CONTRACT 4 — RAG I/O).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.llm_client import LLMClient
from src.retrieval.base import RetrievedChunk, Retriever

# Default prompt: instructs the model to answer *only* from the retrieved
# context, which is what makes faithfulness measurable and reduces hallucination.
DEFAULT_RAG_PROMPT = (
    "Answer the question using only the context below. "
    "If the answer is not contained in the context, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)


@dataclass
class RAGResult:
    """Output of a single RAG query.

    Attributes
    ----------
    answer:
        The model's generated answer.
    retrieved_chunks:
        The chunks supplied to the model as context, as
        :class:`~src.retrieval.base.RetrievedChunk` objects (carrying
        ``doc_id``/``score``) so the result can feed both context-precision
        scoring and source citations.
    """

    answer: str
    retrieved_chunks: List[RetrievedChunk]


class RAGPipeline:
    """A retrieve-then-generate pipeline built on the canonical retriever.

    Parameters
    ----------
    retriever:
        Any object satisfying the :class:`~src.retrieval.base.Retriever`
        contract (the Granite dense retriever, a baseline, or a mock) over an
        already-indexed corpus.
    llm:
        The :class:`~src.llm_client.LLMClient` used for generation.
    top_k:
        Number of retrieved chunks to pass to the model as context.
    prompt_template:
        A ``str.format`` template with ``{context}`` and ``{question}`` fields.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        top_k: int = 4,
        prompt_template: str = DEFAULT_RAG_PROMPT,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.prompt_template = prompt_template

    def _build_prompt(self, question: str, chunks: List[RetrievedChunk]) -> str:
        """Assemble the final prompt from retrieved chunks and the question."""
        context = "\n\n".join(
            f"[{i + 1}] {chunk.text}" for i, chunk in enumerate(chunks)
        )
        return self.prompt_template.format(context=context, question=question)

    def query(self, question: str) -> RAGResult:
        """Run the full retrieve-then-generate flow for a question.

        Retrieves the top-k chunks via the shared retriever, builds a grounded
        prompt, generates an answer, and returns it alongside the chunks that
        supported it.

        Parameters
        ----------
        question:
            The user/probe question.

        Returns
        -------
        RAGResult
            The answer plus the retrieved chunks used to produce it.
        """
        chunks = self.retriever.retrieve(question)[: self.top_k]
        prompt = self._build_prompt(question, chunks)
        answer = self.llm.generate(prompt)
        return RAGResult(answer=answer, retrieved_chunks=chunks)
