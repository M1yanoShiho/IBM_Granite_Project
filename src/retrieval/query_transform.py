"""LLM query-side augmentation (HyDE, Query2Doc) as a ``Retriever`` wrapper.

HyDE (Gao et al., 2023): the LLM writes a hypothetical answer/passage; retrieving
with THAT text recasts search as document-document similarity (for a dense arm the
pseudo-doc is embedded; for a lexical arm its terms expand the query). Query2Doc
(Wang et al., 2023): prepend the pseudo-doc to the original query, keeping the
exact query terms.

Because it wraps *any* :class:`~src.retrieval.base.Retriever` (CONTRACT 1), the
same object is measured on nDCG (``eval.run_benchmark``) AND cover-EM
(``eval.run_rag``) with no harness change. The generating ``LLMClient`` is
injected, so the RAG harness reuses its single generator client instead of
loading a second model.

NB: query rewriting is **not** universally helpful — it hurts when the query
already matches the corpus lexically ("Not All Queries Need Rewriting", 2026), so
these are meant for a when-it-helps / when-it-hurts study, not an always-on
default.
"""

from __future__ import annotations

from typing import Callable, List

from src.retrieval.base import RetrievedChunk, Retriever

HYDE_PROMPT = (
    "Write a short, factual passage that answers the question.\n"
    "Question: {question}\n"
    "Passage:"
)


class HyDETransform:
    """Generate a hypothetical document for ``query`` and search with it.

    Parameters
    ----------
    llm:
        Any object with ``generate(prompt) -> str`` (the project's
        :class:`~src.llm_client.LLMClient`; injected so it can be reused/faked).
    template:
        A ``str.format`` template with a ``{question}`` field.
    """

    def __init__(self, llm, template: str = HYDE_PROMPT) -> None:
        self.llm = llm
        self.template = template

    def __call__(self, query: str) -> str:
        return self.llm.generate(self.template.format(question=query))


class Query2DocTransform(HyDETransform):
    """Query2Doc: concatenate the original query with the generated pseudo-doc, so
    the exact query terms are kept (unlike HyDE, which replaces the query)."""

    def __call__(self, query: str) -> str:
        pseudo_doc = self.llm.generate(self.template.format(question=query))
        return f"{query} {pseudo_doc}"


class TransformingRetriever:
    """Apply a query transform, then delegate to the wrapped retriever.

    Satisfies the :class:`~src.retrieval.base.Retriever` contract, so it drops into
    both eval harnesses unchanged.
    """

    def __init__(self, base: Retriever, transform: Callable[[str], str]) -> None:
        self.base = base
        self.transform = transform

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        return self.base.retrieve(self.transform(query))
