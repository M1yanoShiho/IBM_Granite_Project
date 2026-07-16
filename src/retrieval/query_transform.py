"""LLM query-side augmentation (HyDE, Query2Doc, query decomposition) as a
``Retriever`` wrapper.

HyDE (Gao et al., 2023): the LLM writes a hypothetical answer/passage; retrieving
with THAT text recasts search as document-document similarity (for a dense arm the
pseudo-doc is embedded; for a lexical arm its terms expand the query). Query2Doc
(Wang et al., 2023): prepend the pseudo-doc to the original query, keeping the
exact query terms.

DecomposingRetriever: break a complex multi-fact question into N atomic
sub-questions, retrieve for each independently, then merge the candidate pools via
Reciprocal Rank Fusion (RRF, Cormack et al., 2009). This improves recall (找全) for
information-integration queries where a single retrieval pass misses later facts.

Because each wrapper satisfies :class:`~src.retrieval.base.Retriever` (CONTRACT 1),
the same object is measured on nDCG (``eval.run_benchmark``) AND cover-EM
(``eval.run_rag``) with no harness change. The generating ``LLMClient`` is
injected, so the RAG harness reuses its single generator client instead of
loading a second model.

NB: query rewriting is **not** universally helpful — it hurts when the query
already matches the corpus lexically ("Not All Queries Need Rewriting", 2026), so
these are meant for a when-it-helps / when-it-hurts study, not an always-on
default.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List

from src.prompts.retrieval import DECOMPOSE_PROMPT, HYDE_PROMPT
from src.retrieval.base import RetrievedChunk, Retriever


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


# ---------------------------------------------------------------------------
# Query decomposition + RRF merge
# ---------------------------------------------------------------------------

class QueryDecomposeTransform:
    """Use LLM to decompose a complex query into N atomic sub-questions.

    Parameters
    ----------
    llm:
        Any object with ``generate(prompt) -> str``.
    n_subqueries:
        Number of sub-questions to generate (default: 3).
    template:
        ``str.format`` template with ``{question}`` and ``{n}`` placeholders.
    """

    def __init__(
        self,
        llm,
        n_subqueries: int = 3,
        template: str = DECOMPOSE_PROMPT,
    ) -> None:
        self.llm = llm
        self.n_subqueries = n_subqueries
        self.template = template

    def __call__(self, query: str) -> List[str]:
        raw = self.llm.generate(
            self.template.format(question=query, n=self.n_subqueries)
        )
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        sub_qs: List[str] = []
        for line in lines:
            text = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if text:
                sub_qs.append(text)
        # Fall back to original query if parsing fails
        return sub_qs[: self.n_subqueries] if sub_qs else [query]


def _rrf_merge(
    ranked_lists: List[List[RetrievedChunk]],
    k: int = 60,
    top_k: int = 20,
) -> List[RetrievedChunk]:
    """Reciprocal Rank Fusion (Cormack 2009) over multiple ranked candidate lists.

    ``rrf(d) = Σ_i  1 / (k + rank_i(d))``

    Documents absent from a list contribute 0 for that list. The constant k=60
    controls how steeply rank position is penalised (higher k → flatter weighting).
    """
    rrf_scores: Dict[str, float] = {}
    seen: Dict[str, RetrievedChunk] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, 1):
            rrf_scores[chunk.doc_id] = rrf_scores.get(chunk.doc_id, 0.0) + 1.0 / (k + rank)
            if chunk.doc_id not in seen:
                seen[chunk.doc_id] = chunk

    sorted_ids = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)

    results: List[RetrievedChunk] = []
    for new_rank, doc_id in enumerate(sorted_ids[:top_k], 1):
        base = seen[doc_id]
        results.append(
            RetrievedChunk(
                doc_id=base.doc_id,
                text=base.text,
                score=rrf_scores[doc_id],
                rank=new_rank,
                metadata=base.metadata,
            )
        )
    return results


class DecomposingRetriever:
    """Multi-sub-question retrieval with RRF merge.

    Implements the §5.3 innovation: for complex multi-fact queries, decompose
    into sub-questions → retrieve per sub-question → merge via RRF.

    This improves recall (找全) on information-integration queries where a single
    retrieval pass anchors on one part of the question and misses the rest.

    Parameters
    ----------
    base:
        Any :class:`~src.retrieval.base.Retriever` (BM25, dense, hybrid …).
    decompose:
        A :class:`QueryDecomposeTransform` that maps query → list[sub-question].
    include_original:
        Also retrieve with the unmodified query and include in the RRF merge.
    rrf_k:
        RRF smoothing constant (default: 60, the standard from the original paper).
    top_k:
        Number of candidates to return after merging.
    """

    def __init__(
        self,
        base: Retriever,
        decompose: QueryDecomposeTransform,
        include_original: bool = True,
        rrf_k: int = 60,
        top_k: int = 20,
    ) -> None:
        self.base = base
        self.decompose = decompose
        self.include_original = include_original
        self.rrf_k = rrf_k
        self.top_k = top_k

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        sub_queries = self.decompose(query)

        ranked_lists: List[List[RetrievedChunk]] = []
        if self.include_original:
            ranked_lists.append(self.base.retrieve(query))

        for sub_q in sub_queries:
            ranked_lists.append(self.base.retrieve(sub_q))

        return _rrf_merge(ranked_lists, k=self.rrf_k, top_k=self.top_k)
