"""Cross-encoder reranking — second-stage retrieval.

A bi-encoder retriever (the Granite dense retriever) embeds query and document
*separately* and compares vectors — fast, but it never lets the two texts interact.
A **cross-encoder reranker** scores a ``(query, document)`` pair jointly (full
attention over both), so it ranks a candidate pool more accurately. The standard
two-stage architecture, and the usual way to actually improve a retrieval system:

    dense retrieve top-N  ->  cross-encoder rerank  ->  top-k

Here both stages are IBM Granite (``granite-embedding-*`` bi-encoder +
``granite-embedding-reranker-english-r2``), so the system stays in-family. Reranking
a *hybrid* (granite + BM25) candidate pool can also surface the complementary BM25
finds that rank fusion alone could not.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Sequence

from src.retrieval.base import RetrievedChunk, Retriever
from src.retrieval.corroboration import corroboration_scores
from src.retrieval.fusion import minmax_normalize

DEFAULT_RERANKER_MODEL_ID = "ibm-granite/granite-embedding-reranker-english-r2"


class Reranker:
    """A cross-encoder that re-scores ``(query, document)`` pairs.

    Parameters
    ----------
    model_id:
        The reranker model (default: the Granite English r2 reranker).
    model:
        An already-loaded model exposing ``predict(pairs) -> scores`` (injected in
        tests). ``None`` loads ``model_id`` via sentence-transformers ``CrossEncoder``.

    Notes
    -----
    Loading uses sentence-transformers ``CrossEncoder``; if the Granite reranker's
    model card prescribes a different entry point (e.g. ``AutoModelForSequence
    Classification`` or ``trust_remote_code=True``), adjust :meth:`_load_model` —
    the rest of the class only depends on ``predict``.
    """

    def __init__(self, model_id: str = DEFAULT_RERANKER_MODEL_ID, model=None) -> None:
        self.model_id = model_id
        self._model = model if model is not None else self._load_model()

    def _load_model(self):
        from sentence_transformers import CrossEncoder

        return CrossEncoder(self.model_id, cache_folder=os.getenv("MODEL_CACHE_DIR") or None)

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        """Cross-encoder relevance score for ``query`` against each text."""
        if not texts:
            return []
        return [float(s) for s in self._model.predict([(query, t) for t in texts])]

    def rerank(
        self, query: str, candidates: List[RetrievedChunk], top_k: int
    ) -> List[RetrievedChunk]:
        """Re-rank ``candidates`` by cross-encoder score, best-first, keep ``top_k``.

        The returned chunks carry the **reranker** score (not the first-stage one).
        Stable: equal scores keep their first-stage order.
        """
        if not candidates:
            return []
        scored = list(zip(candidates, self.score(query, [c.text for c in candidates])))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [
            RetrievedChunk(doc_id=c.doc_id, text=c.text, score=float(s))
            for c, s in scored[:top_k]
        ]


class TwoStageRetriever:
    """First-stage retriever + cross-encoder reranker, as one ``Retriever``.

    Parameters
    ----------
    retriever:
        The first stage (any ``Retriever`` — the Granite dense retriever or a
        hybrid) producing the candidate pool.
    reranker:
        The :class:`Reranker` that re-scores the pool.
    top_k:
        Number of reranked results to return.
    candidates:
        How many of the first stage's results to rerank (the rest are dropped). A
        wider pool gives the reranker more to work with at more cost.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        top_k: int = 10,
        candidates: int = 100,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if candidates < 1:
            raise ValueError("candidates must be at least 1.")
        self.retriever = retriever
        self.reranker = reranker
        self.top_k = top_k
        self.candidates = candidates

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """Retrieve a candidate pool, then return the reranked top-k."""
        pool = self.retriever.retrieve(query)[: self.candidates]
        return self.reranker.rerank(query, pool, self.top_k)


class LLMListwiseReranker:
    """RankGPT-style listwise reranking with the project's own generative LLM.

    A cross-encoder (:class:`Reranker`) scores each ``(query, doc)`` pair
    independently; a *listwise* reranker shows the LLM a whole window of candidates
    at once and asks for a permutation, so documents are judged in context. A
    sliding window (size ``window``, stride ``step``) traversed back-to-front
    covers pools larger than the model's context and lets the best candidates
    bubble toward the top (Sun et al., 2023, "Is ChatGPT Good at Search?").

    Exposes the same ``rerank(query, candidates, top_k)`` contract as
    :class:`Reranker`, so :class:`TwoStageRetriever` wraps it unchanged. Cost is
    ~``pool / step`` LLM calls per query, so it is meant for the HPC / small query
    sets (and a modest ``candidates`` pool), not full-corpus latency-sensitive use.

    Parameters
    ----------
    llm:
        Any object with ``generate(prompt) -> str`` (the project's
        :class:`~src.llm_client.LLMClient`; injected so it can be reused/faked).
    window, step:
        Sliding-window size and stride over the candidate pool.
    passage_chars:
        Max characters of each passage shown to the LLM (keeps the prompt bounded).
    """

    def __init__(
        self, llm, window: int = 20, step: int = 10, passage_chars: int = 300
    ) -> None:
        if window < 1:
            raise ValueError("window must be at least 1.")
        if step < 1:
            raise ValueError("step must be at least 1.")
        self.llm = llm
        self.window = window
        self.step = step
        self.passage_chars = passage_chars

    def rerank(
        self, query: str, candidates: List[RetrievedChunk], top_k: int
    ) -> List[RetrievedChunk]:
        """Listwise-rerank ``candidates`` with a back-to-front sliding window.

        Returns the top ``top_k`` chunks carrying a rank-derived, strictly
        descending score (the LLM gives an ordering, not calibrated scores).
        """
        if not candidates:
            return []
        order = list(range(len(candidates)))
        n = len(order)
        # Slide from the tail to the head so each window's best rises toward the front.
        for start in range(max(0, n - self.window), -1, -self.step):
            window_idx = order[start : start + self.window]
            perm = self._rank_window(query, [candidates[i] for i in window_idx])
            order[start : start + self.window] = [window_idx[j] for j in perm]
        return [
            RetrievedChunk(
                doc_id=candidates[i].doc_id,
                text=candidates[i].text,
                score=float(n - rank),
            )
            for rank, i in enumerate(order[:top_k])
        ]

    def _rank_window(self, query: str, docs: List[RetrievedChunk]) -> List[int]:
        """Ask the LLM to order one window; return a permutation of its indices."""
        listing = "\n".join(
            f"[{i + 1}] {doc.text[: self.passage_chars]}" for i, doc in enumerate(docs)
        )
        prompt = (
            f"Rank the {len(docs)} passages below by their relevance to the query, "
            "most relevant first.\n"
            f"Query: {query}\n\n"
            f"{listing}\n\n"
            "Answer with only the ranking as identifiers, e.g. 3 > 1 > 2."
        )
        return self._parse_permutation(self.llm.generate(prompt), len(docs))

    @staticmethod
    def _parse_permutation(text: str, n: int) -> List[int]:
        """Parse identifiers (``[2] > [1] > [3]`` or ``2 > 1 > 3``) into a
        zero-indexed permutation of ``range(n)``.

        Robust to a messy model: out-of-range and duplicate ids are dropped, and
        any positions the model omitted are appended in their original order (an
        identity fallback), so the result is always a valid permutation.
        """
        seen: set[int] = set()
        order: List[int] = []
        for token in re.findall(r"\d+", text):
            idx = int(token) - 1
            if 0 <= idx < n and idx not in seen:
                seen.add(idx)
                order.append(idx)
        order.extend(i for i in range(n) if i not in seen)
        return order


_EXTRACT_PROMPT = (
    "Using ONLY the passage below, answer the question with the shortest exact answer "
    "(a name, place, date, or number). If the passage does not answer it, reply NONE.\n"
    "Question: {question}\n"
    "Passage: {passage}\n"
    "Answer:"
)
_PARAMETRIC_PROMPT = (
    "Answer the question with the shortest exact answer from your own knowledge. "
    "If you are not sure, reply NONE.\n"
    "Question: {question}\n"
    "Answer:"
)


class CorroborationReranker:
    """Rerank by cross-source answer corroboration instead of query relevance.

    Extracts each candidate's answer with the LLM, scores how many OTHER candidates
    (plus the model's own parametric answer) give the same answer
    (:func:`~src.retrieval.corroboration.corroboration_scores`), then blends that --
    min-max normalised -- with the min-max normalised first-stage relevance:
    ``final = alpha * relevance + (1 - alpha) * corroboration`` (``alpha`` is the
    relevance weight, like the hybrid's dense weight; ``alpha = 1`` -> plain first
    stage). A lone counterfactual is relevant but uncorroborated, so it falls below the
    corroborated needle -- the separation a relevance reranker cannot make.

    Same ``rerank(query, candidates, top_k)`` contract as :class:`Reranker`, so
    :class:`TwoStageRetriever` wraps it unchanged. Cost = ``top_n`` (+1 parametric) LLM
    calls/query; only the first ``top_n`` candidates are re-scored, the rest keep their
    first-stage order below. Design: 2026-07-05-corroboration-reranking-design.md.
    """

    def __init__(
        self,
        llm,
        top_n: int = 20,
        alpha: float = 0.5,
        use_parametric: bool = True,
        passage_chars: int = 600,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {alpha}.")
        if top_n < 1:
            raise ValueError("top_n must be at least 1.")
        self.llm = llm
        self.top_n = top_n
        self.alpha = alpha
        self.use_parametric = use_parametric
        self.passage_chars = passage_chars

    def _extract_answer(self, query: str, passage: str) -> str:
        prompt = _EXTRACT_PROMPT.format(question=query, passage=passage[: self.passage_chars])
        return self.llm.generate(prompt).strip()

    def _parametric_answer(self, query: str) -> Optional[str]:
        if not self.use_parametric:
            return None
        return self.llm.generate(_PARAMETRIC_PROMPT.format(question=query)).strip()

    def score_docs_with_answers(
        self, query: str, docs: List[RetrievedChunk]
    ) -> "tuple[List[float], List[float], List[str], Optional[str]]":
        """``(relevance, corroboration, raw answers, parametric answer)`` per doc.

        The raw extracted strings (pre-normalisation) and the parametric answer are
        what the WS-0 runs dump persists, so vote-matching changes (WS-7) can be
        re-simulated offline without re-running the extraction LLM.
        """
        answers = [self._extract_answer(query, d.text) for d in docs]
        parametric = self._parametric_answer(query)
        corr = corroboration_scores(answers, parametric)
        return [d.score for d in docs], corr, answers, parametric

    def score_docs(
        self, query: str, docs: List[RetrievedChunk]
    ) -> "tuple[List[float], List[float]]":
        """Raw ``(relevance, corroboration)`` per doc — the two signals ``rerank`` blends.

        Relevance is the first-stage score; corroboration is the cross-source
        answer-vote count (:func:`~src.retrieval.corroboration.corroboration_scores`).
        Exposed so the offline lambda-sweep (``eval.tune_corroboration``) can extract
        answers ONCE and then sweep the blend weight as pure arithmetic.
        """
        relevance, corr, _, _ = self.score_docs_with_answers(query, docs)
        return relevance, corr

    def rerank(
        self, query: str, candidates: List[RetrievedChunk], top_k: int
    ) -> List[RetrievedChunk]:
        """Re-score the first ``top_n`` candidates by relevance+corroboration; the rest
        keep their order below. Returns the top ``top_k`` chunks (blended score)."""
        if not candidates:
            return []
        window = candidates[: self.top_n]
        relevance, corr = self.score_docs(query, window)
        rel_norm = minmax_normalize({i: relevance[i] for i in range(len(window))})
        cor_norm = minmax_normalize({i: corr[i] for i in range(len(window))})
        final = [
            self.alpha * rel_norm[i] + (1.0 - self.alpha) * cor_norm[i]
            for i in range(len(window))
        ]
        # Stable sort: ties keep first-stage order (Python sort is stable under reverse).
        order = sorted(range(len(window)), key=lambda i: final[i], reverse=True)
        reranked = [
            RetrievedChunk(window[i].doc_id, window[i].text, float(final[i]))
            for i in order
        ]
        return (reranked + candidates[self.top_n :])[:top_k]
