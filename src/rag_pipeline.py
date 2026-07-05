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

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from src.explainability.citations import Citation, attribute_answer
from src.llm_client import LLMClient
from src.retrieval.base import RetrievedChunk, Retriever

# Default prompt: instructs the model to answer *only* from the retrieved
# context, which is what makes faithfulness measurable and reduces hallucination.
DEFAULT_RAG_PROMPT = (
    "Answer the question using only the context below. "
    "Give only the answer itself — the shortest phrase that answers the question, "
    "with no explanation and without repeating or quoting the context. "
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
    citations:
        Sentence-level source attributions for the generated answer.
    abstained:
        ``True`` when the pipeline declines to treat the answer as supported.
    abstain_reason:
        Machine-readable reason for an abstention, or ``None``.
    confidence:
        Optional retrieval-confidence signal, used by corrective variants.
    used_corrective_retrieval:
        ``True`` when a corrective re-retrieval branch was used.
    """

    answer: str
    retrieved_chunks: List[RetrievedChunk]
    citations: List[Citation] = field(default_factory=list)
    abstained: bool = False
    abstain_reason: str | None = None
    confidence: float | None = None
    used_corrective_retrieval: bool = False


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

    @staticmethod
    def _is_unknown_answer(answer: str) -> bool:
        """True when the model DECLINED to answer (the prompt says "say you don't know").

        Bare "unknown"/"n/a"/"none" must be the WHOLE answer -- as a substring it would
        wrongly flag legitimate answers like "The Unknown Soldier". The multi-word
        decline phrases stay substring matches (low false-positive risk).
        """
        normalized = answer.strip().lower().strip(".!?\"' ")
        if normalized in ("unknown", "n/a", "none"):
            return True
        decline_phrases = (
            "i don't know",
            "i do not know",
            "don't know",
            "do not know",
            "not contained in the context",
            "not in the context",
        )
        return any(phrase in normalized for phrase in decline_phrases)

    def _build_result(
        self,
        answer: str,
        chunks: List[RetrievedChunk],
        *,
        confidence: float | None = None,
        used_corrective_retrieval: bool = False,
    ) -> RAGResult:
        citations = attribute_answer(answer, chunks)
        abstained = False
        reason: str | None = None

        if not chunks:
            abstained = True
            reason = "no_retrieved_context"
        elif self._is_unknown_answer(answer):
            abstained = True
            reason = "model_reported_unknown"
        elif not citations:
            abstained = True
            reason = "answer_not_attributed"

        return RAGResult(
            answer=answer,
            retrieved_chunks=chunks,
            citations=citations,
            abstained=abstained,
            abstain_reason=reason,
            confidence=confidence,
            used_corrective_retrieval=used_corrective_retrieval,
        )

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
        # Cap the context at top_k chunks independently of the retriever's own
        # depth: the same retriever may be configured to return more (e.g. when
        # shared with the eval harness), but the prompt should only carry top_k.
        # If the retriever returns fewer, we use what's available (no padding).
        chunks = self.retriever.retrieve(question)[: self.top_k]
        prompt = self._build_prompt(question, chunks)
        answer = self.llm.generate(prompt)
        return self._build_result(answer, chunks)


class CorrectiveRAGPipeline(RAGPipeline):
    """A confidence-gated, adaptive variant of :class:`RAGPipeline`.

    Turns the static single-shot flow into: retrieve -> score retrieval
    confidence -> if it is below ``confidence_threshold``, re-retrieve with a
    ``query_rewriter``-rewritten query and widen the context to
    ``fallback_top_k`` -> generate. Closed-corpus, so the corrective action is a
    re-retrieval (e.g. a HyDE-expanded query), not a web search.

    The confidence signal is deliberately **model-free**: the *margin* of the
    top-1 retrieval score over the top-2, normalised by the top score. It is ~0
    when the leading results are indistinguishable (ambiguous retrieval → correct)
    and near 1 when one document clearly dominates (confident → keep). This is a
    lightweight gate in the spirit of Corrective RAG (Yan et al., 2024), **not**
    their learned retrieval evaluator — named for the family, not a reimplementation.

    Only the pipeline changes, so ``eval.run_rag`` scores it against the vanilla
    pipeline with everything else fixed: the cover-EM delta is attributable to the
    adaptive loop alone.

    Parameters
    ----------
    query_rewriter:
        A ``str -> str`` callable used to rewrite the query on a low-confidence
        first pass (e.g. :class:`~src.retrieval.query_transform.HyDETransform`).
        ``None`` disables correction (the pipeline then matches the vanilla one).
    confidence_threshold:
        Re-retrieve when the first-pass confidence is *below* this value.
    fallback_top_k:
        Context depth used after a corrective re-retrieval (usually wider than
        ``top_k`` to give the generator more to work with).
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        top_k: int = 4,
        prompt_template: str = DEFAULT_RAG_PROMPT,
        query_rewriter: Optional[Callable[[str], str]] = None,
        confidence_threshold: float = 0.5,
        fallback_top_k: int = 8,
    ) -> None:
        super().__init__(retriever, llm, top_k=top_k, prompt_template=prompt_template)
        self.query_rewriter = query_rewriter
        self.confidence_threshold = confidence_threshold
        self.fallback_top_k = fallback_top_k

    @staticmethod
    def _confidence(chunks: List[RetrievedChunk]) -> float:
        """Retrieval confidence in ``[0, 1]``: the top-1 score's margin over top-2.

        ``0.0`` when nothing was retrieved or the top score is non-positive; ``1.0``
        when there is a single candidate. Otherwise ``(s0 - s1) / s0`` clamped to
        ``[0, 1]`` — small when the leading results tie (ambiguous), large when one
        dominates.
        """
        if not chunks:
            return 0.0
        if len(chunks) < 2:
            return 1.0
        top, second = chunks[0].score, chunks[1].score
        if top <= 0:
            return 0.0
        return max(0.0, min(1.0, (top - second) / top))

    def query(self, question: str) -> RAGResult:
        """Retrieve, and if the first pass is low-confidence, correct then generate."""
        chunks = self.retriever.retrieve(question)
        confidence = self._confidence(chunks)
        used_corrective_retrieval = False
        low_confidence_without_rewriter = (
            self.query_rewriter is None and confidence < self.confidence_threshold
        )

        if self.query_rewriter is not None and confidence < self.confidence_threshold:
            chunks = self.retriever.retrieve(self.query_rewriter(question))
            top = chunks[: self.fallback_top_k]
            used_corrective_retrieval = True
        else:
            top = chunks[: self.top_k]
        answer = self.llm.generate(self._build_prompt(question, top))
        result = self._build_result(
            answer,
            top,
            confidence=confidence,
            used_corrective_retrieval=used_corrective_retrieval,
        )
        if low_confidence_without_rewriter and not result.abstained:
            result.abstained = True
            result.abstain_reason = "low_retrieval_confidence"
        return result


class AstuteRAGPipeline(RAGPipeline):
    """Source-aware internal/external consolidation, an Astute-RAG-style variant.

    Where the vanilla pipeline generates straight from the retrieved passages, this
    runs the three steps of Astute RAG (Wang et al., 2024) so a single misleading
    passage cannot decide the answer on its own:

    1. **Elicit** — the LLM writes a passage from its *own* parametric knowledge,
       grounded on the question only (``ELICIT_PROMPT``), giving an independent
       source to check the retrieved ones against.
    2. **Consolidate** — the retrieved passages (tagged ``[Document i]``) and the
       elicited passage (tagged ``[Model knowledge]``) go into one *source-aware*
       prompt (``CONSOLIDATE_PROMPT``); the LLM keeps agreeing facts, flags
       conflicts, and drops what it judges unreliable.
    3. **Finalise** — the answer is generated from the consolidated notes
       (``FINALIZE_PROMPT``), not the raw passages.

    This directly targets the NIAH failure mode measured on the task: the top-k
    often contains a near-duplicate *counterfactual* distractor a reranker cannot
    tell from the needle — consolidation cross-checks it against the other sources
    and the model's own knowledge before answering. Like
    :class:`CorrectiveRAGPipeline`, it is a lightweight, prompt-only member of the
    family (**not** a fine-tuned reimplementation) and reuses the single injected
    ``LLMClient`` for all three calls (three generations per query, no extra load).
    Only the generation flow changes, so ``eval.run_rag`` scores it against the
    vanilla pipeline with retrieval fixed — any cover-EM delta is the consolidation's.
    """

    ELICIT_PROMPT = (
        "Generate a short passage from your own knowledge that answers the question. "
        "If you are unsure, state only what you are confident about.\n\n"
        "Question: {question}\nPassage:"
    )
    CONSOLIDATE_PROMPT = (
        "You are given passages about a question from two kinds of source: DOCUMENTS "
        "retrieved from a corpus (which may be wrong or contradict each other) and "
        "your own MODEL knowledge. Consolidate them — keep facts that agree across "
        "sources, flag conflicts, and drop information you judge unreliable.\n\n"
        "{sources}\n\n"
        "Question: {question}\nConsolidated notes:"
    )
    FINALIZE_PROMPT = (
        "Answer the question using only the consolidated notes below. Give only the "
        "answer itself — the shortest phrase that answers the question, with no "
        "explanation. If the notes do not contain the answer, say you don't know.\n\n"
        "Consolidated notes:\n{consolidated}\n\nQuestion: {question}\nAnswer:"
    )

    def _format_sources(self, chunks: List[RetrievedChunk], internal: str) -> str:
        """Source-tag each passage so the consolidator can weigh reliability: the
        retrieved chunks as ``[Document i]`` and the elicited passage as ``[Model]``."""
        lines = [f"[Document {i + 1}] {chunk.text}" for i, chunk in enumerate(chunks)]
        lines.append(f"[Model knowledge] {internal}")
        return "\n".join(lines)

    def query(self, question: str) -> RAGResult:
        chunks = self.retriever.retrieve(question)[: self.top_k]
        internal = self.llm.generate(self.ELICIT_PROMPT.format(question=question))
        sources = self._format_sources(chunks, internal)
        consolidated = self.llm.generate(
            self.CONSOLIDATE_PROMPT.format(sources=sources, question=question)
        )
        answer = self.llm.generate(
            self.FINALIZE_PROMPT.format(consolidated=consolidated, question=question)
        )
        return self._build_result(answer, chunks)
