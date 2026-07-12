"""RAG generation-line prompts (the ``rag`` phase).

These five prompts control the three RAG pipelines in :mod:`src.rag_pipeline`:

* :data:`DEFAULT_RAG_PROMPT` — the vanilla ``RAGPipeline`` prompt; instructs the model
  to answer *only* from retrieved context (faithfulness is therefore measurable).
* :data:`CITATION_RAG_PROMPT` — the citation-aware variant; emits a structured
  ``Answer: ...`` + ``Evidence: [i], ...`` format that :func:`src.rag_pipeline.parse_citation_output`
  parses back into per-chunk citations (falls back to post-hoc attribution on failure).
* :data:`ELICIT_PROMPT` / :data:`CONSOLIDATE_PROMPT` / :data:`FINALIZE_PROMPT` — the three
  prompts of ``AstuteRAGPipeline`` (Wang et al., 2024): elicit internal knowledge,
  consolidate with source-tagged retrieved passages, finalise from the notes.

All prompts are imported (never re-defined) by their consumers; the central registry
in :mod:`src.prompts` is the single source of truth. See
``docs/superpowers/specs/2026-07-12-prompt-centralization-design.md``.
"""
from __future__ import annotations

DEFAULT_RAG_PROMPT = (
    "Answer the question using only the context below. "
    "Give only the answer itself — the shortest phrase that answers the question, "
    "with no explanation and without repeating or quoting the context. "
    "If the answer is not contained in the context, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

CITATION_RAG_PROMPT = (
    "Answer the question using only the context below.\n"
    "For every claim in your answer, cite the supporting source by its chunk number.\n\n"
    "You must follow this output format EXACTLY:\n"
    "Answer: <the shortest phrase that answers the question>\n"
    "Evidence: [chunk_number_1], [chunk_number_2], ...\n\n"
    "If the answer is not contained in the context, say:\n"
    "Answer: I don't know\n"
    "Evidence: []\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

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