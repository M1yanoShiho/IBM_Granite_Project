"""Retrieval-enhancement prompts (the ``retrieval`` phase).

Four prompts that augment or post-process retrieval:

* :data:`HYDE_PROMPT` — HyDE/Query2Doc pseudo-document generation. Historically the
  project had two identical copies under the names ``HYDE_PROMPT``
  (``src/retrieval/query_transform.py``) and ``Q2D_PROMPT``
  (``eval/niah_selector_pilot.py``). The registry keeps a single canonical definition;
  ``Q2D_PROMPT`` survives only as an identity alias in :mod:`src.prompts.__init__`
  for backwards compatibility.
* :data:`EXTRACT_PROMPT` — extract the shortest answer from a passage; used by the
  corroboration reranker (``src/retrieval/reranker.py``) and the NIAH selector
  pilot (``eval/niah_selector_pilot.py``). The src wording is authoritative (see
  spec §3.3 group 3); legacy eval-line experiments tagged in ``results-summary.md``
  used a slightly different wording.
* :data:`PARAMETRIC_PROMPT` — answer from the model's parametric knowledge alone.
* :data:`LISTWISE_RANK_PROMPT` — the listwise re-ranking prompt previously inlined
  in ``RelevanceReranker._rank_window``; extracted to a named constant so the
  registry is exhaustive. Placeholders: ``{n}`` (window size), ``{query}``,
  ``{listing}`` (pre-formatted ``[i] passage`` block).

See ``docs/superpowers/specs/2026-07-12-prompt-centralization-design.md``.
"""
from __future__ import annotations

HYDE_PROMPT = (
    "Write a short, factual passage that answers the question.\n"
    "Question: {question}\n"
    "Passage:"
)

EXTRACT_PROMPT = (
    "Using ONLY the passage below, answer the question with the shortest exact answer "
    "(a name, place, date, or number). If the passage does not answer it, reply NONE.\n"
    "Question: {question}\n"
    "Passage: {passage}\n"
    "Answer:"
)

PARAMETRIC_PROMPT = (
    "Answer the question with the shortest exact answer from your own knowledge. "
    "If you are not sure, reply NONE.\n"
    "Question: {question}\n"
    "Answer:"
)

LISTWISE_RANK_PROMPT = (
    "Rank the {n} passages below by their relevance to the query, "
    "most relevant first.\n"
    "Query: {query}\n\n"
    "{listing}\n\n"
    "Answer with only the ranking as identifiers, e.g. 3 > 1 > 2."
)

DECOMPOSE_PROMPT = (
    "Break the following question into {n} specific, atomic sub-questions that "
    "together cover all the facts needed to answer it. Each sub-question should "
    "target a single verifiable fact.\n"
    "Question: {question}\n"
    "List the {n} sub-questions, one per line, numbered:"
)