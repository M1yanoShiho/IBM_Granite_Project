# src/niah/filters.py
"""The two mandatory distractor filters (spec §5.1).

Filter 1 — false-negative / answerability: a distractor must NOT answer the query.
  * positive-anchor score threshold (NV-Retriever TopK-MarginPos): keep only
    candidates scored clearly below the true needle (``passes_margin``);
  * an explicit answerability judge (cross-encoder or LLM) as the semantic backstop
    (``answers_query``).
Filter 2 — dual-retriever hardness: keep only candidates ranked highly by BOTH a
  dense and a sparse retriever (``is_hard``).
"""
from __future__ import annotations

from typing import Dict

_ANSWERABILITY_PROMPT = (
    "Does the passage directly answer the question? Reply ONLY 'YES' or 'NO'.\n"
    "Question: {query}\n"
    "Passage: {passage}\n"
    "Answer:"
)


def passes_margin(cand_score: float, positive_score: float, margin: float) -> bool:
    """True if the candidate scores far enough below the gold to be a safe negative."""
    return cand_score < positive_score - margin


def answers_query(passage: str, query: str, judge) -> bool:
    """True if ``judge`` (any ``generate``-able) says the passage answers the query."""
    verdict = judge.generate(_ANSWERABILITY_PROMPT.format(query=query, passage=passage))
    return verdict.strip().upper().startswith("YES")


def is_hard(
    cand_id: str,
    dense_rank: Dict[str, int],
    sparse_rank: Dict[str, int],
    rank_threshold: int,
) -> bool:
    """True if ``cand_id`` is within top-``rank_threshold`` in BOTH rank maps."""
    d = dense_rank.get(cand_id)
    s = sparse_rank.get(cand_id)
    if d is None or s is None:
        return False
    return d <= rank_threshold and s <= rank_threshold


def keep_distractor(
    *,
    cand_score: float,
    positive_score: float,
    margin: float,
    cand_text: str,
    query: str,
    judge,
    cand_id: str,
    dense_rank: Dict[str, int],
    sparse_rank: Dict[str, int],
    rank_threshold: int,
) -> bool:
    """Apply Filter 1 (margin AND not-answering) then Filter 2 (hard in both)."""
    if not passes_margin(cand_score, positive_score, margin):
        return False
    if answers_query(cand_text, query, judge):
        return False
    return is_hard(cand_id, dense_rank, sparse_rank, rank_threshold)
