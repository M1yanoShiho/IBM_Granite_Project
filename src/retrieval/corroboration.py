"""Corroboration scoring for the corroboration reranker (pure — no LLM, no IO).

Scores a candidate by how many OTHER retrieved passages (plus the model's own
parametric answer) independently answer the query with the SAME entity. A lone
factually-wrong passage (a Source-A counterfactual: a relevant-but-wrong entity) is
corroborated by nobody, so it is demoted even though it is semantically relevant --
the signal a relevance reranker cannot provide. Design spec:
docs/superpowers/specs/2026-07-05-corroboration-reranking-design.md.
"""
from __future__ import annotations

import re
from typing import List, Optional

# A too-weak extraction (e.g. a failed "the") must not corroborate everything. Non-numeric
# answers shorter than this (after normalisation) are ignored; short numbers are kept.
_MIN_ANSWER_LEN = 3
_STOPWORDS = {"the", "a", "an", "none", "n/a", "unknown", "it", "yes", "no"}


def normalize_answer(answer: str) -> str:
    """Lowercase, strip a leading article, and strip surrounding punctuation/space."""
    s = answer.strip().lower()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    return s.strip(" \t\n.,;:!?\"'()[]")


def is_valid_answer(answer: str) -> bool:
    """True if ``answer`` counts as an entity vote (not empty/stopword/too-short)."""
    s = normalize_answer(answer)
    if not s or s in _STOPWORDS:
        return False
    if len(s) < _MIN_ANSWER_LEN and not s.isdigit():
        return False
    return True


def corroboration_scores(
    answers: List[str], parametric: Optional[str] = None
) -> List[float]:
    """Raw count of OTHER sources that answer with the same entity, per candidate.

    ``answers[i]`` is candidate ``i``'s extracted answer (or a non-answer like "NONE").
    ``parametric`` is the model's own answer (one extra voter) or ``None``. Each valid
    ``answers[i]`` scores the number of *other* candidates with an equal normalised
    answer, +1 if ``parametric`` matches. Invalid answers score 0. Raw counts — the
    reranker min-max normalises them before blending with relevance.
    """
    norms = [normalize_answer(a) if is_valid_answer(a) else None for a in answers]
    p = normalize_answer(parametric) if (parametric and is_valid_answer(parametric)) else None
    scores: List[float] = []
    for i, ni in enumerate(norms):
        if ni is None:
            scores.append(0.0)
            continue
        votes = sum(1 for j, nj in enumerate(norms) if j != i and nj == ni)
        if p is not None and p == ni:
            votes += 1
        scores.append(float(votes))
    return scores
