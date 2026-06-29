"""Lightweight, model-free text utilities shared across the RAG layers.

Deliberately simple (lower-case, unicode word tokens, stop-word removal,
sentence splitting) so the heuristics that depend on them run CPU-only without
any model. Kept in one place so the callers — ``eval.rag_metrics`` (faithfulness)
and ``src.explainability.citations`` (attribution) — cannot drift apart, which
they previously did (two copies of these helpers with diverging sentence
splitting).
"""

from __future__ import annotations

import re
from typing import List, Sequence

# Function words filtered out of token-overlap comparisons so they don't inflate
# similarity between otherwise-unrelated texts. NOTE: this list is for *overlap*
# heuristics (faithfulness, citation attribution); answer-correctness scoring
# uses SQuAD normalisation instead (see ``eval.rag_metrics.normalize_answer``)
# and must NOT strip content words like "no"/"not".
STOP_WORDS: frozenset = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "nor", "not", "so", "yet", "both", "either", "neither", "each",
    "every", "all", "any", "few", "more", "most", "some", "no", "only",
    "other", "same", "than", "too", "very", "just", "about", "also",
    "it", "its", "that", "this", "these", "those", "he", "she", "they",
    "we", "you", "i", "me", "him", "her", "us", "them", "my", "your",
    "his", "our", "their", "there",
})

# ``\w`` (unicode) keeps accented letters — "Beyoncé" tokenises to "beyoncé"
# rather than being split/dropped the way ``[a-z0-9]+`` did.
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str) -> List[str]:
    """Lower-case, extract unicode word tokens, drop stop words."""
    return [t for t in _WORD_RE.findall(text.lower()) if t not in STOP_WORDS]


def jaccard(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> float:
    """Jaccard similarity between two token sequences (empty/empty -> 1.0)."""
    set_a, set_b = set(tokens_a), set(tokens_b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def split_sentences(text: str) -> List[str]:
    """Split *text* into sentence spans on ``. ! ?``.

    The delimiter is kept attached to its sentence (split on the whitespace that
    *follows* terminal punctuation), so each returned span is verbatim text.
    """
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
