"""Deterministic lenient answer equivalence (selector primitive).

Superset of exact `canonicalize_answer` equality: also treats one answer as equivalent to another
when, after a stronger normalization (leading function words + number-words), they are equal or one
is a contiguous token-subsequence of the other. Does NOT cover abbreviations or synonyms (that needs
NLI / Graph 2.0). Used by the gate's lenient clustering and by the eval-side wrong-answer re-classifier.
"""

import re
from collections.abc import Sequence

from evidence_rag.selector.answer_norm import canonicalize_answer

EQUAL_AFTER_NORM = "equal_after_norm"
GOLD_IN_EXTRACTED = "gold_in_extracted"
EXTRACTED_IN_GOLD = "extracted_in_gold"
DIFFERENT = "different"

_LEADING_FUNCTION_WORDS = frozenset(
    {"the", "a", "an", "in", "on", "at", "of", "to", "for", "by", "with", "from", "as"}
)
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "million": 1_000_000, "billion": 1_000_000_000,
}


def normalize_tokens(text: str) -> tuple[str, ...]:
    tokens = [
        str(_NUMBER_WORDS[token]) if token in _NUMBER_WORDS else token
        for token in re.findall(r"\w+", text.lower())
    ]
    start = 0
    while start < len(tokens) and tokens[start] in _LEADING_FUNCTION_WORDS:
        start += 1
    return tuple(tokens[start:])


def is_sublist(needle: Sequence[str], hay: Sequence[str]) -> bool:
    n = len(needle)
    if n == 0:
        return False
    return any(tuple(hay[i : i + n]) == tuple(needle) for i in range(len(hay) - n + 1))


def reclassify(extracted: str, gold: str) -> str:
    e = normalize_tokens(extracted)
    g = normalize_tokens(gold)
    if e and e == g:
        return EQUAL_AFTER_NORM
    if is_sublist(g, e):
        return GOLD_IN_EXTRACTED
    if is_sublist(e, g):
        return EXTRACTED_IN_GOLD
    return DIFFERENT


def lenient_equivalent(extracted: str, gold: str) -> bool:
    """True if the two answers match under the lenient, deterministic equivalence.

    Symmetric, and a superset of exact canonicalization (so lenient clustering never splits what
    exact clustering would merge). Containment is checked in both directions, so argument order
    does not change the result.
    """
    if canonicalize_answer(extracted) == canonicalize_answer(gold):
        return True
    return reclassify(extracted, gold) != DIFFERENT
