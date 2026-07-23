"""Re-classify probe `wrong` cases into matching-artifact tiers (systematic-debugging).

The needle probe marks an extraction WRONG whenever exact-string canonicalization differs from
gold. Many of those are the model being right in different words. This splits WRONG into
deterministic, recoverable tiers -- equal after a stronger normalization (leading function
words + number-words), or gold/extracted being a contiguous token-subsequence of the other --
versus genuinely DIFFERENT (real QA error or synonym needing semantic matching). No LLM.
`recoverable_rate` sizes how much needle-gold-recovery a better answer-matcher buys for free.
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

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
    """True if the two answers match under a lenient, deterministic equivalence.

    Superset of exact canonicalization: exact-equal, or one is a contiguous token-subsequence
    of the other after leading-function-word + number-word normalization. Does NOT cover
    abbreviations (USA/United States) or synonyms (needs NLI / Graph 2.0) -- those stay a
    reported residual. Reusable by the selector-side answer-matcher and by lenient re-scoring.
    """
    if canonicalize_answer(extracted) == canonicalize_answer(gold):
        return True
    return reclassify(extracted, gold) != DIFFERENT


@dataclass(frozen=True)
class ReclassSummary:
    n: int
    equal_after_norm: int
    gold_in_extracted: int
    extracted_in_gold: int
    different: int
    recoverable: int
    recoverable_rate: float | None


def summarize_reclass(rows: Iterable[Mapping[str, object]]) -> ReclassSummary:
    labels = [reclassify(str(row["extracted"]), str(row["gold_value"])) for row in rows]
    n = len(labels)
    equal = labels.count(EQUAL_AFTER_NORM)
    gold_in = labels.count(GOLD_IN_EXTRACTED)
    extracted_in = labels.count(EXTRACTED_IN_GOLD)
    different = labels.count(DIFFERENT)
    recoverable = equal + gold_in + extracted_in
    return ReclassSummary(
        n=n,
        equal_after_norm=equal,
        gold_in_extracted=gold_in,
        extracted_in_gold=extracted_in,
        different=different,
        recoverable=recoverable,
        recoverable_rate=(recoverable / n) if n else None,
    )
