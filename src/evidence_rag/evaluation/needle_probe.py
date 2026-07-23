"""Needle-extraction failure-mode probe (systematic-debugging Phase 3).

Classifies what the extractor did with the needle passage: recovered the gold answer,
produced a WRONG entity, or abstained (NONE). Split by visibility, this decides the fix for
the ~50% needle-gold-recovery: on VISIBLE needles, mostly-NONE => the extractor abstains on a
readable answer (prompt problem, free to fix, and a bigger model likely won't help); mostly-WRONG
=> genuine QA error (where an 8B extractor could plausibly help). No LLM here; the CLI feeds the
answers the extractor produced.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from evidence_rag.selector.answer_norm import canonicalize_answer, is_valid_answer

RECOVERED = "recovered"
WRONG = "wrong"
NONE = "none"


def classify_extraction(extracted_answers: Sequence[str], gold_value: str) -> str:
    gold = canonicalize_answer(gold_value)
    valid = [canonicalize_answer(answer) for answer in extracted_answers if is_valid_answer(answer)]
    if any(answer == gold for answer in valid):
        return RECOVERED
    if valid:
        return WRONG
    return NONE


@dataclass(frozen=True)
class ProbeOutcome:
    query_id: str
    visible: bool
    outcome: str
    extracted: str
    gold_value: str


@dataclass(frozen=True)
class ProbeSummary:
    n: int
    recovered: int
    wrong: int
    none: int
    n_visible: int
    visible_recovered: int
    visible_wrong: int
    visible_none: int


def summarize_probe(outcomes: Iterable[ProbeOutcome]) -> ProbeSummary:
    items = tuple(outcomes)
    visible = tuple(item for item in items if item.visible)
    return ProbeSummary(
        n=len(items),
        recovered=sum(1 for item in items if item.outcome == RECOVERED),
        wrong=sum(1 for item in items if item.outcome == WRONG),
        none=sum(1 for item in items if item.outcome == NONE),
        n_visible=len(visible),
        visible_recovered=sum(1 for item in visible if item.outcome == RECOVERED),
        visible_wrong=sum(1 for item in visible if item.outcome == WRONG),
        visible_none=sum(1 for item in visible if item.outcome == NONE),
    )
