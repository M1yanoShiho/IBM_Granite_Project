# src/niah/counterfactual.py
"""Source A — counterfactual entity substitution (spec §5.1).

Given a needle passage and its answer entity, ask an LLM for a *type-consistent
but different* entity, then swap it in. The result reads coherently, shares
almost all tokens with the needle (adversarial to BM25/SPLADE) and embeds almost
identically (adversarial to dense), yet no longer answers the query.

Method lineage: entity-based knowledge conflicts (Longpre et al., 2021) /
Faithfulness-QA / WikiContradict.
"""
from __future__ import annotations

_WRONG_ENTITY_PROMPT = (
    "Replace the following answer with a DIFFERENT but same-type, equally plausible "
    "entity (same category: person/place/date/number/organisation). "
    "Reply with ONLY the replacement, nothing else.\n"
    "Answer: {answer}\n"
    "Replacement:"
)


def swap_entity(text: str, old: str, new: str) -> str:
    """Replace every exact-surface occurrence of ``old`` with ``new`` (pure)."""
    return text.replace(old, new)


def propose_wrong_entity(answer: str, llm) -> str:
    """Ask ``llm`` for a type-consistent but different entity than ``answer``."""
    reply = llm.generate(_WRONG_ENTITY_PROMPT.format(answer=answer)).strip()
    if not reply:
        raise ValueError("LLM returned an empty replacement entity.")
    if reply.lower() == answer.strip().lower():
        raise ValueError("LLM returned the same entity; not a counterfactual.")
    return reply


def make_counterfactual(needle_text: str, answer: str, llm) -> str:
    """Build a counterfactual distractor passage from a needle + its answer.

    Raises ``ValueError`` if ``answer`` does not occur in ``needle_text`` (the swap
    would be a no-op, yielding a verbatim copy of the needle — a false negative, not
    a distractor), so callers skip it like the empty/echo cases.
    """
    wrong = propose_wrong_entity(answer, llm)
    out = swap_entity(needle_text, answer, wrong)
    if out == needle_text:
        raise ValueError("answer not found in needle text; counterfactual would be a no-op.")
    return out
