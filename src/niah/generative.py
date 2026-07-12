# src/niah/generative.py
"""Source B — LLM-generated plausible non-answers (spec §5.1).

Multi-attribute self-reflection prompting (SyNeg, 2024) + MCQ-distractor design:
write an on-topic passage that shares the query's terms/entities and is highly
plausible, but does NOT answer the query. Filtered downstream by the same
answerability check (src/niah/filters.py) as Source A.
"""
from __future__ import annotations

from src.prompts.niah import GENERATIVE_DISTRACTOR_PROMPT


def make_generative_distractor(query: str, needle_text: str, llm) -> str:
    """Generate a plausible, on-topic passage that does not answer ``query``."""
    out = llm.generate(GENERATIVE_DISTRACTOR_PROMPT.format(query=query, needle=needle_text)).strip()
    if not out:
        raise ValueError("LLM returned an empty distractor passage.")
    return out
