"""Prompt probe for the twin missed-conflict (systematic-debugging Phase 3).

The gate is blind to ~39% of planted conflicts because the extractor gives the SAME answer to the
near-identical needle/counterfactual twins (they differ by one swapped value). This probe tests
whether a more targeted extraction prompt separates the twins — pulling the SWAPPED value (gold from
the needle, the injected replacement from the counterfactual) instead of a shared non-swapped entity.
If a prompt lowers missed-conflict, that is Graph 2.0's conflict-detection benefit for a prompt change;
if not, the NLI graph is justified. Pure classification here; the CLI runs the LLM.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from evidence_rag.selector.answer_norm import canonicalize_answer, is_valid_answer
from evidence_rag.selector.extraction import EXTRACT_PROMPT

VERBATIM_PROMPT = (
    "Read the passage and copy, word for word, the exact phrase from the passage that answers "
    "the question. Output only that phrase, nothing else. If the passage does not answer it, "
    "reply NONE.\nQuestion: {question}\nPassage: {passage}\nAnswer:"
)
ATTRIBUTE_PROMPT = (
    "The question asks for one specific value. Find that value in the passage and output it "
    "exactly as it appears in the passage, nothing else. If it is not stated, reply NONE.\n"
    "Question: {question}\nPassage: {passage}\nAnswer:"
)
PROMPTS: dict[str, str] = {
    "baseline": EXTRACT_PROMPT,
    "verbatim": VERBATIM_PROMPT,
    "attribute": ATTRIBUTE_PROMPT,
}

# Decoupled two-stage extraction: separate the reasoning (what value does the question want?) from
# the mechanical span-copy, so the extractor locks onto the swapped value in each twin rather than a
# shared entity. Stage A runs once per query (passage-independent); Stage B once per passage.
STAGE_A_PROMPT = (
    "What kind of value does this question ask for? Answer in a few words naming the target "
    "(for example: a person's name, a year, a city, a number).\nQuestion: {question}\nTarget:"
)
STAGE_B_PROMPT = (
    "From the passage below, output the exact {target} that the passage states, copied verbatim "
    "from the passage and nothing else. If the passage does not state it, reply NONE.\n"
    "Passage: {passage}\nAnswer:"
)


@dataclass(frozen=True)
class PairOutcome:
    missed_conflict: bool  # needle and counterfactual collapsed to the same answer
    needle_gold: bool  # needle recovered the gold value
    cf_replacement: bool  # counterfactual recovered the injected replacement value


def classify_pair(
    needle_answer: str,
    cf_answer: str,
    *,
    gold_value: str,
    replacement_value: str,
) -> PairOutcome:
    needle = canonicalize_answer(needle_answer) if is_valid_answer(needle_answer) else None
    cf = canonicalize_answer(cf_answer) if is_valid_answer(cf_answer) else None
    return PairOutcome(
        missed_conflict=needle is not None and needle == cf,
        needle_gold=needle is not None and needle == canonicalize_answer(gold_value),
        cf_replacement=cf is not None and cf == canonicalize_answer(replacement_value),
    )


@dataclass(frozen=True)
class PromptSummary:
    prompt: str
    n: int
    missed_conflict: int
    needle_gold: int
    cf_replacement: int
    missed_conflict_rate: float | None
    needle_gold_rate: float | None
    cf_replacement_rate: float | None


def summarize_prompt(prompt: str, outcomes: Iterable[PairOutcome]) -> PromptSummary:
    items = tuple(outcomes)
    n = len(items)
    missed = sum(1 for outcome in items if outcome.missed_conflict)
    needle_gold = sum(1 for outcome in items if outcome.needle_gold)
    cf_replacement = sum(1 for outcome in items if outcome.cf_replacement)
    return PromptSummary(
        prompt=prompt,
        n=n,
        missed_conflict=missed,
        needle_gold=needle_gold,
        cf_replacement=cf_replacement,
        missed_conflict_rate=(missed / n) if n else None,
        needle_gold_rate=(needle_gold / n) if n else None,
        cf_replacement_rate=(cf_replacement / n) if n else None,
    )
