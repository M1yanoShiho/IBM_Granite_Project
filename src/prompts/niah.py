"""NIAH task-construction + distractor prompts (the ``niah`` phase).

Five prompts used to fabricate NIAH distractors and filter them:

* :data:`WRONG_ENTITY_PROMPT` — Source-A counterfactual entity substitution (base
  wording, authoritative — see spec §3.3 group 2).
* :data:`WRONG_ENTITY_PROMPT_SEALED` — the *sealed* variant used for the test-split
  construction phase to prevent the model having seen the same wording during
  training-time construction. **Independent constant by design** — the spec §3.2
  forbids parameterising sealed variants into the base prompt.
* :data:`GENERATIVE_DISTRACTOR_PROMPT` — Source-B LLM-generated plausible non-answer.
* :data:`ANSWERABILITY_PROMPT` — Filter 1's semantic backstop: judge whether a
  candidate passage answers the query.
* :data:`NON_ANSWER_PROMPT` / :data:`NON_ANSWER_PROMPT_SEALED` — generative
  non-answer distractor construction (train / test wording pair, same sealed
  invariant as the wrong-entity pair).

Note: the RELIABILITY judge prompt and the LABEL_AUDIT prompts are conceptually
*judges*, so they live in :mod:`src.prompts.judge` per spec §3.1 — not here.

See ``docs/superpowers/specs/2026-07-12-prompt-centralization-design.md``.
"""
from __future__ import annotations

WRONG_ENTITY_PROMPT = (
    "Replace the following answer with a DIFFERENT but same-type, equally plausible "
    "entity (same category: person/place/date/number/organisation). "
    "Reply with ONLY the replacement, nothing else.\n"
    "Answer: {answer}\n"
    "Replacement:"
)

WRONG_ENTITY_PROMPT_SEALED = (
    "For a factual robustness test, substitute the answer with one plausible entity "
    "of the same type. Give only that different entity.\nOriginal: {answer}\nSubstitute:"
)

GENERATIVE_DISTRACTOR_PROMPT = (
    "Write a short, fluent passage that is on the same topic as the query and "
    "shares its key terms and entities, BUT does not answer the query (it should "
    "discuss adjacent facts only). Reply with ONLY the passage.\n"
    "Query: {query}\n"
    "A real answer passage (for style reference, do NOT reuse its answer):\n{needle}\n"
    "Passage:"
)

ANSWERABILITY_PROMPT = (
    "Does the passage directly answer the question? Reply ONLY 'YES' or 'NO'.\n"
    "Question: {query}\n"
    "Passage: {passage}\n"
    "Answer:"
)

NON_ANSWER_PROMPT = (
    "Write two short, fluent sentences about how someone should research the topic "
    "in the question. Refer to the subject only in general terms. Do not give or imply "
    "the answer. Do not include names, dates, numbers, places, titles, organizations, "
    "quoted phrases, or causal facts that could answer the question.\n"
    "Question: {question}\nNon-answering research note:"
)

NON_ANSWER_PROMPT_SEALED = (
    "Draft two concise sentences explaining that the query needs source verification "
    "and what kind of record should be consulted. Keep the subject generic. Exclude "
    "every concrete person, date, number, location, work title, institution, and factual "
    "conclusion that might resolve the query.\n"
    "Query: {question}\nVerification note:"
)