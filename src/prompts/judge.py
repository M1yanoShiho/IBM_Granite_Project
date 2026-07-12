"""LLM-as-judge prompts (the ``judge`` phase).

Three prompts that turn the LLM into a graded scorer rather than a content generator:

* :data:`RELIABILITY_PROMPT` — three-axis 0-2 reliability judge (direct support /
  condition coverage / evidence sufficiency) returning a single JSON object. Used
  inside the NIAH selector loop to score candidate passages; placed in the *judge*
  phase rather than ``niah`` per spec §3.1 because it is conceptually a grader, not
  a distractor fabricator.
* :data:`LABEL_AUDIT_PROMPT_A` / :data:`LABEL_AUDIT_PROMPT_B` — the two independent
  wording variants of a 0-4 utility-grade auditor used by
  ``eval/niah_label_audit.py`` for inter-rater (model-vs-model) agreement scoring.
  As with the ``niah.*_SEALED`` pairs, the two variants deliberately use independent
  wording to prevent theaudit's outcome from collapsing onto a single prompt-specific
  artefact; **do not merge** them.

See ``docs/superpowers/specs/2026-07-12-prompt-centralization-design.md``.
"""
from __future__ import annotations

RELIABILITY_PROMPT = (
    "Assess whether the passage itself is usable evidence for the question. Do not "
    "reward topical similarity or fluent writing. Score each field with an integer "
    "from 0 to 2: direct_support (0 none, 1 partial, 2 direct), condition_coverage "
    "(0 misses or violates stated conditions, 1 partial or no explicit condition, "
    "2 covers all stated conditions), and evidence_sufficiency (0 unusable, 1 needs "
    "other evidence, 2 sufficient by itself). Return only a JSON object with exactly "
    "these three keys.\n\n"
    "Example 1\nQuestion: Who wrote The Old Man and the Sea?\n"
    "Passage: Ernest Hemingway wrote The Old Man and the Sea.\n"
    "Candidate answer extracted from this passage: Ernest Hemingway\n"
    'JSON: {{"direct_support": 2, "condition_coverage": 1, '
    '"evidence_sufficiency": 2}}\n\n'
    "Example 2\nQuestion: Who wrote The Old Man and the Sea?\n"
    "Passage: To research the novel's authorship, consult catalogues and publication records.\n"
    "Candidate answer extracted from this passage: NONE\n"
    'JSON: {{"direct_support": 0, "condition_coverage": 1, '
    '"evidence_sufficiency": 0}}\n\n'
    "Example 3\nQuestion: Does the United Kingdom policy after 2025 allow the action?\n"
    "Passage: A United States policy from 2022 allowed the action.\n"
    "Candidate answer extracted from this passage: allowed\n"
    'JSON: {{"direct_support": 0, "condition_coverage": 0, '
    '"evidence_sufficiency": 0}}\n\n'
    "Question: {question}\nPassage: {passage}\n"
    "Candidate answer extracted from this passage: {candidate_answer}\nJSON:"
)

LABEL_AUDIT_PROMPT_A = (
    "Audit this passage against the question and reference answer. Assign one utility "
    "grade: 4 directly and correctly supports the full answer; 3 correct necessary "
    "partial support; 2 topical and plausible but does not answer; 1 unrelated noise; "
    "0 contradicts or supports a wrong answer. Return only GRADE: X.\n"
    "Question: {question}\nReference answer: {answers}\nPassage: {passage}\n"
)

LABEL_AUDIT_PROMPT_B = (
    "For an evidence-selection audit, score whether the text is safe and useful for "
    "answering the query. Use exactly one label: 4 complete valid evidence, 3 required "
    "partial evidence, 2 relevant non-answer, 1 irrelevant, 0 misleading or conflicting. "
    "Reply with the single digit.\nQuery: {question}\nGold answer(s): {answers}\n"
    "Candidate evidence: {passage}\n"
)