"""Invariant tests for the central prompt registry.

Validates the invariants declared in
``docs/superpowers/specs/2026-07-12-prompt-centralization-design.md``:

* Every registry entry is a non-empty ``str``.
* Every registry key is lowercase-dotted ``"<phase>.<purpose>"``.
* Every prompt templates cleanly with its declared placeholders.
* Sealed variants are *independent strings* from their bases (regression guard: a
  well-intentioned future "tidy" PR must not collapse a sealed variant by accident).
* :data:`Q2D_PROMPT` shares identity with :data:`HYDE_PROMPT` (the alias must not
  drift, even textually).
* The citation prompt emits the ``Answer:`` / ``Evidence:`` schema the parser
  (:func:`src.rag_pipeline.parse_citation_output`) expects.
"""
from __future__ import annotations

import re

from src.prompts import (
    CITATION_RAG_PROMPT,
    DEFAULT_RAG_PROMPT,
    HYDE_PROMPT,
    NON_ANSWER_PROMPT,
    NON_ANSWER_PROMPT_SEALED,
    PROMPT_REGISTRY,
    Q2D_PROMPT,
    WRONG_ENTITY_PROMPT,
    WRONG_ENTITY_PROMPT_SEALED,
)


def test_registry_is_non_empty():
    assert len(PROMPT_REGISTRY) >= 18


def test_every_value_is_a_non_empty_string():
    for key, prompt in PROMPT_REGISTRY.items():
        assert isinstance(prompt, str), f"{key}: not a str"
        assert prompt.strip(), f"{key}: empty prompt"


_KEY_RE = re.compile(r"^[a-z_]+\.[a-z_]+$")


def test_registry_keys_are_dotted_phase_purpose():
    for key in PROMPT_REGISTRY:
        assert _KEY_RE.match(key), f"bad key shape: {key!r}"


def test_sealed_variants_differ_from_base():
    # Regression guard: a future "tidy" refactor must not collapse sealed variants
    # into the base prompt by accident (spec §3.2 forbids parameterising them).
    assert WRONG_ENTITY_PROMPT_SEALED != WRONG_ENTITY_PROMPT
    assert NON_ANSWER_PROMPT_SEALED != NON_ANSWER_PROMPT


def test_q2d_alias_shares_identity_with_hyde():
    # The alias must share identity, not merely equal text.
    assert Q2D_PROMPT is HYDE_PROMPT


_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _placeholders(prompt: str) -> list[str]:
    return _PLACEHOLDER_RE.findall(prompt)


def test_every_prompt_formats_with_its_placeholders():
    # Catches stray ``{}`` / ``{1}`` that would break ``.format(**)``.
    for key, prompt in PROMPT_REGISTRY.items():
        names = _placeholders(prompt)
        kwargs = {name: "X" for name in names}
        try:
            prompt.format(**kwargs)
        except (KeyError, IndexError) as exc:
            raise AssertionError(f"{key}: failed to format placeholders {names}: {exc}")


def test_citation_prompt_mentions_evidence_and_answer_tags():
    # The parser (src.rag_pipeline.parse_citation_output) keys on these marks;
    # if a future edit drops them the citation path silently falls back to
    # post-hoc attribution for every query.
    assert "Evidence:" in CITATION_RAG_PROMPT
    assert "Answer:" in CITATION_RAG_PROMPT


def test_default_prompt_takes_context_and_question():
    # The vanilla pipeline's ``_build_prompt`` formats these two fields.
    names = _placeholders(DEFAULT_RAG_PROMPT)
    assert set(names) == {"context", "question"}, (
        f"DEFAULT_RAG_PROMPT placeholders changed: {names}"
    )