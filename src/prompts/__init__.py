"""Central prompt registry — the single source of truth for every LLM prompt
template in the project.

This package consolidates the 22 prompts previously scattered across 8 files
(``src/rag_pipeline.py`` / ``src/retrieval/{reranker,query_transform}.py`` /
``src/niah/{counterfactual,generative,filters}.py`` /
``eval/{niah_selector_pilot,niah_label_audit}.py``) with 4 confirmed duplicate
groups (see ``docs/superpowers/specs/2026-07-12-prompt-centralization-design.md``
§3.3 for the consolidation decisions).

Layout (per spec §3.1, by LLM-call phase):

* :mod:`src.prompts.rag`        — 5 prompts: generation line
* :mod:`src.prompts.retrieval`  — 5 prompts: augmentation / post-processing
* :mod:`src.prompts.niah`       — 6 prompts: distractor fabrication + filtering
* :mod:`src.prompts.judge`      — 3 prompts: LLM-as-judge grading

Total = 19 canonical prompts + 1 backwards-compat alias (:data:`Q2D_PROMPT`).

Public surface:

* Each individual prompt constant (e.g. :data:`DEFAULT_RAG_PROMPT`) for direct import.
* :data:`PROMPT_REGISTRY` — ``{dotted_key: prompt_string}`` map for listing / audit
  / report appendix. Keys follow ``<phase>.<purpose>`` (lowercase, dot-separated);
  adding a prompt requires adding both a constant and a registry entry, which is the
  minimum friction that prevents silent accretion.

Note: :data:`Q2D_PROMPT` is an **alias** ``Q2D_PROMPT = HYDE_PROMPT`` (identity, not
just textual equality) preserved so that ``eval/ramdocs_selector.py`` and
``eval/financebench_selector.py`` continue to import via
``from eval.niah_selector_pilot import Q2D_PROMPT`` unchanged. It is NOT a registry
key — the canonical identity is ``retrieval.hyde``. See spec §3.2.
"""
from __future__ import annotations

from src.prompts.judge import (
    LABEL_AUDIT_PROMPT_A,
    LABEL_AUDIT_PROMPT_B,
    RELIABILITY_PROMPT,
)
from src.prompts.niah import (
    ANSWERABILITY_PROMPT,
    GENERATIVE_DISTRACTOR_PROMPT,
    NON_ANSWER_PROMPT,
    NON_ANSWER_PROMPT_SEALED,
    WRONG_ENTITY_PROMPT,
    WRONG_ENTITY_PROMPT_SEALED,
)
from src.prompts.rag import (
    CITATION_RAG_PROMPT,
    CONSOLIDATE_PROMPT,
    DEFAULT_RAG_PROMPT,
    ELICIT_PROMPT,
    FINALIZE_PROMPT,
)
from src.prompts.retrieval import (
    DECOMPOSE_PROMPT,
    EXTRACT_PROMPT,
    HYDE_PROMPT,
    LISTWISE_RANK_PROMPT,
    PARAMETRIC_PROMPT,
)

# Backwards-compat identity alias. ``HYDE_PROMPT`` and ``Q2D_PROMPT`` are the same
# prompt (the project originally had two byte-identical copies in different files;
# spec §3.3 group 1). The alias shares identity (``is``-equal) so tests can enforce
# that the two never drift apart again; the registry key is ``retrieval.hyde`` only.
Q2D_PROMPT = HYDE_PROMPT

PROMPT_REGISTRY: dict[str, str] = {
    "rag.default": DEFAULT_RAG_PROMPT,
    "rag.citation": CITATION_RAG_PROMPT,
    "rag.elicit": ELICIT_PROMPT,
    "rag.consolidate": CONSOLIDATE_PROMPT,
    "rag.finalize": FINALIZE_PROMPT,
    "retrieval.hyde": HYDE_PROMPT,
    "retrieval.decompose": DECOMPOSE_PROMPT,
    "retrieval.listwise_rank": LISTWISE_RANK_PROMPT,
    "retrieval.extract": EXTRACT_PROMPT,
    "retrieval.parametric": PARAMETRIC_PROMPT,
    "niah.wrong_entity": WRONG_ENTITY_PROMPT,
    "niah.wrong_entity_sealed": WRONG_ENTITY_PROMPT_SEALED,
    "niah.generative": GENERATIVE_DISTRACTOR_PROMPT,
    "niah.answerability": ANSWERABILITY_PROMPT,
    "niah.non_answer": NON_ANSWER_PROMPT,
    "niah.non_answer_sealed": NON_ANSWER_PROMPT_SEALED,
    "judge.reliability": RELIABILITY_PROMPT,
    "judge.label_audit_a": LABEL_AUDIT_PROMPT_A,
    "judge.label_audit_b": LABEL_AUDIT_PROMPT_B,
}

__all__ = [
    "PROMPT_REGISTRY",
    # rag
    "DEFAULT_RAG_PROMPT",
    "CITATION_RAG_PROMPT",
    "ELICIT_PROMPT",
    "CONSOLIDATE_PROMPT",
    "FINALIZE_PROMPT",
    # retrieval
    "HYDE_PROMPT",
    "Q2D_PROMPT",
    "DECOMPOSE_PROMPT",
    "EXTRACT_PROMPT",
    "PARAMETRIC_PROMPT",
    "LISTWISE_RANK_PROMPT",
    # niah
    "WRONG_ENTITY_PROMPT",
    "WRONG_ENTITY_PROMPT_SEALED",
    "GENERATIVE_DISTRACTOR_PROMPT",
    "ANSWERABILITY_PROMPT",
    "NON_ANSWER_PROMPT",
    "NON_ANSWER_PROMPT_SEALED",
    # judge
    "RELIABILITY_PROMPT",
    "LABEL_AUDIT_PROMPT_A",
    "LABEL_AUDIT_PROMPT_B",
]