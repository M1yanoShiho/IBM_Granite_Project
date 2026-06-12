"""Scoring metrics for the NIAH (long-context "needle") diagnostic.

NIAH-specific scoring:

- **Accuracy** — did the model retrieve the injected needle (exact / fuzzy /
  LLM-judged match against the ground-truth answer)?

The RAG answer-quality metrics (**context precision**, **faithfulness**) now
live in ``eval/rag_metrics.py`` — the single source of truth shared by the
primary RAG evaluation and this NIAH diagnostic. They are re-exported below so
existing ``from eval.metrics import score_context_precision`` imports keep working.

Implementations are left as documented skeletons.
"""

from __future__ import annotations

# Re-exported from the canonical RAG-metrics home (see module docstring).
from eval.rag_metrics import score_context_precision, score_faithfulness  # noqa: F401


# --------------------------------------------------------------------------- #
# Accuracy
# --------------------------------------------------------------------------- #
def score_accuracy(prediction: str, ground_truth: str, fuzzy: bool = True) -> float:
    """Score whether the prediction contains the ground-truth answer.

    Parameters
    ----------
    prediction:
        The model's generated answer.
    ground_truth:
        The expected needle answer.
    fuzzy:
        If ``True``, allow case-insensitive substring / fuzzy matching;
        otherwise require an exact match.

    Returns
    -------
    float
        A score in ``[0.0, 1.0]`` (commonly 0.0 or 1.0 for exact matching).
    """
    raise NotImplementedError("TODO: implement (fuzzy) answer matching.")
