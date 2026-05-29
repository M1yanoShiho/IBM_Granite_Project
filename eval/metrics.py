"""Scoring metrics for the NIAH evaluation.

Three complementary metrics:

- **Accuracy** — did the model retrieve the injected needle (exact / fuzzy /
  LLM-judged match against the ground-truth answer)?
- **Context precision** — of the context supplied to the model (relevant for the
  RAG baseline), how much was actually relevant to answering the question?
- **Faithfulness** — is the generated answer grounded in the provided context,
  rather than hallucinated?

Context precision and faithfulness can be computed with the
`RAGAS <https://docs.ragas.io/>`_ framework; thin wrappers are stubbed below.

Implementations are left as documented skeletons.
"""

from __future__ import annotations

from typing import List


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


# --------------------------------------------------------------------------- #
# Context precision
# --------------------------------------------------------------------------- #
def score_context_precision(
    question: str,
    retrieved_chunks: List[str],
    ground_truth: str,
) -> float:
    """Estimate how much of the retrieved context was relevant.

    Parameters
    ----------
    question:
        The probe question.
    retrieved_chunks:
        Context chunks supplied to the model.
    ground_truth:
        The expected answer.

    Returns
    -------
    float
        A precision score in ``[0.0, 1.0]``.
    """
    raise NotImplementedError(
        "TODO: implement context precision (e.g. via RAGAS)."
    )


# --------------------------------------------------------------------------- #
# Faithfulness
# --------------------------------------------------------------------------- #
def score_faithfulness(
    answer: str,
    context: str,
) -> float:
    """Estimate how grounded ``answer`` is in ``context``.

    Parameters
    ----------
    answer:
        The model's generated answer.
    context:
        The context the answer should be grounded in.

    Returns
    -------
    float
        A faithfulness score in ``[0.0, 1.0]`` (higher = less hallucination).
    """
    raise NotImplementedError(
        "TODO: implement faithfulness scoring (e.g. via RAGAS)."
    )
