"""Answer-quality metrics for the RAG evaluation (primary, A+B headline).

Where ``eval/ir_metrics.py`` scores the *retrieval* layer against qrels, this
module scores the *generation* layer of the retrieve-then-generate system:

- **Answer correctness** — does the generated answer match the reference answer
  (exact / fuzzy / LLM-judged)?
- **Context precision** — of the chunks supplied to the model, how many were
  actually relevant to answering the question?
- **Faithfulness** — is the answer grounded in the retrieved context rather than
  hallucinated?

Context precision and faithfulness can be computed with the
`RAGAS <https://docs.ragas.io/>`_ framework; thin wrappers are stubbed here.

These metrics are shared: the long-context NIAH diagnostic (``eval/metrics.py``)
re-exports ``score_context_precision`` / ``score_faithfulness`` from here so
there is a single source of truth.

Implementations are left as documented skeletons.
"""

from __future__ import annotations

from typing import Dict, List


def score_answer_correctness(
    prediction: str, reference: str, fuzzy: bool = True
) -> float:
    """Score whether ``prediction`` matches the reference answer.

    Parameters
    ----------
    prediction:
        The model's generated answer.
    reference:
        The gold answer from the QA benchmark.
    fuzzy:
        If ``True``, allow case-insensitive substring / fuzzy matching (or an
        LLM judge); otherwise require an exact match.

    Returns
    -------
    float
        A score in ``[0.0, 1.0]``.
    """
    raise NotImplementedError("TODO: implement (fuzzy / LLM-judged) answer matching.")


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
    raise NotImplementedError("TODO: implement context precision (e.g. via RAGAS).")


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
    raise NotImplementedError("TODO: implement faithfulness scoring (e.g. via RAGAS).")


def evaluate_rag(
    predictions: Dict[str, str],
    references: Dict[str, str],
    contexts: Dict[str, List[str]],
) -> Dict[str, float]:
    """Compute the full RAG metric suite over a set of answered questions.

    Parameters
    ----------
    predictions:
        ``{question_id: generated_answer}``.
    references:
        ``{question_id: gold_answer}``.
    contexts:
        ``{question_id: [chunk_text, ...]}`` — the context shown to the model,
        for context-precision and faithfulness.

    Returns
    -------
    dict
        ``{metric: mean_value}`` across all questions (answer correctness,
        context precision, faithfulness).
    """
    raise NotImplementedError(
        "TODO: aggregate answer-correctness + context-precision + faithfulness "
        "over all questions into one results dict."
    )
