"""Answer-quality metrics for the RAG evaluation (primary, A+B headline).

Where ``eval/ir_metrics.py`` scores the *retrieval* layer against qrels, this
module scores the *generation* layer of the retrieve-then-generate system. The
definitions follow the established open-domain QA / IR conventions so the numbers
are comparable to published work rather than a bespoke heuristic:

- **Answer correctness** — SQuAD-style normalised **Exact Match** and **token-F1**
  (Rajpurkar et al., 2016), taking the best score over all acceptable gold
  answers (so NQ-style multi-answer questions are scored fairly).
- **Context precision** — **precision@k against qrels**: of the chunks supplied
  to the model, the fraction whose document is judged relevant. This reuses the
  same ground-truth relevance the retrieval benchmark uses, instead of a token
  heuristic that saturates near 1.0.
- **Faithfulness** — a model-free grounding proxy: the fraction of the answer's
  content tokens that appear in the retrieved context.

The answer/faithfulness scores are model-free so they run CPU-only without an
LLM judge; they can later be upgraded to an LLM-as-judge / NLI model without
changing these signatures. ``score_faithfulness`` (and historically
``score_context_precision``) are re-exported by the NIAH diagnostic
(``eval/metrics.py``) so there is a single source of truth.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence, Union

from src.text_utils import tokenize

# Type alias: an answer may have one or several acceptable gold strings.
References = Union[str, Sequence[str]]

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = set(string.punctuation)


def normalize_answer(text: str) -> str:
    """SQuAD canonical answer normalisation.

    Lower-cases, removes punctuation, drops the articles ``a/an/the``, and
    collapses whitespace — the standard pre-processing for QA Exact Match / F1.
    Unicode letters are preserved (so "Beyoncé" stays intact). Crucially it does
    *not* strip a broad stop-word list: removing words like "no"/"not" would flip
    an answer's meaning.
    """
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def _as_reference_list(references: References) -> List[str]:
    """Coerce a single gold string or a list of acceptable golds into a list."""
    if isinstance(references, str):
        return [references]
    return list(references)


def _exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def _token_f1(prediction: str, reference: str) -> float:
    """SQuAD token-level F1 between a prediction and a single reference."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer_correctness(
    prediction: str, references: References, fuzzy: bool = True
) -> float:
    """Score ``prediction`` against one or more acceptable gold answers.

    Parameters
    ----------
    prediction:
        The model's generated answer.
    references:
        The gold answer, or a list of acceptable gold answers (NQ-style). The
        best score over all golds is returned.
    fuzzy:
        ``True`` (default) returns the SQuAD **token-F1** (partial credit);
        ``False`` returns strict **Exact Match** after normalisation.

    Returns
    -------
    float
        A score in ``[0.0, 1.0]``.
    """
    refs = _as_reference_list(references)
    if not refs:
        return 0.0
    score_fn = _token_f1 if fuzzy else _exact_match
    return max(score_fn(prediction, ref) for ref in refs)


def _cover_match(prediction: str, reference: str) -> float:
    """1.0 if the normalised reference appears as a contiguous token span of the
    normalised prediction."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not ref_tokens:
        return float(not pred_tokens)
    if len(ref_tokens) > len(pred_tokens):
        return 0.0
    span = len(ref_tokens)
    return float(
        any(
            pred_tokens[i : i + span] == ref_tokens
            for i in range(len(pred_tokens) - span + 1)
        )
    )


def score_answer_cover(prediction: str, references: References) -> float:
    """Cover-EM / answer-recall: 1.0 if any gold answer is *contained* in the
    prediction (as a contiguous token span, after SQuAD normalisation).

    This is the de-facto metric for **generative** open-domain QA (DPR/RAG): a
    correct answer phrased as a full sentence ("The capital is Paris.") scores 0
    on Exact Match and low on token-F1 — both penalise verbosity — but should
    count as correct. Token-level (not character-substring) containment avoids
    false positives like "cat" matching "category". Returns the best over all
    acceptable gold answers.
    """
    refs = _as_reference_list(references)
    if not refs:
        return 0.0
    return max(_cover_match(prediction, ref) for ref in refs)


def score_answer_claims(prediction: str, references: References, judge=None) -> float:
    """Claim-level answer correctness via entailment.

    ``judge`` is a callable ``(premise, hypothesis) -> bool`` (an NLI model or an
    LLM-as-judge): does the prediction (premise) entail a gold answer (hypothesis)?
    Returns the best over all acceptable golds.

    When ``judge is None`` this falls back to :func:`score_answer_cover`, so the
    default path stays model-free / CPU-only and the metric suite is unchanged
    unless a judge is supplied. Unlike token EM/F1/cover, a judge credits
    semantically-correct answers phrased with a different surface form.
    """
    refs = _as_reference_list(references)
    if not refs:
        return 0.0
    if judge is None:
        return score_answer_cover(prediction, references)
    return max(float(bool(judge(prediction, ref))) for ref in refs)


def score_context_precision(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Union[Iterable[str], Mapping[str, int]],
) -> float:
    """Precision@k of the retrieved context against the gold relevance set.

    Parameters
    ----------
    retrieved_doc_ids:
        Document ids of the chunks supplied to the model, in rank order.
        Duplicate ids (several chunks from the same document) are counted once,
        so the score is precision over *distinct retrieved documents*.
    relevant_doc_ids:
        The gold-relevant document ids. A ``{doc_id: relevance}`` mapping (a row
        of the benchmark qrels) is also accepted — entries with relevance > 0 are
        treated as relevant.

    Returns
    -------
    float
        ``relevant ∩ retrieved`` over ``retrieved`` in ``[0.0, 1.0]``; ``0.0``
        when nothing was retrieved.
    """
    if isinstance(relevant_doc_ids, Mapping):
        relevant = {doc_id for doc_id, rel in relevant_doc_ids.items() if rel > 0}
    else:
        relevant = set(relevant_doc_ids)

    distinct: List[str] = []
    for doc_id in retrieved_doc_ids:
        if doc_id not in distinct:
            distinct.append(doc_id)
    if not distinct:
        return 0.0

    hits = sum(1 for doc_id in distinct if doc_id in relevant)
    return hits / len(distinct)


def score_faithfulness(answer: str, context: str) -> float:
    """Fraction of the answer's content tokens that appear in the context.

    A model-free grounding proxy: higher means more of the answer is supported by
    the retrieved context (less hallucination). An empty answer is trivially
    faithful (``1.0`` — nothing to ground); a non-empty answer with empty context
    is ``0.0``.

    Parameters
    ----------
    answer:
        The model's generated answer.
    context:
        The retrieved context the answer should be grounded in.
    """
    if not answer.strip():
        return 1.0
    if not context.strip():
        return 0.0

    answer_tokens = tokenize(answer)
    if not answer_tokens:  # answer is all function words: no content to ground
        return 1.0
    context_tokens = set(tokenize(context))
    covered = sum(1 for token in answer_tokens if token in context_tokens)
    return covered / len(answer_tokens)


METRIC_NAMES = (
    "answer_em",
    "answer_f1",
    "answer_cover",
    "context_precision",
    "faithfulness",
)


def score_rag_per_query(
    predictions: Dict[str, str],
    references: Dict[str, References],
    contexts: Dict[str, List[str]],
    retrieved_doc_ids: Dict[str, Sequence[str]],
    qrels: Dict[str, Mapping[str, int]],
    judge=None,
) -> Dict[str, Dict[str, float]]:
    """Per-question metric dict ``{qid: {metric: value}}``.

    Keyed by every question present in both ``predictions`` and ``references``.
    This is the input to paired significance testing across retrievers
    (``eval.significance``): export one retriever's column per run, then compare.

    ``judge`` (optional): when supplied, each question also gets an
    ``answer_claims`` entry (:func:`score_answer_claims`); omitted otherwise, so the
    default keys are unchanged.
    """
    per_query: Dict[str, Dict[str, float]] = {}
    for qid in set(predictions) & set(references):
        pred = predictions[qid]
        ref = references[qid]
        scores = {
            "answer_em": score_answer_correctness(pred, ref, fuzzy=False),
            "answer_f1": score_answer_correctness(pred, ref, fuzzy=True),
            "answer_cover": score_answer_cover(pred, ref),
            "context_precision": score_context_precision(
                retrieved_doc_ids.get(qid, []), qrels.get(qid, {})
            ),
            "faithfulness": score_faithfulness(pred, "\n".join(contexts.get(qid, []))),
        }
        if judge is not None:
            scores["answer_claims"] = score_answer_claims(pred, ref, judge=judge)
        per_query[qid] = scores
    return per_query


def evaluate_rag(
    predictions: Dict[str, str],
    references: Dict[str, References],
    contexts: Dict[str, List[str]],
    retrieved_doc_ids: Dict[str, Sequence[str]],
    qrels: Dict[str, Mapping[str, int]],
    judge=None,
) -> Dict[str, float]:
    """Compute the mean RAG metric suite over a set of answered questions.

    Parameters
    ----------
    predictions:
        ``{question_id: generated_answer}``.
    references:
        ``{question_id: gold_answer | [gold_answers]}``.
    contexts:
        ``{question_id: [chunk_text, ...]}`` — the context text shown to the
        model (for faithfulness).
    retrieved_doc_ids:
        ``{question_id: [doc_id, ...]}`` — the documents of the supplied chunks
        (for qrels-based context precision).
    qrels:
        ``{question_id: {doc_id: relevance}}`` — the benchmark relevance
        judgments (for context precision).

    Returns
    -------
    dict
        ``{answer_em, answer_f1, context_precision, faithfulness}`` averaged over
        every question present in both ``predictions`` and ``references``.
    """
    per_query = score_rag_per_query(
        predictions, references, contexts, retrieved_doc_ids, qrels, judge=judge
    )
    names = METRIC_NAMES + (("answer_claims",) if judge is not None else ())
    if not per_query:
        return {metric: 0.0 for metric in names}
    n = len(per_query)
    return {
        metric: round(sum(q[metric] for q in per_query.values()) / n, 6)
        for metric in names
    }
