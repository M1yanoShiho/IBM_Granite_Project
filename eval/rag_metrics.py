"""Answer-quality metrics for the RAG evaluation (primary, A+B headline).

Where ``eval/ir_metrics.py`` scores the *retrieval* layer against qrels, this
module scores the *generation* layer of the retrieve-then-generate system:

- **Answer correctness** — does the generated answer match the reference answer
  (exact / fuzzy matching)?
- **Context precision** — of the chunks supplied to the model, how many were
  actually relevant to answering the question?
- **Faithfulness** — is the answer grounded in the retrieved context rather than
  hallucinated?

The current implementation uses heuristic, model-free scoring (token overlap and
string matching) so it can run CPU-only without an LLM judge.  When the RAGAS
dependency is repaired or a GPU becomes available for LLM-as-judge, these
heuristics can be upgraded to the RAGAS equivalents without changing the public
signatures.

These metrics are shared: the long-context NIAH diagnostic (``eval/metrics.py``)
re-exports ``score_context_precision`` / ``score_faithfulness`` from here so
there is a single source of truth.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List

# Stop words filtered out during token-overlap checks so short function words
# ("the", "is", ...) don't inflate relevance scores.
_STOP_WORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "nor", "not", "so", "yet", "both", "either", "neither", "each",
    "every", "all", "any", "few", "more", "most", "some", "no", "only",
    "other", "same", "than", "too", "very", "just", "about", "also",
    "it", "its", "that", "this", "these", "those", "he", "she", "they",
    "we", "you", "i", "me", "him", "her", "us", "them", "my", "your",
    "his", "our", "their", "there",
}


def _tokenize(text: str) -> List[str]:
    """Lower-case, extract alphanumeric tokens, drop stop words."""
    tokens: List[str] = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS]


def _jaccard(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Jaccard similarity between two token lists."""
    set_a, set_b = set(tokens_a), set(tokens_b)
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def _split_sentences(text: str) -> List[str]:
    """Split *text* into sentences on ``. ! ?``, keeping non-empty parts."""
    parts = re.split(r"[.!?]+", text)
    return [s.strip() for s in parts if s.strip()]


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
        If ``True``, allow case-insensitive substring / fuzzy matching;
        otherwise require an exact match.

    Returns
    -------
    float
        A score in ``[0.0, 1.0]``.
    """
    pred = prediction.strip()
    ref = reference.strip()

    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0

    # --- exact (case-insensitive) -------------------------------------------
    if pred.lower() == ref.lower():
        return 1.0

    if not fuzzy:
        return 0.0

    # --- fuzzy --------------------------------------------------------------
    pred_lower = pred.lower()
    ref_lower = ref.lower()

    # substring containment
    if ref_lower in pred_lower or pred_lower in ref_lower:
        return 1.0

    # token-level Jaccard
    pred_tokens = _tokenize(pred_lower)
    ref_tokens = _tokenize(ref_lower)
    token_score = _jaccard(pred_tokens, ref_tokens)

    # character-level sequence similarity (robust to reordering / extra words)
    char_score = SequenceMatcher(None, pred_lower, ref_lower).ratio()

    # equally-weighted blend of token overlap and character similarity
    return round((token_score + char_score) / 2.0, 6)


def score_context_precision(
    question: str,
    retrieved_chunks: List[str],
    ground_truth: str,
) -> float:
    """Estimate how much of the retrieved context was relevant.

    A chunk is judged relevant when it shares at least one meaningful
    (non-stop-word) token with the combined question + ground-truth query.

    Parameters
    ----------
    question:
        The probe question.
    retrieved_chunks:
        Context chunks supplied to the model (ordered as presented).
    ground_truth:
        The expected answer.

    Returns
    -------
    float
        A precision score in ``[0.0, 1.0]`` — fraction of retrieved chunks
        that are relevant.
    """
    if not retrieved_chunks:
        return 0.0

    query_tokens = set(_tokenize(f"{question} {ground_truth}"))
    if not query_tokens:
        return 0.0

    relevant = 0
    for chunk in retrieved_chunks:
        chunk_tokens = set(_tokenize(chunk))
        if query_tokens & chunk_tokens:  # at least one shared token
            relevant += 1

    return round(relevant / len(retrieved_chunks), 6)


def score_faithfulness(
    answer: str,
    context: str,
) -> float:
    """Estimate how grounded ``answer`` is in ``context``.

    Two signals are averaged:

    1. **Token coverage** — fraction of answer tokens present in the context.
    2. **Sentence support** — fraction of answer sentences whose tokens are at
       least 50% covered by the context.

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
    answer_stripped = answer.strip()
    if not answer_stripped:
        return 1.0  # nothing to hallucinate

    context_stripped = context.strip()
    if not context_stripped:
        return 0.0  # no grounding available

    context_tokens = set(_tokenize(context_stripped))

    # 1. token coverage -------------------------------------------------------
    answer_tokens = _tokenize(answer_stripped)
    if not answer_tokens:
        return 0.0
    token_covered = sum(1 for t in answer_tokens if t in context_tokens)
    token_score = token_covered / len(answer_tokens)

    # 2. sentence support -----------------------------------------------------
    sentences = _split_sentences(answer_stripped)
    if not sentences:
        return token_score

    supported = 0
    for sent in sentences:
        sent_tokens = [t for t in _tokenize(sent)]
        if not sent_tokens:
            supported += 1  # degenerate sentence: give the benefit of the doubt
            continue
        sent_covered = sum(1 for t in sent_tokens if t in context_tokens)
        if sent_covered / len(sent_tokens) >= 0.5:
            supported += 1
    sentence_score = supported / len(sentences)

    return round((token_score + sentence_score) / 2.0, 6)


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
    ids = set(predictions) & set(references) & set(contexts)
    if not ids:
        return {
            "answer_correctness": 0.0,
            "context_precision": 0.0,
            "faithfulness": 0.0,
        }

    correctness_scores: List[float] = []
    precision_scores: List[float] = []
    faithfulness_scores: List[float] = []

    for qid in ids:
        pred = predictions[qid]
        ref = references[qid]
        chunks = contexts[qid]

        correctness_scores.append(score_answer_correctness(pred, ref, fuzzy=True))
        precision_scores.append(score_context_precision(qid, chunks, ref))
        # faithfulness uses the concatenated context as the grounding source
        faithfulness_scores.append(
            score_faithfulness(pred, "\n".join(chunks))
        )

    return {
        "answer_correctness": round(
            sum(correctness_scores) / len(correctness_scores), 6
        ),
        "context_precision": round(
            sum(precision_scores) / len(precision_scores), 6
        ),
        "faithfulness": round(
            sum(faithfulness_scores) / len(faithfulness_scores), 6
        ),
    }
