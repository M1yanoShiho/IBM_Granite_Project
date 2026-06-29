"""Answer provenance / citations.

Maps spans of a generated answer to the retrieved chunks that support them, so
the system can show *where* each claim came from. This is the basis for the
"trust in retrieved outputs" requirement and underpins faithfulness checks.

The current implementation uses token-overlap (Jaccard) between answer sentences
and retrieved chunks.  When a GPU is available this can be upgraded to
embedding-based semantic overlap via ``src.retrieval.embedder.Embedder``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from src.retrieval.base import RetrievedChunk

# Stop words filtered out during tokenisation so short function words don't
# inflate the overlap score between a sentence and a chunk.
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
    """Split *text* into sentence-level spans on ``. ! ?``.

    The delimiter is kept as part of the sentence so the returned span is
    verbatim text from the original answer.
    """
    # Split at punctuation followed by whitespace or end-of-string, keeping
    # the punctuation attached to its sentence.
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]


@dataclass
class Citation:
    """A link between a piece of the answer and its supporting source.

    Attributes
    ----------
    answer_span:
        The portion of the generated answer being attributed.
    source_chunk_id:
        Identifier of the supporting chunk.
    score:
        Confidence/support score for the attribution.
    """

    answer_span: str
    source_chunk_id: str
    score: float


def attribute_answer(
    answer: str,
    retrieved: List[RetrievedChunk],
    token_overlap_threshold: float = 0.1,
) -> List[Citation]:
    """Attribute parts of ``answer`` to the ``retrieved`` chunks that support them.

    The answer is split into sentence-level spans.  Each span is compared
    against every retrieved chunk via token-overlap (Jaccard similarity after
    lower-casing and stop-word removal).  A span is attributed to the chunk
    with the highest overlap, provided the score meets *token_overlap_threshold*.

    Parameters
    ----------
    answer:
        The model's generated answer (one or more sentences).
    retrieved:
        The chunks that were supplied as context to the model, as
        :class:`~src.retrieval.base.RetrievedChunk` objects.
    token_overlap_threshold:
        Minimum Jaccard score for a sentence to be attributed to any chunk
        (default ``0.1``).

    Returns
    -------
    List[Citation]
        One :class:`Citation` per attributable answer sentence, in the order
        the sentences appear in *answer*.
    """
    if not answer.strip() or not retrieved:
        return []

    sentences = _split_sentences(answer)
    if not sentences:
        return []

    # Pre-tokenise every chunk once (avoid re-tokenising per sentence).
    chunk_tokens_list: List[List[str]] = [
        _tokenize(chunk.text) for chunk in retrieved
    ]

    citations: List[Citation] = []
    for sentence in sentences:
        sent_tokens = _tokenize(sentence)
        if not sent_tokens:
            continue

        # best chunk for this sentence
        best_idx = 0
        best_score = 0.0
        for i, chunk_tokens in enumerate(chunk_tokens_list):
            score = _jaccard(sent_tokens, chunk_tokens)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score >= token_overlap_threshold:
            citations.append(
                Citation(
                    answer_span=sentence,
                    source_chunk_id=retrieved[best_idx].doc_id,
                    score=round(best_score, 6),
                )
            )

    return citations
