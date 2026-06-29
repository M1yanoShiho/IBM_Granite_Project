"""Answer provenance / citations.

Maps spans of a generated answer to the retrieved chunks that support them, so
the system can show *where* each claim came from. This is the basis for the
"trust in retrieved outputs" requirement and underpins faithfulness checks.

The current implementation uses token-overlap (Jaccard) between answer sentences
and retrieved chunks, via the shared model-free helpers in
``src.text_utils``. When a GPU is available this can be upgraded to
embedding-based semantic overlap via ``src.retrieval.embedder.Embedder``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.retrieval.base import RetrievedChunk
from src.text_utils import jaccard, split_sentences, tokenize


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

    sentences = split_sentences(answer)
    if not sentences:
        return []

    # Pre-tokenise every chunk once (avoid re-tokenising per sentence).
    chunk_tokens_list: List[List[str]] = [tokenize(chunk.text) for chunk in retrieved]

    citations: List[Citation] = []
    for sentence in sentences:
        sent_tokens = tokenize(sentence)
        if not sent_tokens:
            continue

        # best chunk for this sentence
        best_idx = 0
        best_score = 0.0
        for i, chunk_tokens in enumerate(chunk_tokens_list):
            score = jaccard(sent_tokens, chunk_tokens)
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
