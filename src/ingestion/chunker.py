"""Chunking: split documents into retrievable passages.

Retrieval quality depends heavily on chunking. Chunks must be small enough to
be precise units of relevance, but large enough to stay self-contained.

Two units are supported:

- **Words** (default) — split on whitespace. Dependency-light and reproducible;
  ``chunk_size`` counts whitespace-delimited words.
- **Model tokens** (pass a ``tokenizer``) — split on the embedding model's own
  sub-word tokens via offset mapping, so a chunk never exceeds the model's
  ``max_seq_length`` and gets silently truncated at encode time. Use this for
  long-document corpora (e.g. NQ) where word-vs-token mismatch otherwise both
  truncates chunks and makes a chunk-size ablation meaningless past the limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    """A retrievable passage derived from a source document.

    Attributes
    ----------
    chunk_id:
        Unique identifier (e.g. ``"<doc_id>::<n>"``) — used for citations and
        for scoring against benchmark relevance judgments.
    doc_id:
        Identifier of the parent document.
    text:
        The chunk text.
    """

    chunk_id: str
    doc_id: str
    text: str


def chunk_document(
    doc_id: str,
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    tokenizer=None,
) -> List[Chunk]:
    """Split a document into overlapping :class:`Chunk` objects.

    ``chunk_overlap`` must be smaller than ``chunk_size``: the window advances by
    ``chunk_size - chunk_overlap`` units each step, so an overlap >= size would
    give a non-positive step and never advance (infinite loop). These knobs are
    user-exposed (CLI ``--overlap``, the factory, ablation sweeps), so validate
    up front rather than hang.

    Parameters
    ----------
    chunk_size, chunk_overlap:
        Window size and overlap, counted in **words** when ``tokenizer`` is
        ``None`` (default) or in **model tokens** when a ``tokenizer`` is given.
    tokenizer:
        Optional Hugging Face *fast* tokenizer (anything callable that accepts
        ``return_offsets_mapping=True``). When provided, the document is split on
        that tokenizer's sub-word tokens and each chunk's text is sliced from the
        original string at token boundaries (via the offset mapping, so no decode
        round-trip artefacts). Pass the embedding model's own tokenizer so chunk
        lengths line up with what the model actually encodes.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be at least 1; got {chunk_size}.")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative; got {chunk_overlap}.")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than "
            f"chunk_size ({chunk_size}); otherwise the window never advances."
        )

    if tokenizer is None:
        return _chunk_by_words(doc_id, text, chunk_size, chunk_overlap)
    return _chunk_by_tokens(doc_id, text, chunk_size, chunk_overlap, tokenizer)


def _chunk_by_words(
    doc_id: str, text: str, chunk_size: int, chunk_overlap: int
) -> List[Chunk]:
    """Sliding-window chunking over whitespace-delimited words (the default)."""
    tokens = text.split()
    if not tokens:
        return []

    chunks: List[Chunk] = []
    n = 0
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(tokens):
        chunk_text = " ".join(tokens[start : start + chunk_size])
        chunks.append(Chunk(chunk_id=f"{doc_id}::{n}", doc_id=doc_id, text=chunk_text))
        n += 1
        start += step

    return chunks


def _chunk_by_tokens(
    doc_id: str, text: str, chunk_size: int, chunk_overlap: int, tokenizer
) -> List[Chunk]:
    """Sliding-window chunking over the model's sub-word tokens.

    Uses the tokenizer's character offset mapping to slice each chunk straight
    out of the original ``text``, so the chunk count respects ``chunk_size`` *in
    model tokens* and the chunk text is the exact source substring (no decode
    artefacts). Requires a fast tokenizer that supports ``return_offsets_mapping``.
    """
    try:
        encoding = tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
    except (TypeError, NotImplementedError) as exc:  # slow tokenizer / no offsets
        raise ValueError(
            "Token-aware chunking needs a fast tokenizer that supports "
            "return_offsets_mapping=True."
        ) from exc

    # Keep only real, non-empty spans (defensive against any (0, 0) markers).
    offsets = [(s, e) for s, e in encoding["offset_mapping"] if e > s]
    if not offsets:
        return []

    chunks: List[Chunk] = []
    n = 0
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(offsets):
        window = offsets[start : start + chunk_size]
        char_start = window[0][0]
        char_end = window[-1][1]
        chunk_text = text[char_start:char_end]
        chunks.append(Chunk(chunk_id=f"{doc_id}::{n}", doc_id=doc_id, text=chunk_text))
        n += 1
        start += step

    return chunks
