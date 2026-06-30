"""SPLADE learned-sparse encoding.

Turns text into a SPARSE term-weight vector over the model's vocabulary — the
learned-sparse counterpart to BM25's term frequencies. Used as the (stronger) lexical
arm of the convex hybrid (Phase 2): a ``SpladeEncoder`` feeds a ``SparseIndex``
(``src/retrieval/sparse_index.py``), wrapped by a ``SparseRetriever``.

The SPLADE recipe (Formal et al.): from the MLM logits ``[batch, seq, vocab]``,

    w_{b,j} = max over positions i of ( log(1 + relu(logits_{b,i,j})) * mask_{b,i} ),

i.e. max-pool over the sequence of log-saturated ReLU activations, with padding masked
out. :func:`splade_pool` is that pure transform (the TDD unit); :class:`SpladeEncoder`
wraps a tokenizer + ``AutoModelForMaskedLM`` around it.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import torch

DEFAULT_SPLADE_MODEL_ID = "naver/splade-cocondenser-ensembledistil"

TermWeights = Dict[int, float]  # {term_id: weight}, a sparse vocabulary vector


def splade_pool(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """SPLADE pooling: ``max_i log(1 + relu(logits)) * mask`` over the sequence.

    Parameters
    ----------
    logits:
        MLM logits, shape ``[batch, seq, vocab]``.
    attention_mask:
        ``[batch, seq]``; padding positions (0) are zeroed so they never win the max.

    Returns
    -------
    Tensor ``[batch, vocab]`` of non-negative term weights.
    """
    activated = torch.log1p(torch.relu(logits))          # log(1 + relu(x)) >= 0
    masked = activated * attention_mask.unsqueeze(-1)     # zero out padding positions
    return masked.max(dim=1).values                       # max-pool over the sequence


class SpladeEncoder:
    """Encode text into sparse SPLADE term weights via a masked-language model.

    Parameters
    ----------
    model_id:
        Hugging Face model id (default ``naver/splade-cocondenser-ensembledistil``;
        overridable via the ``SPLADE_MODEL_ID`` env var).
    model, tokenizer:
        Injected for tests — a fake MLM exposing ``(**inputs).logits`` and a fake
        tokenizer returning ``input_ids`` / ``attention_mask`` tensors. When both are
        omitted, the real ``AutoModelForMaskedLM`` / ``AutoTokenizer`` load lazily
        (honouring ``MODEL_CACHE_DIR``).
    """

    def __init__(self, model_id: str | None = None, *, model=None, tokenizer=None) -> None:
        self.model_id = model_id or os.getenv("SPLADE_MODEL_ID") or DEFAULT_SPLADE_MODEL_ID
        if model is not None and tokenizer is not None:
            self._model, self._tokenizer = model, tokenizer
        else:
            self._model, self._tokenizer = self._load()

    def _load(self):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        cache = os.getenv("MODEL_CACHE_DIR") or None
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=cache)
        model = AutoModelForMaskedLM.from_pretrained(self.model_id, cache_dir=cache)
        model.eval()
        return model, tokenizer

    def encode(self, texts: Sequence[str]) -> List[TermWeights]:
        """Encode a batch of texts into sparse ``{term_id: weight}`` dicts (weight > 0)."""
        texts = list(texts)
        if not texts:
            return []
        inputs = self._tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
        pooled = splade_pool(logits, inputs["attention_mask"])
        out: List[TermWeights] = []
        for row in pooled:
            nz = torch.nonzero(row > 0.0, as_tuple=False).flatten().tolist()
            out.append({int(j): float(row[j]) for j in nz})
        return out
