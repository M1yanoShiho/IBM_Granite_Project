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
        # use_safetensors avoids torch.load, which transformers blocks on torch < 2.6
        # (the HPC's torch 2.5.1) for CVE-2025-32434. The weights load from the model's
        # .safetensors file instead of pytorch_model.bin.
        model = AutoModelForMaskedLM.from_pretrained(
            self.model_id, cache_dir=cache, use_safetensors=True
        )
        model.eval()
        if torch.cuda.is_available():
            model = model.to("cuda")  # encode the corpus on GPU (CPU is far too slow at scale)
        return model, tokenizer

    @property
    def vocab_size(self) -> int:
        """Vocabulary size = the MLM head's output dimension (the term-id space)."""
        return int(self._model.config.vocab_size)

    def encode(
        self, texts: Sequence[str], batch_size: int | None = None
    ) -> List[TermWeights]:
        """Encode texts into sparse ``{term_id: weight}`` dicts (weight > 0), in batches.

        ``batch_size`` bounds GPU memory: the intermediate logits are ``[batch, seq,
        vocab]`` with ``vocab`` ~30k, so encoding the whole corpus at once OOMs. Defaults
        to the ``SPLADE_BATCH_SIZE`` env var, else 32. Pooled rows move to CPU before
        sparsifying, freeing GPU memory between batches.
        """
        texts = list(texts)
        if not texts:
            return []
        bs = batch_size or int(os.getenv("SPLADE_BATCH_SIZE", "32"))
        device = getattr(self._model, "device", None)
        out: List[TermWeights] = []
        for start in range(0, len(texts), bs):
            inputs = self._tokenizer(
                texts[start : start + bs],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            if device is not None:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits
            pooled = splade_pool(logits, inputs["attention_mask"]).cpu()
            for row in pooled:
                nz = torch.nonzero(row > 0.0, as_tuple=False).flatten().tolist()
                out.append({int(j): float(row[j]) for j in nz})
        return out
