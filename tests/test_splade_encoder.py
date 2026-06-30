"""Tests for src/retrieval/splade_encoder.py — SPLADE sparse encoding.

``splade_pool`` is the pure pooling recipe (max over the sequence of
log(1+relu(logits)), padding masked out); ``SpladeEncoder.encode`` is tested with an
injected fake model + tokenizer so no SPLADE model is downloaded. The real model
loads only on the HPC run.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from src.retrieval.splade_encoder import SpladeEncoder, splade_pool


def test_splade_pool_max_over_sequence_of_log1p_relu() -> None:
    # V=3, one example, two positions, all unmasked.
    logits = torch.tensor([[[1.0, -1.0, 0.0], [2.0, 0.0, -3.0]]])
    mask = torch.tensor([[1, 1]])
    pooled = splade_pool(logits, mask)
    assert tuple(pooled.shape) == (1, 3)
    assert pooled[0, 0].item() == pytest.approx(math.log(3.0))  # max(log2, log3)
    assert pooled[0, 1].item() == pytest.approx(0.0)
    assert pooled[0, 2].item() == pytest.approx(0.0)


def test_splade_pool_masks_padding_positions() -> None:
    # Position 1 is padding (mask 0) -> its large logits must not contribute.
    logits = torch.tensor([[[1.0, 0.0], [9.0, 9.0]]])
    mask = torch.tensor([[1, 0]])
    pooled = splade_pool(logits, mask)
    assert pooled[0, 0].item() == pytest.approx(math.log(2.0))  # from position 0 only
    assert pooled[0, 1].item() == pytest.approx(0.0)


class _FakeTokenizer:
    """Returns canned input_ids/attention_mask tensors (ignores the texts)."""

    def __init__(self, input_ids, attention_mask) -> None:
        self._batch = {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }

    def __call__(self, texts, padding=True, truncation=True, return_tensors="pt"):
        return dict(self._batch)


class _FakeMaskedLM:
    """Returns canned logits regardless of inputs; exposes a vocab_size config."""

    def __init__(self, logits) -> None:
        self._logits = logits
        self.config = SimpleNamespace(vocab_size=int(logits.shape[-1]))

    def __call__(self, **inputs):
        return SimpleNamespace(logits=self._logits)


def test_encode_produces_sparse_term_weights() -> None:
    logits = torch.tensor(
        [
            [[2.0, -1.0, 0.0, -5.0], [0.0, 3.0, -1.0, -5.0]],   # text 0, both positions
            [[1.0, -1.0, 4.0, -5.0], [9.0, 9.0, 9.0, 9.0]],     # text 1, pos 1 masked out
        ]
    )
    tok = _FakeTokenizer([[1, 2], [3, 0]], [[1, 1], [1, 0]])
    enc = SpladeEncoder(model=_FakeMaskedLM(logits), tokenizer=tok)

    out = enc.encode(["text zero", "text one"])

    assert out[0] == pytest.approx({0: math.log(3.0), 1: math.log(4.0)})
    assert out[1] == pytest.approx({0: math.log(2.0), 2: math.log(5.0)})


def test_encode_empty_returns_empty() -> None:
    enc = SpladeEncoder(
        model=_FakeMaskedLM(torch.zeros((0, 0, 0))),
        tokenizer=_FakeTokenizer([], []),
    )
    assert enc.encode([]) == []


def test_encoder_exposes_vocab_size() -> None:
    enc = SpladeEncoder(
        model=_FakeMaskedLM(torch.zeros((1, 1, 7))),
        tokenizer=_FakeTokenizer([[1]], [[1]]),
    )
    assert enc.vocab_size == 7
