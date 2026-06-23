"""Regression tests for ``LLMClient.generate`` (no real model load).

``generate()`` must work across transformers versions: 4.x's
``apply_chat_template`` returns a bare tensor, while 5.x returns a
``BatchEncoding`` (dict-like). Passing the latter straight to
``model.generate(input_ids=...)`` crashes on ``inputs_tensor.shape`` — the bug
this locks down. The tests drive ``generate()`` with fakes, bypassing the heavy
model download in ``__init__``.
"""

from __future__ import annotations

import torch

from src.llm_client import GenerationConfig, LLMClient


class _FakeBatchEncoding(dict):
    """Mimics a transformers BatchEncoding: dict-like, has ``.to()`` and
    attribute access for keys, but no ``.shape`` (so ``.shape`` raises
    AttributeError exactly like the real one)."""

    def to(self, device):
        return self

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _ChatTokenizer:
    """Instruct-model tokenizer whose ``apply_chat_template`` returns a
    BatchEncoding (the transformers 5.x behaviour that broke ``generate``)."""

    chat_template = "{{ messages }}"

    def apply_chat_template(self, messages, add_generation_prompt, return_tensors):
        return _FakeBatchEncoding(input_ids=torch.tensor([[1, 2, 3]]))

    def decode(self, ids, skip_special_tokens):
        return "decoded answer"


class _PlainTokenizer:
    """Base-model tokenizer: no chat template; ``__call__`` returns a
    BatchEncoding."""

    chat_template = None

    def __call__(self, text, return_tensors):
        return _FakeBatchEncoding(input_ids=torch.tensor([[1, 2, 3]]))

    def decode(self, ids, skip_special_tokens):
        return "decoded answer"


class _FakeModel:
    device = "cpu"

    def generate(self, input_ids, **kwargs):
        # transformers reads input_ids.shape internally; a BatchEncoding has no
        # .shape, so this reproduces the real crash if generate() fails to unwrap.
        assert hasattr(input_ids, "shape"), "input_ids must be a tensor, not a dict"
        return torch.tensor([[1, 2, 3, 4, 5]])


def _client_with(tokenizer):
    client = LLMClient.__new__(LLMClient)  # bypass __init__ (no model download)
    client._tokenizer = tokenizer
    client._client = _FakeModel()
    client.config = GenerationConfig()
    return client


def test_generate_unwraps_chat_template_batchencoding() -> None:
    # transformers 5.x apply_chat_template returns a BatchEncoding; generate must
    # unwrap it to the input_ids tensor before calling model.generate.
    client = _client_with(_ChatTokenizer())

    assert client.generate("hello") == "decoded answer"


def test_generate_handles_base_model_tokenizer() -> None:
    client = _client_with(_PlainTokenizer())

    assert client.generate("hello") == "decoded answer"
