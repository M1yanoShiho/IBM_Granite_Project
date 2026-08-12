from __future__ import annotations

import builtins
import importlib
import sys
from dataclasses import dataclass
from typing import Any

import pytest

from evidence_rag.selector.dual_head import (
    MAX_LENGTH,
    audit_token_lengths,
    fingerprint_dual_head_model,
    masked_dual_head_bce,
    tokenize_question_candidates,
    weights_sha256,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def _ids(text: str) -> list[int]:
        return list(range(len(text.split())))

    def __call__(
        self,
        first: list[str],
        second: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, list[list[int]]]:
        self.calls.append({"first": first, "second": second, **kwargs})
        if second is None:
            return {"input_ids": [self._ids(text) for text in first]}
        rows: list[list[int]] = []
        for question, candidate in zip(first, second, strict=True):
            question_ids = self._ids(question)
            candidate_ids = self._ids(candidate)
            # Three pair-special tokens mimic the only property the audit needs.
            raw = question_ids + [-1] + candidate_ids + [-2, -3]
            if kwargs.get("truncation") == "only_second" and len(raw) > kwargs["max_length"]:
                keep = kwargs["max_length"] - len(question_ids) - 3
                if keep < 0:
                    raise ValueError("question does not fit")
                raw = question_ids + [-1] + candidate_ids[:keep] + [-2, -3]
            rows.append(raw)
        return {"input_ids": rows}


def test_importing_module_does_not_import_optional_ml_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "evidence_rag.selector.dual_head"
    original_import = builtins.__import__
    seen: list[str] = []

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {"torch", "transformers"}:
            seen.append(name)
            raise AssertionError(f"eager optional import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    sys.modules.pop(module_name, None)
    importlib.import_module(module_name)
    assert seen == []


def test_tokenizer_contract_preserves_question_and_only_truncates_candidate() -> None:
    tokenizer = FakeTokenizer()
    encoded = tokenize_question_candidates(
        tokenizer,
        question="which city",
        candidate_text="candidate text",
        return_tensors=None,
        padding=False,
    )
    assert len(encoded["input_ids"]) == 1
    assert tokenizer.calls[-1] == {
        "first": ["which city"],
        "second": ["candidate text"],
        "add_special_tokens": True,
        "truncation": "only_second",
        "max_length": 512,
        "padding": False,
        "return_tensors": None,
    }


def test_input_boundary_accepts_only_aligned_non_blank_question_candidate_text() -> None:
    tokenizer = FakeTokenizer()
    with pytest.raises(ValueError, match="batch sizes differ"):
        tokenize_question_candidates(
            tokenizer,
            question=["q1", "q2"],
            candidate_text=["d1"],
        )
    with pytest.raises(ValueError, match="question.*non-blank"):
        tokenize_question_candidates(tokenizer, question=" ", candidate_text="candidate")
    with pytest.raises(ValueError, match="candidate_text.*non-blank"):
        tokenize_question_candidates(tokenizer, question="question", candidate_text="")


def test_length_audit_measures_candidate_only_truncation() -> None:
    tokenizer = FakeTokenizer()
    long_candidate = " ".join(f"token-{index}" for index in range(MAX_LENGTH + 20))
    (audit,) = audit_token_lengths(
        tokenizer,
        question="short question",
        candidate_text=long_candidate,
    )
    assert audit.question_tokens == 2
    assert audit.candidate_tokens == MAX_LENGTH + 20
    assert audit.raw_pair_tokens == MAX_LENGTH + 25
    assert audit.encoded_pair_tokens == MAX_LENGTH
    assert audit.candidate_truncated is True
    assert tokenizer.calls[-1]["truncation"] == "only_second"


def _torch() -> Any:
    return pytest.importorskip("torch")


def test_masked_bce_uses_independent_masks() -> None:
    torch = _torch()
    protect_logits = torch.tensor([0.0, 0.0], requires_grad=True)
    harm_logits = torch.tensor([0.0, 0.0], requires_grad=True)
    losses = masked_dual_head_bce(
        protect_logits=protect_logits,
        harm_logits=harm_logits,
        protect_labels=torch.tensor([1.0, float("nan")]),
        harm_labels=torch.tensor([float("nan"), 0.0]),
        protect_mask=torch.tensor([1, 0]),
        harm_mask=torch.tensor([0, 1]),
    )
    assert losses.protect_count == 1
    assert losses.harm_count == 1
    assert losses.protect.item() == pytest.approx(0.693147, rel=1e-5)
    assert losses.harm.item() == pytest.approx(0.693147, rel=1e-5)
    assert torch.isfinite(losses.total)


def test_empty_masks_return_graph_connected_zero_without_nan() -> None:
    torch = _torch()
    protect_logits = torch.tensor([0.2, -0.4], requires_grad=True)
    harm_logits = torch.tensor([-0.1, 0.5], requires_grad=True)
    losses = masked_dual_head_bce(
        protect_logits=protect_logits,
        harm_logits=harm_logits,
        protect_labels=torch.tensor([float("nan"), float("nan")]),
        harm_labels=torch.tensor([float("nan"), float("nan")]),
        protect_mask=torch.tensor([0, 0]),
        harm_mask=torch.tensor([False, False]),
    )
    assert losses.protect_count == 0
    assert losses.harm_count == 0
    assert losses.protect.item() == 0.0
    assert losses.harm.item() == 0.0
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert torch.equal(protect_logits.grad, torch.zeros_like(protect_logits))
    assert torch.equal(harm_logits.grad, torch.zeros_like(harm_logits))


def test_active_nan_label_is_refused_but_masked_nan_is_allowed() -> None:
    torch = _torch()
    with pytest.raises(ValueError, match="non-finite active"):
        masked_dual_head_bce(
            protect_logits=torch.tensor([0.0]),
            harm_logits=torch.tensor([0.0]),
            protect_labels=torch.tensor([float("nan")]),
            harm_labels=torch.tensor([0.0]),
            protect_mask=torch.tensor([1]),
            harm_mask=torch.tensor([0]),
        )


def test_real_torch_model_has_shared_encoder_and_two_independent_sigmoid_heads() -> None:
    torch = _torch()
    module = importlib.import_module("evidence_rag.selector.dual_head")

    class TinyEncoder(torch.nn.Module):  # type: ignore[name-defined,misc]
        @dataclass
        class Config:
            hidden_size: int = 3

        def __init__(self) -> None:
            super().__init__()
            self.config = self.Config()
            self.projection = torch.nn.Linear(1, 3)

        def forward(self, input_ids: Any) -> Any:
            hidden = self.projection(input_ids.float().unsqueeze(-1))
            return type("EncoderOutput", (), {"last_hidden_state": hidden})()

    class TensorTokenizer(FakeTokenizer):
        def __call__(self, first: list[str], second: list[str] | None = None, **kwargs: Any) -> Any:
            if kwargs.get("return_tensors") == "pt":
                return {"input_ids": torch.tensor([[1, 2], [3, 4]])[: len(first)]}
            return super().__call__(first, second, **kwargs)

    model_class = module._dual_head_model_class(torch)
    encoder = TinyEncoder()
    model = model_class(
        tokenizer=TensorTokenizer(),
        encoder=encoder,
        model_id="tiny",
        revision="revision",
    )
    assert model.protect_head is not model.harm_head
    assert model.protect_head.weight.data_ptr() != model.harm_head.weight.data_ptr()
    output = model(question=["q1", "q2"], candidate_text=["d1", "d2"])
    assert output.protect_logits.shape == (2,)
    assert output.harm_logits.shape == (2,)
    assert torch.all((output.protect_scores >= 0) & (output.protect_scores <= 1))
    assert torch.all((output.harm_scores >= 0) & (output.harm_scores <= 1))
    # A public forward call with privileged or generic pre-tokenized inputs is rejected by the
    # keyword-only question/candidate_text signature rather than becoming a hidden feature path.
    with pytest.raises(TypeError):
        model(input_ids=torch.tensor([[1, 2]]))


def test_weight_fingerprint_is_deterministic_and_covers_both_heads() -> None:
    torch = _torch()

    class TinyModule(torch.nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.protect_head = torch.nn.Linear(2, 1)
            self.harm_head = torch.nn.Linear(2, 1)
            self.model_id = "tiny"
            self.revision = "abc123"
            self.max_length = MAX_LENGTH

    model = TinyModule()
    before = weights_sha256(model)
    assert len(before) == 64
    assert weights_sha256(model) == before
    with torch.no_grad():
        model.harm_head.weight.add_(1)
    after = weights_sha256(model)
    assert after != before
    fingerprint = fingerprint_dual_head_model(model)
    assert fingerprint.model_id == "tiny"
    assert fingerprint.revision == "abc123"
    assert fingerprint.weights_sha256 == after
    assert fingerprint.to_dict()["max_length"] == 512
