from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

import evidence_rag.selector.deberta_contradiction as deberta_module
from evidence_rag.selector.deberta_contradiction import (
    DEFAULT_CONTRADICTION_MODEL_ID,
    DEFAULT_CONTRADICTION_MODEL_REVISION,
    DebertaContradictionScorer,
)
from evidence_rag.selector.reliability_mis import SelectorBackendError


class FakeScalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class FakeInput:
    def __init__(self) -> None:
        self.moves: list[str] = []

    def to(self, device: str) -> FakeInput:
        self.moves.append(device)
        return self


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[str], dict[str, Any]]] = []
        self.inputs: list[FakeInput] = []

    def __call__(
        self,
        premises: list[str],
        hypotheses: list[str],
        **kwargs: Any,
    ) -> dict[str, FakeInput]:
        self.calls.append((premises, hypotheses, kwargs))
        value = FakeInput()
        self.inputs.append(value)
        return {"input_ids": value}


class FakeModel:
    def __init__(
        self,
        id2label: object,
        probability_batches: list[list[list[float]]],
        *,
        fail_on_call: int | None = None,
    ) -> None:
        self.config = SimpleNamespace(id2label=id2label)
        self.probability_batches = probability_batches
        self.fail_on_call = fail_on_call
        self.call_count = 0
        self.eval_count = 0
        self.to_calls: list[str] = []
        self.device = "fake-device"

    def to(self, device: str) -> FakeModel:
        self.to_calls.append(device)
        self.device = device
        return self

    def eval(self) -> None:
        self.eval_count += 1

    def __call__(self, **_inputs: Any) -> SimpleNamespace:
        self.call_count += 1
        if self.fail_on_call == self.call_count:
            raise RuntimeError("inference failed")
        rows = self.probability_batches[self.call_count - 1]
        return SimpleNamespace(logits=rows)


class FakeTorch:
    def __init__(self) -> None:
        self.inference_count = 0

    def inference_mode(self) -> Any:
        self.inference_count += 1
        return nullcontext()

    @staticmethod
    def softmax(logits: list[list[float]], *, dim: int) -> list[list[FakeScalar]]:
        assert dim == -1
        return [[FakeScalar(value) for value in row] for row in logits]


class FakeLoader:
    def __init__(self, value: object) -> None:
        self.value = value
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def from_pretrained(self, model_id: str, **kwargs: Any) -> object:
        self.calls.append((model_id, kwargs))
        return self.value


def install_fake_modules(
    monkeypatch: pytest.MonkeyPatch,
    model: FakeModel,
) -> tuple[FakeTokenizer, FakeLoader, FakeLoader, FakeTorch]:
    tokenizer = FakeTokenizer()
    tokenizer_loader = FakeLoader(tokenizer)
    model_loader = FakeLoader(model)
    transformers = SimpleNamespace(
        AutoTokenizer=tokenizer_loader,
        AutoModelForSequenceClassification=model_loader,
    )
    torch = FakeTorch()

    def import_module(name: str) -> object:
        if name == "transformers":
            return transformers
        if name == "torch":
            return torch
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(deberta_module.importlib, "import_module", import_module)
    return tokenizer, tokenizer_loader, model_loader, torch


def test_constructor_and_empty_batch_do_not_import_or_load(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_import(name: str) -> object:
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(deberta_module.importlib, "import_module", forbidden_import)
    scorer = DebertaContradictionScorer()
    assert scorer.score_pairs(()) == ()


def test_fixed_revision_label_lookup_batching_and_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel(
        {0: "neutral", 1: "CONTRADICTION", 2: "entailment"},
        [
            [[0.2, 0.7, 0.1], [0.8, 0.1, 0.1]],
            [[0.1, 0.6, 0.3]],
        ],
    )
    tokenizer, tokenizer_loader, model_loader, torch = install_fake_modules(monkeypatch, model)
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "test-token")
    monkeypatch.setenv("MODEL_CACHE_DIR", "/tmp/model-cache")

    scorer = DebertaContradictionScorer(device="cpu", batch_size=2)
    pairs = (("p1", "h1"), ("p2", "h2"), ("p3", "h3"))
    assert scorer.score_pairs(pairs) == pytest.approx((0.7, 0.1, 0.6))

    assert tokenizer_loader.calls == [
        (
            DEFAULT_CONTRADICTION_MODEL_ID,
            {
                "revision": DEFAULT_CONTRADICTION_MODEL_REVISION,
                "token": "test-token",
                "cache_dir": "/tmp/model-cache",
            },
        )
    ]
    assert model_loader.calls[0][0] == DEFAULT_CONTRADICTION_MODEL_ID
    assert model_loader.calls[0][1]["revision"] == DEFAULT_CONTRADICTION_MODEL_REVISION
    assert model_loader.calls[0][1]["device_map"] is None
    assert tokenizer.calls[0][0:2] == (["p1", "p2"], ["h1", "h2"])
    assert tokenizer.calls[1][0:2] == (["p3"], ["h3"])
    assert tokenizer.calls[0][2] == {
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
        "max_length": 512,
    }
    assert model.to_calls == ["cpu"]
    assert model.eval_count == 1
    assert all(value.moves == ["cpu"] for value in tokenizer.inputs)
    assert torch.inference_count == 2


def test_loaded_backend_is_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel(
        {0: "contradiction", 1: "neutral", 2: "entailment"},
        [[[0.2, 0.5, 0.3]], [[0.8, 0.1, 0.1]]],
    )
    _tokenizer, tokenizer_loader, model_loader, _torch = install_fake_modules(monkeypatch, model)
    scorer = DebertaContradictionScorer()

    assert scorer.score_pairs((("p1", "h1"),)) == (0.2,)
    assert scorer.score_pairs((("p2", "h2"),)) == (0.8,)
    assert len(tokenizer_loader.calls) == 1
    assert len(model_loader.calls) == 1
    assert model.eval_count == 1


def test_default_batch_size_splits_thirty_three_pairs_into_thirty_two_plus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_rows = [[0.25, 0.5, 0.25] for _ in range(32)]
    model = FakeModel(
        {0: "contradiction", 1: "neutral", 2: "entailment"},
        [first_rows, [[0.75, 0.2, 0.05]]],
    )
    tokenizer, _tokenizer_loader, _model_loader, _torch = install_fake_modules(monkeypatch, model)
    pairs = tuple((f"p-{index}", f"h-{index}") for index in range(33))

    result = DebertaContradictionScorer().score_pairs(pairs)
    assert result == (0.25,) * 32 + (0.75,)
    assert [len(call[0]) for call in tokenizer.calls] == [32, 1]


@pytest.mark.parametrize(
    "id2label",
    [
        {0: "neutral", 1: "entailment"},
        {0: "contradiction", 1: "contradiction_alt", 2: "neutral"},
        None,
    ],
)
def test_invalid_contradiction_label_mapping_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch,
    id2label: object,
) -> None:
    model = FakeModel(id2label, [[[0.2, 0.5, 0.3]]])
    install_fake_modules(monkeypatch, model)
    with pytest.raises(SelectorBackendError, match="contradiction label|id2label"):
        DebertaContradictionScorer().score_pairs((("p", "h"),))


def test_second_batch_failure_discards_partial_result(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel(
        {0: "contradiction", 1: "neutral", 2: "entailment"},
        [[[0.8, 0.1, 0.1]], [[0.2, 0.5, 0.3]]],
        fail_on_call=2,
    )
    install_fake_modules(monkeypatch, model)
    scorer = DebertaContradictionScorer(batch_size=1)
    with pytest.raises(SelectorBackendError, match="inference"):
        scorer.score_pairs((("p1", "h1"), ("p2", "h2")))


@pytest.mark.parametrize("probability", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_probability_fails(monkeypatch: pytest.MonkeyPatch, probability: float) -> None:
    model = FakeModel(
        {0: "contradiction", 1: "neutral", 2: "entailment"},
        [[[probability, 0.5, 0.5]]],
    )
    install_fake_modules(monkeypatch, model)
    with pytest.raises(SelectorBackendError, match="probability"):
        DebertaContradictionScorer().score_pairs((("p", "h"),))


def test_wrong_batch_length_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel(
        {0: "contradiction", 1: "neutral", 2: "entailment"},
        [[[0.8, 0.1, 0.1]]],
    )
    install_fake_modules(monkeypatch, model)
    with pytest.raises(SelectorBackendError, match="batch length"):
        DebertaContradictionScorer().score_pairs((("p1", "h1"), ("p2", "h2")))


def test_model_load_failure_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_import(_name: str) -> object:
        raise ImportError("optional dependency unavailable")

    monkeypatch.setattr(deberta_module.importlib, "import_module", failing_import)
    with pytest.raises(SelectorBackendError, match="load"):
        DebertaContradictionScorer().score_pairs((("p", "h"),))


def test_invalid_local_configuration_fails_before_model_loading() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        DebertaContradictionScorer(batch_size=0)
    with pytest.raises(ValueError, match="max_length"):
        DebertaContradictionScorer(max_length=0)
