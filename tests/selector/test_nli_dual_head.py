from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

from evidence_rag.contracts.models import CandidateSet, EvidenceCandidate, Query
from evidence_rag.selector.dual_head import (
    load_dual_head_checkpoint,
    masked_dual_head_bce,
    save_dual_head_checkpoint,
)
from evidence_rag.selector.models import CandidateRiskScore
from evidence_rag.selector.nli_dual_head import (
    NLI_ARCHITECTURE_VERSION,
    NLI_LABEL_MAP,
    load_nli_dual_head_model,
    pairwise_logistic_loss,
)
from evidence_rag.selector.risk_controlled import RiskControlledSelector


def _torch() -> Any:
    return pytest.importorskip("torch")


def _fake_transformers(
    torch: Any,
    *,
    id2label: dict[int, str] | None = None,
    classifier_outputs: int = 3,
    missing_module: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    state: dict[str, Any] = {}

    class TensorTokenizer:
        truncation_side = "left"

        def __call__(
            self,
            first: list[str],
            second: list[str] | None = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            del second, kwargs
            return {
                "input_ids": torch.tensor(
                    [[index + 1, index + 2] for index in range(len(first))],
                    dtype=torch.long,
                )
            }

    class TinyEncoder(torch.nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(1, 4)

        def forward(self, input_ids: Any) -> Any:
            hidden = self.projection(input_ids.float().unsqueeze(-1))
            return type("EncoderOutput", (), {"last_hidden_state": hidden})()

    class CountingPooler(torch.nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, hidden: Any) -> Any:
            self.calls += 1
            return hidden[:, 0, :]

    class CountingDropout(torch.nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, pooled: Any) -> Any:
            self.calls += 1
            return pooled

    class TinySequenceClassifier(torch.nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.config = type(
                "Config",
                (),
                {
                    "id2label": id2label or {0: "CONTRADICTION", 1: "ENTAILMENT", 2: "NEUTRAL"},
                    "num_labels": 3,
                },
            )()
            self.deberta = TinyEncoder()
            self.pooler = CountingPooler()
            self.dropout = CountingDropout()
            self.classifier = torch.nn.Linear(4, classifier_outputs)
            if missing_module is not None:
                delattr(self, missing_module)

        def forward(self, input_ids: Any) -> Any:
            hidden = self.deberta(input_ids=input_ids).last_hidden_state
            pooled = self.dropout(self.pooler(hidden))
            return type("ClassifierOutput", (), {"logits": self.classifier(pooled)})()

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path: str, **kwargs: Any) -> TensorTokenizer:
            assert path == "local/snapshot"
            assert kwargs["revision"] == "frozen-revision"
            assert kwargs["local_files_only"] is True
            assert kwargs["trust_remote_code"] is False
            tokenizer = TensorTokenizer()
            state["tokenizer"] = tokenizer
            return tokenizer

    class AutoModelForSequenceClassification:
        @staticmethod
        def from_pretrained(path: str, **kwargs: Any) -> TinySequenceClassifier:
            assert path == "local/snapshot"
            assert kwargs["revision"] == "frozen-revision"
            assert kwargs["local_files_only"] is True
            assert kwargs["trust_remote_code"] is False
            assert kwargs["use_safetensors"] is True
            model = TinySequenceClassifier()
            state["source_model"] = model
            return model

    transformers = type(
        "FakeTransformers",
        (),
        {
            "AutoTokenizer": AutoTokenizer,
            "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        },
    )()
    return transformers, state


def _load_fake_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    id2label: dict[int, str] | None = None,
    classifier_outputs: int = 3,
    missing_module: str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    torch = _torch()
    module = importlib.import_module("evidence_rag.selector.nli_dual_head")
    transformers, state = _fake_transformers(
        torch,
        id2label=id2label,
        classifier_outputs=classifier_outputs,
        missing_module=missing_module,
    )

    def optional_module(name: str, *, extra: str = "granite") -> Any:
        del extra
        return torch if name == "torch" else transformers

    monkeypatch.setattr(module, "_optional_module", optional_module)
    model = module.load_nli_dual_head_model(
        "local/snapshot",
        revision="frozen-revision",
        identity_model_id="cross-encoder/nli-deberta-v3-base",
        local_files_only=True,
    )
    return torch, model, state


def test_importing_module_does_not_import_optional_ml_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "evidence_rag.selector.nli_dual_head"
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


def test_loader_restores_nli_probabilities_and_independent_classifier_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch, model, state = _load_fake_model(monkeypatch)
    source = state["source_model"]

    assert model.architecture_version == NLI_ARCHITECTURE_VERSION
    assert dict(model.label_map) == dict(NLI_LABEL_MAP)
    assert model.model_id == "cross-encoder/nli-deberta-v3-base"
    assert state["tokenizer"].truncation_side == "right"
    assert torch.equal(model.protect_classifier.weight, source.classifier.weight)
    assert torch.equal(model.harm_classifier.weight, source.classifier.weight)
    assert model.protect_classifier.weight.data_ptr() != model.harm_classifier.weight.data_ptr()

    encoded = model.tokenize(
        question=["q1", "q2"],
        candidate_text=["candidate one", "candidate two"],
    )
    source.eval()
    with torch.no_grad():
        expected = torch.softmax(source(**encoded).logits, dim=-1)
    source.pooler.calls = 0
    source.dropout.calls = 0

    with torch.no_grad():
        output = model(
            question=["q1", "q2"],
            candidate_text=["candidate one", "candidate two"],
        )
    assert source.pooler.calls == 1
    assert source.dropout.calls == 1
    assert output.protect_logits.shape == output.harm_logits.shape == (2,)
    assert torch.allclose(output.protect_scores, expected[:, 1], atol=1e-7, rtol=1e-6)
    assert torch.allclose(output.harm_scores, expected[:, 0], atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize(
    ("cuda_available", "expected"),
    ((False, "cpu"), (True, "cuda")),
)
def test_loader_resolves_auto_device_from_cuda_availability(
    cuda_available: bool,
    expected: str,
) -> None:
    module = importlib.import_module("evidence_rag.selector.nli_dual_head")
    fake_torch = type(
        "FakeTorch",
        (),
        {
            "cuda": type(
                "FakeCuda",
                (),
                {"is_available": staticmethod(lambda: cuda_available)},
            )()
        },
    )()

    assert module._resolve_device(fake_torch, "auto") == expected


def test_loader_preserves_explicit_or_unspecified_device() -> None:
    module = importlib.import_module("evidence_rag.selector.nli_dual_head")
    fake_torch = object()

    assert module._resolve_device(fake_torch, None) is None
    assert module._resolve_device(fake_torch, "cuda:2") == "cuda:2"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        (
            {"id2label": {0: "entailment", 1: "contradiction", 2: "neutral"}},
            "label map",
        ),
        ({"classifier_outputs": 2}, "three logits"),
        ({"missing_module": "pooler"}, "pooler"),
    ),
)
def test_loader_rejects_incompatible_sequence_classifier_contract(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _load_fake_model(monkeypatch, **kwargs)


def test_loader_validates_identity_before_optional_imports() -> None:
    with pytest.raises(ValueError, match="model_id_or_path"):
        load_nli_dual_head_model(" ")
    with pytest.raises(ValueError, match="identity_model_id"):
        load_nli_dual_head_model("local/snapshot", identity_model_id=" ")


def test_pairwise_loss_has_frozen_value_gradients_and_empty_zero() -> None:
    torch = _torch()
    protect_clean = torch.zeros(2, requires_grad=True)
    protect_cf = torch.zeros(2, requires_grad=True)
    harm_clean = torch.zeros(2, requires_grad=True)
    harm_cf = torch.zeros(2, requires_grad=True)

    loss = pairwise_logistic_loss(
        protect_clean_logits=protect_clean,
        protect_counterfactual_logits=protect_cf,
        harm_clean_logits=harm_clean,
        harm_counterfactual_logits=harm_cf,
    )
    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
    loss.backward()
    assert torch.all(protect_clean.grad < 0)
    assert torch.all(protect_cf.grad > 0)
    assert torch.all(harm_clean.grad > 0)
    assert torch.all(harm_cf.grad < 0)

    empty = [torch.empty(0, requires_grad=True) for _ in range(4)]
    empty_loss = pairwise_logistic_loss(
        protect_clean_logits=empty[0],
        protect_counterfactual_logits=empty[1],
        harm_clean_logits=empty[2],
        harm_counterfactual_logits=empty[3],
    )
    assert empty_loss.item() == 0.0
    empty_loss.backward()
    assert all(value.grad is not None for value in empty)
    assert all(torch.equal(value.grad, torch.zeros_like(value)) for value in empty)


def test_pairwise_loss_rejects_shape_mismatch_and_nonfinite_logits() -> None:
    torch = _torch()
    with pytest.raises(ValueError, match="shape"):
        pairwise_logistic_loss(
            protect_clean_logits=torch.zeros(2),
            protect_counterfactual_logits=torch.zeros(1),
            harm_clean_logits=torch.zeros(2),
            harm_counterfactual_logits=torch.zeros(2),
        )
    with pytest.raises(ValueError, match="finite"):
        pairwise_logistic_loss(
            protect_clean_logits=torch.tensor([float("nan")]),
            protect_counterfactual_logits=torch.zeros(1),
            harm_clean_logits=torch.zeros(1),
            harm_counterfactual_logits=torch.zeros(1),
        )


def test_existing_masked_bce_keeps_2wiki_harm_classifier_gradient_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch, model, _ = _load_fake_model(monkeypatch)
    model.train()
    output = model(
        question=["q1", "q2"],
        candidate_text=["candidate one", "candidate two"],
    )
    losses = masked_dual_head_bce(
        protect_logits=output.protect_logits,
        harm_logits=output.harm_logits,
        protect_labels=torch.tensor([1.0, 1.0]),
        harm_labels=torch.tensor([float("nan"), float("nan")]),
        protect_mask=torch.tensor([1, 1]),
        harm_mask=torch.tensor([0, 0]),
        weight_normalization="active_count",
    )
    losses.total.backward()

    assert torch.any(model.protect_classifier.weight.grad != 0)
    assert torch.equal(
        model.harm_classifier.weight.grad,
        torch.zeros_like(model.harm_classifier.weight),
    )
    assert torch.any(model.encoder.projection.weight.grad != 0)


def test_existing_checkpoint_helpers_roundtrip_nli_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    torch, model, _ = _load_fake_model(monkeypatch)
    pytest.importorskip("safetensors.torch")
    model.eval()
    with torch.no_grad():
        before = model(question="q1", candidate_text="candidate").protect_scores.clone()
    checkpoint = tmp_path / "nli-dual-head.safetensors"
    save_dual_head_checkpoint(model, checkpoint)

    with torch.no_grad():
        model.protect_classifier.weight.add_(4.0)
    load_dual_head_checkpoint(model, checkpoint)
    with torch.no_grad():
        after = model(question="q1", candidate_text="candidate").protect_scores
    assert torch.equal(after, before)


def test_nli_scores_feed_existing_zero_to_cap_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch, model, _ = _load_fake_model(monkeypatch)
    model.eval()
    with torch.no_grad():
        output = model(
            question=["question"] * 10,
            candidate_text=[f"candidate {rank}" for rank in range(1, 11)],
        )
    scores = {
        "q1": {
            f"ev-{rank:02d}": CandidateRiskScore(
                protect_score=float(output.protect_scores[rank - 1]),
                harm_score=float(output.harm_scores[rank - 1]),
            )
            for rank in range(1, 11)
        }
    }
    threshold = max(score.safe_score() or 0.0 for score in scores["q1"].values())
    candidates = CandidateSet(
        query_id="q1",
        candidates=tuple(
            EvidenceCandidate(
                evidence_id=f"ev-{rank:02d}",
                document_id=f"doc-{rank:02d}",
                chunk_id=f"chunk-{rank:02d}",
                text=f"candidate {rank}",
                source_uri=f"fixture://{rank}",
                retrieval_score=1.0 - rank / 100.0,
                retrieval_rank=rank,
            )
            for rank in range(1, 11)
        ),
    )
    result, trace = RiskControlledSelector(
        scores_by_query=scores,
        safe_threshold=threshold,
        max_delete=2,
    ).select_with_trace(Query(query_id="q1", text="question"), candidates, max_selected=10)

    assert 1 <= len(trace.dropped_evidence_ids) <= 2
    assert len(result.items) == 10 - len(trace.dropped_evidence_ids)
    assert {item.evidence_id for item in result.items} <= set(scores["q1"])
