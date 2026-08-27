"""NLI-aware dual-head scorer for the Lean Selector experiment.

This module restores the sequence-classification path that the legacy raw-CLS scorer discarded:
one shared DeBERTa encoder, the pretrained context pooler, one shared dropout view, and two
independent copies of the pretrained three-class NLI classifier.  Its public input boundary stays
identical to :mod:`evidence_rag.selector.dual_head`: only ``question`` and ``candidate_text`` may
reach the model.

Torch and Transformers remain optional dependencies and are imported only by the functions that
need them.  BCE, tokenization, checkpoint I/O, and the delete-only policy deliberately remain in
their existing modules rather than being duplicated here.
"""

from __future__ import annotations

import copy
import importlib
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from evidence_rag.selector.dual_head import (
    DEFAULT_MODEL_ID,
    MAX_LENGTH,
    DualHeadOutput,
    TokenLengthAudit,
    audit_token_lengths,
    tokenize_question_candidates,
)

NLI_ARCHITECTURE_VERSION = "nli-aware-independent-dual-head-v1"
NLI_LABEL_MAP: Mapping[str, int] = MappingProxyType(
    {"contradiction": 0, "entailment": 1, "neutral": 2}
)

_EXPECTED_ID2LABEL = {index: label for label, index in NLI_LABEL_MAP.items()}


def _optional_module(name: str, *, extra: str = "granite") -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:  # pragma: no cover - depends on the optional runtime
        raise RuntimeError(
            f"NLI dual-head Selector requires optional dependency {name!r}; "
            f"install the project with the {extra!r} extra"
        ) from exc


def _validate_model_identity(
    model_id_or_path: str,
    *,
    identity_model_id: str | None,
) -> None:
    if not isinstance(model_id_or_path, str) or not model_id_or_path.strip():
        raise ValueError("model_id_or_path must be a non-blank string")
    if identity_model_id is not None and (
        not isinstance(identity_model_id, str) or not identity_model_id.strip()
    ):
        raise ValueError("identity_model_id must be a non-blank string when provided")


def _validated_id2label(config: Any) -> dict[int, str]:
    raw = getattr(config, "id2label", None)
    if not isinstance(raw, Mapping):
        raise ValueError("NLI checkpoint config.id2label must be a mapping")
    normalized: dict[int, str] = {}
    for raw_index, raw_label in raw.items():
        if isinstance(raw_index, bool):
            raise ValueError("NLI checkpoint label IDs must be integer class indices")
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError("NLI checkpoint label IDs must be integer class indices") from exc
        if isinstance(raw_label, str):
            label = raw_label.strip().lower()
        else:
            raise ValueError("NLI checkpoint labels must be strings")
        if index in normalized:
            raise ValueError("NLI checkpoint config.id2label contains duplicate class indices")
        normalized[index] = label
    if normalized != _EXPECTED_ID2LABEL:
        raise ValueError(
            "NLI checkpoint label map must be 0=contradiction, 1=entailment, 2=neutral"
        )
    num_labels = getattr(config, "num_labels", 3)
    if isinstance(num_labels, bool) or int(num_labels) != 3:
        raise ValueError("NLI checkpoint must expose exactly three labels")
    return normalized


def _required_module(model: Any, name: str) -> Any:
    module = getattr(model, name, None)
    if module is None:
        raise ValueError(f"NLI checkpoint is missing required {name!r} module")
    return module


def _validate_classifier(classifier: Any) -> None:
    out_features = getattr(classifier, "out_features", None)
    if isinstance(out_features, bool) or out_features != 3:
        raise ValueError("NLI checkpoint classifier must output exactly three logits")


def _nli_dual_head_model_class(torch: Any) -> type[Any]:
    """Create the real ``nn.Module`` only after Torch has been requested."""

    class _NliDualHeadModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(
            self,
            *,
            tokenizer: Any,
            encoder: Any,
            pooler: Any,
            dropout: Any,
            classifier: Any,
            model_id: str,
            revision: str | None,
        ) -> None:
            super().__init__()
            self.tokenizer = tokenizer
            self.encoder = encoder
            self.pooler = pooler
            self.dropout = dropout
            self.protect_classifier = copy.deepcopy(classifier)
            self.harm_classifier = copy.deepcopy(classifier)
            self.model_id = model_id
            self.revision = revision
            self.max_length = MAX_LENGTH
            self.architecture_version = NLI_ARCHITECTURE_VERSION
            self.label_map = NLI_LABEL_MAP

        def tokenize(
            self,
            *,
            question: str | Sequence[str],
            candidate_text: str | Sequence[str],
            return_tensors: str | None = "pt",
            padding: bool | str = True,
        ) -> Mapping[str, Any]:
            return tokenize_question_candidates(
                self.tokenizer,
                question=question,
                candidate_text=candidate_text,
                return_tensors=return_tensors,
                padding=padding,
            )

        def token_length_audit(
            self,
            *,
            question: str | Sequence[str],
            candidate_text: str | Sequence[str],
        ) -> tuple[TokenLengthAudit, ...]:
            return audit_token_lengths(
                self.tokenizer,
                question=question,
                candidate_text=candidate_text,
            )

        @staticmethod
        def _hidden_state(encoder_output: Any) -> Any:
            hidden = getattr(encoder_output, "last_hidden_state", None)
            if hidden is None:
                try:
                    hidden = encoder_output[0]
                except (IndexError, KeyError, TypeError) as exc:
                    raise TypeError("encoder output has no last_hidden_state") from exc
            if getattr(hidden, "ndim", None) != 3:
                raise ValueError(
                    "encoder last_hidden_state must have shape [batch, tokens, hidden]"
                )
            return hidden

        @staticmethod
        def _three_logits(classifier: Any, pooled: Any, *, head: str) -> Any:
            logits = classifier(pooled)
            if getattr(logits, "ndim", None) != 2 or int(logits.shape[-1]) != 3:
                raise ValueError(f"{head} NLI classifier must return shape [batch, 3]")
            return logits

        def forward(
            self,
            *,
            question: str | Sequence[str],
            candidate_text: str | Sequence[str],
        ) -> DualHeadOutput:
            encoded = self.tokenize(
                question=question,
                candidate_text=candidate_text,
                return_tensors="pt",
                padding=True,
            )
            try:
                device = next(self.parameters()).device
            except StopIteration as exc:  # pragma: no cover - the backbone always has parameters
                raise RuntimeError("NLI dual-head model unexpectedly has no parameters") from exc
            model_inputs = {
                name: value.to(device) if hasattr(value, "to") else value
                for name, value in encoded.items()
            }
            hidden = self._hidden_state(self.encoder(**model_inputs))
            pooled = self.dropout(self.pooler(hidden))
            protect_nli_logits = self._three_logits(
                self.protect_classifier,
                pooled,
                head="protect",
            )
            harm_nli_logits = self._three_logits(
                self.harm_classifier,
                pooled,
                head="harm",
            )

            protect_logits = protect_nli_logits[:, 1] - torch.logsumexp(
                protect_nli_logits[:, [0, 2]], dim=-1
            )
            harm_logits = harm_nli_logits[:, 0] - torch.logsumexp(
                harm_nli_logits[:, [1, 2]], dim=-1
            )
            return DualHeadOutput(
                protect_logits=protect_logits,
                harm_logits=harm_logits,
                protect_scores=torch.sigmoid(protect_logits),
                harm_scores=torch.sigmoid(harm_logits),
            )

    return _NliDualHeadModel


def _resolve_device(torch: Any, device: str | None) -> str | None:
    if device == "auto":
        return "cuda" if bool(torch.cuda.is_available()) else "cpu"
    return device


def load_nli_dual_head_model(
    model_id_or_path: str = DEFAULT_MODEL_ID,
    *,
    revision: str | None = None,
    identity_model_id: str | None = None,
    local_files_only: bool = False,
    device: str | None = None,
) -> Any:
    """Load the pretrained DeBERTa NLI path and split its classifier into two heads.

    The two classifier copies start with identical values but independent parameter storage.  The
    returned model starts in evaluation mode; training callers must explicitly call ``train()``.
    """

    _validate_model_identity(model_id_or_path, identity_model_id=identity_model_id)
    torch = _optional_module("torch")
    transformers = _optional_module("transformers")
    load_kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "trust_remote_code": False,
    }
    if revision is not None:
        load_kwargs["revision"] = revision
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id_or_path, **load_kwargs)
    tokenizer.truncation_side = "right"
    source_model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_id_or_path,
        use_safetensors=True,
        **load_kwargs,
    )
    _validated_id2label(source_model.config)
    encoder = _required_module(source_model, "deberta")
    pooler = _required_module(source_model, "pooler")
    dropout = _required_module(source_model, "dropout")
    classifier = _required_module(source_model, "classifier")
    _validate_classifier(classifier)

    model_class = _nli_dual_head_model_class(torch)
    model = model_class(
        tokenizer=tokenizer,
        encoder=encoder,
        pooler=pooler,
        dropout=dropout,
        classifier=classifier,
        model_id=identity_model_id or model_id_or_path,
        revision=revision,
    )
    resolved_device = _resolve_device(torch, device)
    if resolved_device is not None:
        model.to(resolved_device)
    model.eval()
    return model


def pairwise_logistic_loss(
    *,
    protect_clean_logits: Any,
    protect_counterfactual_logits: Any,
    harm_clean_logits: Any,
    harm_counterfactual_logits: Any,
) -> Any:
    """Return the frozen clean/counterfactual pair objective averaged over aligned pairs."""

    torch = _optional_module("torch")
    expected = tuple(protect_clean_logits.shape)
    named = {
        "protect_counterfactual_logits": protect_counterfactual_logits,
        "harm_clean_logits": harm_clean_logits,
        "harm_counterfactual_logits": harm_counterfactual_logits,
    }
    if len(expected) != 1:
        raise ValueError(f"protect_clean_logits must be one-dimensional, got shape {expected}")
    for name, value in named.items():
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} shape {tuple(value.shape)} does not match {expected}")

    tensors = (
        protect_clean_logits,
        protect_counterfactual_logits,
        harm_clean_logits,
        harm_counterfactual_logits,
    )
    for name, value in zip(
        (
            "protect_clean_logits",
            "protect_counterfactual_logits",
            "harm_clean_logits",
            "harm_counterfactual_logits",
        ),
        tensors,
        strict=True,
    ):
        if not bool(torch.all(torch.isfinite(value)).item()):
            raise ValueError(f"{name} must contain only finite values")

    if not expected[0]:
        return sum((value.sum() for value in tensors), start=protect_clean_logits.sum() * 0) * 0

    protect_direction = torch.nn.functional.softplus(
        -(protect_clean_logits - protect_counterfactual_logits)
    )
    harm_direction = torch.nn.functional.softplus(-(harm_counterfactual_logits - harm_clean_logits))
    return (0.5 * (protect_direction + harm_direction)).mean()
