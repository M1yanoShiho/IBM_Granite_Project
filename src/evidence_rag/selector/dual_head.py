"""Conservative dual-head scorer used by the adaptive Selector experiments.

The optional ML stack is deliberately imported only inside the functions that need it.  Importing
``evidence_rag.selector.dual_head`` must therefore remain safe in the project's lightweight core
environment, where neither torch nor transformers is installed.

The public scoring boundary is intentionally narrow: callers provide only ``question`` and
``candidate_text``.  Provenance, source-parent IDs, gold labels and retrieval metadata are useful
for splitting, supervision and evaluation, but allowing any of them through this boundary would
turn privileged audit information into a model feature.
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

DEFAULT_MODEL_ID = "cross-encoder/nli-deberta-v3-base"
MAX_LENGTH = 512
FINGERPRINT_SCHEMA_VERSION = "selector-dual-head-fingerprint-v1"
WeightNormalization = Literal["active_weight_sum", "active_count"]


@dataclass(frozen=True)
class TokenLengthAudit:
    """Token lengths for one question/candidate pair before and after truncation."""

    question_tokens: int
    candidate_tokens: int
    raw_pair_tokens: int
    encoded_pair_tokens: int
    candidate_truncated: bool


@dataclass(frozen=True)
class DualHeadOutput:
    """Logits and independent sigmoid scores for a batch of candidates."""

    protect_logits: Any
    harm_logits: Any
    protect_scores: Any
    harm_scores: Any


@dataclass(frozen=True)
class DualHeadLoss:
    """Masked losses and the number of effective labels seen by each head."""

    total: Any
    protect: Any
    harm: Any
    protect_count: int
    harm_count: int


@dataclass(frozen=True)
class DualHeadModelFingerprint:
    """Identity needed to bind an R004/R005 artifact to the exact initialized weights."""

    schema_version: str
    model_id: str
    revision: str | None
    max_length: int
    weights_sha256: str

    def to_dict(self) -> dict[str, str | int | None]:
        return asdict(self)


def _optional_module(name: str, *, extra: str = "granite") -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:  # pragma: no cover - depends on the optional runtime
        raise RuntimeError(
            f"dual-head Selector requires optional dependency {name!r}; "
            f"install the project with the {extra!r} extra"
        ) from exc


def _text_batch(value: str | Sequence[str], *, field: str) -> tuple[str, ...]:
    items = (value,) if isinstance(value, str) else tuple(value)
    if not items:
        raise ValueError(f"{field} must contain at least one text")
    if any(not isinstance(item, str) for item in items):
        raise TypeError(f"every {field} value must be a string")
    if any(not item.strip() for item in items):
        raise ValueError(f"every {field} value must be non-blank")
    return items


def _question_candidate_batches(
    *,
    question: str | Sequence[str],
    candidate_text: str | Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    questions = _text_batch(question, field="question")
    candidates = _text_batch(candidate_text, field="candidate_text")
    if len(questions) != len(candidates):
        raise ValueError(
            f"question and candidate_text batch sizes differ: {len(questions)} != {len(candidates)}"
        )
    return questions, candidates


def tokenize_question_candidates(
    tokenizer: Any,
    *,
    question: str | Sequence[str],
    candidate_text: str | Sequence[str],
    return_tensors: str | None = "pt",
    padding: bool | str = True,
) -> Mapping[str, Any]:
    """Tokenize the only legal scorer inputs with the frozen truncation contract.

    The question is sequence one and ``truncation='only_second'`` means that the candidate is
    the only sequence that may be shortened.  Passing ``truncation=True`` here would let a long
    candidate silently remove question tokens and change the task the scorer is solving.
    """

    questions, candidates = _question_candidate_batches(
        question=question,
        candidate_text=candidate_text,
    )
    encoded = tokenizer(
        list(questions),
        list(candidates),
        add_special_tokens=True,
        truncation="only_second",
        max_length=MAX_LENGTH,
        padding=padding,
        return_tensors=return_tensors,
    )
    if not isinstance(encoded, Mapping):
        raise TypeError("tokenizer must return a mapping of model input names to values")
    return encoded


def audit_token_lengths(
    tokenizer: Any,
    *,
    question: str | Sequence[str],
    candidate_text: str | Sequence[str],
) -> tuple[TokenLengthAudit, ...]:
    """Return per-pair truncation measurements without running the encoder.

    R004 uses these records to report the raw length distribution and the exact fraction whose
    candidate text was shortened.  The encoded call goes through the same helper as inference,
    so the audit cannot accidentally measure a different truncation policy.
    """

    questions, candidates = _question_candidate_batches(
        question=question,
        candidate_text=candidate_text,
    )
    question_ids = tokenizer(
        list(questions),
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_tensors=None,
    )["input_ids"]
    candidate_ids = tokenizer(
        list(candidates),
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_tensors=None,
    )["input_ids"]
    raw_pair_ids = tokenizer(
        list(questions),
        list(candidates),
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_tensors=None,
    )["input_ids"]
    encoded_pair_ids = tokenize_question_candidates(
        tokenizer,
        question=questions,
        candidate_text=candidates,
        return_tensors=None,
        padding=False,
    )["input_ids"]
    rows = zip(
        question_ids,
        candidate_ids,
        raw_pair_ids,
        encoded_pair_ids,
        strict=True,
    )
    return tuple(
        TokenLengthAudit(
            question_tokens=len(question_row),
            candidate_tokens=len(candidate_row),
            raw_pair_tokens=len(raw_row),
            encoded_pair_tokens=len(encoded_row),
            candidate_truncated=len(encoded_row) < len(raw_row),
        )
        for question_row, candidate_row, raw_row, encoded_row in rows
    )


def _dual_head_model_class(torch: Any) -> type[Any]:
    """Create the real ``nn.Module`` only after torch has been requested by a caller."""

    class _DualHeadModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(
            self,
            *,
            tokenizer: Any,
            encoder: Any,
            model_id: str,
            revision: str | None,
        ) -> None:
            super().__init__()
            hidden_size = int(encoder.config.hidden_size)
            if hidden_size <= 0:
                raise ValueError("encoder config.hidden_size must be positive")
            self.tokenizer = tokenizer
            self.encoder = encoder
            # These must remain two distinct parameter sets.  A two-logit softmax would make
            # protect and harm mutually exclusive, which is precisely not the experiment.
            self.protect_head = torch.nn.Linear(hidden_size, 1)
            self.harm_head = torch.nn.Linear(hidden_size, 1)
            self.model_id = model_id
            self.revision = revision
            self.max_length = MAX_LENGTH

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
            except StopIteration as exc:  # pragma: no cover - an encoder always has parameters
                raise RuntimeError("dual-head model unexpectedly has no parameters") from exc
            model_inputs = {
                name: value.to(device) if hasattr(value, "to") else value
                for name, value in encoded.items()
            }
            encoder_output = self.encoder(**model_inputs)
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
            pooled = hidden[:, 0, :]
            protect_logits = self.protect_head(pooled).squeeze(-1)
            harm_logits = self.harm_head(pooled).squeeze(-1)
            return DualHeadOutput(
                protect_logits=protect_logits,
                harm_logits=harm_logits,
                protect_scores=torch.sigmoid(protect_logits),
                harm_scores=torch.sigmoid(harm_logits),
            )

    return _DualHeadModel


def load_dual_head_model(
    model_id_or_path: str = DEFAULT_MODEL_ID,
    *,
    revision: str | None = None,
    identity_model_id: str | None = None,
    local_files_only: bool = False,
    device: str | None = None,
) -> Any:
    """Load one shared DeBERTa encoder and attach two independent one-logit heads.

    The returned object is a real ``torch.nn.Module``.  It starts in evaluation mode so R004's
    timing probe is deterministic with respect to dropout; a training caller must explicitly
    call ``model.train()`` before optimisation.
    """

    if not isinstance(model_id_or_path, str) or not model_id_or_path.strip():
        raise ValueError("model_id_or_path must be a non-blank string")
    if identity_model_id is not None and (
        not isinstance(identity_model_id, str) or not identity_model_id.strip()
    ):
        raise ValueError("identity_model_id must be a non-blank string when provided")
    torch = _optional_module("torch")
    transformers = _optional_module("transformers")
    load_kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "trust_remote_code": False,
    }
    if revision is not None:
        load_kwargs["revision"] = revision
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id_or_path, **load_kwargs)
    # Make which side of a long candidate survives deterministic.  The question remains intact
    # because the actual pair call separately freezes truncation="only_second".
    tokenizer.truncation_side = "right"
    encoder = transformers.AutoModel.from_pretrained(
        model_id_or_path,
        use_safetensors=True,
        **load_kwargs,
    )
    model_class = _dual_head_model_class(torch)
    model = model_class(
        tokenizer=tokenizer,
        encoder=encoder,
        model_id=identity_model_id or model_id_or_path,
        revision=revision,
    )
    if device is not None:
        model.to(device)
    model.eval()
    return model


def _validate_loss_tensor_shapes(
    *,
    protect_logits: Any,
    harm_logits: Any,
    protect_labels: Any,
    harm_labels: Any,
    protect_mask: Any,
    harm_mask: Any,
    protect_weights: Any | None = None,
    harm_weights: Any | None = None,
) -> None:
    expected = tuple(protect_logits.shape)
    named = {
        "harm_logits": harm_logits,
        "protect_labels": protect_labels,
        "harm_labels": harm_labels,
        "protect_mask": protect_mask,
        "harm_mask": harm_mask,
    }
    if protect_weights is not None:
        named["protect_weights"] = protect_weights
    if harm_weights is not None:
        named["harm_weights"] = harm_weights
    if len(expected) != 1:
        raise ValueError(f"protect_logits must be one-dimensional, got shape {expected}")
    for name, value in named.items():
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} shape {tuple(value.shape)} does not match logits {expected}")


def _masked_bce_component(
    torch: Any,
    *,
    logits: Any,
    labels: Any,
    mask: Any,
    weights: Any | None,
    weight_normalization: WeightNormalization,
    name: str,
) -> tuple[Any, int]:
    mask_values = mask.detach()
    is_binary_mask = (mask_values == 0) | (mask_values == 1)
    if not bool(torch.all(is_binary_mask).item()):
        raise ValueError(f"{name}_mask must contain only 0/1 or bool values")
    active = mask_values.to(device=logits.device, dtype=torch.bool)
    labels_on_device = labels.to(device=logits.device, dtype=logits.dtype)
    active_labels = labels_on_device[active]
    if active_labels.numel() and not bool(torch.all(torch.isfinite(active_labels)).item()):
        raise ValueError(f"{name}_labels contains a non-finite active value")
    if active_labels.numel() and not bool(
        torch.all((active_labels >= 0) & (active_labels <= 1)).item()
    ):
        raise ValueError(f"{name}_labels active values must lie in [0, 1]")

    # Masked positions may deliberately carry NaN as a visible "not labelled" sentinel.  Replace
    # them before BCE: BCE(NaN) * 0 is still NaN and is the common empty-mask failure this helper
    # exists to prevent.  Dividing by clamp_min(1) leaves a graph-connected exact zero when there
    # are no effective labels, so total.backward() remains legal and produces zero gradients.
    safe_labels = torch.where(active, labels_on_device, torch.zeros_like(logits))
    elementwise = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        safe_labels,
        reduction="none",
    )
    active_float = active.to(dtype=logits.dtype)
    if weights is None:
        active_weights = active_float
    else:
        weights_on_device = weights.to(device=logits.device, dtype=logits.dtype)
        observed_weights = weights_on_device[active]
        if observed_weights.numel() and not bool(
            torch.all(torch.isfinite(observed_weights)).item()
        ):
            raise ValueError(f"{name}_weights contains a non-finite active value")
        if observed_weights.numel() and not bool(torch.all(observed_weights > 0).item()):
            raise ValueError(f"{name}_weights active values must be strictly positive")
        # As with masked labels, an explicit NaN sentinel outside the active mask is harmless
        # only after replacement.  Multiplying NaN by zero would otherwise contaminate loss.
        active_weights = torch.where(active, weights_on_device, torch.zeros_like(logits))
    active_count = active_float.sum()
    denominator = (
        active_weights.sum() if weight_normalization == "active_weight_sum" else active_count
    )
    loss = (elementwise * active_weights).sum() / denominator.clamp_min(1)
    return loss, int(active.sum().detach().cpu().item())


def masked_dual_head_bce(
    *,
    protect_logits: Any,
    harm_logits: Any,
    protect_labels: Any,
    harm_labels: Any,
    protect_mask: Any,
    harm_mask: Any,
    protect_weights: Any | None = None,
    harm_weights: Any | None = None,
    weight_normalization: WeightNormalization = "active_weight_sum",
) -> DualHeadLoss:
    """Compute two independent masked BCE losses and their graph-connected sum.

    Optional weights are per-example.  ``active_weight_sum`` (the default) divides each head's
    weighted numerator by the sum of its active weights, preserving the original R004 behavior.
    ``active_count`` instead divides by the number of active labels.  R005 uses that explicit
    mode because its source/head/class weights are already normalized to have active-example
    mean one; dividing by their batch-local sum would otherwise cancel a class weight entirely
    in a single-class microbatch.

    Masked labels and weights may carry NaN sentinels.  They are replaced before arithmetic so
    an empty mask remains a graph-connected exact zero; every active weight must be finite and
    strictly positive.
    """

    if weight_normalization not in ("active_weight_sum", "active_count"):
        raise ValueError("weight_normalization must be 'active_weight_sum' or 'active_count'")
    torch = _optional_module("torch")
    _validate_loss_tensor_shapes(
        protect_logits=protect_logits,
        harm_logits=harm_logits,
        protect_labels=protect_labels,
        harm_labels=harm_labels,
        protect_mask=protect_mask,
        harm_mask=harm_mask,
        protect_weights=protect_weights,
        harm_weights=harm_weights,
    )
    protect_loss, protect_count = _masked_bce_component(
        torch,
        logits=protect_logits,
        labels=protect_labels,
        mask=protect_mask,
        weights=protect_weights,
        weight_normalization=weight_normalization,
        name="protect",
    )
    harm_loss, harm_count = _masked_bce_component(
        torch,
        logits=harm_logits,
        labels=harm_labels,
        mask=harm_mask,
        weights=harm_weights,
        weight_normalization=weight_normalization,
        name="harm",
    )
    return DualHeadLoss(
        total=protect_loss + harm_loss,
        protect=protect_loss,
        harm=harm_loss,
        protect_count=protect_count,
        harm_count=harm_count,
    )


def _tensor_bytes(torch: Any, tensor: Any) -> memoryview:
    cpu_tensor = tensor.detach().cpu().contiguous()
    # NumPy has no native bfloat16 dtype.  Viewing every tensor as bytes also makes the path
    # uniform and avoids a second dtype-dependent serialization format.
    byte_array = cpu_tensor.view(torch.uint8).numpy()
    return memoryview(byte_array).cast("B")


def weights_sha256(model: Any) -> str:
    """Hash names, shapes, dtypes and values of every tensor in ``state_dict``."""

    torch = _optional_module("torch")
    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name]
        spec = f"{name}|{tuple(tensor.shape)}|{tensor.dtype}".encode()
        raw = _tensor_bytes(torch, tensor)
        digest.update(len(spec).to_bytes(8, "big"))
        digest.update(spec)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def fingerprint_dual_head_model(model: Any) -> DualHeadModelFingerprint:
    """Return a machine-serializable identity for the current encoder and both heads."""

    if int(getattr(model, "max_length", -1)) != MAX_LENGTH:
        raise ValueError(f"model max_length must remain frozen at {MAX_LENGTH}")
    model_id = getattr(model, "model_id", None)
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model has no non-blank model_id")
    revision = getattr(model, "revision", None)
    if revision is not None and not isinstance(revision, str):
        raise TypeError("model revision must be a string or None")
    return DualHeadModelFingerprint(
        schema_version=FINGERPRINT_SCHEMA_VERSION,
        model_id=model_id,
        revision=revision,
        max_length=MAX_LENGTH,
        weights_sha256=weights_sha256(model),
    )


def save_dual_head_checkpoint(model: Any, path: Path) -> DualHeadModelFingerprint:
    """Write one complete sanity/training state dict as a write-once safetensors file."""

    torch = _optional_module("torch")
    safetensors = _optional_module("safetensors.torch")
    destination = Path(path)
    if destination.suffix != ".safetensors":
        raise ValueError("dual-head checkpoint path must end in .safetensors")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite dual-head checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: tensor.detach().to(device="cpu").contiguous()
        for name, tensor in model.state_dict().items()
    }
    if not state:
        raise ValueError("dual-head checkpoint state dict must not be empty")
    if any(not bool(torch.all(torch.isfinite(tensor)).item()) for tensor in state.values()):
        raise ValueError("dual-head checkpoint contains a non-finite tensor")
    safetensors.save_file(state, str(destination))
    return fingerprint_dual_head_model(model)


def load_dual_head_checkpoint(model: Any, path: Path) -> DualHeadModelFingerprint:
    """Strictly load a safetensors state dict and return the resulting full-state fingerprint."""

    safetensors = _optional_module("safetensors.torch")
    source = Path(path)
    if source.suffix != ".safetensors" or not source.is_file():
        raise ValueError(f"missing .safetensors dual-head checkpoint: {source}")
    state = safetensors.load_file(str(source), device="cpu")
    if not state:
        raise ValueError("dual-head checkpoint state dict must not be empty")
    model.load_state_dict(state, strict=True)
    return fingerprint_dual_head_model(model)
