"""B1 -- NLI entailment judge used for claim attribution.

Wraps a dedicated cross-encoder NLI model instead of asking the generative
model to grade itself (plan section 3, "支持关系判定"): the whole point of the
verification pass is that it must not inherit the generator's own hallucination.
LLM-as-a-judge stays a v2 second opinion.

Direction is fixed and load-bearing: the evidence is the premise, the claim is
the hypothesis. "Evidence entails claim" is what attribution needs; the reverse
would accept a claim that merely implies the evidence.
"""

import importlib
import os
from typing import Any, Literal, Protocol

NLILabel = Literal["entailment", "neutral", "contradiction"]

DEFAULT_NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-base"


class NLIModel(Protocol):
    """Injectable entailment judge -- tests pass a deterministic fake so no
    model weights are loaded (mirrors ``TextGenerator`` in ``granite.py``)."""

    def classify(self, premise: str, hypothesis: str) -> NLILabel: ...


def normalize_nli_label(raw: str) -> NLILabel:
    """Map a model's own label vocabulary onto our three-way label.

    Checkpoints disagree on both casing and ordering (``ENTAILMENT``,
    ``entailment``, ``LABEL_0``...), so we read ``config.id2label`` and normalize
    here rather than hard-coding one checkpoint's index order.
    """
    label = raw.strip().lower()
    if label.startswith("entail"):
        return "entailment"
    if label.startswith("contradict"):
        return "contradiction"
    return "neutral"


class DebertaNLIModel:
    """Local Hugging Face cross-encoder NLI model (``cross-encoder/nli-deberta-v3-*``).

    Weights are loaded on the first ``classify`` call, not in ``__init__``, so
    constructing a Verifier stays free in tests and in CLI paths that end up
    never verifying anything.
    """

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.model_id = model_id or os.getenv("NLI_MODEL_ID") or DEFAULT_NLI_MODEL_ID
        self.device = device or os.getenv("LLM_DEVICE", "auto")
        self.max_length = max_length
        self._tokenizer: Any = None
        self._model: Any = None

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as exc:  # pragma: no cover - depends on optional runtime deps
            raise RuntimeError(
                "DebertaNLIModel requires the optional 'transformers' package."
            ) from exc

        token = os.getenv("HUGGINGFACE_API_KEY") or None
        cache_dir = os.getenv("MODEL_CACHE_DIR") or None
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_id,
            token=token,
            cache_dir=cache_dir,
        )
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            token=token,
            cache_dir=cache_dir,
            device_map="auto" if self.device == "auto" else None,
        )
        if self.device != "auto":
            model = model.to(self.device)
        model.eval()
        self._tokenizer, self._model = tokenizer, model
        return tokenizer, model

    def classify(self, premise: str, hypothesis: str) -> NLILabel:
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:  # pragma: no cover - depends on optional runtime deps
            raise RuntimeError("DebertaNLIModel requires the optional 'torch' package.") from exc

        tokenizer, model = self._ensure_loaded()
        inputs = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        model_device = getattr(model, "device", None)
        if model_device is not None:
            inputs = {key: value.to(model_device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
        predicted = int(logits[0].argmax().item())
        return normalize_nli_label(str(model.config.id2label[predicted]))
