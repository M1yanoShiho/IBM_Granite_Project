"""Fixed, lazy DeBERTa adapter for contradiction probabilities."""

from __future__ import annotations

import importlib
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

from evidence_rag.selector.reliability_mis import SelectorBackendError

DEFAULT_CONTRADICTION_MODEL_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
DEFAULT_CONTRADICTION_MODEL_REVISION = "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 32


def _contradiction_index(id2label: object) -> int:
    if not isinstance(id2label, Mapping):
        raise SelectorBackendError("NLI model config has no valid id2label mapping")

    matches: list[int] = []
    for raw_index, raw_label in id2label.items():
        normalized = str(raw_label).strip().casefold().replace("_", " ")
        if normalized.startswith("contradict"):
            try:
                matches.append(int(raw_index))
            except (TypeError, ValueError) as exc:
                raise SelectorBackendError(
                    "NLI contradiction label has a non-integer index"
                ) from exc
    if len(matches) != 1:
        raise SelectorBackendError(
            "NLI model must expose exactly one contradiction label in config.id2label"
        )
    return matches[0]


def _probability_value(raw_value: object) -> float:
    value = raw_value.item() if hasattr(raw_value, "item") else raw_value
    try:
        probability = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise SelectorBackendError("NLI backend returned a non-numeric probability") from exc
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise SelectorBackendError("NLI backend returned an invalid probability")
    return probability


class DebertaContradictionScorer:
    """Map ordered statement pairs to contradiction probabilities.

    Weight loading is delayed until the first non-empty call.  The production
    checkpoint and revision are fixed by defaults; composition does not expose
    either value as an experiment parameter.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_CONTRADICTION_MODEL_ID,
        model_revision: str = DEFAULT_CONTRADICTION_MODEL_REVISION,
        *,
        device: str | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if not model_id:
            raise ValueError("model_id must not be empty")
        if not model_revision:
            raise ValueError("model_revision must not be empty")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_id = model_id
        self.model_revision = model_revision
        self.device = device or os.getenv("LLM_DEVICE", "auto")
        self.max_length = max_length
        self.batch_size = batch_size
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._contradiction_label_index: int | None = None

    def _ensure_loaded(self) -> tuple[Any, Any, Any, int]:
        if (
            self._tokenizer is not None
            and self._model is not None
            and self._torch is not None
            and self._contradiction_label_index is not None
        ):
            return (
                self._tokenizer,
                self._model,
                self._torch,
                self._contradiction_label_index,
            )

        try:
            transformers = importlib.import_module("transformers")
            torch = importlib.import_module("torch")
            token = os.getenv("HUGGINGFACE_API_KEY") or None
            cache_dir = os.getenv("MODEL_CACHE_DIR") or None
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_id,
                revision=self.model_revision,
                token=token,
                cache_dir=cache_dir,
            )
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                revision=self.model_revision,
                token=token,
                cache_dir=cache_dir,
                device_map="auto" if self.device == "auto" else None,
            )
            if self.device != "auto":
                model = model.to(self.device)
            model.eval()
            label_index = _contradiction_index(getattr(model.config, "id2label", None))
        except SelectorBackendError:
            raise
        except Exception as exc:
            raise SelectorBackendError("failed to load DeBERTa contradiction backend") from exc

        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        self._contradiction_label_index = label_index
        return tokenizer, model, torch, label_index

    @staticmethod
    def _move_inputs(inputs: Mapping[str, Any], model: Any) -> dict[str, Any]:
        model_device = getattr(model, "device", None)
        if model_device is None:
            return dict(inputs)
        return {
            key: value.to(model_device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> tuple[float, ...]:
        ordered_pairs = tuple(pairs)
        if not ordered_pairs:
            return ()

        tokenizer, model, torch, label_index = self._ensure_loaded()
        probabilities: list[float] = []
        try:
            for start in range(0, len(ordered_pairs), self.batch_size):
                batch = ordered_pairs[start : start + self.batch_size]
                premises = [premise for premise, _hypothesis in batch]
                hypotheses = [hypothesis for _premise, hypothesis in batch]
                raw_inputs = tokenizer(
                    premises,
                    hypotheses,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                if not isinstance(raw_inputs, Mapping):
                    raise SelectorBackendError("NLI tokenizer returned a non-mapping batch")
                inputs = self._move_inputs(raw_inputs, model)
                with torch.inference_mode():
                    logits = model(**inputs).logits
                    probability_rows = list(torch.softmax(logits, dim=-1))
                if len(probability_rows) != len(batch):
                    raise SelectorBackendError("NLI backend returned the wrong batch length")
                probabilities.extend(
                    _probability_value(row[label_index]) for row in probability_rows
                )
        except SelectorBackendError:
            raise
        except Exception as exc:
            raise SelectorBackendError("DeBERTa contradiction inference failed") from exc

        if len(probabilities) != len(ordered_pairs):
            raise SelectorBackendError("NLI backend returned the wrong number of probabilities")
        return tuple(probabilities)
