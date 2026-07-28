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

# Production backend selection (G2). TRUE is the default because it won the G1
# six-arm triage on both axes that matter -- ASQA entailment recall 0.747 and
# derived citation precision 0.966, vs MiniCheck 0.620 / 0.886 -- and its
# probability sweep is usable where Granite's degenerates. MiniCheck stays as a
# CPU fallback (no GPU / no 21GB budget); DeBERTa is kept for parity with the
# earlier calibration rounds. Select with NLI_BACKEND=true|minicheck|deberta.
NLI_BACKEND_ENV = "NLI_BACKEND"
DEFAULT_NLI_BACKEND = "true"

# Operating threshold for the two binary (support-probability) backends, read off
# the G1 TRUE P(entail) sweep (docs/generator/verifier-triage-results-hpc.md):
# at 0.50 TRUE holds ASQA entailment recall 0.747 against a 0.007 hard-neutral
# false-positive rate -- the max-recall point that still keeps FP <= 0.007, i.e.
# the derived-citation-precision 0.966 operating point the G2 decision was made on.
DEFAULT_BINARY_ENTAIL_THRESHOLD = 0.50


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

    produces_contradiction = True
    """Three-way checkpoint: a returned ``contradiction`` is a real judgement, so
    ``ClaimVerification.contradicted`` is genuinely computed under this backend
    (unlike the binary backends below)."""

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


def _binary_entailment_label(p_entail: float, threshold: float) -> NLILabel:
    """Collapse a support probability onto the NLI vocabulary.

    Binary backends have no contradiction class, so they only ever emit
    ``entailment`` or ``neutral`` -- never ``contradiction``. Under such a
    backend ``ClaimVerification.contradicted`` is therefore **not computed**
    (it stays False for lack of a contradiction judgement, not because a
    contradiction was ruled out); this is the annotation the team agreed to and
    it is documented in ``docs/generator/verifier-backends.md``.
    """
    return "entailment" if p_entail >= threshold else "neutral"


class TrueNLIModel:
    """Production B1 backend: ``google/t5_xxl_true_nli_mixture`` (the TRUE judge
    ALCE's own citation evaluation uses). Ported verbatim from the ``true`` arm
    of ``scripts/verifier_triage.py`` -- the exact implementation the G1 numbers
    came from -- so the production path cannot drift from what was measured.

    11B params, ~21GB in bf16, GPU-only. Weights load lazily on the first
    ``classify`` so constructing a Verifier stays free in tests/CLI.

    Binary: emits ``entailment``/``neutral`` only, never ``contradiction`` --
    see ``_binary_entailment_label``.
    """

    produces_contradiction = False

    def __init__(
        self,
        model_id: str = "google/t5_xxl_true_nli_mixture",
        threshold: float = DEFAULT_BINARY_ENTAIL_THRESHOLD,
        max_length: int = 2048,
    ) -> None:
        self.model_id = model_id
        self.threshold = threshold
        self.max_length = max_length
        self._tokenizer: Any = None
        self._model: Any = None
        self._ids: tuple[int, int] | None = None

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is None:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")

            cache_dir = os.getenv("MODEL_CACHE_DIR") or None
            token = os.getenv("HUGGINGFACE_API_KEY") or None
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_id, cache_dir=cache_dir, token=token
            )
            self._model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id,
                cache_dir=cache_dir,
                token=token,
                dtype=torch.bfloat16,
                device_map="auto",
            )
            self._model.eval()
            # decoder vocabulary ids for '0' (not entailed) and '1' (entailed)
            self._ids = (
                int(self._tokenizer("0", add_special_tokens=False).input_ids[0]),
                int(self._tokenizer("1", add_special_tokens=False).input_ids[0]),
            )
        return self._tokenizer, self._model

    def score(self, premise: str, hypothesis: str) -> float:
        torch = importlib.import_module("torch")
        tokenizer, model = self._ensure_loaded()
        assert self._ids is not None
        text = f"premise: {premise} hypothesis: {hypothesis}"
        inputs = tokenizer(text, max_length=self.max_length, truncation=True, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        decoder_input_ids = torch.zeros((1, 1), dtype=torch.long, device=model.device)
        with torch.no_grad():
            logits = model(**inputs, decoder_input_ids=decoder_input_ids).logits.squeeze(1)
        pair = logits[0, torch.tensor(self._ids, device=logits.device)].float().cpu()
        return float(torch.softmax(pair, dim=-1)[1].item())

    def classify(self, premise: str, hypothesis: str) -> NLILabel:
        return _binary_entailment_label(self.score(premise, hypothesis), self.threshold)


class MiniCheckNLIModel:
    """CPU-fallback B1 backend: ``lytang/MiniCheck-Flan-T5-Large``, a
    grounding-specific verifier. Ported verbatim from the ``minicheck`` arm of
    ``scripts/verifier_triage.py``.

    Input format is the checkpoint's own: ``predict: {doc}</s>{claim}``, one
    decoder step, support probability = softmax over decoder ids (3 = unsupported,
    209 = supported). Binary -- no contradiction class.

    The checkpoint ships only ``pytorch_model.bin``; ``use_safetensors=False``
    stops transformers from stalling on a safetensors-conversion lookup, and the
    .bin path needs torch >= 2.6 (CVE-2025-32434), which the deployment already has.
    """

    produces_contradiction = False
    UNSUPPORTED_ID = 3
    SUPPORTED_ID = 209

    def __init__(
        self,
        model_id: str = "lytang/MiniCheck-Flan-T5-Large",
        threshold: float = DEFAULT_BINARY_ENTAIL_THRESHOLD,
        max_length: int = 2048,
    ) -> None:
        self.model_id = model_id
        self.threshold = threshold
        self.max_length = max_length
        self._tokenizer: Any = None
        self._model: Any = None

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is None:
            transformers = importlib.import_module("transformers")

            cache_dir = os.getenv("MODEL_CACHE_DIR") or None
            token = os.getenv("HUGGINGFACE_API_KEY") or None
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_id, cache_dir=cache_dir, token=token
            )
            self._model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id, cache_dir=cache_dir, token=token, use_safetensors=False
            )
            self._model.eval()
        return self._tokenizer, self._model

    def score(self, premise: str, hypothesis: str) -> float:
        torch = importlib.import_module("torch")
        tokenizer, model = self._ensure_loaded()
        text = "predict: " + tokenizer.eos_token.join([premise, hypothesis])
        inputs = tokenizer(text, max_length=self.max_length, truncation=True, return_tensors="pt")
        model_device = getattr(model, "device", None)
        if model_device is not None:
            inputs = {key: value.to(model_device) for key, value in inputs.items()}
            decoder_input_ids = torch.zeros((1, 1), dtype=torch.long, device=model_device)
        else:
            decoder_input_ids = torch.zeros((1, 1), dtype=torch.long)
        with torch.no_grad():
            logits = model(**inputs, decoder_input_ids=decoder_input_ids).logits.squeeze(1)
        pair = logits[:, torch.tensor([self.UNSUPPORTED_ID, self.SUPPORTED_ID], device=logits.device)]
        return float(torch.softmax(pair, dim=-1)[0, 1].item())

    def classify(self, premise: str, hypothesis: str) -> NLILabel:
        return _binary_entailment_label(self.score(premise, hypothesis), self.threshold)


def build_nli_model(name: str | None = None) -> NLIModel:
    """Construct the production NLI backend. Defaults to TRUE (the G2 decision);
    override with the ``NLI_BACKEND`` env var or an explicit name.

    No weights load here -- every backend defers loading to its first
    ``classify`` -- so this is safe to call on CPU and in tests that inject a fake.
    """
    resolved = (name or os.getenv(NLI_BACKEND_ENV) or DEFAULT_NLI_BACKEND).strip().lower()
    if resolved == "true":
        return TrueNLIModel()
    if resolved == "minicheck":
        return MiniCheckNLIModel()
    if resolved == "deberta":
        return DebertaNLIModel()
    raise ValueError(f"unknown NLI backend {resolved!r} (expected true|minicheck|deberta)")
