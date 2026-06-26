"""Self-hosted Granite LLM client for the RAG generation layer.

IBM confirmed there is **no watsonx.ai API access** for this project. Instead we
use the **open-source IBM Granite models (Apache 2.0) downloaded from Hugging
Face** and run them locally / on the university HPC. This module wraps a local
``transformers`` causal-LM behind a single uniform interface
(:class:`LLMClient`) so the rest of the system stays independent of inference
details.

Roles (see ``.env.example``):

- **Generation / RAG answers** — a Granite *generative* model, e.g.
  ``ibm-granite/granite-4.1-3b`` (lightweight, fine on CPU for development) up to
  ``ibm-granite/granite-4.1-8b-base`` (needs a GPU; ~512K-token context, useful
  for the long-context "needle" experiments).
- **Dense-retrieval embeddings** are handled separately in
  ``src/retrieval/embedder.py`` — *not* here.

Configuration is read from environment variables (see ``.env.example``):

- ``GRANITE_MODEL_ID``     Hugging Face repo id of the generative model.
- ``HUGGINGFACE_API_KEY``  optional; only for gated/private models or rate limits.
- ``MODEL_CACHE_DIR``      optional; where to cache downloaded weights
                           (point at shared scratch on the HPC to avoid re-downloads).
- ``LLM_DEVICE``           ``"auto"`` (let ``accelerate`` place it), ``"cuda"``, or ``"cpu"``.

Verified: this client has been run against real Granite models on the University
HPC (BluePebble) — ``granite-4.1-3b`` and ``granite-4.1-8b`` both load and generate
correctly (see ``docs/hpc-deployment.md`` / ``docs/dev-log.md``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Load variables from a local .env file if present.
load_dotenv()

# Lightweight default — small enough to develop against on a CPU.
DEFAULT_MODEL_ID = "ibm-granite/granite-4.1-3b"


@dataclass
class GenerationConfig:
    """Decoding parameters.

    Attributes
    ----------
    max_new_tokens:
        Maximum number of tokens to generate.
    temperature:
        Sampling temperature; ``0.0`` selects greedy/deterministic decoding,
        which is what we want for reproducible evaluation.
    top_p:
        Nucleus sampling probability mass (only used when ``temperature > 0``).
    """

    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0


class LLMClient:
    """Uniform wrapper around a locally hosted Granite generative model.

    Parameters
    ----------
    model_id:
        Hugging Face repo id. Defaults to the ``GRANITE_MODEL_ID`` env var, then
        to :data:`DEFAULT_MODEL_ID`.
    config:
        Decoding parameters. Defaults to :class:`GenerationConfig`.
    device:
        ``"auto"`` | ``"cuda"`` | ``"cpu"``. Defaults to the ``LLM_DEVICE`` env
        var, then ``"auto"``.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_id = model_id or os.getenv("GRANITE_MODEL_ID") or DEFAULT_MODEL_ID
        self.config = config or GenerationConfig()
        self.device = device or os.getenv("LLM_DEVICE", "auto")
        self._tokenizer = None
        self._client = self._init_client()

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    def _init_client(self):
        """Download (if needed) and load the tokenizer + model from Hugging Face.

        Returns the loaded model; the tokenizer is stored on ``self._tokenizer``.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = os.getenv("HUGGINGFACE_API_KEY") or None
        cache_dir = os.getenv("MODEL_CACHE_DIR") or None

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, token=token, cache_dir=cache_dir
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            token=token,
            cache_dir=cache_dir,
            dtype="auto",
            # "auto" lets accelerate place layers across available devices;
            # an explicit device is moved with .to() below instead.
            device_map="auto" if self.device == "auto" else None,
        )
        if self.device != "auto":
            model = model.to(self.device)
        model.eval()
        return model

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(self, prompt: str) -> str:
        """Send a single prompt to the model and return only the new text.

        Parameters
        ----------
        prompt:
            The full prompt (e.g. retrieved context plus the question).

        Returns
        -------
        str
            The model's generated answer (input echo stripped).
        """
        import torch

        tok = self._tokenizer
        # Instruct models carry a chat template; base models do not — fall back
        # to feeding the raw prompt text in that case.
        if getattr(tok, "chat_template", None):
            encoded = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
            # transformers 4.x returns a bare tensor here; 5.x returns a
            # BatchEncoding (dict-like). Unwrap to the input_ids tensor so the
            # model.generate(input_ids=...) call below gets a tensor on both.
            input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        else:
            input_ids = tok(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self._client.device)

        gen_kwargs = {"max_new_tokens": self.config.max_new_tokens}
        if self.config.temperature > 0:
            gen_kwargs.update(
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self._client.generate(input_ids=input_ids, **gen_kwargs)

        # Decode only the newly generated tokens, not the echoed prompt.
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        return tok.decode(new_tokens, skip_special_tokens=True).strip()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"LLMClient(model_id={self.model_id!r}, device={self.device!r})"
