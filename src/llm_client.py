"""LLM client wrapper for IBM Granite and baseline models.

Provides a single, uniform interface (:class:`LLMClient`) for sending prompts
to different model providers so the evaluation harness does not need to know
provider-specific details.

Primary target: **IBM Granite** via watsonx.ai (using ``langchain-ibm``).
Baselines: open-source models (e.g. via Hugging Face) for comparison.

Credentials are read from environment variables (see ``.env.example``):

- ``WATSONX_API_KEY``
- ``WATSONX_PROJECT_ID``
- ``WATSONX_URL``
- ``HUGGINGFACE_API_KEY`` (optional, for baselines)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Load variables from a local .env file if present.
load_dotenv()


@dataclass
class GenerationConfig:
    """Decoding parameters shared across providers.

    Attributes
    ----------
    max_new_tokens:
        Maximum number of tokens to generate.
    temperature:
        Sampling temperature; 0.0 for deterministic retrieval evaluation.
    top_p:
        Nucleus sampling probability mass.
    """

    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0


class LLMClient:
    """Uniform wrapper around a chat/generation model.

    Parameters
    ----------
    provider:
        One of ``"watsonx"`` (IBM Granite) or ``"huggingface"`` (baselines).
    model_id:
        Provider-specific model identifier. Defaults to the value of the
        ``GRANITE_MODEL_ID`` env var when ``provider == "watsonx"``.
    config:
        Decoding parameters. Defaults to :class:`GenerationConfig`.
    """

    def __init__(
        self,
        provider: str = "watsonx",
        model_id: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> None:
        self.provider = provider
        self.model_id = model_id or os.getenv("GRANITE_MODEL_ID")
        self.config = config or GenerationConfig()
        self._client = self._init_client()

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    def _init_client(self):
        """Instantiate the underlying provider client.

        For ``watsonx`` this should construct a ``WatsonxLLM`` (from
        ``langchain_ibm``) using the API key, project ID, and URL from the
        environment. For ``huggingface`` it should construct the appropriate
        baseline client.

        Returns
        -------
        object
            The initialized provider-specific client.
        """
        raise NotImplementedError(
            "TODO: initialize the watsonx / huggingface client from env vars."
        )

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(self, prompt: str) -> str:
        """Send a single prompt to the model and return its text response.

        Parameters
        ----------
        prompt:
            The full prompt (typically the haystack context plus the probe
            question).

        Returns
        -------
        str
            The model's generated answer.
        """
        raise NotImplementedError("TODO: call the underlying client.")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"LLMClient(provider={self.provider!r}, model_id={self.model_id!r})"
