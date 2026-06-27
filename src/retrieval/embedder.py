"""Embedding interface for dense retrieval.

A uniform wrapper that turns text into dense vectors. The primary target is an
**IBM Granite embedding model** (open-source, self-hosted from Hugging Face); a
``sentence-transformers`` model is supported as an open-source comparison point.

Keeping embedding behind a single interface lets the retriever and the
evaluation harness swap embedding backends without code changes — important
for the system-vs-baseline comparison.
"""

from __future__ import annotations

import os
from typing import List, Sequence

import numpy as np

DEFAULT_GRANITE_EMBEDDING_MODEL_ID = "ibm-granite/granite-embedding-english-r2"
DEFAULT_BASELINE_EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Uniform text-embedding interface.

    Parameters
    ----------
    backend:
        ``"granite"`` (IBM Granite embedding model, self-hosted from Hugging
        Face) or ``"sentence-transformers"`` (open-source baseline embeddings).
    model_id:
        Backend-specific embedding model identifier.
    query_prefix:
        String prepended to every query before encoding. Useful for
        instruction-tuned models that distinguish query vs. passage roles.
        ``granite-embedding-english-r2`` model card shows no recommended
        prefix, so the default is empty.
    doc_prefix:
        String prepended to every document/passage before encoding.
    """

    def __init__(
        self,
        backend: str = "granite",
        model_id: str | None = None,
        query_prefix: str = "",
        doc_prefix: str = "",
    ) -> None:
        self._validate_backend(backend)
        self.backend = backend
        self.model_id = model_id or self._default_model_id(backend)
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self._model = self._init_model()

    @staticmethod
    def _validate_backend(backend: str) -> None:
        if backend not in {"granite", "sentence-transformers"}:
            raise ValueError(
                "Unsupported embedding backend. Expected 'granite' or "
                "'sentence-transformers'."
            )

    @staticmethod
    def _default_model_id(backend: str) -> str:
        if backend == "granite":
            return (
                os.getenv("GRANITE_EMBEDDING_MODEL_ID")
                or DEFAULT_GRANITE_EMBEDDING_MODEL_ID
            )
        if backend == "sentence-transformers":
            return (
                os.getenv("BASELINE_EMBEDDING_MODEL_ID")
                or DEFAULT_BASELINE_EMBEDDING_MODEL_ID
            )
        raise AssertionError(f"Unexpected backend after validation: {backend}")

    def _init_model(self):
        """Instantiate the underlying sentence-transformers-compatible model."""
        from sentence_transformers import SentenceTransformer

        cache_folder = os.getenv("MODEL_CACHE_DIR") or None
        return SentenceTransformer(self.model_id, cache_folder=cache_folder)

    @property
    def tokenizer(self):
        """The model's underlying Hugging Face tokenizer.

        Exposed so token-aware chunking (``src.ingestion.chunker``) can split
        documents on the *same* sub-word tokens this model encodes, instead of
        on whitespace words — keeping chunk lengths aligned with the model.
        """
        return self._model.tokenizer

    @property
    def max_seq_length(self) -> int | None:
        """Max input length, in tokens, the model encodes before truncating.

        Returns ``None`` if the model does not advertise a limit. Used to cap
        token-aware chunk sizes so chunks are never silently truncated at encode
        time.
        """
        getter = getattr(self._model, "get_max_seq_length", None)
        if getter is not None:
            value = getter()
            if value:
                return int(value)
        value = getattr(self._model, "max_seq_length", None)
        return int(value) if value else None

    def _encode(self, texts: Sequence[str]) -> List[List[float]]:
        # convert_to_numpy lets sentence-transformers return one ndarray via its
        # fast C path; np.asarray(...).tolist() then converts the whole batch to
        # plain Python floats in one shot (cheaper than a per-element map(float)
        # over a list of tensors). np.asarray also normalises the stand-in list
        # output used in tests, so both paths yield List[List[float]].
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors).tolist()

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch of documents/chunks into dense vectors."""
        if not texts:
            return []
        prefixed = [self.doc_prefix + t for t in texts] if self.doc_prefix else list(texts)
        return self._encode(prefixed)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query into a dense vector."""
        prefixed = self.query_prefix + text if self.query_prefix else text
        return self._encode([prefixed])[0]
