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

    def _encode(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=False,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in vectors]

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
