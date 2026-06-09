"""Embedding interface for dense retrieval.

A uniform wrapper that turns text into dense vectors. The primary target is an
**IBM Granite embedding model** (via watsonx.ai); a ``sentence-transformers``
model is supported as an open-source comparison point.

Keeping embedding behind a single interface lets the retriever and the
evaluation harness swap embedding backends without code changes — important
for the system-vs-baseline comparison.
"""

from __future__ import annotations

from typing import List, Sequence


class Embedder:
    """Uniform text-embedding interface.

    Parameters
    ----------
    backend:
        ``"granite"`` (IBM watsonx.ai embedding model) or
        ``"sentence-transformers"`` (open-source baseline embeddings).
    model_id:
        Backend-specific embedding model identifier.
    """

    def __init__(self, backend: str = "granite", model_id: str | None = None) -> None:
        self.backend = backend
        self.model_id = model_id
        self._model = self._init_model()

    def _init_model(self):
        """Instantiate the underlying embedding model from config / env."""
        raise NotImplementedError(
            "TODO: initialise Granite (watsonx) or sentence-transformers embeddings."
        )

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch of documents/chunks into dense vectors."""
        raise NotImplementedError("TODO: batch-embed documents.")

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query into a dense vector."""
        raise NotImplementedError("TODO: embed a query.")
