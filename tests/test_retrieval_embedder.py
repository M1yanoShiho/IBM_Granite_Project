from __future__ import annotations

import pytest

from src.retrieval.embedder import Embedder


class FakeSentenceTransformer:
    created_model_ids: list[str] = []
    created_cache_folders: list[str | None] = []

    def __init__(self, model_id: str, cache_folder: str | None = None) -> None:
        self.created_model_ids.append(model_id)
        self.created_cache_folders.append(cache_folder)

    def encode(
        self,
        texts,
        convert_to_numpy: bool = False,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
    ):
        if isinstance(texts, str):
            texts = [texts]
        return [
            [float(len(text)), float(len(text.split()))]
            for text in texts
        ]


@pytest.fixture(autouse=True)
def reset_fake_model() -> None:
    FakeSentenceTransformer.created_model_ids.clear()
    FakeSentenceTransformer.created_cache_folders.clear()


def test_sentence_transformers_backend_embeds_documents(monkeypatch) -> None:
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = Embedder(
        backend="sentence-transformers",
        model_id="sentence-transformers/fake-model",
    )
    vectors = embedder.embed_documents(["alpha", "alpha beta"])

    assert FakeSentenceTransformer.created_model_ids == [
        "sentence-transformers/fake-model"
    ]
    assert vectors == [[5.0, 1.0], [10.0, 2.0]]


def test_granite_backend_uses_environment_default(monkeypatch) -> None:
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        FakeSentenceTransformer,
    )
    monkeypatch.setenv(
        "GRANITE_EMBEDDING_MODEL_ID",
        "ibm-granite/test-granite-embedding",
    )

    embedder = Embedder(backend="granite")
    vector = embedder.embed_query("granite retrieval")

    assert FakeSentenceTransformer.created_model_ids == [
        "ibm-granite/test-granite-embedding"
    ]
    assert vector == [17.0, 2.0]


def test_embedder_uses_model_cache_dir(monkeypatch) -> None:
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        FakeSentenceTransformer,
    )
    monkeypatch.setenv("MODEL_CACHE_DIR", "/tmp/hf-cache")

    Embedder(
        backend="sentence-transformers",
        model_id="sentence-transformers/fake-model",
    )

    assert FakeSentenceTransformer.created_cache_folders == ["/tmp/hf-cache"]


def test_embedder_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported embedding backend"):
        Embedder(backend="unknown", model_id="fake-model")

