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


# --- instruction-prefix tests ---


def test_no_prefix_by_default(monkeypatch) -> None:
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeSentenceTransformer)
    embedder = Embedder(backend="sentence-transformers", model_id="fake")
    assert embedder.query_prefix == ""
    assert embedder.doc_prefix == ""
    assert embedder.embed_query("hello") == [5.0, 1.0]


def test_query_prefix_prepended_to_embed_query(monkeypatch) -> None:
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeSentenceTransformer)
    embedder = Embedder(
        backend="sentence-transformers",
        model_id="fake",
        query_prefix="query: ",
    )
    # "query: hello" -> len=12, words=2
    assert embedder.embed_query("hello") == [12.0, 2.0]


def test_doc_prefix_prepended_to_embed_documents(monkeypatch) -> None:
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeSentenceTransformer)
    embedder = Embedder(
        backend="sentence-transformers",
        model_id="fake",
        doc_prefix="passage: ",
    )
    # "passage: hello" -> len=14, words=2
    assert embedder.embed_documents(["hello"]) == [[14.0, 2.0]]


def test_query_prefix_does_not_affect_embed_documents(monkeypatch) -> None:
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeSentenceTransformer)
    embedder = Embedder(
        backend="sentence-transformers",
        model_id="fake",
        query_prefix="query: ",
    )
    # doc_prefix is empty, so documents are unchanged
    assert embedder.embed_documents(["hello"]) == [[5.0, 1.0]]


def test_granite_backend_with_instruction_prefix(monkeypatch) -> None:
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setenv("GRANITE_EMBEDDING_MODEL_ID", "ibm-granite/test-granite-embedding")
    prefix = "Represent the query for retrieval: "
    embedder = Embedder(backend="granite", query_prefix=prefix)
    full = prefix + "what is DNA"
    assert embedder.embed_query("what is DNA") == [float(len(full)), float(len(full.split()))]


# --- token-aware-chunking support: tokenizer + max_seq_length ---


def test_embedder_exposes_tokenizer_and_max_seq_length(monkeypatch) -> None:
    class FakeSTWithTokenizer:
        tokenizer = "FAKE-TOKENIZER"

        def __init__(self, model_id, cache_folder=None) -> None:
            pass

        def get_max_seq_length(self):
            return 256

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSTWithTokenizer
    )
    embedder = Embedder(backend="sentence-transformers", model_id="fake")

    assert embedder.tokenizer == "FAKE-TOKENIZER"
    assert embedder.max_seq_length == 256


def test_embedder_max_seq_length_falls_back_to_attribute(monkeypatch) -> None:
    class FakeSTAttrOnly:
        tokenizer = "T"
        max_seq_length = 128  # no get_max_seq_length() method

        def __init__(self, model_id, cache_folder=None) -> None:
            pass

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeSTAttrOnly)
    embedder = Embedder(backend="sentence-transformers", model_id="fake")

    assert embedder.max_seq_length == 128


def test_different_model_ids_are_recorded(monkeypatch) -> None:
    # Verify that distinct model_id values are passed through to the underlying
    # SentenceTransformer — the mechanism that lets us compare Granite variants.
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeSentenceTransformer)
    Embedder(backend="granite", model_id="ibm-granite/granite-embedding-english-r2")
    Embedder(backend="granite", model_id="ibm-granite/granite-embedding-small-english-r2")
    assert FakeSentenceTransformer.created_model_ids == [
        "ibm-granite/granite-embedding-english-r2",
        "ibm-granite/granite-embedding-small-english-r2",
    ]

