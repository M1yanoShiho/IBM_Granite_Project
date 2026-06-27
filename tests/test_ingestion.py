from __future__ import annotations

import re
import tempfile
from pathlib import Path

import numpy as np
import faiss
import pytest

from src.ingestion.chunker import Chunk, chunk_document
from src.ingestion.loaders import load_text_file, load_documents
from src.ingestion.indexer import FaissIndex, VectorIndexer


def test_chunk_document_chunk_id_format() -> None:
    chunks = chunk_document("doc1", "word " * 600, chunk_size=512, chunk_overlap=50)
    assert chunks[0].chunk_id == "doc1::0"
    assert chunks[1].chunk_id == "doc1::1"


def test_chunk_document_first_chunk_length() -> None:
    chunks = chunk_document("doc1", "word " * 600, chunk_size=512, chunk_overlap=50)
    assert len(chunks[0].text.split()) == 512


def test_chunk_document_empty_text_returns_empty_list() -> None:
    assert chunk_document("doc1", "") == []


def test_chunk_document_short_text_returns_one_chunk() -> None:
    chunks = chunk_document("doc1", "hello world", chunk_size=512, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].doc_id == "doc1"


def test_chunk_document_overlap_ge_size_raises_instead_of_hanging() -> None:
    # overlap >= size makes the window step non-positive; guard against the
    # infinite loop rather than hang (these knobs are user-exposed via sweeps).
    with pytest.raises(ValueError, match="must be smaller than"):
        chunk_document("doc1", "word " * 10, chunk_size=256, chunk_overlap=256)


def test_chunk_document_invalid_size_raises() -> None:
    with pytest.raises(ValueError, match="chunk_size must be at least 1"):
        chunk_document("doc1", "word " * 10, chunk_size=0, chunk_overlap=0)


class FakeOffsetTokenizer:
    """Stand-in HF fast tokenizer: each whitespace-run word is one token, with a
    character offset span — enough to exercise token-aware chunking offline."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        return {"offset_mapping": offsets}


def test_chunk_document_token_mode_slices_by_token_offsets() -> None:
    # 5 tokens, size 2 / overlap 0 -> windows [a b], [c d], [e]; each chunk text
    # is sliced from the original string at the token char boundaries.
    chunks = chunk_document(
        "d", "a b c d e", chunk_size=2, chunk_overlap=0, tokenizer=FakeOffsetTokenizer()
    )
    assert [c.text for c in chunks] == ["a b", "c d", "e"]
    assert chunks[0].chunk_id == "d::0"


def test_chunk_document_token_mode_respects_overlap() -> None:
    # size 2 / overlap 1 -> step 1 -> sliding windows [a b], [b c], [c d], [d].
    chunks = chunk_document(
        "d", "a b c d", chunk_size=2, chunk_overlap=1, tokenizer=FakeOffsetTokenizer()
    )
    assert [c.text for c in chunks] == ["a b", "b c", "c d", "d"]


def test_chunk_document_token_mode_empty_text_returns_empty() -> None:
    assert chunk_document("d", "", tokenizer=FakeOffsetTokenizer()) == []


def test_chunk_document_token_mode_requires_offset_support() -> None:
    def slow_tokenizer(text, add_special_tokens=False, return_offsets_mapping=False):
        raise NotImplementedError("slow tokenizers can't return offsets")

    with pytest.raises(ValueError, match="fast tokenizer"):
        chunk_document("d", "a b c", chunk_size=2, chunk_overlap=0, tokenizer=slow_tokenizer)



def test_load_text_file_returns_content() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("hello world")
        tmp_path = f.name
    assert load_text_file(tmp_path) == "hello world"


def test_load_documents_yields_txt_and_md() -> None:
    with tempfile.TemporaryDirectory() as d:
        Path(d, "a.txt").write_text("text a", encoding="utf-8")
        Path(d, "b.md").write_text("text b", encoding="utf-8")
        Path(d, "c.pdf").write_text("skip me", encoding="utf-8")
        results = dict(load_documents(d))
    assert results == {"a": "text a", "b": "text b"}



def test_faiss_index_search_returns_retrieved_chunks() -> None:
    chunks = [
        Chunk(chunk_id="doc1::0", doc_id="doc1", text="granite retrieval"),
        Chunk(chunk_id="doc1::1", doc_id="doc1", text="bm25 baseline"),
    ]
    dim = 4
    raw_index = faiss.IndexFlatIP(dim)
    vectors = np.array([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0]], dtype="float32")
    raw_index.add(vectors)

    index = FaissIndex(raw_index, chunks)
    results = index.search([1.0, 0.0, 0.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0].doc_id == "doc1"
    assert results[0].score > results[1].score


def test_vector_indexer_build_rejects_empty_chunks() -> None:
    # Empty chunk list must raise a clear error, not the cryptic
    # "IndexError: tuple index out of range" from reading dim off a 0-row array.
    indexer = VectorIndexer(None)  # embedder unused: we fail before embedding
    with pytest.raises(ValueError, match="zero chunks"):
        indexer.build([])


def test_vector_indexer_save_load_round_trips_on_non_ascii_path() -> None:
    # FAISS's C++ narrow-char file API cannot open non-ASCII paths on Windows;
    # VectorIndexer.save/load must round-trip through unicode-safe Python I/O.
    raw_index = faiss.IndexFlatIP(4)
    raw_index.add(np.array([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0]], dtype="float32"))
    chunks = [
        Chunk(chunk_id="d::0", doc_id="d", text="alpha"),
        Chunk(chunk_id="d::1", doc_id="d", text="beta"),
    ]
    indexer = VectorIndexer(None)  # embedder is unused by save/load

    with tempfile.TemporaryDirectory() as tmp:
        nonascii_dir = Path(tmp) / "索引（无）"
        nonascii_dir.mkdir()
        stem = str(nonascii_dir / "idx")

        indexer.save(FaissIndex(raw_index, chunks), stem)
        loaded = indexer.load(stem)

    results = loaded.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert results[0].doc_id == "d"