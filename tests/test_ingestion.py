from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import faiss

from src.ingestion.chunker import Chunk, chunk_document
from src.ingestion.loaders import load_text_file, load_documents
from src.ingestion.indexer import FaissIndex


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