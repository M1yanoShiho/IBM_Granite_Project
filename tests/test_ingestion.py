from __future__ import annotations

import re
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import faiss
import pytest

from src.ingestion.chunker import Chunk, chunk_document
from src.ingestion.loaders import (
    LoadedDocument,
    caption_images,
    load_directory,
    load_documents,
    load_pdf,
    load_text_file,
)
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


class _FakeEmbedder:
    """Embedder stub returning fixed vectors per text — no model, exact control."""

    def __init__(self, vectors: dict) -> None:
        self._vectors = vectors

    def embed_documents(self, texts):
        return [self._vectors[t] for t in texts]

    def embed_query(self, text):
        return self._vectors[text]


def _onehot_chunks_and_embedder():
    # Three orthogonal one-hot docs; a query equal to beta's vector must rank d2.
    chunks = [
        Chunk(chunk_id="d1::0", doc_id="d1", text="alpha"),
        Chunk(chunk_id="d2::0", doc_id="d2", text="beta"),
        Chunk(chunk_id="d3::0", doc_id="d3", text="gamma"),
    ]
    embedder = _FakeEmbedder(
        {
            "alpha": [1.0, 0.0, 0.0, 0.0],
            "beta": [0.0, 1.0, 0.0, 0.0],
            "gamma": [0.0, 0.0, 1.0, 0.0],
        }
    )
    return chunks, embedder


def test_vector_indexer_defaults_to_exact_flat_index() -> None:
    # Default behaviour is unchanged: an exact inner-product flat index.
    chunks, embedder = _onehot_chunks_and_embedder()
    index = VectorIndexer(embedder).build(chunks)
    assert isinstance(index._index, faiss.IndexFlatIP)
    assert index.search([0.0, 1.0, 0.0, 0.0], top_k=1)[0].doc_id == "d2"


def test_vector_indexer_builds_hnsw_index_returning_nearest() -> None:
    # An HNSW (ANN) index — for scale — still returns the true nearest neighbour on
    # clean data, and is a *different* faiss index type than the exact flat one.
    chunks, embedder = _onehot_chunks_and_embedder()
    index = VectorIndexer(embedder, index_type="hnsw").build(chunks)
    assert isinstance(index._index, faiss.IndexHNSWFlat)
    assert index.search([0.0, 1.0, 0.0, 0.0], top_k=1)[0].doc_id == "d2"


def test_vector_indexer_builds_ivf_index_and_trains_it() -> None:
    # IVF needs training before it can be searched; build() must train it. nlist=1
    # (one exhaustive cell) keeps the tiny test deterministic.
    chunks, embedder = _onehot_chunks_and_embedder()
    index = VectorIndexer(embedder, index_type="ivf", nlist=1, nprobe=1).build(chunks)
    assert isinstance(index._index, faiss.IndexIVFFlat)
    assert index._index.is_trained
    assert index.search([0.0, 1.0, 0.0, 0.0], top_k=1)[0].doc_id == "d2"


def test_vector_indexer_builds_ivfpq_compressed_index() -> None:
    # IVFPQ compresses vectors via Product Quantization (so millions of docs fit in
    # RAM). It trains + searches; PQ is lossy, so we assert the index TYPE + that it
    # trains and returns a result, not the exact nearest neighbour.
    n, dim = 64, 8
    rng = np.random.default_rng(0)
    vecs = {f"t{i}": rng.standard_normal(dim).astype("float32").tolist() for i in range(n)}
    chunks = [Chunk(chunk_id=f"d{i}::0", doc_id=f"d{i}", text=f"t{i}") for i in range(n)]
    index = VectorIndexer(
        _FakeEmbedder(vecs), index_type="ivfpq", nlist=4, nprobe=4, pq_m=2, pq_nbits=4
    ).build(chunks)
    assert isinstance(index._index, faiss.IndexIVFPQ)
    assert index._index.is_trained
    assert len(index.search(vecs["t0"], top_k=1)) == 1


def test_vector_indexer_rejects_unknown_index_type() -> None:
    with pytest.raises(ValueError, match="index_type"):
        VectorIndexer(None, index_type="bogus")


def test_ann_index_save_load_round_trips(tmp_path) -> None:
    # faiss serialization must round-trip an ANN index too (not just flat).
    chunks, embedder = _onehot_chunks_and_embedder()
    indexer = VectorIndexer(embedder, index_type="hnsw")
    index = indexer.build(chunks)
    stem = str(tmp_path / "idx")

    indexer.save(index, stem)
    loaded = indexer.load(stem)

    assert loaded.search([0.0, 1.0, 0.0, 0.0], top_k=1)[0].doc_id == "d2"


# ---------------------------------------------------------------------------
# Chunk metadata (multimodal provenance riding along on ordinary text chunks)
# ---------------------------------------------------------------------------

def test_chunk_metadata_defaults_to_empty_dict() -> None:
    # Legacy callers (no metadata argument) must be byte-identical to before.
    assert Chunk("d::0", "d", "hello").metadata == {}
    assert chunk_document("d", "hello world")[0].metadata == {}


def test_chunk_document_stamps_metadata_on_every_chunk_as_independent_copies() -> None:
    meta = {"source_type": "pdf", "file_name": "r.pdf", "page_number": 3}
    chunks = chunk_document("d", "word " * 600, chunk_size=512, chunk_overlap=50, metadata=meta)
    assert len(chunks) > 1
    assert all(c.metadata == meta for c in chunks)
    # Each chunk owns a copy: later per-chunk annotation cannot cross-contaminate.
    assert chunks[0].metadata is not chunks[1].metadata
    assert chunks[0].metadata is not meta


def test_chunk_document_token_mode_passes_metadata() -> None:
    chunks = chunk_document(
        "d", "a b c", chunk_size=2, chunk_overlap=0,
        tokenizer=FakeOffsetTokenizer(), metadata={"source_type": "txt"},
    )
    assert chunks and all(c.metadata == {"source_type": "txt"} for c in chunks)


def test_chunk_unpickles_from_pre_metadata_state() -> None:
    # ``.meta`` index pickles written before the metadata field existed carry
    # no "metadata" key; loading them must not AttributeError.
    old = Chunk.__new__(Chunk)
    old.__setstate__({"chunk_id": "d::0", "doc_id": "d", "text": "t"})
    assert old.metadata == {}


# ---------------------------------------------------------------------------
# Multimodal loaders. docling / torch / transformers are faked via sys.modules:
# these tests must run on machines without the ML stack (mirroring the loaders'
# lazy-import design), and fakes give exact control over pages and captions.
# ---------------------------------------------------------------------------

class _FakePdfDoc:
    """Docling document stub: 3 pages; page 2 exports empty (figure-only)."""

    pages = {1: None, 2: None, 3: None}

    def export_to_markdown(self, page_no=None):
        if page_no is None:
            return "# whole doc"
        return "" if page_no == 2 else f"# Page {page_no}"


def _install_fake_docling(monkeypatch, doc) -> None:
    module = types.ModuleType("docling.document_converter")

    class DocumentConverter:
        def convert(self, path):
            return types.SimpleNamespace(document=doc)

    module.DocumentConverter = DocumentConverter
    monkeypatch.setitem(sys.modules, "docling", types.ModuleType("docling"))
    monkeypatch.setitem(sys.modules, "docling.document_converter", module)


def test_load_pdf_emits_one_markdown_record_per_nonblank_page(tmp_path, monkeypatch) -> None:
    _install_fake_docling(monkeypatch, _FakePdfDoc())
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-")

    records = load_pdf(pdf)

    assert [r.doc_id for r in records] == ["report::p1", "report::p3"]  # blank p2 skipped
    assert records[0].text == "# Page 1"
    assert records[0].metadata == {
        "source_type": "pdf", "file_name": "report.pdf", "page_number": 1,
    }


def test_load_pdf_falls_back_to_whole_document_without_per_page_export(tmp_path, monkeypatch) -> None:
    class OldDoc(_FakePdfDoc):
        def export_to_markdown(self):  # older docling-core: no page_no kwarg
            return "# whole doc"

    _install_fake_docling(monkeypatch, OldDoc())
    pdf = tmp_path / "old.pdf"
    pdf.write_bytes(b"%PDF-")

    records = load_pdf(pdf)

    assert len(records) == 1
    assert records[0].doc_id == "old"
    assert records[0].text == "# whole doc"
    assert records[0].metadata["page_number"] is None


def test_load_pdf_missing_file_raises_before_docling_import() -> None:
    with pytest.raises(FileNotFoundError):
        load_pdf("no/such.pdf")


def _install_fake_vision_stack(monkeypatch, caption="A bar chart.", generate_exc=None) -> dict:
    """Fake torch + transformers; returns counters for loads / empty_cache."""
    calls = {"loads": 0, "empty_cache": 0}

    torch_mod = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    torch_mod.no_grad = _NoGrad
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: calls.__setitem__("empty_cache", calls["empty_cache"] + 1),
    )
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    class _Inputs(dict):
        def to(self, device):
            return self

    class _Ids:
        shape = (1, 3)

        def __getitem__(self, key):  # output_ids[0][n:] -> the "new" tokens
            return "new-token-ids"

    class _Processor:
        def apply_chat_template(self, conversation, **kwargs):
            return _Inputs(input_ids=_Ids())

        def decode(self, ids, skip_special_tokens=True):
            return f"  {caption}  "  # loader must strip

    class _Model:
        device = "cpu"

        def generate(self, **kwargs):
            if generate_exc is not None:
                raise generate_exc
            return [_Ids()]

        def eval(self):
            return self

        def to(self, device):
            return self

    tf_mod = types.ModuleType("transformers")

    class AutoProcessor:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return _Processor()

    class AutoModelForImageTextToText:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["loads"] += 1
            return _Model()

    tf_mod.AutoProcessor = AutoProcessor
    tf_mod.AutoModelForImageTextToText = AutoModelForImageTextToText
    monkeypatch.setitem(sys.modules, "transformers", tf_mod)
    return calls


def _write_png(path: Path) -> None:
    from PIL import Image

    Image.new("RGB", (4, 4), "white").save(path)


def test_caption_images_returns_caption_records_and_frees_vram(tmp_path, monkeypatch) -> None:
    calls = _install_fake_vision_stack(monkeypatch, caption="A bar chart of revenue.")
    img1, img2 = tmp_path / "chart.png", tmp_path / "photo.jpg"
    _write_png(img1)
    _write_png(img2)

    records = caption_images([img1, img2])

    assert [r.text for r in records] == ["A bar chart of revenue."] * 2
    assert records[0].doc_id == "chart"
    assert records[0].metadata == {
        "source_type": "image", "file_name": "chart.png", "image_path": str(img1.resolve()),
    }
    assert calls["loads"] == 1        # one model load for the whole batch
    assert calls["empty_cache"] == 1  # VRAM released after captioning


def test_caption_images_frees_vram_even_when_generation_fails(tmp_path, monkeypatch) -> None:
    # The cluster-VRAM contract: the finally-block release must run on the
    # error path too, or a crashed ingestion job pins multi-GB of GPU memory.
    calls = _install_fake_vision_stack(monkeypatch, generate_exc=RuntimeError("CUDA OOM"))
    img = tmp_path / "x.png"
    _write_png(img)

    with pytest.raises(RuntimeError, match="OOM"):
        caption_images([img])
    assert calls["empty_cache"] == 1


def test_caption_images_missing_file_raises_before_model_load(tmp_path, monkeypatch) -> None:
    calls = _install_fake_vision_stack(monkeypatch)
    with pytest.raises(FileNotFoundError):
        caption_images([tmp_path / "nope.png"])
    assert calls["loads"] == 0


def test_caption_images_empty_input_returns_empty_without_model_load(monkeypatch) -> None:
    calls = _install_fake_vision_stack(monkeypatch)
    assert caption_images([]) == []
    assert calls["loads"] == 0


# ---------------------------------------------------------------------------
# load_directory: extension routing + one vision batch per call
# ---------------------------------------------------------------------------

def test_load_directory_routes_by_extension_and_batches_images(tmp_path, monkeypatch) -> None:
    (tmp_path / "a.txt").write_text("text a", encoding="utf-8")
    (tmp_path / "b.md").write_text("text b", encoding="utf-8")
    (tmp_path / "c.pdf").write_bytes(b"%PDF-")
    (tmp_path / "d.png").write_bytes(b"png")
    (tmp_path / "e.jpg").write_bytes(b"jpg")
    (tmp_path / "f.docx").write_text("unsupported", encoding="utf-8")

    from src.ingestion.loaders import dispatch

    monkeypatch.setattr(dispatch, "build_converter", lambda: "CONVERTER")

    def fake_load_pdf(path, *, converter=None):
        assert converter == "CONVERTER"  # built once, shared across PDFs
        return [LoadedDocument(
            f"{path.stem}::p1", "# md",
            {"source_type": "pdf", "file_name": path.name, "page_number": 1},
        )]

    monkeypatch.setattr(dispatch, "load_pdf", fake_load_pdf)

    batches = []

    def fake_caption_images(paths, **kwargs):
        batches.append(list(paths))
        return [LoadedDocument(
            p.stem, f"caption of {p.name}",
            {"source_type": "image", "file_name": p.name, "image_path": str(p)},
        ) for p in paths]

    monkeypatch.setattr(dispatch, "caption_images", fake_caption_images)

    records = dispatch.load_directory(tmp_path)

    # text/pdf in sorted order, then images; unsupported .docx skipped.
    assert [r.doc_id for r in records] == ["a", "b", "c::p1", "d", "e"]
    assert records[0].metadata == {"source_type": "txt", "file_name": "a.txt"}
    # All images captioned in a single batch: one model load + release per call.
    assert len(batches) == 1
    assert [p.name for p in batches[0]] == ["d.png", "e.jpg"]


def test_load_directory_without_pdfs_never_builds_docling_converter(tmp_path, monkeypatch) -> None:
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")

    from src.ingestion.loaders import dispatch

    monkeypatch.setattr(
        dispatch, "build_converter",
        lambda: pytest.fail("docling converter built although no PDF is present"),
    )
    records = dispatch.load_directory(tmp_path)
    assert [r.doc_id for r in records] == ["a"]


def test_load_directory_rejects_non_directory(tmp_path) -> None:
    with pytest.raises(NotADirectoryError):
        load_directory(tmp_path / "missing")