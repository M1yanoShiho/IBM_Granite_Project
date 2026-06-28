"""Vector index construction and persistence.

Embeds chunks and builds a **persistent** FAISS index that the retriever loads
at query time. Persistence (save/load) is what distinguishes a real system from
the throwaway, rebuilt-every-run index used in early experiments.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Sequence

import faiss
import numpy as np

from src.ingestion.chunker import Chunk
from src.retrieval.base import RetrievedChunk
from src.retrieval.embedder import Embedder


class FaissIndex:
    """Wraps a FAISS index + chunk metadata.

    DenseRetriever calls index.search(query_vector, top_k) and expects
    List[RetrievedChunk] back — this class provides that interface.
    """

    def __init__(self, faiss_index, chunks: List[Chunk]) -> None:
        self._index = faiss_index
        self._chunks = chunks

    def search(self, query_vector: List[float], top_k: int) -> List[RetrievedChunk]:
        q = np.array([query_vector], dtype="float32")
        scores, ids = self._index.search(q, top_k)
        results = []
        for rank, i in enumerate(ids[0]):
            if i < 0:
                continue
            chunk = self._chunks[i]
            results.append(RetrievedChunk(
                doc_id=chunk.doc_id,
                text=chunk.text,
                score=float(scores[0][rank]),
            ))
        return results


class VectorIndexer:
    """Builds and persists a FAISS vector index over chunks.

    Parameters
    ----------
    embedder:
        Encodes chunk texts into vectors (normalised -> inner product = cosine).
    index_type:
        FAISS index structure:

        - ``"flat"`` (default): exact brute-force inner-product search
          (``IndexFlatIP``). Exact but O(N) per query — fine for small corpora,
          but slow and memory-heavy at millions of documents.
        - ``"hnsw"``: an approximate-nearest-neighbour graph (``IndexHNSWFlat``).
          Orders of magnitude faster at scale, ~no training, very high recall.
        - ``"ivf"``: an inverted-file ANN index (``IndexIVFFlat``); needs
          training, recall/speed traded via ``nprobe``.

        ANN indexes are what make "find the needle in a *large* haystack"
        practical — the default stays ``"flat"`` so existing results reproduce.
    hnsw_m:
        HNSW graph degree (neighbours per node); higher = better recall, more memory.
    ef_search:
        HNSW search breadth; higher = better recall, slower. The HNSW recall/speed knob.
    nlist:
        IVF number of cells (clusters); clamped to the corpus size for tiny inputs.
    nprobe:
        IVF cells probed per query; higher = better recall, slower. The IVF knob.
    """

    _INDEX_TYPES = ("flat", "hnsw", "ivf")

    def __init__(
        self,
        embedder: Embedder,
        index_type: str = "flat",
        *,
        hnsw_m: int = 32,
        ef_search: int = 64,
        nlist: int = 100,
        nprobe: int = 8,
    ) -> None:
        if index_type not in self._INDEX_TYPES:
            raise ValueError(
                f"Unknown index_type {index_type!r}; expected one of {list(self._INDEX_TYPES)}."
            )
        self.embedder = embedder
        self.index_type = index_type
        self.hnsw_m = hnsw_m
        self.ef_search = ef_search
        self.nlist = nlist
        self.nprobe = nprobe

    def _new_index(self, dim: int, n_vectors: int):
        """Construct the empty FAISS index for ``index_type`` (inner product)."""
        if self.index_type == "flat":
            return faiss.IndexFlatIP(dim)
        if self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = self.ef_search
            return index
        # ivf — can't have more cells than vectors; probe at least one cell.
        nlist = max(1, min(self.nlist, n_vectors))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = max(1, min(self.nprobe, nlist))
        return index

    def build(self, chunks: Sequence[Chunk]) -> FaissIndex:
        """Embed chunks and build an in-memory FAISS index (see ``index_type``).

        Raises
        ------
        ValueError
            If ``chunks`` is empty — surfaced as a clear error rather than the
            cryptic ``IndexError: tuple index out of range`` from reading the
            embedding dimension off a zero-row array.
        """
        chunks = list(chunks)
        if not chunks:
            raise ValueError(
                "Cannot build an index from zero chunks; provide at least one "
                "chunk (check that the corpus is non-empty and produced chunkable text)."
            )
        texts = [c.text for c in chunks]
        vectors = np.array(self.embedder.embed_documents(texts), dtype="float32")
        dim = vectors.shape[1]
        index = self._new_index(dim, len(chunks))
        if not index.is_trained:  # IVF needs training; flat/HNSW do not
            index.train(vectors)
        index.add(vectors)
        return FaissIndex(index, chunks)

    def save(self, faiss_index_obj: FaissIndex, path: str | Path) -> None:
        """Persist the index and chunk metadata to disk.

        Serialises the FAISS index in memory and writes the bytes with Python's
        unicode-safe I/O, instead of handing a path to ``faiss.write_index``:
        FAISS's C++ narrow-char file API cannot open non-ASCII paths on Windows.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Path(str(path) + ".faiss").write_bytes(
            faiss.serialize_index(faiss_index_obj._index).tobytes()
        )
        with open(str(path) + ".meta", "wb") as f:
            pickle.dump(faiss_index_obj._chunks, f)

    def load(self, path: str | Path) -> FaissIndex:
        """Load a previously persisted index from disk (unicode-safe; see save)."""
        path = Path(path)
        data = Path(str(path) + ".faiss").read_bytes()
        index = faiss.deserialize_index(np.frombuffer(data, dtype="uint8"))
        with open(str(path) + ".meta", "rb") as f:
            chunks = pickle.load(f)
        return FaissIndex(index, chunks)