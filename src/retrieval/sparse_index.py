"""scipy.sparse CSR index for learned-sparse retrieval (SPLADE).

The sparse-vector counterpart to :class:`~src.ingestion.indexer.FaissIndex`: it stores
the corpus as a CSR matrix ``[n_docs x vocab]`` of term weights and scores a query by a
sparse dot product. Used by ``SparseRetriever`` (the SPLADE arm of the convex hybrid).

Generic over any sparse vectors ``{term_id: weight}`` — it does **not** depend on the
SPLADE encoder, so it stays a small, independently-testable unit.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import sparse

SparseVector = Dict[int, float]  # {term_id: weight}


class SparseIndex:
    """A CSR term-weight matrix over documents, searchable by sparse dot product.

    Parameters
    ----------
    matrix:
        ``scipy.sparse.csr_matrix`` of shape ``[n_docs, vocab_size]``.
    doc_ids:
        Document ids parallel to the matrix rows.
    vocab_size:
        Number of columns (vocabulary size).
    """

    def __init__(
        self, matrix: sparse.csr_matrix, doc_ids: Sequence[str], vocab_size: int
    ) -> None:
        self.matrix = matrix
        self.doc_ids = list(doc_ids)
        self.vocab_size = vocab_size

    @classmethod
    def build(
        cls,
        doc_vectors: Sequence[SparseVector],
        doc_ids: Sequence[str],
        vocab_size: int,
    ) -> "SparseIndex":
        """Build a CSR index from per-document ``{term_id: weight}`` vectors."""
        doc_ids = list(doc_ids)
        doc_vectors = list(doc_vectors)
        if len(doc_vectors) != len(doc_ids):
            raise ValueError("doc_vectors and doc_ids must have the same length.")
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        for i, vec in enumerate(doc_vectors):
            for term, weight in vec.items():
                rows.append(i)
                cols.append(term)
                data.append(weight)
        matrix = sparse.csr_matrix(
            (data, (rows, cols)), shape=(len(doc_ids), vocab_size)
        )
        return cls(matrix, doc_ids, vocab_size)

    def search(self, query: SparseVector, top_k: int) -> List[Tuple[int, float]]:
        """Top-k ``(doc_index, score)`` by sparse dot product; only score > 0.

        Deterministic: ties break to the smaller doc index (stable descending sort).
        An empty query, or a query sharing no term with any document, returns ``[]``.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not query:
            return []
        q = sparse.csr_matrix(
            (list(query.values()), ([0] * len(query), list(query.keys()))),
            shape=(1, self.vocab_size),
        )
        scores = (self.matrix @ q.T).toarray().ravel()  # dense [n_docs]
        order = np.argsort(-scores, kind="stable")
        out: List[Tuple[int, float]] = []
        for i in order[:top_k]:
            score = float(scores[i])
            if score <= 0.0:
                break  # stable descending order -> everything after this is <= 0 too
            out.append((int(i), score))
        return out
