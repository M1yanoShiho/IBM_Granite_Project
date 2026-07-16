"""Shared retrieval contracts — the single source of truth for the retrieval
hand-off between team members (see ``docs/interfaces.md``).

Both implementations (``DenseRetriever``, ``BM25Retriever``) and consumers
(``eval/run_benchmark.py``, ``src/explainability``) import these types from here,
so the parts integrate without coordination drift. **Do not change the shape of
these without team agreement** — other people are coding against them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, runtime_checkable


@dataclass
class RetrievedChunk:
    """A single retrieval result (the unit every retriever returns).

    Attributes
    ----------
    doc_id:
        Identifier of the source chunk/passage. Scored against the benchmark's
        qrels (which are keyed by ``doc_id``).
    text:
        The chunk text.
    score:
        Similarity / relevance score; higher = more relevant.
    rank:
        1-based position in the ranked list returned by this retriever.
        Defaults to 0 (unset) for callers that do not populate it.
    metadata:
        Provenance fields carried from the source chunk: ``source_parent_id``,
        ``company``, ``date``, ``document_type``, ``version``, etc.
        Used by the Selector interface; empty for legacy callers.
    """

    doc_id: str
    text: str
    score: float
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Retriever(Protocol):
    """CONTRACT 1 — the uniform retriever interface.

    Every retriever (the Granite dense retriever, the sentence-transformers
    dense baseline, and the BM25 baseline) must implement ``retrieve`` so the
    evaluation harness (``eval/run_benchmark.py``) can treat them
    interchangeably.

    This is a structural ``Protocol``: a class satisfies it simply by defining a
    matching ``retrieve`` method — no explicit subclassing required. Because it
    is ``@runtime_checkable`` you can also assert ``isinstance(obj, Retriever)``
    in tests.

    Example
    -------
    >>> class MyRetriever:
    ...     def retrieve(self, query: str) -> List[RetrievedChunk]:
    ...         ...
    >>> isinstance(MyRetriever(), Retriever)
    True
    """

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """Return the chunks most relevant to ``query``, ranked best-first."""
        ...
