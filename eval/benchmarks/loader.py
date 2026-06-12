"""Load standard IR benchmarks into a uniform (corpus, queries, qrels) form.

Supports the datasets named in the project brief:

- **BEIR** — a heterogeneous suite of retrieval benchmarks.
- **MS MARCO** — passage ranking with relevance labels.
- **Natural Questions** — open-domain QA over Wikipedia.

Each loader returns:

- ``corpus``:  ``{doc_id: text}``
- ``queries``: ``{query_id: text}``
- ``qrels``:   ``{query_id: {doc_id: relevance}}``  (the gold relevance judgments)

This uniform shape lets ``eval.run_benchmark`` evaluate any retriever the same
way, against the same ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class BenchmarkData:
    """A loaded benchmark split.

    Attributes
    ----------
    corpus:
        ``{doc_id: text}`` — the documents/passages to search.
    queries:
        ``{query_id: text}`` — the evaluation queries.
    qrels:
        ``{query_id: {doc_id: relevance}}`` — human relevance judgments
        (drives the retrieval metrics).
    answers:
        ``{query_id: gold_answer}`` — optional free-text answers, present only
        for answer-bearing QA benchmarks (e.g. NQ, MS MARCO QA). Required for the
        RAG evaluation (``eval/run_rag.py``); retrieval-only sets like SciFact
        leave this ``None``. See meeting question Q5.
    """

    corpus: Dict[str, str]
    queries: Dict[str, str]
    qrels: Dict[str, Dict[str, int]]
    answers: Optional[Dict[str, str]] = None


def load_benchmark(name: str, split: str = "test") -> BenchmarkData:
    """Load a named benchmark into a :class:`BenchmarkData`.

    Parameters
    ----------
    name:
        One of e.g. ``"scifact"``, ``"nq"``, ``"msmarco"`` (BEIR dataset names)
        or another supported benchmark identifier.
    split:
        Dataset split to load (e.g. ``"test"``).
    """
    raise NotImplementedError(
        "TODO: download/load the benchmark (e.g. via the `beir` or `datasets` "
        "library) and return corpus, queries, and qrels."
    )
