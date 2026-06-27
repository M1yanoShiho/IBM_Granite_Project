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

import ir_datasets


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
    if name == "nq":
        dataset = ir_datasets.load("dpr-w100/natural-questions/dev")
    else:
        # support multi dataset under beir
        dataset_id = f"beir/{name}/{split}"
        try:
            dataset = ir_datasets.load(dataset_id)
        except KeyError:
            dataset = ir_datasets.load(f"beir/{name}")

    corpus: Dict[str, str] = {}
    for doc in dataset.docs_iter():
        corpus[doc.doc_id] = doc.text

    queries: Dict[str, str] = {}
    answers: Dict[str, str] = {}
    for query in dataset.queries_iter():
        queries[query.query_id] = query.text
        gold = getattr(query, "answers", None)
        if gold:
            answers[query.query_id] = gold[0]

    qrels: Dict[str, Dict[str, int]] = {}
    for qrel in dataset.qrels_iter():
        # set default queryId to avoid null inner dict
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = int(qrel.relevance)

    final_answers = answers or None

    # print statistics info
    docs_count = len(corpus)
    queries_count = len(queries)
    qrels_count = sum(len(docs) for docs in qrels.values())
    # average length in characters (len() of a str counts characters, not words)
    avg_doc_chars = sum(len(t) for t in corpus.values()) / docs_count if docs_count else 0
    avg_query_chars = sum(len(t) for t in queries.values()) / queries_count if queries_count else 0
    # average relevant documents per query
    avg_rel_per_q = qrels_count / queries_count if queries_count else 0

    print()
    print(f"Dataset: {name}({split})")
    print(f"    Corpus: {docs_count} documents (avg {avg_doc_chars:.0f} chars)")
    print(f"    Queries: {queries_count} queries (avg {avg_query_chars:.0f} chars)")
    print(f"    Qrels: {qrels_count} relevance judgments")
    print(f"    Average relevant documents per query: {avg_rel_per_q:.2f}")
    if final_answers is not None:
        print(f"    Answers: {len(final_answers)} gold answers")
    else:
        print(f"    Answers: None (retrieval-only dataset)")
    print()

    return BenchmarkData(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        answers=final_answers,
    )
