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
from typing import Dict, List, Optional

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
        ``{query_id: [gold_answer, ...]}`` — optional free-text answers, present
        only for answer-bearing QA benchmarks (e.g. NQ, MS MARCO QA). A query may
        have several acceptable golds (NQ aliases), so this is a *list* per query;
        the answer-correctness metric scores against the best match. Required for
        the RAG evaluation (``eval/run_rag.py``); retrieval-only sets like SciFact
        leave this ``None``. See meeting question Q5.
    """

    corpus: Dict[str, str]
    queries: Dict[str, str]
    qrels: Dict[str, Dict[str, int]]
    answers: Optional[Dict[str, List[str]]] = None


# Answer-bearing QA benchmarks served from the DPR Wikipedia split (dpr-w100):
# they all share the SAME ~21M-passage corpus, so adding one is nearly free once
# the corpus is cached. Maps our short names to the ir_datasets ids.
_DPR_QA_DATASETS = {
    "nq": "dpr-w100/natural-questions/dev",
    "trivia": "dpr-w100/trivia-qa/dev",
    "triviaqa": "dpr-w100/trivia-qa/dev",
}


def load_benchmark(
    name: str,
    split: str = "test",
    max_queries: Optional[int] = None,
    max_docs: Optional[int] = None,
) -> BenchmarkData:
    """Load a named benchmark into a :class:`BenchmarkData`.

    Parameters
    ----------
    name:
        One of e.g. ``"scifact"``, ``"msmarco"`` (a BEIR dataset name) or a
        dpr-w100 QA set ``"nq"`` / ``"trivia"`` (see ``_DPR_QA_DATASETS``).
    split:
        Dataset split to load (e.g. ``"test"``).
    max_queries:
        If set, keep only the first ``max_queries`` queries (by sorted id), with
        their qrels/answers. Lets a huge set (NQ's ~21M passages) be run on a
        small subset first. ``None`` = all queries.
    max_docs:
        If set, cap the corpus at ``max_docs`` *distractor* documents — the gold
        documents of the kept queries are always included on top, so retrieval
        stays valid. ``None`` = keep the whole corpus (only feasible for small
        datasets).
    """
    if name in _DPR_QA_DATASETS:
        dataset = ir_datasets.load(_DPR_QA_DATASETS[name])
    else:
        # support multi dataset under beir
        dataset_id = f"beir/{name}/{split}"
        try:
            dataset = ir_datasets.load(dataset_id)
        except KeyError:
            dataset = ir_datasets.load(f"beir/{name}")

    queries: Dict[str, str] = {}
    answers: Dict[str, List[str]] = {}
    for query in dataset.queries_iter():
        queries[query.query_id] = query.text
        gold = getattr(query, "answers", None)
        if gold:
            # Keep every acceptable gold (NQ aliases), not just the first, so
            # answer-correctness can score against the best match.
            answers[query.query_id] = list(gold)

    qrels: Dict[str, Dict[str, int]] = {}
    for qrel in dataset.qrels_iter():
        # set default queryId to avoid null inner dict
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = int(qrel.relevance)

    # Subsample queries first (deterministic: first max_queries by sorted id) so a
    # huge corpus (e.g. NQ's ~21M passages) can be run on a small, valid subset.
    if max_queries is not None:
        kept = set(sorted(queries)[:max_queries])
        queries = {q: t for q, t in queries.items() if q in kept}
        qrels = {q: rels for q, rels in qrels.items() if q in kept}
        answers = {q: a for q, a in answers.items() if q in kept}

    # The documents the kept queries are judged against MUST be in the corpus,
    # else retrieval can never succeed and the eval is meaningless. Always keep
    # those; fill the remainder with up to ``max_docs`` distractor documents
    # (``None`` = keep the whole corpus, only feasible for small datasets).
    required_docs = {doc_id for rels in qrels.values() for doc_id in rels}
    corpus: Dict[str, str] = {}
    n_distractors = 0
    n_required_found = 0
    for doc in dataset.docs_iter():
        if doc.doc_id in required_docs:
            if doc.doc_id not in corpus:
                corpus[doc.doc_id] = doc.text
                n_required_found += 1
        elif max_docs is None or n_distractors < max_docs:
            corpus[doc.doc_id] = doc.text
            n_distractors += 1
        if (
            max_docs is not None
            and n_distractors >= max_docs
            and n_required_found >= len(required_docs)
        ):
            break

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
