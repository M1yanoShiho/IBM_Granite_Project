""" Information-retrieval metrics for the primary evaluation.
Standard, literature-comparable metrics computed from a retriever's ranked
results against benchmark relevance judgments (qrels):
- **Precision@k** — of the top-k retrieved, how many are relevant.
- **Recall@k** — of all relevant documents, how many appear in the top-k.
- **nDCG@k** — rank-aware quality (relevant hits near the top count more).
- **MRR** — mean reciprocal rank of the first relevant hit.
These are the metrics the brief asks for ("precision/recall metrics on
benchmark datasets") and the basis for the system-vs-baseline comparison.
"""
from __future__ import annotations
from typing import Dict, List

from ranx import Qrels as RanxQrels, Run as RanxRun, evaluate

# ---------------------------------------------------------------------------
# Type aliases (契约 2 — interfaces.md)
# Run:   {query_id: {doc_id: score}}   — retriever output, higher = more relevant
# Qrels: {query_id: {doc_id: relevance}} — benchmark ground-truth labels
# ---------------------------------------------------------------------------
Run = Dict[str, Dict[str, float]]
Qrels = Dict[str, Dict[str, int]]


def _to_ranx(run: Run, qrels: Qrels) -> tuple[RanxQrels, RanxRun]:
    """Convert raw dicts to ranx objects.

    Centralised so every public function pays the conversion cost once
    and we avoid repeating the same two lines everywhere.
    """
    return RanxQrels(qrels), RanxRun(run)


def precision_at_k(run: Run, qrels: Qrels, k: int = 10) -> float:
    """Mean Precision@k over all queries.

    Fraction of the top-k retrieved documents that are relevant.
    Measures how much noise is in the returned results.

    Args:
        run:   retriever output {query_id: {doc_id: score}}.
        qrels: ground-truth relevance {query_id: {doc_id: relevance}}.
        k:     cutoff rank (default 10).

    Returns:
        Mean Precision@k across all queries, in [0, 1].
    """
    q, r = _to_ranx(run, qrels)
    return evaluate(q, r, f"precision@{k}")


def recall_at_k(run: Run, qrels: Qrels, k: int = 10) -> float:
    """Mean Recall@k over all queries.

    Fraction of all relevant documents that appear in the top-k.
    Measures coverage — how many relevant docs the retriever actually finds.

    Args:
        run:   retriever output {query_id: {doc_id: score}}.
        qrels: ground-truth relevance {query_id: {doc_id: relevance}}.
        k:     cutoff rank (default 10).

    Returns:
        Mean Recall@k across all queries, in [0, 1].
    """
    q, r = _to_ranx(run, qrels)
    return evaluate(q, r, f"recall@{k}")


def ndcg_at_k(run: Run, qrels: Qrels, k: int = 10) -> float:
    """Mean nDCG@k over all queries.

    Normalised Discounted Cumulative Gain — rank-aware metric that rewards
    placing relevant documents higher in the list. A relevant hit at rank 1
    is worth more than the same hit at rank 10.

    Args:
        run:   retriever output {query_id: {doc_id: score}}.
        qrels: ground-truth relevance {query_id: {doc_id: relevance}}.
        k:     cutoff rank (default 10).

    Returns:
        Mean nDCG@k across all queries, in [0, 1].
    """
    q, r = _to_ranx(run, qrels)
    return evaluate(q, r, f"ndcg@{k}")


def mrr(run: Run, qrels: Qrels) -> float:
    """Mean Reciprocal Rank over all queries.

    For each query, takes the reciprocal of the rank of the first relevant
    document (1/rank), then averages across queries. Useful when the user
    cares mainly about whether the top result is relevant.

    Args:
        run:   retriever output {query_id: {doc_id: score}}.
        qrels: ground-truth relevance {query_id: {doc_id: relevance}}.

    Returns:
        MRR across all queries, in (0, 1].
    """
    q, r = _to_ranx(run, qrels)
    return evaluate(q, r, "mrr")


def evaluate_run(
    run: Run,
    qrels: Qrels,
    k_values: List[int] | None = None,
) -> Dict[str, float]:
    """Compute the full metric suite for a retriever run.

    Evaluates Precision, Recall, nDCG at each k in k_values, plus MRR.
    This is the main entry point for P6 (run_benchmark.py) after building
    a Run from retriever output via the chunk→doc max-pool aggregation
    defined in interfaces.md契约 3.

    Args:
        run:      retriever output {query_id: {doc_id: score}}.
        qrels:    ground-truth relevance {query_id: {doc_id: relevance}}.
        k_values: list of cutoff ranks to evaluate at.
                  Defaults to [1, 3, 5, 10] — consistent with RULER/HELMET
                  reporting convention.

    Returns:
        Dict mapping metric name to value, e.g.:
        {
            "precision@1": 0.72, "precision@3": 0.61, ...,
            "recall@1":    0.18, "recall@3":    0.41, ...,
            "ndcg@1":      0.72, "ndcg@3":      0.65, ...,
            "mrr":         0.74,
        }

    Example:
        >>> qrels = {"q1": {"d1": 1, "d2": 0, "d3": 1}}
        >>> run   = {"q1": {"d1": 0.9, "d2": 0.8, "d3": 0.3}}
        >>> evaluate_run(run, qrels)
        {'precision@1': 1.0, 'precision@3': 0.67, ..., 'mrr': 1.0}
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    q, r = _to_ranx(run, qrels)

    metrics = [f"precision@{k}" for k in k_values]
    metrics += [f"recall@{k}" for k in k_values]
    metrics += [f"ndcg@{k}" for k in k_values]
    metrics += ["mrr"]

    return dict(evaluate(q, r, metrics))


def per_query_scores(run: Run, qrels: Qrels, metric: str = "ndcg@10") -> Dict[str, float]:
    """Per-query scores for a single metric, keyed by query id (not averaged).

    ``evaluate_run`` returns the *mean* over queries; this returns the score for
    each individual query — what the failure analysis (which queries does a
    retriever win/lose) and paired significance testing (``eval.significance``)
    need. ``metric`` is any ranx metric string, e.g. ``"ndcg@10"`` or ``"recall@10"``.

    ranx's ``return_mean=False`` yields one score per query, in ``qrels`` order, so
    zipping with ``qrels`` keys recovers the ``{query_id: score}`` mapping. The mean
    of the returned values equals the corresponding aggregate metric by construction.

    Args:
        run:    retriever output {query_id: {doc_id: score}}.
        qrels:  ground-truth relevance {query_id: {doc_id: relevance}}.
        metric: a single ranx metric name (default ``"ndcg@10"``).

    Returns:
        ``{query_id: score}`` for every query in ``qrels``.
    """
    q, r = _to_ranx(run, qrels)
    scores = evaluate(q, r, metric, return_mean=False)
    return {query_id: float(score) for query_id, score in zip(qrels.keys(), scores)}