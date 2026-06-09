"""Information-retrieval metrics for the primary evaluation.

Standard, literature-comparable metrics computed from a retriever's ranked
results against benchmark relevance judgments (qrels):

- **Precision@k** — of the top-k retrieved, how many are relevant.
- **Recall@k** — of all relevant documents, how many appear in the top-k.
- **nDCG@k** — rank-aware quality (relevant hits near the top count more).
- **MRR** — mean reciprocal rank of the first relevant hit.

These are the metrics the brief asks for ("precision/recall metrics on
benchmark datasets") and the basis for the system-vs-baseline comparison.
A mature implementation may delegate to a vetted library (e.g. ``ranx`` or
``pytrec_eval``); thin wrappers are stubbed here.
"""

from __future__ import annotations

from typing import Dict, List


# A run is {query_id: {doc_id: score}}; qrels is {query_id: {doc_id: relevance}}.
Run = Dict[str, Dict[str, float]]
Qrels = Dict[str, Dict[str, int]]


def precision_at_k(run: Run, qrels: Qrels, k: int = 10) -> float:
    """Mean Precision@k over all queries."""
    raise NotImplementedError("TODO: compute precision@k.")


def recall_at_k(run: Run, qrels: Qrels, k: int = 10) -> float:
    """Mean Recall@k over all queries."""
    raise NotImplementedError("TODO: compute recall@k.")


def ndcg_at_k(run: Run, qrels: Qrels, k: int = 10) -> float:
    """Mean nDCG@k over all queries."""
    raise NotImplementedError("TODO: compute nDCG@k.")


def mrr(run: Run, qrels: Qrels) -> float:
    """Mean Reciprocal Rank over all queries."""
    raise NotImplementedError("TODO: compute MRR.")


def evaluate_run(run: Run, qrels: Qrels, k_values: List[int] | None = None) -> Dict[str, float]:
    """Compute the full metric suite for a run and return a ``{metric: value}`` dict."""
    raise NotImplementedError(
        "TODO: assemble precision/recall/nDCG@k + MRR into one results dict."
    )
