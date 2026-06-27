"""Per-query failure-mode analysis for retriever comparisons (Bharat point 2).

Once the aggregate numbers show two retrievers are statistically tied (e.g.
granite_dense ~= gte_dense), the mean hides the interesting behaviour. This module
answers, per query:

- :func:`head_to_head` — do A and B win the *same* queries (high score
  correlation = redundant) or *different* ones (low/negative correlation =
  complementary, so a hybrid could beat both)? Plus win/loss/tie counts.
- :func:`top_disagreements` — the specific queries where each retriever wins by
  the largest margin (qualitative inspection material for the report).
- :func:`upsets` — queries where a supposedly-weaker retriever (BM25) still beats
  a stronger one (Granite): "where does dense retrieval fail?".
- :func:`query_features` + :func:`summarize_by_bucket` — categorise the wins/losses
  by cheap query features (length, presence of digits), e.g. "on numeric queries A
  leads by X".

All core functions are pure over per-query score dicts
(``{query_id: score}``, from :func:`eval.ir_metrics.per_query_scores` /
``run_benchmark --per-query-out``), so they need no models. The CLI reads the wide
per-query CSV and runs locally — no GPU.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Hashable, List, Tuple

import numpy as np

from eval.significance import load_per_query_csv


def _shared_qids(scores_a: Dict[str, float], scores_b: Dict[str, float]) -> List[str]:
    """Query ids scored by both retrievers, in ``scores_a`` order."""
    shared = [qid for qid in scores_a if qid in scores_b]
    if not shared:
        raise ValueError("No shared query ids between the two retrievers.")
    return shared


def per_query_deltas(
    scores_a: Dict[str, float], scores_b: Dict[str, float]
) -> Dict[str, float]:
    """Per-query ``a - b`` over the shared queries, keyed by query id."""
    return {qid: scores_a[qid] - scores_b[qid] for qid in _shared_qids(scores_a, scores_b)}


@dataclass
class HeadToHead:
    """Win/loss/tie counts and score correlation between two retrievers."""

    a: str
    b: str
    a_wins: int
    b_wins: int
    ties: int
    n: int
    correlation: float  # Pearson of paired scores; high => redundant, low => complementary
    mean_delta: float   # mean(a - b)


def head_to_head(
    scores_a: Dict[str, float],
    scores_b: Dict[str, float],
    *,
    name_a: str = "A",
    name_b: str = "B",
    tol: float = 1e-9,
) -> HeadToHead:
    """Per-query win/loss/tie counts + score correlation between two retrievers.

    A tie is ``|a - b| <= tol``. The correlation is Pearson over the shared
    per-query scores: high means the two rank queries alike (redundant); low or
    negative means they win different queries (complementary — the case where a
    hybrid could beat either alone). Returns ``nan`` correlation if either side is
    constant (undefined).
    """
    shared = _shared_qids(scores_a, scores_b)
    a = np.array([scores_a[qid] for qid in shared], dtype=float)
    b = np.array([scores_b[qid] for qid in shared], dtype=float)
    delta = a - b
    correlation = float("nan")
    if a.std() > 0 and b.std() > 0:
        correlation = float(np.corrcoef(a, b)[0, 1])
    return HeadToHead(
        a=name_a,
        b=name_b,
        a_wins=int(np.count_nonzero(delta > tol)),
        b_wins=int(np.count_nonzero(delta < -tol)),
        ties=int(np.count_nonzero(np.abs(delta) <= tol)),
        n=len(shared),
        correlation=correlation,
        mean_delta=float(delta.mean()),
    )


def top_disagreements(
    scores_a: Dict[str, float],
    scores_b: Dict[str, float],
    n: int = 10,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """The ``n`` queries where A beats B by the most, and B beats A by the most.

    Returns ``(a_wins_most, b_wins_most)``, each a list of ``(query_id, delta)``
    with ``delta = a - b``: ``a_wins_most`` is sorted by largest positive margin,
    ``b_wins_most`` by largest negative margin. Ties (delta 0) appear in neither.
    """
    deltas = per_query_deltas(scores_a, scores_b)
    ordered = sorted(deltas.items(), key=lambda kv: kv[1])
    b_wins_most = [(qid, d) for qid, d in ordered[:n] if d < 0]
    a_wins_most = [(qid, d) for qid, d in reversed(ordered[-n:]) if d > 0]
    return a_wins_most, b_wins_most


def upsets(
    weak_scores: Dict[str, float], strong_scores: Dict[str, float]
) -> List[Tuple[str, float]]:
    """Queries where the supposedly weaker retriever beats the stronger one.

    For "where does BM25 beat Granite?" call ``upsets(bm25, granite)``. Returns
    ``(query_id, margin)`` with ``margin = weak - strong > 0``, largest first —
    the concrete failures of the stronger retriever, worth inspecting by hand.
    """
    wins = [(qid, d) for qid, d in per_query_deltas(weak_scores, strong_scores).items() if d > 0]
    return sorted(wins, key=lambda kv: kv[1], reverse=True)


def query_features(text: str) -> Dict[str, object]:
    """Cheap, deterministic query features for bucketing wins/losses.

    ``n_words`` (whitespace tokens) and ``has_digit`` (any digit character).
    Deliberately simple — enough to categorise *where* a retriever wins/loses
    (short keyword vs long, numeric vs not), not a full NLP pipeline.
    """
    return {"n_words": len(text.split()), "has_digit": any(c.isdigit() for c in text)}


@dataclass
class BucketStat:
    """Count and mean per-query delta within one bucket."""

    n: int
    mean_delta: float


def summarize_by_bucket(
    deltas: Dict[str, float], buckets: Dict[str, Hashable]
) -> Dict[Hashable, BucketStat]:
    """Mean per-query delta within each bucket.

    ``deltas`` is ``{query_id: a - b}``; ``buckets`` is ``{query_id: label}`` (e.g.
    derived from :func:`query_features`). Returns ``{label: BucketStat(n,
    mean_delta)}`` so you can report "on numeric queries A leads by X; on text-only
    queries by Y". Only query ids present in both maps are counted.
    """
    grouped: Dict[Hashable, List[float]] = {}
    for qid, delta in deltas.items():
        if qid in buckets:
            grouped.setdefault(buckets[qid], []).append(delta)
    return {label: BucketStat(len(v), float(np.mean(v))) for label, v in grouped.items()}


def _length_bucket(n_words: int) -> str:
    """Coarse query-length bucket for failure analysis."""
    if n_words <= 5:
        return "1-5"
    if n_words <= 15:
        return "6-15"
    return "16+"


def _format_head_to_head(h: HeadToHead) -> str:
    if h.correlation != h.correlation:  # nan
        read = "n/a (constant scores)"
    elif h.correlation >= 0.7:
        read = "redundant (high corr -> they win the same queries)"
    else:
        read = "complementary (low/neg corr -> they win different queries)"
    return (
        f"{h.a} vs {h.b}  (n={h.n})\n"
        f"  {h.a} wins: {h.a_wins}   {h.b} wins: {h.b_wins}   ties: {h.ties}\n"
        f"  mean delta (a-b): {h.mean_delta:+.4f}   score correlation: {h.correlation:.3f}\n"
        f"  -> {read}"
    )


def main(argv: List[str] | None = None) -> None:
    """CLI: per-query failure analysis of retriever A vs B from a per-query CSV.

    Prints head-to-head (wins/ties + correlation) and the top queries each side
    wins. With ``--dataset`` it also loads the query text and buckets the A-B delta
    by query length and digit-presence. Runs locally — no GPU.

    Example::

        python -m eval.failure_analysis --per-query-csv results/scifact_per_query.csv \\
            --a granite_dense --b gte_dense --dataset scifact
    """
    parser = argparse.ArgumentParser(
        prog="python -m eval.failure_analysis",
        description="Per-query failure-mode analysis between two retrievers.",
    )
    parser.add_argument("--per-query-csv", type=Path, required=True)
    parser.add_argument("--a", default="granite_dense", help="Retriever A (default: %(default)s).")
    parser.add_argument("--b", default="gte_dense", help="Retriever B (default: %(default)s).")
    parser.add_argument("--top", type=int, default=10, help="Top disagreements each way (default: %(default)s).")
    parser.add_argument(
        "--dataset",
        default=None,
        help="If set, load this benchmark's query text and bucket the delta by "
        "query length / digit-presence (needs the ir_datasets cache).",
    )
    args = parser.parse_args(argv)

    per_query = load_per_query_csv(args.per_query_csv)
    for name in (args.a, args.b):
        if name not in per_query:
            parser.error(f"{name!r} not in CSV; available: {sorted(per_query)}")
    a, b = per_query[args.a], per_query[args.b]

    print(_format_head_to_head(head_to_head(a, b, name_a=args.a, name_b=args.b)))

    a_top, b_top = top_disagreements(a, b, n=args.top)
    print(f"\nTop {args.top} queries where {args.a} beats {args.b}:")
    for qid, d in a_top:
        print(f"  {qid:<14} {d:+.4f}")
    print(f"\nTop {args.top} queries where {args.b} beats {args.a}:")
    for qid, d in b_top:
        print(f"  {qid:<14} {d:+.4f}")

    if args.dataset:
        from eval.benchmarks.loader import load_benchmark

        queries = load_benchmark(args.dataset).queries
        deltas = per_query_deltas(a, b)
        feats = {qid: query_features(queries[qid]) for qid in deltas if qid in queries}
        digit = {qid: ("has_digit" if feats[qid]["has_digit"] else "no_digit") for qid in feats}
        length = {qid: _length_bucket(int(feats[qid]["n_words"])) for qid in feats}
        print(f"\n{args.a} - {args.b} delta by digit-presence:")
        for label, st in sorted(summarize_by_bucket(deltas, digit).items()):
            print(f"  {label:<10} n={st.n:<5} mean_delta={st.mean_delta:+.4f}")
        print(f"\n{args.a} - {args.b} delta by query length (words):")
        for label, st in sorted(summarize_by_bucket(deltas, length).items()):
            print(f"  {label:<10} n={st.n:<5} mean_delta={st.mean_delta:+.4f}")


if __name__ == "__main__":
    main()
