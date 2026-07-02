# eval/build_niah_task.py
"""Build a hard NIAH task from a source benchmark (spec §5.1).

For each query: gold docs (qrels ``rel > 0``) = needles; build distractors from
**Source A** (a counterfactual per needle) and **Source C** (mined topical
negatives), keep those passing the §5.1 filters, inject the freshly-generated
ones, and return a ``NiahTask`` whose qrels still mark ONLY the needles relevant.

The two sources take *different* filters, on purpose:

- **Source A (counterfactual)** — a ~1-token edit of a needle, so it embeds
  near-identically (hard for both retrievers *by construction*) and scores *high*,
  close to the needle. The positive-anchor margin is therefore inapplicable (it
  would reject exactly these high-similarity distractors); non-relevance is
  guaranteed by the broken fact and confirmed by the answerability judge. So
  Source A is gated by the judge alone.
- **Source C (mined)** — a real corpus doc surfaced by the retrievers, where a high
  score may signal a false negative (an actually-relevant doc). Here the full
  filter applies: margin + not-answering + dual-retriever hardness.

``build_task`` takes precomputed *per-query* ranks/scores + injected models so it
is unit testable; :func:`compute_retrieval_signals` derives those from real
retrievers and :func:`main` wires the whole pipeline (load → retrieve → build →
gate → persist).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from src.niah.assembly import inject
from src.niah.counterfactual import make_counterfactual
from src.niah.filters import answers_query, keep_distractor
from src.niah.mining import mine_topical
from src.niah.types import (
    SOURCE_COUNTERFACTUAL,
    SOURCE_MINED,
    Distractor,
    NiahExample,
    NiahTask,
)


def build_task(
    *,
    corpus: Dict[str, str],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    answers: Dict[str, List[str]],
    llm,
    judge,
    dense_rank: Dict[str, Dict[str, int]],
    sparse_rank: Dict[str, Dict[str, int]],
    cand_scores: Dict[str, Dict[str, float]],
    positive_scores: Dict[str, float],
    margin: float,
    rank_threshold: int,
    mined_ids: Dict[str, List[str]] | None = None,
) -> NiahTask:
    """Assemble a NIAH task from Sources A (counterfactual) and C (mined).

    ``positive_scores`` is the per-query anchor (the needle's own retrieval score)
    used by the Source-C margin guard. ``mined_ids`` are the Source-C candidate doc
    ids per query (already in the corpus, so they are recorded but not re-injected).
    """
    mined = mined_ids or {}
    examples: List[NiahExample] = []
    injected: List[Distractor] = []

    for qid, query in queries.items():
        needle_ids = [doc_id for doc_id, rel in qrels.get(qid, {}).items() if rel > 0]
        gold_answer = (answers.get(qid) or [None])[0]
        pos = positive_scores.get(qid, 0.0)
        ex = NiahExample(query_id=qid, query=query, needle_ids=needle_ids)

        # Source A — counterfactual per needle (freshly generated -> injected);
        # gated by the answerability judge only (see module docstring).
        if gold_answer:
            for nid in needle_ids:
                cand_id = f"{qid}__{nid}__cf0"
                try:
                    text = make_counterfactual(corpus[nid], gold_answer, llm)
                except (ValueError, KeyError):
                    continue  # answer absent / echo / no-op swap -> skip this needle
                if not answers_query(text, query, judge):
                    d = Distractor(cand_id, text, SOURCE_COUNTERFACTUAL, nid)
                    ex.distractors.append(d)
                    injected.append(d)

        # Source C — mined topical negatives (already in the corpus). Full filter.
        needle_set = set(needle_ids)
        for mined_id in mined.get(qid, []):
            if mined_id in needle_set or mined_id not in corpus:
                continue
            if keep_distractor(
                cand_score=cand_scores.get(qid, {}).get(mined_id, 0.0),
                positive_score=pos,
                margin=margin,
                cand_text=corpus[mined_id],
                query=query,
                judge=judge,
                cand_id=mined_id,
                dense_rank=dense_rank.get(qid, {}),
                sparse_rank=sparse_rank.get(qid, {}),
                rank_threshold=rank_threshold,
            ):
                ex.distractors.append(
                    Distractor(mined_id, corpus[mined_id], SOURCE_MINED, "")
                )

        examples.append(ex)

    return NiahTask(
        corpus=inject(corpus, injected),
        queries=dict(queries),
        qrels={q: dict(r) for q, r in qrels.items()},
        examples=examples,
    )


# --------------------------------------------------------------------------- #
# Real-retriever signals (the inputs build_task consumes)
# --------------------------------------------------------------------------- #
@dataclass
class RetrievalSignals:
    """Per-query ranks/scores derived from the real dense + sparse retrievers."""

    dense_rank: Dict[str, Dict[str, int]]
    sparse_rank: Dict[str, Dict[str, int]]
    cand_scores: Dict[str, Dict[str, float]]
    positive_scores: Dict[str, float]
    mined_ids: Dict[str, List[str]]


def _rank_and_score(chunks):
    """Doc-level rank (1-indexed, best-first) + score from a retriever's chunks.

    Chunks arrive best-first, so the first appearance of a ``doc_id`` is its best
    rank and score (contract-3 max pooling is monotone in the best chunk).
    """
    rank: Dict[str, int] = {}
    score: Dict[str, float] = {}
    for chunk in chunks:
        if chunk.doc_id not in rank:
            rank[chunk.doc_id] = len(rank) + 1
            score[chunk.doc_id] = chunk.score
    return rank, score


def compute_retrieval_signals(
    dense, sparse, queries, qrels, top_n: int, mine_k: int
) -> RetrievalSignals:
    """Run both retrievers per query and derive the signals ``build_task`` needs.

    The positive anchor is the needle's own dense score (fallback: the top dense
    score when the needle is outside the retrieved depth). Candidate scores are on
    the dense scale, matching the anchor. Mined ids are dense∪sparse top-``mine_k``
    minus the needles (Source C).
    """
    dense_rank: Dict[str, Dict[str, int]] = {}
    sparse_rank: Dict[str, Dict[str, int]] = {}
    cand_scores: Dict[str, Dict[str, float]] = {}
    positive_scores: Dict[str, float] = {}
    mined_ids: Dict[str, List[str]] = {}
    for qid, query in queries.items():
        needle_ids = {d for d, rel in qrels.get(qid, {}).items() if rel > 0}
        d_rank, d_score = _rank_and_score(dense.retrieve(query)[:top_n])
        s_rank, _ = _rank_and_score(sparse.retrieve(query)[:top_n])
        dense_rank[qid] = d_rank
        sparse_rank[qid] = s_rank
        cand_scores[qid] = d_score
        found = [d_score[n] for n in needle_ids if n in d_score]
        positive_scores[qid] = (
            max(found) if found else (max(d_score.values()) if d_score else 0.0)
        )
        mined_ids[qid] = mine_topical(query, dense, sparse, k=mine_k, exclude_ids=needle_ids)
    return RetrievalSignals(dense_rank, sparse_rank, cand_scores, positive_scores, mined_ids)


def write_task_json(task: NiahTask, path: Path) -> None:
    """Persist a ``NiahTask`` as JSON (distractor text lives in ``corpus``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "corpus": task.corpus,
        "queries": task.queries,
        "qrels": task.qrels,
        "examples": [
            {
                "query_id": e.query_id,
                "query": e.query,
                "needle_ids": e.needle_ids,
                "distractors": [
                    {
                        "doc_id": d.doc_id,
                        "source": d.source,
                        "parent_needle_id": d.parent_needle_id,
                    }
                    for d in e.distractors
                ],
            }
            for e in task.examples
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_niah_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m eval.build_niah_task",
        description="Build a hard NIAH task (Sources A + C) from a benchmark.",
    )
    p.add_argument("--dataset", default="nq")
    p.add_argument("--split", default="dev")
    p.add_argument("--max-queries", type=int, default=200, dest="max_queries")
    p.add_argument("--max-docs", type=int, default=200_000, dest="max_docs")
    p.add_argument(
        "--sparse-arm", default="splade", choices=["splade", "bm25"], dest="sparse_arm"
    )
    p.add_argument(
        "--top-n", type=int, default=200, dest="top_n",
        help="Retrieval depth used to derive ranks/scores (default: %(default)s).",
    )
    p.add_argument(
        "--mine-k", type=int, default=20, dest="mine_k",
        help="Source-C mined candidates per retriever arm (default: %(default)s).",
    )
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--rank-threshold", type=int, default=10, dest="rank_threshold")
    p.add_argument(
        "--gate-k", type=int, default=10, dest="gate_k",
        help="recall@k cut-off for the hardness gate (default: %(default)s).",
    )
    p.add_argument(
        "--index-type", default="hnsw", choices=["flat", "hnsw", "ivf"], dest="index_type"
    )
    p.add_argument("--out", type=Path, default=Path("results/niah_task.json"))
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """Load a benchmark, build a hard NIAH task from real retrievers, gate, persist."""
    args = _parse_niah_args(argv)
    # Heavy deps imported lazily so unit tests of the logic above stay light.
    from eval.benchmarks.loader import load_benchmark
    from eval.run_benchmark import BenchmarkConfig, _build_retrievers
    from src.llm_client import LLMClient
    from src.niah.hardness_gate import gate_report, recall_at_k

    data = load_benchmark(
        args.dataset, split=args.split,
        max_queries=args.max_queries, max_docs=args.max_docs,
    )
    config = BenchmarkConfig(
        dataset=args.dataset,
        split=args.split,
        retrievers=["granite_dense", args.sparse_arm],
        k_values=[args.top_n],          # retrieve deep enough for ranks + the gate
        index_type=args.index_type,
        chunk_unit="token",
    )
    retrievers = _build_retrievers(config, data)
    dense, sparse = retrievers["granite_dense"], retrievers[args.sparse_arm]

    signals = compute_retrieval_signals(
        dense, sparse, data.queries, data.qrels, args.top_n, args.mine_k
    )
    llm = LLMClient()  # one client, reused as the generator AND the answerability judge
    task = build_task(
        corpus=data.corpus, queries=data.queries, qrels=data.qrels,
        answers=data.answers or {}, llm=llm, judge=llm,
        dense_rank=signals.dense_rank, sparse_rank=signals.sparse_rank,
        cand_scores=signals.cand_scores, positive_scores=signals.positive_scores,
        margin=args.margin, rank_threshold=args.rank_threshold,
        mined_ids=signals.mined_ids,
    )

    # Hardness gate (CONSERVATIVE): the dense baseline is indexed over the ORIGINAL
    # corpus, which already contains the mined Source-C distractors. The injected
    # Source-A counterfactuals are not in that index, but adding them could only
    # push the needle further down — so "not saturated" here guarantees the built
    # task is at least as hard. (A full gate would re-index over task.corpus.)
    per_q = {}
    for qid, query in task.queries.items():
        needles = {d for d, rel in task.qrels.get(qid, {}).items() if rel > 0}
        ranked = [c.doc_id for c in dense.retrieve(query)[: args.gate_k]]
        per_q[qid] = recall_at_k(ranked, needles, args.gate_k)
    report = gate_report(per_q)

    write_task_json(task, args.out)
    n_distractors = sum(len(e.distractors) for e in task.examples)
    print(
        f"Built NIAH task: {len(task.queries)} queries, {n_distractors} distractors, "
        f"{len(task.corpus)} corpus docs -> {args.out}"
    )
    print(
        f"Hardness gate (dense recall@{args.gate_k}, conservative): "
        f"mean_recall={report['mean_recall']:.3f} saturated={report['saturated']} "
        f"passes_gate={report['passes_gate']}"
    )
    if report["saturated"]:
        print(
            "WARNING: baseline saturates the task -> increase distractor count/hardness "
            "(raise --mine-k, or add Source B) and rebuild."
        )


if __name__ == "__main__":
    main()
