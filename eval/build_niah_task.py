# eval/build_niah_task.py
"""Build a hard NIAH task from a source benchmark (spec §5.1).

For each query: gold docs (qrels ``rel > 0``) = needles; build distractors from
**Source A** (a counterfactual per needle) and **Source C** (mined topical
negatives), keep those passing the §5.1 filters, inject the freshly-generated
ones, and return a ``NiahTask`` whose qrels still mark ONLY the needles relevant.

``build_task`` takes precomputed *per-query* ranks/scores + injected models so it
is unit testable; ``main`` (below) computes them from the real retrievers.
"""
from __future__ import annotations

from typing import Dict, List

from src.niah.assembly import inject
from src.niah.counterfactual import make_counterfactual
from src.niah.filters import answers_query, keep_distractor, passes_margin
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

    Parameters of note:
    - ``positive_scores`` — the per-query anchor (the needle's own retrieval score);
      distractors must score below it by ``margin`` (NV-Retriever positive-anchor).
    - ``mined_ids`` — Source-C candidate doc ids per query. They are already in the
      corpus and were surfaced by the retrievers, so Filter 2 (hardness) is implicit
      and only Filter 1 (margin + not-answering) is applied; they are recorded as
      distractors but not re-injected.
    """
    mined = mined_ids or {}
    examples: List[NiahExample] = []
    injected: List[Distractor] = []

    for qid, query in queries.items():
        needle_ids = [doc_id for doc_id, rel in qrels.get(qid, {}).items() if rel > 0]
        gold_answer = (answers.get(qid) or [None])[0]
        pos = positive_scores.get(qid, 0.0)
        ex = NiahExample(query_id=qid, query=query, needle_ids=needle_ids)

        # Source A — counterfactual, one per needle (freshly generated -> injected).
        if gold_answer:
            for nid in needle_ids:
                cand_id = f"{qid}__{nid}__cf0"
                try:
                    text = make_counterfactual(corpus[nid], gold_answer, llm)
                except (ValueError, KeyError):
                    continue  # answer absent / echo / no-op swap -> skip this needle
                if keep_distractor(
                    cand_score=cand_scores.get(qid, {}).get(cand_id, 0.0),
                    positive_score=pos,
                    margin=margin,
                    cand_text=text,
                    query=query,
                    judge=judge,
                    cand_id=cand_id,
                    dense_rank=dense_rank.get(qid, {}),
                    sparse_rank=sparse_rank.get(qid, {}),
                    rank_threshold=rank_threshold,
                ):
                    d = Distractor(cand_id, text, SOURCE_COUNTERFACTUAL, nid)
                    ex.distractors.append(d)
                    injected.append(d)

        # Source C — mined topical negatives: already in the corpus with real ranks,
        # so Filter 2 is implicit; apply only Filter 1 (margin + not-answering).
        needle_set = set(needle_ids)
        for mined_id in mined.get(qid, []):
            if mined_id in needle_set or mined_id not in corpus:
                continue
            if passes_margin(
                cand_scores.get(qid, {}).get(mined_id, 0.0), pos, margin
            ) and not answers_query(corpus[mined_id], query, judge):
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
