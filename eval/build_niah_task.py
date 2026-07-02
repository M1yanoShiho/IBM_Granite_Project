# eval/build_niah_task.py
"""Build a hard NIAH task from a source benchmark (spec §5.1, MVP = Source A).

For each query: gold docs = needles; make one counterfactual distractor per needle
(Source A), keep those passing Filter 1 + Filter 2, inject them, and return a
``NiahTask`` whose qrels still mark ONLY the needles relevant.

``build_task`` takes precomputed ranks/scores + injected models so it is unit
testable; ``main`` (below) computes them from the real retrievers.
"""
from __future__ import annotations

from typing import Dict, List

from src.niah.assembly import inject
from src.niah.counterfactual import make_counterfactual
from src.niah.filters import keep_distractor
from src.niah.types import SOURCE_COUNTERFACTUAL, Distractor, NiahExample, NiahTask


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
    positive_score: float,
    margin: float,
    rank_threshold: int,
) -> NiahTask:
    examples: List[NiahExample] = []
    all_distractors: List[Distractor] = []

    for qid, query in queries.items():
        needle_ids = list(qrels.get(qid, {}))
        gold_answer = (answers.get(qid) or [None])[0]
        ex = NiahExample(query_id=qid, query=query, needle_ids=needle_ids)
        if gold_answer:
            for nid in needle_ids:
                cand_id = f"{qid}__{nid}__cf0"
                try:
                    text = make_counterfactual(corpus[nid], gold_answer, llm)
                except (ValueError, KeyError):
                    continue  # skip needles we cannot safely counterfactual
                score = cand_scores.get(qid, {}).get(cand_id, 0.0)
                if keep_distractor(
                    cand_score=score, positive_score=positive_score, margin=margin,
                    cand_text=text, query=query, judge=judge,
                    cand_id=cand_id,
                    dense_rank=dense_rank.get(qid, {}),
                    sparse_rank=sparse_rank.get(qid, {}),
                    rank_threshold=rank_threshold,
                ):
                    d = Distractor(cand_id, text, SOURCE_COUNTERFACTUAL, nid)
                    ex.distractors.append(d)
                    all_distractors.append(d)
        examples.append(ex)

    return NiahTask(
        corpus=inject(corpus, all_distractors),
        queries=dict(queries),
        qrels={q: dict(r) for q, r in qrels.items()},
        examples=examples,
    )
