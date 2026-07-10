"""Controlled NQ/DPR pilot for relevance-aware evidence selection."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

from src.retrieval.corroboration import is_valid_answer, normalize_answer


LABEL_GAINS = (0, 1, 3, 7, 15)
CORE_FEATURES = (
    "relevance_score",
    "relevance_normalized",
    "original_rank",
    "reciprocal_rank",
    "exact_vote_count",
    "vote_ratio",
    "parametric_agreement",
    "extraction_failure",
)
FULL_FEATURES = CORE_FEATURES + (
    "source_dedup_vote_count",
    "dominant_answer_agreement",
    "answer_length",
)


def utility_grade(*, source: str, relevance: int | None) -> int:
    """Map controlled-NIAH provenance to the canonical five-level utility label."""

    if source == "counterfactual":
        return 0
    if source == "generative_non_answer":
        return 2
    if source != "dpr":
        raise ValueError(f"Unknown candidate source {source!r}")
    mapping = {2: 4, 1: 3, 0: 1, -1: 2}
    if relevance not in mapping:
        raise ValueError(f"Unsupported DPR relevance {relevance!r}")
    return mapping[relevance]


def minmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [0.0] * len(values)
    return [(value - low) / (high - low) for value in values]


def rank_candidates(rows: Sequence[Mapping[str, object]], score_key: str) -> list[dict[str, object]]:
    """Rank by score with the preregistered deterministic tie-break."""

    return [
        dict(row)
        for row in sorted(
            rows,
            key=lambda row: (
                -float(row[score_key]),
                int(row.get("original_rank", 10**9)),
                str(row["candidate_id"]),
            ),
        )
    ]


def add_group_features(
    rows: Sequence[Mapping[str, object]], *, parametric_answer: str | None
) -> list[dict[str, object]]:
    """Add cost-matched corroboration and cheap reliability features."""

    answers = [str(row.get("extracted_answer", "")) for row in rows]
    normalized = [normalize_answer(answer) if is_valid_answer(answer) else None for answer in answers]
    parametric = (
        normalize_answer(parametric_answer)
        if parametric_answer and is_valid_answer(parametric_answer)
        else None
    )
    valid_count = sum(answer is not None for answer in normalized)
    frequency = Counter(answer for answer in normalized if answer is not None)
    dominant = max(frequency, key=lambda answer: (frequency[answer], answer)) if frequency else None
    relevance = [float(row["relevance_score"]) for row in rows]
    relevance_norm = minmax(relevance)
    output: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        item = dict(row)
        answer = normalized[index]
        exact_votes = 0
        source_votes: set[str] = set()
        if answer is not None:
            own_parent = str(row.get("source_parent_id", ""))
            for other_index, other_answer in enumerate(normalized):
                if other_index == index or other_answer != answer:
                    continue
                exact_votes += 1
                other_parent = str(rows[other_index].get("source_parent_id", ""))
                if other_parent != own_parent:
                    source_votes.add(other_parent)
            if parametric == answer:
                exact_votes += 1
        item.update(
            {
                "relevance_normalized": relevance_norm[index],
                "reciprocal_rank": 1.0 / max(1, int(row["original_rank"])),
                "exact_vote_count": float(exact_votes),
                "vote_ratio": float(exact_votes) / max(1, valid_count),
                "parametric_agreement": float(answer is not None and answer == parametric),
                "extraction_failure": float(answer is None),
                "source_dedup_vote_count": float(
                    len(source_votes) + (1 if answer is not None and answer == parametric else 0)
                ),
                "dominant_answer_agreement": float(answer is not None and answer == dominant),
                "answer_length": float(len(answer or "")),
            }
        )
        output.append(item)
    return output


def _dcg(grades: Sequence[int]) -> float:
    return sum(LABEL_GAINS[grade] / math.log2(rank + 2) for rank, grade in enumerate(grades))


def selector_metrics(
    groups: Mapping[str, Sequence[Mapping[str, object]]], *, score_key: str, k: int = 10
) -> dict[str, float]:
    ndcgs: list[float] = []
    required_recalls: list[float] = []
    harmful = direct = selected_count = 0
    for rows in groups.values():
        ranked = rank_candidates(rows, score_key)
        selected = ranked[:k]
        grades = [int(row["utility_grade"]) for row in selected]
        ideal = sorted((int(row["utility_grade"]) for row in rows), reverse=True)[:k]
        denominator = _dcg(ideal)
        ndcgs.append(_dcg(grades) / denominator if denominator else 0.0)
        required_total = sum(int(row["utility_grade"]) >= 3 for row in rows)
        required_selected = sum(int(row["utility_grade"]) >= 3 for row in selected)
        required_recalls.append(required_selected / required_total if required_total else 1.0)
        harmful += sum(int(row["utility_grade"]) == 0 for row in selected)
        direct += sum(int(row["utility_grade"]) == 4 for row in selected)
        selected_count += len(selected)
    return {
        f"ndcg@{k}": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        f"required_evidence_recall@{k}": (
            sum(required_recalls) / len(required_recalls) if required_recalls else 0.0
        ),
        f"harmful_rate@{k}": harmful / selected_count if selected_count else 0.0,
        f"direct_support_precision@{k}": direct / selected_count if selected_count else 0.0,
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_dpr_pool(*, split_manifest: Path, dataset_id: str, out: Path) -> None:
    """Materialize the 1,100-query DPR candidate pools used by the pilot."""

    import ir_datasets

    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))["datasets"]["niah"]
    split_by_qid: dict[str, str] = {
        qid: "train" for qid in manifest["pilot_train_query_ids"]
    }
    needle_by_qid: dict[str, str] = {}
    parent_by_qid: dict[str, str] = {}
    for split in ("train", "dev", "test"):
        for row in manifest["splits"][split]:
            qid = str(row["query_id"])
            needle_by_qid[qid] = str(row["needle_doc_id"])
            parent_by_qid[qid] = str(row["parent_page_id"])
            if split != "train":
                split_by_qid[qid] = split
    selected = set(split_by_qid)
    dataset = ir_datasets.load(dataset_id)
    questions: dict[str, str] = {}
    aliases: dict[str, list[str]] = {}
    for query in dataset.queries_iter():
        qid = str(query.query_id)
        if qid in selected:
            questions[qid] = str(query.text)
            aliases[qid] = [str(answer) for answer in query.answers]
    qrels: dict[str, list[tuple[str, int]]] = {qid: [] for qid in selected}
    needed_docs: set[str] = set()
    for qrel in dataset.qrels_iter():
        qid = str(qrel.query_id)
        if qid not in selected:
            continue
        doc_id = str(qrel.doc_id)
        qrels[qid].append((doc_id, int(qrel.relevance)))
        needed_docs.add(doc_id)
    documents: dict[str, tuple[str, str]] = {}
    for document in dataset.docs_iter():
        doc_id = str(document.doc_id)
        if doc_id in needed_docs:
            documents[doc_id] = (str(document.title), str(document.text))
            if len(documents) == len(needed_docs):
                break
    missing_docs = needed_docs.difference(documents)
    if missing_docs:
        raise ValueError(f"DPR pool is missing {len(missing_docs)} candidate documents")
    rows: list[dict[str, object]] = []
    for qid in sorted(selected):
        candidates = [
            {
                "candidate_id": doc_id,
                "text": documents[doc_id][1],
                "title": documents[doc_id][0],
                "source_parent_id": re.sub(r"\s+", " ", documents[doc_id][0]).strip().casefold(),
                "source": "dpr",
                "dpr_relevance": relevance,
                "dpr_rank": rank,
                "utility_grade": utility_grade(source="dpr", relevance=relevance),
            }
            for rank, (doc_id, relevance) in enumerate(qrels[qid], start=1)
        ]
        rows.append(
            {
                "query_id": qid,
                "split": split_by_qid[qid],
                "question": questions[qid],
                "answers": aliases[qid],
                "needle_doc_id": needle_by_qid[qid],
                "needle_parent_id": parent_by_qid[qid],
                "candidates": candidates,
            }
        )
    _write_jsonl(out, rows)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--split-manifest", type=Path, required=True)
    prepare.add_argument("--dataset-id", default="dpr-w100/natural-questions/dev")
    prepare.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "prepare":
        prepare_dpr_pool(
            split_manifest=args.split_manifest, dataset_id=args.dataset_id, out=args.out
        )


if __name__ == "__main__":
    main()
